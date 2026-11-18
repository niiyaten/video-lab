"""Sparse Lucas–Kanade tracking with manual seeds or auto corner detection."""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .video_io import Mp4Writer, decoded_frames, open_capture, read_info

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class FlowOptions:
    max_points: int = 150
    min_distance_px: int = 12
    quality_level: float = 0.01
    trail_length: int = 15
    redetect_interval: int = 30
    fb_threshold_px: float = 1.5
    show_ids: bool = False
    output_fps: float | None = None  # playback speed of the annotated MP4 only
    roi: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1; x1/y1 exclusive
    initial_points: tuple[tuple[float, float], ...] | None = None  # None: whole-frame auto-detection
    auto_replenish: bool | None = None  # None: auto only if initial_points is None
    start_frame: int = 0  # selected frame is counted from 0 in original recording


@dataclass
class _Track:
    xy: tuple[float, float]
    history: deque[tuple[int, int]] = field(default_factory=deque)


def run_sparse_flow(
    source: str | Path,
    destination_video: str | Path,
    destination_csv: str | Path,
    options: FlowOptions | None = None,
    on_progress: Progress | None = None,
) -> tuple[int, int]:
    """Return (processed frames, unique created point IDs).

    Pixel coordinates are image coordinates: +x right, +y down.
    Displacements are per input frame; speed_px_s uses metadata FPS.
    Lost IDs are not reused. Confidence/measurement quality is not guaranteed.
    """
    opt = options or FlowOptions()
    if opt.max_points < 1 or opt.min_distance_px < 1 or opt.trail_length < 1 or opt.redetect_interval < 1:
        raise ValueError("点数・最小間隔・軌跡長・点の追加間隔は1以上にしてください。")
    if opt.start_frame < 0:
        raise ValueError("開始フレームは0以上にしてください。")
    if not (0 < opt.quality_level <= 1) or opt.fb_threshold_px <= 0:
        raise ValueError("quality_levelと前後追跡誤差閾値を確認してください。")
    info = read_info(source)
    export_fps = float(opt.output_fps) if opt.output_fps is not None else info.fps
    if not np.isfinite(export_fps) or export_fps <= 0:
        raise ValueError("結果動画の再生FPSは0より大きい有限の値にしてください。")
    roi = opt.roi or (0, 0, info.width, info.height)
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= info.width and 0 <= y0 < y1 <= info.height):
        raise ValueError("ROIの座標は動画の範囲内で指定してください。")

    if opt.initial_points is not None:
        if not opt.initial_points:
            raise ValueError("手動モードでは追跡点を1点以上選んでください。")
        if len(opt.initial_points) > opt.max_points:
            raise ValueError("初期追跡点数が最大追跡点数を超えています。")
        for px, py in opt.initial_points:
            if not (np.isfinite(px) and np.isfinite(py) and x0 <= px < x1 and y0 <= py < y1):
                raise ValueError("手動追跡点が画像またはROIの範囲外です。")
    replenish = (opt.initial_points is None) if opt.auto_replenish is None else opt.auto_replenish
    cap = open_capture(source)
    tracks: dict[int, _Track] = {}
    next_id = 1
    prev_gray: np.ndarray | None = None
    frame_index = -1

    def add_points(gray: np.ndarray) -> list[int]:
        nonlocal next_id
        room = opt.max_points - len(tracks)
        if room <= 0:
            return []
        mask = np.zeros((info.height, info.width), np.uint8)
        mask[y0:y1, x0:x1] = 255
        for track in tracks.values():
            cx, cy = track.xy
            cv2.circle(mask, (round(cx), round(cy)), opt.min_distance_px, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=room, qualityLevel=opt.quality_level,
            minDistance=opt.min_distance_px, mask=mask, blockSize=7,
        )
        ids: list[int] = []
        if corners is not None:
            for corner in corners.reshape(-1, 2):
                px, py = float(corner[0]), float(corner[1])
                pid = next_id
                next_id += 1
                tracks[pid] = _Track(xy=(px, py), history=deque([(round(px), round(py))], maxlen=opt.trail_length))
                ids.append(pid)
        return ids

    video_path = Path(destination_video)
    csv_path = Path(destination_csv)
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer_csv = csv.writer(csv_file)
            writer_csv.writerow([
                "frame", "time_s", "point_id", "x_px", "y_px",
                "dx_px", "dy_px", "speed_px_s", "status",
            ])
            with Mp4Writer(video_path, info.width, info.height, export_fps) as writer_mp4:
                for frame_index, frame in enumerate(decoded_frames(cap)):
                    if frame_index < opt.start_frame:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if frame_index == opt.start_frame and opt.initial_points is not None:
                        for px, py in opt.initial_points:
                            tracks[next_id] = _Track(xy=(float(px), float(py)),
                                history=deque([(round(px), round(py))], maxlen=opt.trail_length))
                            writer_csv.writerow([frame_index, frame_index / info.fps, next_id,
                                                 px, py, "", "", "", 1])
                            next_id += 1
                    motion: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
                    if prev_gray is not None and tracks:
                        ids = list(tracks)
                        old_points = np.array([tracks[pid].xy for pid in ids], dtype=np.float32).reshape(-1, 1, 2)
                        kwargs = {
                            "winSize": (21, 21),
                            "maxLevel": 3,
                            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                        }
                        new_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_points, None, **kwargs)
                        backward_points, backward_status, _ = (None, None, None)
                        if new_points is not None:
                            backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                                gray, prev_gray, new_points, None, **kwargs,
                            )
                        for j, pid in enumerate(ids):
                            old_xy = tracks[pid].xy
                            valid = (
                                new_points is not None and backward_points is not None
                                and forward_status is not None and backward_status is not None
                                and forward_status[j, 0] == 1 and backward_status[j, 0] == 1
                            )
                            if valid:
                                xy_new = tuple(float(v) for v in new_points[j, 0])
                                fb_error = float(np.linalg.norm(old_points[j, 0] - backward_points[j, 0]))
                                valid = (
                                    np.isfinite(xy_new).all() and fb_error <= opt.fb_threshold_px
                                    and x0 <= xy_new[0] < x1 and y0 <= xy_new[1] < y1
                                )
                            if not valid:
                                writer_csv.writerow([frame_index, frame_index / info.fps, pid,
                                                     old_xy[0], old_xy[1], "", "", "", 0])
                                del tracks[pid]
                                continue
                            tracks[pid].xy = xy_new
                            tracks[pid].history.append((round(xy_new[0]), round(xy_new[1])))
                            motion[pid] = (old_xy, xy_new)
                            dx = xy_new[0] - old_xy[0]
                            dy = xy_new[1] - old_xy[1]
                            writer_csv.writerow([frame_index, frame_index / info.fps, pid,
                                                 xy_new[0], xy_new[1], dx, dy,
                                                 float(np.hypot(dx, dy) * info.fps), 1])

                    # Automatically replenish points; new points have no prior displacement.
                    if (opt.initial_points is None and frame_index == opt.start_frame) or (
                        replenish and (frame_index - opt.start_frame) > 0
                        and ((frame_index - opt.start_frame) % opt.redetect_interval == 0
                             or len(tracks) < opt.max_points // 4)
                    ):
                        added = add_points(gray)
                        for pid in added:
                            x, y = tracks[pid].xy
                            writer_csv.writerow([frame_index, frame_index / info.fps, pid,
                                                 x, y, "", "", "", 1])

                    # Overlays are drawn after optical flow computation.
                    for pid, track in tracks.items():
                        if len(track.history) >= 2:
                            pts = np.asarray(list(track.history), dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(frame, [pts], False, (0, 210, 255), 1, cv2.LINE_AA)
                        cx, cy = round(track.xy[0]), round(track.xy[1])
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
                        if pid in motion:
                            old_xy, new_xy = motion[pid]
                            cv2.arrowedLine(frame,
                                            (round(old_xy[0]), round(old_xy[1])),
                                            (round(new_xy[0]), round(new_xy[1])),
                                            (255, 150, 0), 1, cv2.LINE_AA, tipLength=0.3)
                        if opt.show_ids:
                            cv2.putText(frame, str(pid), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    if opt.roi:
                        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (255, 210, 0), 1)
                    writer_mp4.write(frame)
                    prev_gray = gray
                    if on_progress and ((frame_index + 1) % 15 == 0 or frame_index + 1 == info.frame_count):
                        on_progress(frame_index + 1, info.frame_count)
    except Exception:
        video_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()

    if frame_index < opt.start_frame:
        video_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise ValueError("読み取れるフレームがありませんでした。")
    if on_progress:
        on_progress(frame_index + 1, info.frame_count)
    return frame_index - opt.start_frame + 1, next_id - 1
