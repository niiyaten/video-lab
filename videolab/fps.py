"""Export a selected range of decoded frames at a chosen playback FPS."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from .segments import validate_frame_interval
from .video_io import Mp4Writer, decoded_frames, open_capture, read_info

Progress = Callable[[int, int], None]


def change_playback_fps(
    source: str | Path,
    destination: str | Path,
    target_fps: float,
    on_progress: Progress | None = None,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    """Save [start_frame, end_frame) at target_fps, keeping all selected frames.

    Frame indices are zero-based, the end is exclusive. No interpolation or
    thinning is performed. Source stays untouched; audio is omitted.
    The source is decoded sequentially to avoid inaccurate AVI random seeks.
    """
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("出力FPSは0より大きい有限の値を指定してください。")
    info = read_info(source)
    # A missing count still permits the original full-video export path.
    if info.frame_count > 0:
        stop = info.frame_count if end_frame is None else end_frame
        interval = validate_frame_interval(start_frame, stop, info.frame_count)
        expected = interval.count
    else:
        if start_frame != 0 or end_frame is not None:
            raise ValueError("フレーム数を取得できない動画では範囲指定を使用できません。")
        stop = None
        expected = 0
    cap = open_capture(source)
    count = 0
    try:
        with Mp4Writer(destination, info.width, info.height, target_fps) as writer:
            for original_frame, frame in enumerate(decoded_frames(cap)):
                if original_frame < start_frame:
                    continue
                if stop is not None and original_frame >= stop:
                    break
                writer.write(frame)
                count += 1
                if on_progress and (count % 15 == 0 or (expected and count == expected)):
                    on_progress(count, expected)
    finally:
        cap.release()
    if count == 0 or (expected and count != expected):
        Path(destination).unlink(missing_ok=True)
        raise ValueError(
            f"選択範囲を最後まで読み込めませんでした（要求 {expected:,} フレーム、取得 {count:,} フレーム）。"
        )
    if on_progress:
        on_progress(count, expected)
    return count
