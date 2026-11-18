import csv

import cv2
import numpy as np

from videolab.fps import change_playback_fps
from videolab.optical_flow import FlowOptions, run_sparse_flow
from videolab.video_io import read_info


def _generate_input(path, *, fps=20.0, n_frames=16, size=(128, 96)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    noise = rng.integers(40, 140, size=(25, 25), dtype=np.uint8)
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[25:50, 20 + i:45 + i, :] = noise[:, :, None]
        writer.write(frame)
    writer.release()


def test_fps_keeps_all_frames_and_changes_duration(tmp_path):
    source = tmp_path / "input.avi"
    dest = tmp_path / "output.mp4"
    _generate_input(source, fps=20, n_frames=16)
    count = change_playback_fps(source, dest, target_fps=8.0)
    info = read_info(dest)
    assert count == 16
    assert info.frame_count == 16
    assert abs(info.fps - 8.0) < 0.05
    assert abs(info.duration_s - 2.0) < 0.05


def test_sparse_flow_outputs_csv_and_video(tmp_path):
    source = tmp_path / "input.avi"
    dest = tmp_path / "flow.mp4"
    csv_path = tmp_path / "flow.csv"
    _generate_input(source, fps=20, n_frames=16)
    frame_count, points = run_sparse_flow(source, dest, csv_path, FlowOptions(max_points=40, min_distance_px=5, output_fps=10.0))
    assert frame_count == 16
    assert read_info(dest).frame_count == 16
    assert abs(read_info(dest).fps - 10.0) < 0.05
    assert points > 0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert any(row["status"] == "1" and row["speed_px_s"] != "" for row in rows)
    assert min(int(row["point_id"]) for row in rows) >= 1


def test_roi_validation(tmp_path):
    source = tmp_path / "input.avi"
    _generate_input(source)
    try:
        run_sparse_flow(source, tmp_path / "bad.mp4", tmp_path / "bad.csv", FlowOptions(roi=(0, 0, 500, 20)))
    except ValueError as exc:
        assert "ROI" in str(exc)
    else:
        raise AssertionError("invalid ROI must fail")
