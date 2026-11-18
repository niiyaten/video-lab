"""Frame/time range exports, boundaries, and FPS without an application cap."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from videolab.fps import change_playback_fps
from videolab.segments import seconds_to_interval, validate_frame_interval
from videolab.video_io import open_capture, read_info


def make_video(path, fps=120., count=24):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for k in range(count):
        frame = np.full((48, 64, 3), k * 10, np.uint8)
        writer.write(frame)
    writer.release()


def read_mean_frames(path):
    cap = open_capture(path)
    try:
        values = []
        while True:
            ok, frame = cap.read()
            if not ok:
                return values
            values.append(float(frame.mean()))
    finally:
        cap.release()


def test_export_selected_interval_exact_frames_and_duration(tmp_path):
    input_path = tmp_path / "source.avi"
    output_path = tmp_path / "segment.mp4"
    make_video(input_path, 120., 24)
    calls = []
    count = change_playback_fps(input_path, output_path, 30., lambda n, t: calls.append((n, t)),
                                start_frame=6, end_frame=15)
    meta = read_info(output_path)
    assert count == meta.frame_count == 9
    assert meta.fps == pytest.approx(30., abs=0.05)
    assert meta.duration_s == pytest.approx(0.3, abs=0.01)
    vals = read_mean_frames(output_path)
    assert len(vals) == 9
    assert vals[0] == pytest.approx(60., abs=5.)
    assert vals[-1] == pytest.approx(140., abs=5.)
    assert calls[-1] == (9, 9)


def test_single_selected_frame(tmp_path):
    source, out = tmp_path / "src.avi", tmp_path / "single.mp4"
    make_video(source)
    count = change_playback_fps(source, out, 120., start_frame=23, end_frame=24)
    assert count == 1
    assert read_info(out).frame_count == 1


def test_no_app_fps_ceiling_5000(tmp_path):
    source, out = tmp_path / "src.avi", tmp_path / "fast.mp4"
    make_video(source)
    count = change_playback_fps(source, out, 5000.0, start_frame=2, end_frame=8)
    assert count == 6
    assert read_info(out).frame_count == 6
    assert read_info(out).fps == pytest.approx(5000., abs=0.1)


def test_seconds_to_frame_range_cfr():
    assert seconds_to_interval(2., 4., 120., 1200) == validate_frame_interval(240, 480, 1200)
    assert seconds_to_interval(0., 10., 120., 1200).count == 1200
    assert seconds_to_interval(0.041666666666666664, 0.125, 120., 1200).count == 10
    assert seconds_to_interval(.001, .012, 120., 1200) == validate_frame_interval(1, 2, 1200)
    assert seconds_to_interval(1 / 119.88, 3 / 119.88, 119.88, 100) == validate_frame_interval(1, 3, 100)


@pytest.mark.parametrize("start,stop,total", [(-1, 2, 10), (2, 2, 10), (5, 4, 10), (0, 11, 10)])
def test_invalid_frame_ranges(start, stop, total):
    with pytest.raises(ValueError):
        validate_frame_interval(start, stop, total)


def test_invalid_export_does_not_create_output(tmp_path):
    source, out = tmp_path / "src.avi", tmp_path / "bad.mp4"
    make_video(source)
    with pytest.raises(ValueError):
        change_playback_fps(source, out, 30., start_frame=20, end_frame=30)
    assert not out.exists()
    for fps in [0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            change_playback_fps(source, out, fps)
    assert not out.exists()


def test_seconds_bad_intervals():
    for start, stop in [(-1., 2.), (1., 1.), (1., 0.), (0., 10.1), (math.nan, 1.)]:
        with pytest.raises(ValueError):
            seconds_to_interval(start, stop, 120., 1200)
