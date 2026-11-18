"""Progress callbacks are based on real units of work rather than elapsed-time guesses."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from videolab.edge_counter import EdgeOptions, analyze_edges
from videolab.fps import change_playback_fps
from videolab.preview import make_browser_preview
from videolab.progress import snapshot
from videolab.video_io import read_info


def _make_video(path: Path, count: int = 28, fps: float = 120):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for n in range(count):
        writer.write(np.full((48, 64, 3), n * 3, np.uint8))
    writer.release()


def test_progress_snapshot_remaining_estimate():
    s = snapshot(25, 100, 10.0)
    assert s.fraction == pytest.approx(0.25)
    assert s.remaining_s == pytest.approx(30.0)
    assert snapshot(10, 0, 10).fraction is None
    assert snapshot(10, 0, 10).remaining_s is None
    assert snapshot(200, 100, 10).fraction == pytest.approx(1)
    assert snapshot(0, 100, 10).remaining_s is None


def test_preview_reports_real_ffmpeg_frames(tmp_path):
    src, dst = tmp_path/'source.avi', tmp_path/'converted.mp4'
    _make_video(src)
    events = []
    result = make_browser_preview(src, dst, max_frames=15, on_progress=lambda n, t: events.append((n, t)))
    assert result.frame_count == 15
    assert len(events) > 0
    assert events[-1] == (15, 15)
    assert all(0 <= n <= 15 and t == 15 for n, t in events)
    assert all(events[j][0] <= events[j + 1][0] for j in range(len(events)-1))
    assert read_info(dst).fps == pytest.approx(120)


def test_fps_progress_includes_skipped_frames(tmp_path):
    src, dst = tmp_path/'source.avi', tmp_path/'trim.mp4'
    _make_video(src)
    events = []
    exported = change_playback_fps(src, dst, target_fps=30, start_frame=20,
                                   end_frame=27, on_scan_progress=lambda n, t: events.append((n, t)))
    assert exported == 7
    assert (20, 27) in events
    assert events[-1] == (27, 27)
    assert all(t == 27 for _,t in events)
    assert read_info(dst).frame_count == 7


def test_edge_progress_stages_monotonic():
    frame = np.zeros((100, 100, 3), np.uint8)
    cv2.rectangle(frame, (12, 15), (72, 68), (255, 255, 255), -1)
    events = []
    result = analyze_edges(frame, EdgeOptions(), on_progress=lambda n, total, name: events.append((n, total, name)))
    assert len(result.stats) > 0
    fractions = [n / total for n, total, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1
    assert any('輪郭集計' in name for _, _, name in events)
