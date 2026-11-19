"""Tests for verified keyframe seeking and safe fallbacks.

The fixture includes B-frame reordering and intra-frame MJPEG in AVI.
Full video data stays on disk; only frame metadata is indexed.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

import videolab.fps_seek as seeker
from videolab.fps_seek import (
    FastSeekUnavailable,
    _build_packet_index,
    _parse_boundary_pts,
    export_fps_with_seek,
)


@pytest.fixture(scope="module")
def sample_videos(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fast-seek")
    mp4 = tmp / "bframes.mp4"
    avi = tmp / "mjpeg.avi"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-y", "-f", "lavfi", "-i", "testsrc2=size=144x80:rate=120",
        "-frames:v", "420", "-c:v", "libx264", "-g", "48", "-bf", "2",
        "-pix_fmt", "yuv420p", str(mp4),
    ], check=True, capture_output=True)

    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"),
                             120, (144, 80))
    if not writer.isOpened():
        pytest.skip("MJPEG AVI writer unavailable")
    for i in range(420):
        frame = np.zeros((80, 144, 3), dtype=np.uint8)
        frame[:, :, 0] = i % 256
        cv2.putText(frame, str(i), (8, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return mp4, avi


def _inspect(path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    try:
        return (int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                float(capture.get(cv2.CAP_PROP_FPS)))
    finally:
        capture.release()


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe unavailable")
@pytest.mark.parametrize("kind", [0, 1])
def test_verified_late_range(sample_videos, tmp_path, kind):
    video = sample_videos[kind]
    out = tmp_path / f"range-{kind}.mp4"
    stages = []
    updates = []
    report = export_fps_with_seek(
        video, out, 30,
        start_frame=320, end_frame=380, preset="ultrafast",
        on_stage=stages.append,
        on_progress=lambda done, total: updates.append((done, total)),
    )
    assert report.method == "seek_transcode"
    assert report.frame_count == 60
    assert _inspect(out) == pytest.approx((60, 30))
    assert updates[-1] == (60, 60)
    assert any("キーフレーム" in s for s in stages)
    assert any("先頭・末尾フレーム" in s for s in stages)
    assert any("FPSを検証" in s for s in stages)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe unavailable")
def test_single_frame_near_end(sample_videos, tmp_path):
    video, _ = sample_videos
    out = tmp_path / "single.mp4"
    report = export_fps_with_seek(
        video, out, 120, start_frame=399, end_frame=400,
        preset="ultrafast",
    )
    assert report.method == "seek_transcode"
    assert report.frame_count == 1
    assert _inspect(out) == pytest.approx((1, 120))


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe unavailable")
def test_to_final_frame_fractional_rate(sample_videos, tmp_path):
    video, _ = sample_videos
    out = tmp_path / "end.mp4"
    report = export_fps_with_seek(
        video, out, 29.97, start_frame=320, end_frame=420,
        preset="ultrafast",
    )
    assert report.method == "seek_transcode"
    count, fps = _inspect(out)
    assert count == 100
    assert fps == pytest.approx(29.97, rel=0.00002)


def test_fallback_when_ffprobe_missing(sample_videos, tmp_path, monkeypatch):
    _, video = sample_videos
    monkeypatch.setattr(seeker, "_ffprobe_path", lambda: None)
    stages = []
    out = tmp_path / "fallback.mp4"
    report = export_fps_with_seek(
        video, out, 30, start_frame=300, end_frame=330,
        preset="ultrafast", on_stage=stages.append,
    )
    assert report.method == "transcode"
    assert report.frame_count == 30
    assert _inspect(out) == pytest.approx((30, 30))
    assert any("ffprobeが未導入" in stage for stage in stages)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe unavailable")
def test_bad_boundary_verification_falls_back(sample_videos, tmp_path, monkeypatch):
    video, _ = sample_videos
    monkeypatch.setattr(seeker, "_parse_boundary_pts", lambda _log, _n: (-7, -7))
    stages = []
    out = tmp_path / "verified-fallback.mp4"
    report = export_fps_with_seek(
        video, out, 30, start_frame=320, end_frame=380,
        preset="ultrafast", on_stage=stages.append,
    )
    assert report.method == "transcode"
    assert report.frame_count == 60
    assert _inspect(out) == pytest.approx((60, 30))
    assert any("高速シーク不可" in stage for stage in stages)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe unavailable")
def test_packet_index_matches_true_frame_order(sample_videos):
    video, _ = sample_videos
    st = video.stat()
    index = _build_packet_index(
        shutil.which("ffprobe"), str(video.resolve()), st.st_size,
        st.st_mtime_ns, 420,
    )
    assert len(index.pts) == 420
    assert all(left < right for left, right in zip(index.pts, index.pts[1:]))
    assert index.keyframes[0] == 0
    assert index.keyframes[-1] < 420


def test_boundary_parser_rejects_unverified_output():
    with pytest.raises(FastSeekUnavailable):
        _parse_boundary_pts("No frame timestamps in this output", 60)


def test_source_file_cannot_be_overwritten(sample_videos):
    video, _ = sample_videos
    with pytest.raises(ValueError):
        export_fps_with_seek(video, video, 30, start_frame=1, end_frame=20)
