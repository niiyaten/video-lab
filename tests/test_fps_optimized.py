"""Integration tests for native FFmpeg FPS export using synthetic 120 FPS video."""
from __future__ import annotations

import hashlib
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from videolab.fps_fast import export_fps_optimized


@pytest.fixture(scope='module')
def samples(tmp_path_factory):
    workdir = tmp_path_factory.mktemp('fps-export')
    avi = workdir / 'input.avi'
    mp4 = workdir / 'input.mp4'
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*'MJPG'), 120, (160, 96))
    if not writer.isOpened():
        pytest.skip('MJPG AVI writer not available in current OpenCV')
    for frame_idx in range(180):
        img = np.zeros((96, 160, 3), dtype=np.uint8)
        img[:, :, 0] = frame_idx
        cv2.putText(img, str(frame_idx), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
               '-y', '-i', str(avi), '-c:v', 'libx264', '-preset', 'ultrafast',
               '-pix_fmt', 'yuv420p', str(mp4)]
    subprocess.run(command, check=True)
    return avi, mp4


def _inspect(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    try:
        return cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def test_full_h264_uses_lossless_remux(samples, tmp_path):
    _, source = samples
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / 'full30.mp4'
    progress = []
    report = export_fps_optimized(source, output, 30, lambda a, b: progress.append((a,b)))
    assert report.method == 'stream_copy'
    fps, count = _inspect(output)
    assert count == report.frame_count == 180
    assert fps == pytest.approx(30, rel=0.00002)
    assert progress and progress[-1] == (180, 180)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_partial_range_uses_native_ffmpeg(samples, tmp_path):
    _, source = samples
    output = tmp_path / 'range11988.mp4'
    report = export_fps_optimized(source, output, 119.88, start_frame=25, end_frame=125)
    fps, count = _inspect(output)
    assert report.method == 'transcode'
    assert count == report.frame_count == 100
    assert fps == pytest.approx(119.88, rel=0.00002)


def test_mjpeg_avi_and_high_fps(samples, tmp_path):
    source, _ = samples
    output = tmp_path / 'high5000.mp4'
    report = export_fps_optimized(source, output, 5000, start_frame=25, end_frame=125)
    fps, count = _inspect(output)
    assert report.method == 'transcode'
    assert count == report.frame_count == 100
    assert fps == pytest.approx(5000, rel=0.00002)


def test_input_file_is_never_overwritten(samples):
    source, _ = samples
    with pytest.raises(ValueError):
        export_fps_optimized(source, source, 30)
