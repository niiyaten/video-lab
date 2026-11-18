"""AVI->MP4 compatibility conversions must NOT silently change capture FPS."""
from pathlib import Path
import json
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from videolab.preview import make_browser_preview
from videolab.video_io import read_info


def create_mjpeg_avi(path: Path, fps: float, count: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (64, 48))
    assert writer.isOpened()
    for n in range(count):
        image = np.full((48, 64, 3), (n * 4) % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()


@pytest.mark.parametrize('capture_fps,frame_count', [(120., 32), (60., 25), (800., 28)])
def test_default_conversion_preserves_fps_frame_count_and_duration(tmp_path, capture_fps, frame_count):
    src, dst = tmp_path/'source.avi', tmp_path/'preview.mp4'
    create_mjpeg_avi(src, capture_fps, frame_count)
    original_bytes = src.read_bytes()
    source_info = read_info(src)

    result = make_browser_preview(src, dst)
    output_info = read_info(dst)
    assert result.path == dst
    assert result.playback_fps == pytest.approx(source_info.fps, rel=0.0001)
    assert result.frame_count == source_info.frame_count == output_info.frame_count == frame_count
    assert output_info.fps == pytest.approx(source_info.fps, rel=0.0001)
    assert output_info.duration_s == pytest.approx(source_info.duration_s, abs=0.0001)
    assert src.read_bytes() == original_bytes


def test_partial_conversion_truncates_only_not_fps(tmp_path):
    src, dst = tmp_path/'source.avi', tmp_path/'preview.mp4'
    create_mjpeg_avi(src, 120., 25)
    result = make_browser_preview(src, dst, max_frames=10)
    assert result.frame_count == 10
    assert result.playback_fps == pytest.approx(120.)
    assert read_info(dst).duration_s == pytest.approx(10 / 120., abs=0.0001)


def test_source_119_88_fps_retained_without_rounding_to_whole_number(tmp_path):
    src, dst = tmp_path/'source.avi', tmp_path/'preview.mp4'
    create_mjpeg_avi(src, 120000 / 1001, 45)
    src_fps = read_info(src).fps
    assert src_fps == pytest.approx(120000 / 1001, abs=.05)
    make_browser_preview(src, dst)
    assert read_info(dst).fps == pytest.approx(src_fps, abs=0.005)


def test_metadata_and_audio_preserved_where_mp4_supports(tmp_path):
    input_avi, src, dst = tmp_path/'vid.avi', tmp_path/'source.avi', tmp_path/'preview.mp4'
    create_mjpeg_avi(input_avi, 120., 48)
    # Avi with audio and an MP4-compatible global comment metadata tag.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(input_avi),
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=0.4',
        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'pcm_s16le',
        '-metadata', 'comment=VideoLab_test', '-shortest', str(src),
    ], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    make_browser_preview(src, dst)
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format_tags=comment:stream=codec_type',
        '-of', 'json', str(dst),
    ], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    info = json.loads(probe.stdout)
    assert info.get('format', {}).get('tags', {}).get('comment') == 'VideoLab_test'
    assert {s['codec_type'] for s in info['streams']} == {'video', 'audio'}


def test_preview_errors(tmp_path):
    src, dst = tmp_path/'source.avi', tmp_path/'preview.mp4'
    src.write_bytes(b'not an AVI')
    with pytest.raises(RuntimeError):
        make_browser_preview(src, dst)
    assert not dst.exists()
    with pytest.raises(ValueError):
        make_browser_preview(src, src)
    with pytest.raises(ValueError):
        make_browser_preview(src, dst, max_frames=-1)
