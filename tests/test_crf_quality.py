"""CRF must be user-selectable for every MP4 re-encoding route."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np
import pytest

from videolab.encoding import DEFAULT_CRF, validate_crf
from videolab.fps_fast import export_fps_optimized
from videolab.fps_seek import export_fps_with_seek
from videolab.preview import make_browser_preview
from videolab.video_io import read_info


@pytest.fixture
def noisy_avi(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"),
                             120, (128, 96))
    assert writer.isOpened()
    rng = np.random.default_rng(2026)
    for frame_no in range(48):
        # Non-flat footage makes the quality/size tradeoff easy to detect.
        frame = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        cv2.putText(frame, str(frame_no), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    return source


def test_crf_validator_and_defaults():
    assert DEFAULT_CRF == 23
    assert validate_crf(0) == 0
    assert validate_crf(18) == 18
    assert validate_crf(23) == 23
    assert validate_crf(51) == 51


@pytest.mark.parametrize("value", [-1, 52, 2.5, 23.0, True, "23", None])
def test_bad_crf_rejected(value):
    with pytest.raises(ValueError, match="CRF"):
        validate_crf(value)


def test_avi_preview_crf_changes_compression_but_preserves_fps_frames(noisy_avi, tmp_path):
    original = noisy_avi.read_bytes()
    low = tmp_path / "preview-crf18.mp4"
    high = tmp_path / "preview-crf35.mp4"
    a = make_browser_preview(noisy_avi, low, crf=18)
    b = make_browser_preview(noisy_avi, high, crf=35)
    assert a.frame_count == b.frame_count == 48
    assert read_info(low).fps == pytest.approx(120)
    assert read_info(high).fps == pytest.approx(120)
    assert low.stat().st_size > high.stat().st_size
    assert noisy_avi.read_bytes() == original


def test_sequential_fps_change_respects_crf(noisy_avi, tmp_path):
    low, high = tmp_path / "fps-crf18.mp4", tmp_path / "fps-crf35.mp4"
    a = export_fps_optimized(
        noisy_avi, low, 30,
        start_frame=8, end_frame=40, preset="ultrafast", crf=18,
    )
    b = export_fps_optimized(
        noisy_avi, high, 30,
        start_frame=8, end_frame=40, preset="ultrafast", crf=35,
    )
    assert a.method == b.method == "transcode"
    assert a.frame_count == b.frame_count == 32
    assert read_info(low).fps == pytest.approx(30)
    assert low.stat().st_size > high.stat().st_size


def test_seek_or_fallback_respects_crf(noisy_avi, tmp_path):
    low, high = tmp_path / "seek-crf18.mp4", tmp_path / "seek-crf35.mp4"
    a = export_fps_with_seek(
        noisy_avi, low, 30,
        start_frame=18, end_frame=40, preset="ultrafast", crf=18,
    )
    b = export_fps_with_seek(
        noisy_avi, high, 30,
        start_frame=18, end_frame=40, preset="ultrafast", crf=35,
    )
    assert a.method in {"seek_transcode", "transcode"}
    assert b.method in {"seek_transcode", "transcode"}
    assert a.frame_count == b.frame_count == 22
    assert read_info(low).frame_count == read_info(high).frame_count == 22
    assert low.stat().st_size > high.stat().st_size


def test_invalid_crf_must_not_create_export(noisy_avi, tmp_path):
    preview = tmp_path / "preview.mp4"
    sequential = tmp_path / "sequential.mp4"
    seek = tmp_path / "seek.mp4"
    with pytest.raises(ValueError, match="CRF"):
        make_browser_preview(noisy_avi, preview, crf=52)
    with pytest.raises(ValueError, match="CRF"):
        export_fps_optimized(noisy_avi, sequential, 30, crf=-1)
    with pytest.raises(ValueError, match="CRF"):
        export_fps_with_seek(noisy_avi, seek, 30, crf=52)
    assert not preview.exists()
    assert not sequential.exists()
    assert not seek.exists()
