"""Video decoding, metadata, and browser-compatible MP4 encoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float | None:
        return self.frame_count / self.fps if self.frame_count > 0 else None


def open_capture(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("動画を開けません。形式やコーデックを確認してください。")
    return cap


def read_info(path: str | Path) -> VideoInfo:
    cap = open_capture(path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("動画の解像度を取得できません。")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("動画のFPS情報を取得できません。正しいFPSを持つ動画を指定してください。")
        return VideoInfo(width=width, height=height, fps=fps, frame_count=max(0, count))
    finally:
        cap.release()


def decoded_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


class Mp4Writer:
    """Write BGR frames to H.264 / yuv420p MP4 without storing them in RAM.

    Odd source dimensions are padded to even pixels, never interpolated.
    Audio is intentionally omitted in the first version.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("幅・高さ・出力FPSは正の値である必要があります。")
        self.path = Path(path)
        self.width = width
        self.height = height
        self._closed = False
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path),
            (width, height),
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            fps=float(fps),
            codec="libx264",
            quality=7,
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=[
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags", "+faststart",
            ],
        )
        self._writer.send(None)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("出力フレームはBGR uint8で元動画と同じ縦横サイズにしてください。")
        self._writer.send(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._writer.close()

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.close()
        finally:
            if exc_type is not None:
                self.path.unlink(missing_ok=True)
