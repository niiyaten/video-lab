"""Manual optical-flow point placement and optional local corner snapping."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image

from .video_io import decoded_frames, open_capture


def read_exact_frame(path: str | Path, frame_index: int) -> np.ndarray:
    """Read the N-th decoded frame sequentially, avoiding unreliable AVI seeking."""
    if frame_index < 0:
        raise ValueError("フレーム番号は0以上です。")
    cap = open_capture(path)
    try:
        for i, frame in enumerate(decoded_frames(cap)):
            if i == frame_index:
                return frame
    finally:
        cap.release()
    raise ValueError(f"フレーム {frame_index} を読み取れません。")


def map_click_to_frame(event: Mapping, source_width: int, source_height: int) -> tuple[float, float]:
    """Convert rendered-image click coordinates back to original-frame pixels."""
    display_width = float(event.get("width", 0))
    display_height = float(event.get("height", 0))
    if not all(np.isfinite(v) and v > 0 for v in (display_width, display_height, source_width, source_height)):
        raise ValueError("画像サイズを取得できません。")
    x, y = float(event["x"]), float(event["y"])
    if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0 or x >= display_width or y >= display_height:
        raise ValueError("クリック位置が画像の範囲外です。")
    return (min(source_width - 1.0, (x + 0.5) * source_width / display_width - 0.5),
            min(source_height - 1.0, (y + 0.5) * source_height / display_height - 0.5))


def snap_to_corner(
    image_bgr: np.ndarray,
    requested_xy: tuple[float, float],
    radius_px: int = 15,
    quality_level: float = 0.01,
) -> tuple[float, float] | None:
    """Find the strongest trackable Shi-Tomasi corner near a manual click.

    Returns None if no suitable feature is found; does not silently move the
    selection to a different area or assume a textureless pixel is trackable.
    """
    if radius_px < 1 or not (0 < quality_level <= 1):
        raise ValueError("探索半径・特徴点品質しきい値を確認してください。")
    height, width = image_bgr.shape[:2]
    x, y = requested_xy
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("クリック位置が画像の範囲外です。")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (round(x), round(y)), radius_px, 255, -1)
    result = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=quality_level,
                                      minDistance=1, mask=mask, blockSize=3)
    if result is None:
        return None
    px, py = (float(v) for v in result.reshape(-1, 2)[0])
    if np.hypot(px - x, py - y) > radius_px + 0.5:
        return None
    return px, py


def make_point_picker_image(
    image_bgr: np.ndarray,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    max_width: int = 850,
) -> tuple[Image.Image, int]:
    """Return annotated PIL image and its pixel width (for fixed-size display)."""
    if max_width < 20:
        raise ValueError("表示幅が小さすぎます。")
    height, width = image_bgr.shape[:2]
    factor = min(1.0, max_width / width)
    out_width = max(1, round(width * factor))
    out_height = max(1, round(height * factor))
    annotated = cv2.resize(image_bgr, (out_width, out_height), interpolation=cv2.INTER_AREA)
    for pid, (x, y) in enumerate(points, start=1):
        at = (round((x + .5) * factor - .5), round((y + .5) * factor - .5))
        cv2.circle(annotated, at, 5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(annotated, at, 4, (0, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, str(pid), (at[0] + 9, at[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, str(pid), (at[0] + 9, at[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)), out_width
