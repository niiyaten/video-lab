"""Reusable, in-memory OpenCV operations for factory image measurements.

Images use OpenCV's BGR order. This pure-Python API is intended for future
frame processing and does not depend on Streamlit or browser JavaScript.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourStats:
    count: int
    total_area_px: int
    largest_area_px: int


def _require_image(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a nonempty numpy.ndarray.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
        raise ValueError("Expected an 8-bit grayscale or BGR image.")
    if image.dtype != np.uint8:
        raise ValueError("Expected image dtype uint8.")
    return image


def alpha_beta(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Apply saturating contrast/brightness transform WITHOUT abs conversion."""
    _require_image(image)
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0; beta must be finite.")
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def binarize(image: np.ndarray, threshold: int = 127, inverted: bool = False) -> np.ndarray:
    """Return single-channel 0/255 binary image after grayscale conversion."""
    _require_image(image)
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be 0..255.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 and image.shape[2] == 3 else image.reshape(image.shape[:2])
    method = cv2.THRESH_BINARY_INV if inverted else cv2.THRESH_BINARY
    _, result = cv2.threshold(gray, threshold, 255, method)
    return result


def contour_overlay(
    image: np.ndarray, threshold: int = 127,
    inverted: bool = False, min_area_px: int = 25,
) -> tuple[np.ndarray, ContourStats]:
    """Color 8-connected-component boundaries red, return pixel-count stats.

    The area statistic is a number of pixels, not cv2.contourArea's geometric
    area. Boundaries use the foreground mask, excluding subthreshold regions.
    """
    _require_image(image)
    if min_area_px < 1:
        raise ValueError("min_area_px must be >= 1.")
    binary = binarize(image, threshold, inverted)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    included = np.zeros(count, dtype=bool)
    for label in range(1, count):
        included[label] = stats[label, cv2.CC_STAT_AREA] >= min_area_px
    foreground = np.where(included[labels], 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.drawContours(output, contours, -1, (0, 0, 255), thickness=1)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count) if included[label]]
    return output, ContourStats(len(areas), sum(areas), max(areas, default=0))


def _common_window(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> tuple[slice, slice, slice, slice] | None:
    ah, aw = a.shape[:2]
    bh, bw = b.shape[:2]
    ax0, ay0 = max(dx, 0), max(dy, 0)
    ax1, ay1 = min(dx + bw, aw), min(dy + bh, ah)
    if ax1 <= ax0 or ay1 <= ay0:
        return None
    return slice(ay0, ay1), slice(ax0, ax1), slice(ay0 - dy, ay1 - dy), slice(ax0 - dx, ax1 - dx)


def overlay(a: np.ndarray, b: np.ndarray, opacity_b: float = 0.5, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Overlay B at native scale onto A with integer (dx,dy) pixel translation."""
    _require_image(a); _require_image(b)
    if a.ndim != 3 or a.shape[2] != 3 or b.ndim != 3 or b.shape[2] != 3:
        raise ValueError("Overlay expects 3-channel BGR images.")
    if not 0 <= opacity_b <= 1:
        raise ValueError("opacity_b must be 0..1.")
    result = a.copy()
    window = _common_window(a, b, int(dx), int(dy))
    if window is not None:
        ay, ax, by, bx = window
        result[ay, ax] = np.clip((1 - opacity_b) * a[ay, ax].astype(float) + opacity_b * b[by, bx], 0, 255).astype(np.uint8)
    return result


def difference(a: np.ndarray, b: np.ndarray, gain: float = 3.0, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Display absolute average BGR difference as dark-to-orange false color.

    Pixels outside B's overlap are black, not treated as changed pixels.
    """
    _require_image(a); _require_image(b)
    if a.ndim != 3 or a.shape[2] != 3 or b.ndim != 3 or b.shape[2] != 3:
        raise ValueError("Difference expects 3-channel BGR images.")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a positive finite value.")
    result = np.zeros_like(a)
    window = _common_window(a, b, int(dx), int(dy))
    if window is not None:
        ay, ax, by, bx = window
        delta = np.abs(a[ay, ax].astype(np.float32) - b[by, bx].astype(np.float32)).mean(axis=2) * gain
        v = np.clip(delta, 0, 255)
        result[ay, ax, 0] = np.clip(v * .12, 0, 255).astype(np.uint8)
        result[ay, ax, 1] = np.clip(v * .55, 0, 255).astype(np.uint8)
        result[ay, ax, 2] = v.astype(np.uint8)
    return result


def units_per_pixel(known_length: float, p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Scale based on two points from a known-distance feature."""
    if not np.isfinite(known_length) or known_length <= 0:
        raise ValueError("known_length must be finite and > 0.")
    d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if not np.isfinite(d) or d <= 0:
        raise ValueError("Calibration points must be distinct and finite.")
    return float(known_length) / d
