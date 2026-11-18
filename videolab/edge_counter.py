"""Edge Counter-style contour extraction for images and decoded video frames.

The processing order intentionally follows the July 2026 Edge Counter v25/v26 design:
brightness/contrast -> optional blur -> threshold/Canny -> morphology ->
findContours -> contour selection -> 1px contour mask -> ROI/border exclusion.

All coordinates and areas refer to the original image's pixel coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .image_processing import alpha_beta

InputMode = Literal["Binary", "Canny"]
TargetMode = Literal["All", "Max", "Top N"]
PointMode = Literal["None", "Edge Pixels", "Left Envelope", "Right Envelope"]


@dataclass(frozen=True)
class EdgeOptions:
    alpha: float = 1.0
    beta: float = 0.0
    blur_mode: str = "OFF"  # OFF / Gaussian / Median
    blur_kernel: int = 3
    input_mode: InputMode = "Binary"
    threshold: int = 127
    invert: bool = False
    canny_low: int = 50
    canny_high: int = 150
    morph_mode: str = "None"  # None / Open / Close / Dilate / Erode
    morph_kernel: int = 3
    morph_iterations: int = 1
    min_area: float = 0.0
    max_area: float = 0.0  # 0 = no upper bound
    target: TargetMode = "All"
    top_n: int = 5
    contour_points: PointMode = "None"
    # ROI is x0,y0,x1,y1 half-open, in ORIGINAL image coordinates.
    roi: tuple[int, int, int, int] | None = None
    remove_border: bool = True
    border_margin: int = 1
    draw_largest_bbox: bool = False


@dataclass
class EdgeResult:
    adjusted: np.ndarray
    contour_input: np.ndarray
    contour_mask: np.ndarray
    overlay: np.ndarray
    contours: list[np.ndarray]
    stats: list[dict[str, int | float]]
    point_rows: list[dict[str, int]]


def _validate(options: EdgeOptions) -> None:
    if options.blur_mode not in ("OFF", "Gaussian", "Median"):
        raise ValueError("Invalid blur_mode")
    if options.input_mode not in ("Binary", "Canny"):
        raise ValueError("Invalid input_mode")
    if options.morph_mode not in ("None", "Open", "Close", "Dilate", "Erode"):
        raise ValueError("Invalid morph_mode")
    if options.target not in ("All", "Max", "Top N"):
        raise ValueError("Invalid target")
    if options.contour_points not in ("None", "Edge Pixels", "Left Envelope", "Right Envelope"):
        raise ValueError("Invalid contour_points")
    if not 0 <= options.threshold <= 255 or not 0 <= options.canny_low <= 255 or not 0 <= options.canny_high <= 255:
        raise ValueError("Thresholds must be in 0..255")
    if options.canny_low > options.canny_high:
        raise ValueError("Canny low threshold must not exceed high threshold")
    if options.blur_kernel < 1 or options.blur_kernel % 2 != 1:
        raise ValueError("Blur kernel must be odd and >=1")
    if options.morph_kernel < 1 or options.morph_iterations < 1:
        raise ValueError("Invalid morphology kernel or iterations")
    if options.min_area < 0 or options.max_area < 0 or (options.max_area and options.max_area < options.min_area):
        raise ValueError("Invalid area bounds")
    if options.top_n < 1 or options.border_margin < 0:
        raise ValueError("Invalid count or border margin")


def _border_and_roi_mask(shape: tuple[int, int], options: EdgeOptions) -> np.ndarray:
    h, w = shape
    allowed = np.full((h, w), 255, dtype=np.uint8)
    if options.roi is not None:
        x0, y0, x1, y1 = options.roi
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("ROI must be inside image and have nonzero size")
        allowed[:] = 0
        allowed[y0:y1, x0:x1] = 255
        if options.remove_border and options.border_margin:
            m = options.border_margin
            allowed[y0:min(y0 + m, y1), x0:x1] = 0
            allowed[max(y0, y1 - m):y1, x0:x1] = 0
            allowed[y0:y1, x0:min(x0 + m, x1)] = 0
            allowed[y0:y1, max(x0, x1 - m):x1] = 0
    if options.remove_border and options.border_margin:
        m = options.border_margin
        allowed[:m, :] = 0
        allowed[-m:, :] = 0
        allowed[:, :m] = 0
        allowed[:, -m:] = 0
    return allowed


def _find_contours(mask: np.ndarray) -> list[np.ndarray]:
    # Return contours with stable sorted-by-area order (largest first).
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def analyze_edges(image: np.ndarray, options: EdgeOptions | None = None) -> EdgeResult:
    """Analyze one frame without mutating input or choosing new points implicitly."""
    o = options or EdgeOptions()
    _validate(o)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected nonempty uint8 image")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.copy()
    else:
        raise ValueError("Expected grayscale or BGR image")
    adjusted = alpha_beta(bgr, o.alpha, o.beta)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    if o.blur_mode != "OFF" and o.blur_kernel > 1:
        if o.blur_mode == "Gaussian":
            gray = cv2.GaussianBlur(gray, (o.blur_kernel, o.blur_kernel), 0)
        else:
            gray = cv2.medianBlur(gray, o.blur_kernel)
    if o.input_mode == "Binary":
        method = cv2.THRESH_BINARY_INV if o.invert else cv2.THRESH_BINARY
        _, candidate = cv2.threshold(gray, o.threshold, 255, method)
    else:
        candidate = cv2.Canny(gray, o.canny_low, o.canny_high)
    if o.morph_mode != "None":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (o.morph_kernel, o.morph_kernel))
        morph = {"Open": cv2.MORPH_OPEN, "Close": cv2.MORPH_CLOSE}
        if o.morph_mode in morph:
            candidate = cv2.morphologyEx(candidate, morph[o.morph_mode], k, iterations=o.morph_iterations)
        elif o.morph_mode == "Dilate":
            candidate = cv2.dilate(candidate, k, iterations=o.morph_iterations)
        else:
            candidate = cv2.erode(candidate, k, iterations=o.morph_iterations)
    raw_contours = _find_contours(candidate)
    filtered = [c for c in raw_contours if cv2.contourArea(c) >= o.min_area and
                (o.max_area == 0 or cv2.contourArea(c) <= o.max_area)]
    if o.target == "Max":
        selected = filtered[:1]
    elif o.target == "Top N":
        selected = filtered[:o.top_n]
    else:
        selected = filtered

    # Draw selected contours to a 1-pixel mask first, then zero ROI and frame edges.
    # This avoids introducing a new artificial ROI boundary prior to contour detection.
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, selected, -1, 255, 1)
    allowed = _border_and_roi_mask(gray.shape, o)
    mask = cv2.bitwise_and(mask, allowed)

    overlay = adjusted.copy()
    overlay[mask > 0] = (0, 0, 255)  # red, BGR
    if o.draw_largest_bbox and selected:
        x, y, w, h = cv2.boundingRect(selected[0])
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 1)
    if o.roi is not None:
        x0, y0, x1, y1 = o.roi
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), 1)

    contour_rows: list[dict[str, int | float]] = []
    points: list[dict[str, int]] = []
    for idx, contour in enumerate(selected, 1):
        x, y, w, h = cv2.boundingRect(contour)
        # Only pixels owned by this contour survive ROI/border removal.
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, 1)
        contour_mask = cv2.bitwise_and(contour_mask, allowed)
        yy, xx = np.nonzero(contour_mask)
        contour_rows.append({
            "contour_id": idx, "area_px2": float(cv2.contourArea(contour)),
            "perimeter_px": float(cv2.arcLength(contour, True)),
            "x_px": int(x), "y_px": int(y), "width_px": int(w), "height_px": int(h),
            "visible_edge_pixels": int(len(xx)),
        })
        if o.contour_points == "None":
            continue
        if o.contour_points == "Edge Pixels":
            coordinates = zip(xx.tolist(), yy.tolist())
        else:
            # Column envelope: one extreme Y per X. This is lightweight and
            # does not attempt to bridge discontinuous contours.
            xy = {}
            for px, py in zip(xx.tolist(), yy.tolist()):
                if px not in xy or ((o.contour_points == "Left Envelope") and py < xy[px]) or ((o.contour_points == "Right Envelope") and py > xy[px]):
                    xy[px] = py
            coordinates = sorted(xy.items())
        points.extend({"contour_id": idx, "x_px": px, "y_px": py} for px, py in coordinates)

    return EdgeResult(
        adjusted=adjusted, contour_input=candidate, contour_mask=mask,
        overlay=overlay, contours=selected, stats=contour_rows, point_rows=points,
    )
