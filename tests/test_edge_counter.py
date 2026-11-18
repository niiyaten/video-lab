"""Factory Edge Counter regression tests."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from videolab.edge_counter import EdgeOptions, analyze_edges


def rect_image() -> np.ndarray:
    a = np.zeros((110, 150, 3), np.uint8)
    cv2.rectangle(a, (20, 25), (70, 85), (255, 255, 255), -1)
    return a


def test_defaults_match_final_edge_counter_design():
    o = EdgeOptions()
    assert o.blur_mode == "OFF"
    assert o.contour_points == "None"
    assert not o.draw_largest_bbox
    assert o.target == "All"


def test_rectangle_contour_and_points_are_accurate():
    a = rect_image()
    result = analyze_edges(a, EdgeOptions(contour_points="Edge Pixels", min_area=100))
    assert len(result.contours) == 1
    assert result.stats[0]["area_px2"] == pytest.approx(3000)
    assert (result.contour_mask[25, 30] == 255)
    assert (result.contour_mask[0, 0] == 0)
    assert len(result.point_rows) == result.stats[0]["visible_edge_pixels"]
    assert len(result.point_rows) > 100


def test_top_n_and_max_filter():
    a = rect_image()
    cv2.rectangle(a, (95, 35), (120, 65), (255, 255, 255), -1)
    assert len(analyze_edges(a, EdgeOptions(target="All", min_area=10)).stats) == 2
    assert len(analyze_edges(a, EdgeOptions(target="Max", min_area=10)).stats) == 1
    assert len(analyze_edges(a, EdgeOptions(target="Top N", top_n=1, min_area=10)).stats) == 1
    assert len(analyze_edges(a, EdgeOptions(target="All", min_area=800)).stats) == 1


def test_roi_and_border_exclusion_keep_raw_contour_selection():
    a = rect_image()
    o = EdgeOptions(roi=(30, 30, 75, 75), remove_border=True, border_margin=2,
                    contour_points="Edge Pixels")
    result = analyze_edges(a, o)
    assert len(result.stats) == 1  # selected before ROI clipping
    assert np.count_nonzero(result.contour_mask) > 0
    assert np.count_nonzero(result.contour_mask[:30]) == 0
    assert np.count_nonzero(result.contour_mask[:, :30]) == 0
    assert np.count_nonzero(result.contour_mask[30:32]) == 0
    assert np.count_nonzero(result.contour_mask[73:75]) == 0
    assert all(30 < p["x_px"] < 75 and 30 < p["y_px"] < 75 for p in result.point_rows)


def test_roi_mask_does_not_create_artificial_rectangle_boundary():
    a = np.full((100, 100, 3), 255, dtype=np.uint8)
    o = EdgeOptions(roi=(20, 20, 70, 70), remove_border=True, border_margin=2)
    result = analyze_edges(a, o)
    assert np.count_nonzero(result.contour_mask) == 0  # only original full-image border exists


def test_canny_and_envelope_modes():
    a = rect_image()
    for mode in ("Left Envelope", "Right Envelope"):
        r = analyze_edges(a, EdgeOptions(input_mode="Canny", contour_points=mode))
        assert r.contour_mask.shape == a.shape[:2]
        assert all("x_px" in p and "y_px" in p for p in r.point_rows)


def test_options_validation_and_input_not_mutated():
    a = rect_image()
    original = a.copy()
    analyze_edges(a, EdgeOptions(alpha=1.2, beta=5, morph_mode="Close"))
    assert np.array_equal(a, original)
    with pytest.raises(ValueError):
        analyze_edges(a, EdgeOptions(roi=(0, 0, 160, 90)))
    with pytest.raises(ValueError):
        analyze_edges(a, EdgeOptions(blur_kernel=2))
    with pytest.raises(ValueError):
        analyze_edges(a, EdgeOptions(canny_low=200, canny_high=100))
