"""Reference tests for Python / OpenCV image pipeline."""
import cv2
import numpy as np
import pytest

from videolab.image_processing import alpha_beta, binarize, contour_overlay, difference, overlay, units_per_pixel


def test_calibration_arbitrary_known_distance():
    assert units_per_pixel(10, (30, 20), (130, 20)) == pytest.approx(.1)
    assert units_per_pixel(12.7, (0, 0), (3, 4)) == pytest.approx(2.54)
    with pytest.raises(ValueError):
        units_per_pixel(0, (0, 0), (100, 0))


def test_alpha_beta_clips_negative_values_without_abs():
    image = np.array([[0, 10, 127, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(alpha_beta(image, 2, -15), np.array([[0, 5, 239, 255]], dtype=np.uint8))


def test_threshold_and_contour_count():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (20, 20), (250, 250, 250), thickness=-1)
    cv2.rectangle(image, (50, 40), (69, 59), (250, 250, 250), thickness=-1)
    assert binarize(image, 127).shape == (100, 120)
    rendered, stats = contour_overlay(image, min_area_px=20)
    assert stats.count == 2
    assert stats.total_area_px == 121 + 400
    assert stats.largest_area_px == 400
    assert rendered[10, 10, 2] == 255


def test_alignment_changes_diff():
    a = np.zeros((60, 70, 3), dtype=np.uint8)
    b = np.zeros_like(a)
    a[12:30, 13:40] = 200
    b[12:30, 18:45] = 200
    assert np.count_nonzero(difference(a, b, gain=1, dx=-5)) == 0
    assert np.count_nonzero(difference(a, b, gain=1, dx=0)) > 0
    np.testing.assert_array_equal(overlay(a, b, opacity_b=0.5, dx=-5)[5:50, 5:50], a[5:50, 5:50])
