"""Playback FPS and multiplier are two synchronized views of one setting."""
import math

import pytest

from videolab.playback_speed import (
    MIN_OUTPUT_FPS,
    fps_to_multiplier,
    multiplier_to_fps,
)


@pytest.mark.parametrize(
    "source_fps,target_fps,expected_multiplier",
    [
        (120, 30, 0.25),
        (120, 120, 1),
        (120, 240, 2),
        (119.88, 59.94, 0.5),
        (1000, 1, 0.001),
        (120, 5000, 5000 / 120),
    ],
)
def test_fps_updates_multiplier(source_fps, target_fps, expected_multiplier):
    assert fps_to_multiplier(target_fps, source_fps) == pytest.approx(expected_multiplier)
    assert multiplier_to_fps(expected_multiplier, source_fps) == pytest.approx(target_fps)


@pytest.mark.parametrize(
    "source_fps,multiplier,expected_fps",
    [
        (120, 0.5, 60),
        (120, 2, 240),
        (29.97, 1, 29.97),
        (119.88, 2.5, 299.7),
        (800, 0.125, 100),
    ],
)
def test_multiplier_updates_fps(source_fps, multiplier, expected_fps):
    assert multiplier_to_fps(multiplier, source_fps) == pytest.approx(expected_fps)
    assert fps_to_multiplier(expected_fps, source_fps) == pytest.approx(multiplier)


def test_minimum_fps_is_supported():
    source_fps = 120
    multiplier = fps_to_multiplier(MIN_OUTPUT_FPS, source_fps)
    assert multiplier_to_fps(multiplier, source_fps) == pytest.approx(MIN_OUTPUT_FPS)


@pytest.mark.parametrize("bad_value", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_multiplier(bad_value):
    with pytest.raises(ValueError):
        multiplier_to_fps(bad_value, 120)


@pytest.mark.parametrize("bad_value", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_source_fps(bad_value):
    with pytest.raises(ValueError):
        fps_to_multiplier(30, bad_value)
    with pytest.raises(ValueError):
        multiplier_to_fps(2, bad_value)


def test_rejects_too_slow_output():
    with pytest.raises(ValueError):
        multiplier_to_fps(1e-9, 120)


def test_rejects_extreme_multiplier_overflow():
    with pytest.raises(ValueError):
        multiplier_to_fps(1e308, 1e308)
