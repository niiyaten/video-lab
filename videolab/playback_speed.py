"""Conversions for coupled output-FPS and playback-speed inputs.

The selected source frames remain unchanged; only their playback timestamps
are rescaled by the FPS export pipeline.
"""
from __future__ import annotations

import math

MIN_OUTPUT_FPS = 0.001


def _positive_finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}は0より大きい有限値にしてください。")
    return value


def fps_to_multiplier(target_fps: float, source_fps: float) -> float:
    """E.g. 30 FPS output from 120 FPS source means 0.25x speed."""
    source = _positive_finite(source_fps, "元動画FPS")
    target = _positive_finite(target_fps, "出力FPS")
    if target < MIN_OUTPUT_FPS:
        raise ValueError(f"出力FPSは{MIN_OUTPUT_FPS:g}以上にしてください。")
    return target / source


def multiplier_to_fps(multiplier: float, source_fps: float) -> float:
    """Convert requested speed into output FPS, without dropping frames."""
    source = _positive_finite(source_fps, "元動画FPS")
    multiplier = _positive_finite(multiplier, "再生倍率")
    target = source * multiplier
    if not math.isfinite(target):
        raise ValueError("変換後のFPSが大きすぎます。")
    # Tolerate only rounding at the already supported minimum of 0.001 FPS.
    if target < MIN_OUTPUT_FPS * (1 - 1e-12):
        raise ValueError(f"出力FPSは{MIN_OUTPUT_FPS:g}以上にしてください。")
    return max(target, MIN_OUTPUT_FPS)
