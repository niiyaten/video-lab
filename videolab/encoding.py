"""Shared H.264/FFmpeg quality settings used by VideoLab MP4 exporters."""
from __future__ import annotations

DEFAULT_CRF = 23
MIN_CRF = 0
MAX_CRF = 51


def validate_crf(crf: int) -> int:
    """Require an integer x264 constant-rate-factor in the supported range.

    CRF controls the encoder only: stream-copy paths retain original coded
    video, and pixel-format conversion to yuv420p can still be lossy at CRF 0.
    """
    if isinstance(crf, bool) or not isinstance(crf, int) or not MIN_CRF <= crf <= MAX_CRF:
        raise ValueError("H.264画質（CRF）は0〜51の整数で指定してください。")
    return crf
