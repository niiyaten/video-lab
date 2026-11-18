"""Frame-accurate export interval helpers for constant-frame-rate video.

Intervals always use zero-based half-open frame indices: [start, stop).
Time inputs use source-video seconds, NOT output-playback seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameInterval:
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


def validate_frame_interval(start: int, stop: int, total_frames: int) -> FrameInterval:
    """Check an exact [start, stop) interval against source metadata."""
    if not all(type(v) is int for v in (start, stop, total_frames)):
        raise ValueError("フレーム番号は整数で指定してください。")
    if total_frames <= 0:
        raise ValueError("元動画の総フレーム数を取得できません。")
    if not (0 <= start < stop <= total_frames):
        raise ValueError(f"範囲は0 <= 開始 < 終了 <= {total_frames}フレームで指定してください。")
    return FrameInterval(start, stop)


def seconds_to_interval(start_s: float, stop_s: float, source_fps: float, total_frames: int) -> FrameInterval:
    """Select frames with source timestamps in [start_s, stop_s).

    For CFR video, frame k occurs at k/source_fps seconds. Both boundaries
    use ceil so that frames whose timestamps are outside the interval are
    not included. Tiny epsilon absorbs floating-point jitter at exact ticks.
    """
    if not all(math.isfinite(float(v)) for v in (start_s, stop_s, source_fps)) or source_fps <= 0:
        raise ValueError("秒数と元動画FPSには有限の正しい数値を指定してください。")
    if start_s < 0 or stop_s <= start_s:
        raise ValueError("開始秒数は0以上、終了秒数は開始より後にしてください。")
    duration = total_frames / source_fps
    if stop_s > duration + 1e-7:
        raise ValueError("終了秒数が元動画の推定時間を超えています。")
    start = min(total_frames, max(0, math.ceil(start_s * source_fps - 1e-7)))
    stop = min(total_frames, max(0, math.ceil(stop_s * source_fps - 1e-7)))
    return validate_frame_interval(start, stop, total_frames)
