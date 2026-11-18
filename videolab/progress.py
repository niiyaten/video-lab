"""Pure progress math. Estimated remaining time is not a promised completion time."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ProgressSnapshot:
    processed: int
    total: int
    elapsed_s: float
    fraction: float | None
    remaining_s: float | None


def snapshot(processed: int, total: int, elapsed_s: float) -> ProgressSnapshot:
    """Compute a progress snapshot, with unknown totals represented by None.

    The caller must provide actual completed work; no synthetic percentages.
    """
    processed = max(0, int(processed))
    total = max(0, int(total))
    elapsed = max(0.0, float(elapsed_s)) if isfinite(elapsed_s) else 0.0
    if total == 0:
        return ProgressSnapshot(processed, 0, elapsed, None, None)
    fraction = min(processed / total, 1.0)
    remain = (total - processed) * elapsed / processed if processed > 0 else None
    return ProgressSnapshot(processed, total, elapsed, fraction, remain)
