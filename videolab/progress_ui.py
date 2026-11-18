"""Streamlit progress indicators shared by long-running jobs."""
from __future__ import annotations

import time

import streamlit as st

from .progress import snapshot


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class JobProgress:
    """Visual status; actual processed counts arrive from computation callbacks.

    The 100% state is only displayed after the output file has been finalized.
    """

    def __init__(self, label: str, unit: str = "フレーム") -> None:
        self.label = label
        self.unit = unit
        self.start = time.monotonic()
        self.last_render = 0.0
        self.processed = 0
        self.total = 0
        self.bar = st.progress(0.0, text=f"{label}：開始しています…")
        self.detail = st.empty()

    def update(self, processed: int, total: int, stage: str | None = None) -> None:
        self.processed, self.total = max(0, int(processed)), max(0, int(total))
        now = time.monotonic()
        if now - self.last_render < 0.2 and processed != total:
            return
        self.last_render = now
        snap = snapshot(self.processed, self.total, now - self.start)
        name = stage or self.label
        if snap.fraction is None:
            # No reliable total (e.g. unknown frame count). Don't show a fake percentage.
            self.bar.progress(0.0, text=f"{name}：{snap.processed:,} {self.unit}処理済み（総数不明）")
            self.detail.caption(f"経過時間 {_duration(snap.elapsed_s)} / 進捗率・残り時間は算出できません")
            return
        # Processing a frame does not guarantee the output has finished encoding.
        percent = min(99, int(snap.fraction * 100))
        self.bar.progress(percent / 100.0, text=f"{name}：{percent}%（{snap.processed:,} / {snap.total:,} {self.unit}）")
        eta = f" / 残り目安 {_duration(snap.remaining_s)}" if snap.remaining_s is not None and snap.processed < snap.total else ""
        self.detail.caption(f"経過時間 {_duration(snap.elapsed_s)}{eta} ※ 残り時間は概算")

    def finish(self, summary: str = "完了") -> None:
        self.bar.progress(1.0, text=f"{self.label}：100%（{summary}）")
        self.detail.caption(f"完了 / 経過時間 {_duration(time.monotonic() - self.start)}")

    def fail(self) -> None:
        self.bar.progress(0.0, text=f"{self.label}：失敗しました")
        self.detail.caption(f"処理は完了していません / 経過時間 {_duration(time.monotonic() - self.start)}")
