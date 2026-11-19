"""Verified keyframe seek for large, frame-indexed video ranges.

Builds a lightweight packet PTS index with ffprobe (no video decoding).
Uses FFmpeg input-side seeking to the preceding keyframe, then selects by
original frame presentation timestamp, rather than relative frame index.
The encoded frame count AND first/last decoded input PTS are verified.
If any condition is untrusted, fall back to the sequential v9 exporter.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import math
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Callable

import imageio_ffmpeg

from .fps_fast import FPSExportResult, _verify, export_fps_optimized
from .segments import validate_frame_interval
from .video_io import read_info

Progress = Callable[[int, int], None]
Stage = Callable[[str], None]


class FastSeekUnavailable(RuntimeError):
    """No provably correct frame-to-packet index is available."""


@dataclass(frozen=True)
class PacketIndex:
    time_base: Fraction
    pts: tuple[int, ...]  # Presentation order, one timestamp per frame.
    keyframes: tuple[int, ...]  # Frame numbers of packet keyframes.


def _ffprobe_path() -> str | None:
    # imageio-ffmpeg ships ffmpeg, but often not ffprobe.
    # Try PATH and next to the shipped executable.
    found = shutil.which("ffprobe")
    if found:
        return found
    binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    for name in ("ffprobe.exe", "ffprobe"):
        probe = binary.with_name(name)
        if probe.is_file():
            return str(probe)
    return None


@lru_cache(maxsize=4)
def _build_packet_index(
    probe_exe: str, source: str, size: int, mtime_ns: int, expected_frames: int,
) -> PacketIndex:
    """Packet PTS lookup is I/O-oriented and cached by file identity.

    Reject streams with missing/repeated PTS, a different packet/frame count,
    or no keyframe. They must use the safe sequential decoder.
    """
    del size, mtime_ns  # Used for cache key invalidation.
    metadata = subprocess.run(
        [probe_exe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=time_base",
         "-of", "default=noprint_wrappers=1:nokey=1", source],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    if metadata.returncode or not metadata.stdout.strip():
        raise FastSeekUnavailable("ffprobeから動画のタイムベースを取得できません")
    try:
        time_base = Fraction(metadata.stdout.strip().splitlines()[0])
    except (ValueError, ZeroDivisionError) as exc:
        raise FastSeekUnavailable("タイムベースが不正です") from exc
    if time_base <= 0:
        raise FastSeekUnavailable("タイムベースが不正です")

    args = [probe_exe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts,flags", "-of", "csv=p=0", source]
    records: list[tuple[int, bool]] = []
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as err:
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err,
                              text=True, encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                fields = line.rstrip("\r\n").split(",", 2)
                if len(fields) < 2:
                    continue
                try:
                    pts = int(fields[0])
                except ValueError as exc:
                    process.kill()
                    process.wait()
                    raise FastSeekUnavailable("入力動画にPTSのないパケットがあります") from exc
                records.append((pts, "K" in fields[1]))
                if len(records) > expected_frames:
                    process.kill()
                    process.wait()
                    raise FastSeekUnavailable("フレーム数とパケット数が一致しません")
            returncode = process.wait()
        if returncode:
            err.seek(0)
            raise FastSeekUnavailable("ffprobeによるフレーム索引作成に失敗: " + err.read()[-400:])

    if len(records) != expected_frames:
        raise FastSeekUnavailable(
            f"フレーム数とパケット数が不一致（{expected_frames:,} / {len(records):,}）"
        )
    records.sort(key=lambda item: item[0])  # B-frames arrive in decode order.
    pts = tuple(record[0] for record in records)
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise FastSeekUnavailable("重複または逆転した表示時刻があります")
    keyframes = tuple(i for i, record in enumerate(records) if record[1])
    if not keyframes:
        raise FastSeekUnavailable("キーフレームが見つかりません")
    return PacketIndex(time_base=time_base, pts=pts, keyframes=keyframes)


def _parse_boundary_pts(stderr_text: str, expected: int) -> tuple[int, int]:
    """showinfo is applied to only first/last frames by a split/select filter."""
    values = [int(m.group(1)) for m in re.finditer(
        r"\[Parsed_showinfo[^\n]*?\]\s*n:\s*\d+\s+pts:\s*(-?\d+)", stderr_text
    )]
    wanted = 1 if expected == 1 else 2
    if len(values) != wanted:
        raise FastSeekUnavailable(
            f"境界検証用のフレーム時刻が取得できません（期待{wanted}、取得{len(values)}）"
        )
    return values[0], values[-1]


def _run_verified_seek(
    command: list[str], expected: int,
    on_progress: Progress | None, on_stage: Stage | None,
) -> str:
    """Run FFmpeg, retaining only a small stderr log (two showinfo frames).

    Streamlit UI callback always runs on the main/caller thread.
    """
    finished = 0
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as err:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err,
                              text=True, encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            messages: queue.Queue[str | None] = queue.Queue()

            def consume() -> None:
                try:
                    for line in process.stdout:
                        messages.put(line)
                finally:
                    messages.put(None)

            reader = threading.Thread(target=consume, daemon=True)
            reader.start()
            while True:
                try:
                    line = messages.get(timeout=0.5)
                except queue.Empty:
                    if finished == 0 and on_stage:
                        on_stage("高速シーク後のデコード中（出力フレーム待ち）")
                    continue
                if line is None:
                    break
                key, sep, value = line.strip().partition("=")
                if not sep or key != "frame":
                    continue
                try:
                    frame = int(value)
                except ValueError:
                    continue
                if frame > finished:
                    if finished == 0 and on_stage:
                        on_stage("指定区間をエンコード中")
                    finished = frame
                    if on_progress:
                        on_progress(min(frame, expected), expected)
            returncode = process.wait()
            reader.join(timeout=1)
        err.seek(0)
        log = err.read()
    if returncode:
        raise FastSeekUnavailable("高速シーク変換に失敗: " + log[-1000:])
    if on_stage:
        on_stage("MP4ファイルの書き込み・確定中")
    return log


def _export_using_index(
    source: Path, destination: Path,
    target_fps: float, start: int, stop: int, preset: str,
    index: PacketIndex,
    on_progress: Progress | None, on_stage: Stage | None,
) -> FPSExportResult:
    expected = stop - start
    preceding = [i for i in index.keyframes if i <= start]
    if not preceding:
        raise FastSeekUnavailable("選択区間以前のキーフレームが見つかりません")
    keyframe = preceding[-1]
    # Seek slightly before the selected keyframe to account for decoder
    # pre-roll and demuxer timestamp rounding.
    seek_seconds = max(
        0.0,
        float((index.pts[keyframe] - index.pts[0]) * index.time_base) - 0.1,
    )
    if on_stage:
        on_stage(f"キーフレーム {keyframe:,} 付近へ高速シーク中")
    rate = Fraction(str(target_fps)).limit_denominator(1_000_000)
    output_rate = f"{rate.numerator}/{rate.denominator}"
    time_base = f"{rate.denominator}/{rate.numerator}"
    first_pts = index.pts[start]
    last_pts = index.pts[stop - 1]
    end_option = f":end_pts={index.pts[stop]}" if stop < len(index.pts) else ""
    endpoint = f"eq(n\\,0)" if expected == 1 else f"eq(n\\,0)+eq(n\\,{expected - 1})"
    graph = (
        f"[0:v:0]trim=start_pts={first_pts}{end_option},split=2[main][check];"
        f"[check]select='{endpoint}',showinfo,nullsink;"
        f"[main]settb=AVTB,setpts=N/({output_rate}*TB),"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2[out]"
    )
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "info",
        "-y", "-nostats", "-stats_period", "0.25", "-progress", "pipe:1",
        "-ss", f"{seek_seconds:.8f}", "-copyts", "-i", str(source),
        "-filter_complex", graph, "-map", "[out]", "-an",
        "-r", output_rate, "-fps_mode:v", "cfr",
        "-enc_time_base:v", time_base,
        "-c:v", "libx264", "-preset", preset, "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    log = _run_verified_seek(cmd, expected, on_progress, on_stage)
    if on_stage:
        on_stage("先頭・末尾フレームの時刻を照合中")
    actual_first, actual_last = _parse_boundary_pts(log, expected)
    if (actual_first, actual_last) != (first_pts, last_pts):
        raise FastSeekUnavailable(
            f"開始・終了フレームが一致しません: "
            f"期待PTS {first_pts}/{last_pts}、実際 {actual_first}/{actual_last}"
        )
    if on_stage:
        on_stage("出力フレーム数とFPSを検証中")
    _verify(destination, expected, target_fps)
    if on_progress:
        on_progress(expected, expected)
    return FPSExportResult(expected, target_fps, "seek_transcode")


def export_fps_with_seek(
    source: str | Path,
    destination: str | Path,
    target_fps: float,
    on_progress: Progress | None = None,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    preset: str = "veryfast",
    on_stage: Stage | None = None,
) -> FPSExportResult:
    """Prefer verified keyframe seek for partial ranges; otherwise fall back."""
    src, dst = Path(source), Path(destination)
    if src.resolve() == dst.resolve():
        raise ValueError("元動画と出力先に同じファイルを指定できません")
    info = read_info(src)
    if info.frame_count <= 0:
        raise ValueError("元動画のフレーム数を取得できません")
    interval = validate_frame_interval(
        start_frame, info.frame_count if end_frame is None else end_frame,
        info.frame_count,
    )
    # The original v9 exporter already stream-copies whole H.264 video.
    # Seeking is useful only when the desired segment starts later.
    if interval.start > 0:
        try:
            probe_exe = _ffprobe_path()
            if probe_exe is None:
                raise FastSeekUnavailable("ffprobeが未導入です")
            if on_stage:
                on_stage("フレーム時刻とキーフレームの索引を確認中")
            stat = src.stat()
            index = _build_packet_index(
                probe_exe, str(src.resolve()), stat.st_size, stat.st_mtime_ns,
                info.frame_count,
            )
            if on_progress:
                on_progress(0, interval.count)
            return _export_using_index(
                src, dst, target_fps, interval.start, interval.stop, preset,
                index, on_progress, on_stage,
            )
        except (FastSeekUnavailable, OSError, subprocess.SubprocessError) as exc:
            # Never deliver a file with an unverified range.
            dst.unlink(missing_ok=True)
            if on_stage:
                on_stage(f"高速シーク不可（{exc}）。先頭から正確に読み込み直します")
            if on_progress:
                on_progress(0, interval.count)
    return export_fps_optimized(
        src, dst, target_fps, on_progress,
        start_frame=interval.start, end_frame=interval.stop,
        preset=preset, on_stage=on_stage,
    )
