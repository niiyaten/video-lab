"""Produce a browser-compatible MP4 without deliberately changing video timing.

The original is kept intact. For constant-frame-rate (CFR) recordings, input
frame rate, frame order and duration are maintained. Variable-frame-rate (VFR)
recordings are not a primary target: when source FPS is readable they are
normalized to that FPS; otherwise input timestamps are passed through.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

import imageio_ffmpeg


@dataclass(frozen=True)
class PreviewInfo:
    path: Path
    playback_fps: float  # Encoded media FPS; normally the same as the input.
    frame_count: int


def make_browser_preview(
    source: str | Path,
    destination: str | Path,
    *,
    max_frames: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> PreviewInfo:
    """Transcode to H.264 MP4 retaining source timing as much as possible.

    - For constant-frame-rate camera videos, uses the source's declared FPS
      to set frame timestamps and output FPS (never an arbitrary 30 FPS).
    - Does NOT intentionally drop/duplicate frames; output count is checked.
    - If source FPS cannot be read through OpenCV, FFmpeg input timestamps
      are passed through without overriding the frame rate.
    - The default is to convert the *entire* input video, without truncation.
    - Global metadata and first audio stream (if any) are retained where the
      destination container/codec supports them; audio is re-encoded to AAC.
    - H.264/yuv420p and padding odd dimensions are *not* lossless. Perform
      metrology on the original camera recording, never the preview.

    Args:
        max_frames: 0 means all source frames; positive number converts the
            first N source frames. This intentionally shortens duration.
    """
    from .video_io import read_info

    src, dest = Path(source), Path(destination)
    if not src.is_file():
        raise ValueError("元動画が見つかりません。")
    if not isinstance(max_frames, int) or max_frames < 0:
        raise ValueError("最大フレーム数は0以上の整数にしてください。")
    if src.resolve() == dest.resolve():
        raise ValueError("元動画と出力動画のパスを分けてください。")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)

    try:
        source_info = read_info(src)
    except (ValueError, RuntimeError):
        source_info = None

    # FFmpeg may round an AVI's fractional nominal rate (e.g. 119.88 to 120).
    # For fixed-cadence industrial recordings, explicitly set the *original*
    # fps and assign one presentation timestamp per decoded source frame.
    if source_info is not None:
        source_fps = str(Fraction(source_info.fps).limit_denominator(100_000))
        video_filters = f"pad=ceil(iw/2)*2:ceil(ih/2)*2,settb=AVTB,setpts=N/({source_fps}*TB)"
        fps_args = ["-r", source_fps, "-fps_mode", "cfr"]
    else:
        video_filters = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        fps_args = ["-fps_mode", "passthrough"]

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-nostats", "-stats_period", "0.25", "-progress", "pipe:1",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0",
        "-vf", video_filters, *fps_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
    ]
    if max_frames:
        command += ["-frames:v", str(max_frames), "-shortest"]
    command += [str(dest)]
    try:
        # FFmpeg sends real encoded frame counts to stdout via -progress.
        # stderr is spooled to disk to avoid pipe deadlocks on long files.
        expected = (min(max_frames, source_info.frame_count) if max_frames else source_info.frame_count) if source_info else max_frames
        processed = 0
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as err_file:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err_file,
                                  text=True, encoding="utf-8", errors="replace") as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    key, _, value = line.strip().partition("=")
                    if key == "frame":
                        try:
                            processed = max(processed, int(value))
                        except ValueError:
                            pass
                    elif key == "progress" and on_progress is not None:
                        on_progress(processed, expected)
                returncode = proc.wait()
            if returncode != 0:
                err_file.seek(0)
                detail = err_file.read()[-800:]
                raise RuntimeError("互換MP4の作成に失敗しました。コーデックを確認してください。\n" + detail)
        info = read_info(dest)
        if info.frame_count <= 0:
            raise RuntimeError("変換後の動画内にフレームが見つかりませんでした。")
        if source_info is not None:
            if abs(info.fps / source_info.fps - 1) > 0.0001:
                raise RuntimeError(
                    f"元動画のFPSを保持できませんでした：元 {source_info.fps:g} → MP4 {info.fps:g} FPS"
                )
            if source_info.frame_count > 0:
                expected_frames = min(max_frames, source_info.frame_count) if max_frames else source_info.frame_count
                if info.frame_count != expected_frames:
                    raise RuntimeError(
                        f"フレーム数が一致しません：元 {expected_frames} → MP4 {info.frame_count}。"
                        " 元動画を変更せず変換を中止しました。"
                    )
        if on_progress is not None:
            on_progress(info.frame_count, expected)
        return PreviewInfo(dest, info.fps, info.frame_count)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
