"""Fast, frame-exact FPS export for constant-frame-rate industrial videos.

Whole-file H.264 may be remuxed with packet timestamp edits (lossless).
Other inputs and any subrange use a single native FFmpeg decode/filter/encode
pipeline (no BGR round-trip through Python). Source video is never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from typing import Callable

import cv2
import imageio_ffmpeg

from .encoding import DEFAULT_CRF, validate_crf
from .segments import validate_frame_interval
from .video_io import read_info

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class FPSExportResult:
    frame_count: int
    fps: float
    method: str  # stream_copy | transcode


def _h264_source(path: Path) -> bool:
    if path.suffix.lower() not in {'.mp4', '.m4v', '.mov', '.avi'}:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = ''.join(chr((code >> (8 * i)) & 255) for i in range(4)).lower()
        return fourcc in {'avc1', 'avc3', 'h264', 'x264'}
    finally:
        cap.release()


def _run_ffmpeg(
    command: list[str],
    expected: int,
    on_progress: Progress | None,
    *,
    on_stage: Callable[[str], None] | None = None,
    waiting_stage: str = "変換処理中",
) -> None:
    """Read FFmpeg progress on a worker thread, render status on caller thread.

    A frame-based trim may spend minutes decoding earlier input frames before
    encoding the first selected frame. In that phase FFmpeg reports frame=0;
    a heartbeat keeps the stage/elapsed-time visible without inventing a
    percentage. Streamlit calls stay on the main thread.
    """
    done = 0
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                              encoding='utf-8', errors='replace', text=True) as process:
            assert process.stdout is not None
            messages: queue.Queue[str | None] = queue.Queue()

            def consume_stdout() -> None:
                try:
                    for line in process.stdout:
                        messages.put(line)
                finally:
                    messages.put(None)

            reader = threading.Thread(target=consume_stdout, daemon=True)
            reader.start()
            while True:
                try:
                    line = messages.get(timeout=0.5)
                except queue.Empty:
                    # No output frame yet: show the scan stage and elapsed time.
                    if done == 0 and on_stage:
                        on_stage(waiting_stage)
                    continue
                if line is None:
                    break
                key, sep, value = line.strip().partition('=')
                if not sep:
                    continue
                if key == 'frame':
                    try:
                        n = int(value)
                    except ValueError:
                        continue
                    if n > done:
                        done = n
                        if on_progress:
                            on_progress(min(done, expected), expected)
                elif key == 'progress' and done == 0 and on_stage:
                    on_stage(waiting_stage)
            returncode = process.wait()
            reader.join(timeout=1)
        if returncode != 0:
            errors.seek(0)
            raise RuntimeError('FFmpeg変換失敗: ' + errors.read()[-1200:])
    if on_progress:
        on_progress(expected, expected)


def _verify(path: Path, source_frames: int, target_fps: float) -> None:
    info = read_info(path)
    if info.frame_count != source_frames:
        raise RuntimeError(f'フレーム数が一致しません: {source_frames} → {info.frame_count}')
    # MP4 stream-copy can introduce sub-frame timestamp rounding.
    if abs(info.fps - target_fps) / target_fps > 0.00002:
        raise RuntimeError(f'出力FPSが一致しません: {target_fps:g} → {info.fps:g}')


def export_fps_optimized(
    source: str | Path,
    destination: str | Path,
    target_fps: float,
    on_progress: Progress | None = None,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    preset: str = 'veryfast',
    crf: int = DEFAULT_CRF,
    on_stage: Callable[[str], None] | None = None,
) -> FPSExportResult:
    """Write exactly [start_frame,end_frame) with new CFR playback timestamps.

    - Auto stream-copy is used only on full H.264 files, and verified.
    - Native FFmpeg transcode is frame-exact even for subranges, but must decode
      frames before the selected start because arbitrary AVI seeks are unreliable.
    - Audio is not included. The original media is never modified.
    """
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError('出力FPSは正の有限値にしてください。')
    if preset not in {'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'}:
        raise ValueError('不正なエンコード速度設定です。')
    crf = validate_crf(crf)
    src, dst = Path(source), Path(destination)
    if src.resolve() == dst.resolve():
        raise ValueError('元動画と出力先は別のファイルを指定してください。')
    info = read_info(src)
    if info.frame_count <= 0:
        raise ValueError('元動画のフレーム数が不明です。')
    interval = validate_frame_interval(start_frame,
                                      info.frame_count if end_frame is None else end_frame,
                                      info.frame_count)
    expected = interval.count
    dst.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(str(target_fps)).limit_denominator(1_000_000)
    output_rate = f'{rate.numerator}/{rate.denominator}'
    time_base = f'{rate.denominator}/{rate.numerator}'
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    base = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-nostats',
            '-stats_period', '0.25', '-progress', 'pipe:1', '-i', str(src),
            '-map', '0:v:0', '-an']
    whole = (interval.start == 0 and interval.stop == info.frame_count)
    if whole and _h264_source(src):
        ratio = (Fraction(str(info.fps)).limit_denominator(1_000_000) / rate)
        fraction = f'{ratio.numerator}/{ratio.denominator}'
        # Bitstream setts changes PTS, DTS and packet duration, no decoding.
        command = [*base, '-c:v', 'copy',
                   '-bsf:v', f'setts=pts=PTS*{fraction}:dts=DTS*{fraction}:duration=DURATION*{fraction}',
                   '-movflags', '+faststart', str(dst)]
        try:
            if on_stage:
                on_stage('再圧縮せずにMP4へ書き込み中')
            _run_ffmpeg(
                command, expected, on_progress, on_stage=on_stage,
                waiting_stage='再圧縮せずにMP4へ書き込み中',
            )
            _verify(dst, expected, target_fps)
            return FPSExportResult(expected, target_fps, 'stream_copy')
        except (RuntimeError, ValueError):
            dst.unlink(missing_ok=True)
            # Fall back to the native FFmpeg transcode for exotic streams.
            if on_progress:
                on_progress(0, expected)
    vf = (f'trim=start_frame={interval.start}:end_frame={interval.stop},'
          f'settb=AVTB,setpts=N/({output_rate}*TB),'
          'pad=ceil(iw/2)*2:ceil(ih/2)*2')
    command = [*base, '-vf', vf, '-r', output_rate, '-fps_mode:v', 'cfr',
               '-enc_time_base:v', time_base,
               '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst)]
    try:
        if interval.start:
            waiting_stage = (
                f'開始フレーム {interval.start:,} まで読み込み中'
                '（この間の出力進捗は0%です）'
            )
        else:
            waiting_stage = 'FFmpegでデコード・エンコードを準備中'
        if on_stage:
            on_stage(waiting_stage)
        _run_ffmpeg(
            command, expected, on_progress, on_stage=on_stage,
            waiting_stage=waiting_stage,
        )
        if on_stage:
            on_stage('出力ファイルのフレーム数・FPSを検証中')
        _verify(dst, expected, target_fps)
        return FPSExportResult(expected, target_fps, 'transcode')
    except Exception:
        dst.unlink(missing_ok=True)
        raise
