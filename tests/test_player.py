"""Tests for the standalone frame viewer and Streamlit embed builder."""

from pathlib import Path

from videolab.player import MAX_EMBED_BYTES, build_player_html


def test_standalone_template_has_interactive_controls():
    html = build_player_html()
    for marker in ('id="file"', 'id="sourceFps"', 'id="playFps"', 'id="timeline"',
                   'data-step="-10"', 'data-step="10"', 'ArrowLeft', 'ArrowRight',
                   'requestAnimationFrame', 'window.VIDEO_LAB_INITIAL'):
        assert marker in html
    assert 'videoData' not in html.split('window.VIDEO_LAB_INITIAL =', 1)[1].split(';', 1)[0]


def test_embed_small_video_and_source_fps(tmp_path: Path):
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'video test bytes')
    html = build_player_html(file_path=video, source_fps=800.0)
    assert '"fps": 800.0' in html
    assert '"name": "sample.mp4"' in html
    assert 'dmlkZW8gdGVzdCBieXRlcw==' in html


def test_dont_embed_large_video(tmp_path: Path):
    video = tmp_path / 'large.mp4'
    with video.open('wb') as f:
        f.seek(MAX_EMBED_BYTES)
        f.write(b'0')
    html = build_player_html(file_path=video, source_fps=100.0)
    assert '"fps": 100.0' in html
    assert '"videoData"' not in html


def test_escaped_name_cannot_inject_script(tmp_path: Path):
    video = tmp_path / 'evil_<script>.mp4'
    video.write_bytes(b'ok')
    html = build_player_html(file_path=video)
    assert '"name": "evil_<script>.mp4"' not in html
    assert 'evil_\\u003cscript\\u003e.mp4' in html


def test_player_allows_high_fps_without_2000_cap():
    html = build_player_html(source_fps=10000., capture_fps=10000.)
    assert '"fps": 10000.0' in html
    assert '"captureFps": 10000.0' in html
    assert 'max="2000"' not in html
    assert '元動画FPSは0.01～2000に設定' not in html
