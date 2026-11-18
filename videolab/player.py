"""Interactive in-browser video viewer. JavaScript lives in assets/frame_player.html."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

PLAYER_TEMPLATE = Path(__file__).resolve().parent.parent / 'assets' / 'frame_player.html'
MAX_EMBED_BYTES = 16 * 1024 * 1024

MIME_TYPES = {
    '.mp4': 'video/mp4',
    '.m4v': 'video/mp4',
    '.mov': 'video/quicktime',
    '.avi': 'video/x-msvideo',
    '.mkv': 'video/x-matroska',
}


def build_player_html(*, file_path: str | Path | None = None,
                      source_fps: float | None = None,
                      capture_fps: float | None = None) -> str:
    """Generate the offline-capable UI. Embed small uploads for one-click playback.

    Large videos can be opened using the in-player file chooser without copying
    their contents through Streamlit's websocket/iframe markup.
    """
    data: dict[str, str | float] = {}
    if source_fps is not None and math.isfinite(source_fps) and source_fps > 0:
        data['fps'] = round(float(source_fps), 7)
    if capture_fps is not None and math.isfinite(capture_fps) and capture_fps > 0:
        data['captureFps'] = round(float(capture_fps), 7)
    if file_path is not None:
        path = Path(file_path)
        if path.stat().st_size <= MAX_EMBED_BYTES:
            data['videoData'] = base64.b64encode(path.read_bytes()).decode('ascii')
            data['mime'] = MIME_TYPES.get(path.suffix.lower(), 'application/octet-stream')
            data['name'] = path.name
    # JSON is quoted so uploaded names can never break out into executable JS.
    js = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e')
    return PLAYER_TEMPLATE.read_text(encoding='utf-8').replace(
        '/*__INIT__*/', 'window.VIDEO_LAB_INITIAL = ' + js + ';'
    )
