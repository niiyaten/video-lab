"""Smoke checks for the self-contained offline image-measurement viewer."""
from pathlib import Path


ASSET = Path(__file__).parents[1] / "assets" / "image_lab.html"
APP = Path(__file__).parents[1] / "app.py"


def test_image_lab_controls_present():
    html = ASSET.read_text(encoding="utf-8")
    for feature in (
        'id="fileA"', 'id="fileB"', 'id="fileRef"',
        'id="knownLength"', 'id="customUnit"',
        'id="alpha"', 'id="beta"', 'id="threshold"',
        'id="modeCal"', 'id="modeDistance"', 'id="modeAngle"',
        'value="binary"', 'value="contour"',
        'value="overlay"', 'value="difference"',
        'id="savePNG"', 'id="saveCSV"',
    ):
        assert feature in html, feature


def test_image_lab_accessible_without_video_upload():
    code = APP.read_text(encoding="utf-8")
    assert code.index('with image_tab:') < code.index('st.stop()')
    assert 'assets" / "image_lab.html' in code


def test_no_network_or_third_party_js_required():
    html = ASSET.read_text(encoding="utf-8")
    assert "<script src=" not in html
    assert 'fetch(' not in html
    assert 'XMLHttpRequest' not in html
