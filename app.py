"""Streamlit front end. All image processing lives in videolab/."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit.components.v1 as components

from videolab.edge_ui import render_edge_counter_tab
from videolab.edge_ui import render_edge_counter_tab
from videolab.fps import change_playback_fps
from videolab.segments import seconds_to_interval
from videolab.optical_flow import FlowOptions, run_sparse_flow
from videolab.player import MAX_EMBED_BYTES, build_player_html
from videolab.preview import make_browser_preview
from videolab.point_selection import (
    make_point_picker_image, map_click_to_frame, read_exact_frame, snap_to_corner,
)
from videolab.video_io import read_info

@st.cache_data(show_spinner=False, max_entries=4)
def load_selection_frame(path: str, signature: str, index: int):
    # Signature distinguishes a newly uploaded file stored at the same path.
    return read_exact_frame(path, index)


st.set_page_config(page_title="Video Lab", page_icon="🎞️", layout="wide")
st.title("Video Lab | 動画計測ツール")
st.caption("ローカル実行用：動画再生・寸法校正・画像処理・エッジ解析・FPS変更・オプティカルフロー解析。")

if "workdir" not in st.session_state:
    st.session_state.workdir = str(Path(tempfile.mkdtemp(prefix="videolab_")))
if "results" not in st.session_state:
    st.session_state.results = {}

st.markdown("## 動画を読み込む")
st.caption("AVIなどブラウザ非対応動画も、Python側で再生用MP4に変換できます。大容量AVIはPC上のパスを指定するとメモリへのアップロードを避けられます。")
source_method = st.radio(
    "元動画の指定方法", ["ファイル選択（ブラウザ）", "PC上のファイルパス（ローカル実行向け）"],
    horizontal=True,
)
source_path = None
video = None
workdir = Path(st.session_state.workdir)
if source_method == "ファイル選択（ブラウザ）":
    uploaded = st.file_uploader(
        "FPS変更・解析にも使用する動画を選択", type=["mp4", "avi", "mov", "mkv", "m4v"],
        help="大容量動画はローカルパス指定の方が負荷を抑えられます。",
    )
    if uploaded is not None:
        buffer = uploaded.getbuffer()
        signature = (uploaded.name, len(buffer), hashlib.sha256(buffer).hexdigest())
        if st.session_state.get("signature") != signature:
            source_path = workdir / ("input" + Path(uploaded.name).suffix.lower())
            source_path.write_bytes(buffer)
            st.session_state.signature = signature
            st.session_state.source_path = str(source_path)
            st.session_state.results = {}
        source_path = Path(st.session_state.source_path)
else:
    input_path = st.text_input(
        "PC上の動画の絶対パス", placeholder=r"D:\Videos\camera_800fps.avi",
        help="Streamlitを起動しているPC上のファイルです。外部公開したWebサイトから利用者のPCを直接参照する機能ではありません。",
    ).strip().strip('"')
    if input_path:
        candidate = Path(input_path).expanduser()
        if candidate.is_file():
            signature = (str(candidate.resolve()), candidate.stat().st_size, candidate.stat().st_mtime_ns)
            if st.session_state.get("signature") != signature:
                st.session_state.signature = signature
                st.session_state.results = {}
            source_path = candidate
        else:
            st.error("指定したファイルが見つかりません。Streamlitを起動しているPC上のパスを指定してください。")

if source_path is not None:
    try:
        video = read_info(source_path)
    except (ValueError, RuntimeError) as exc:
        st.warning(f"OpenCVでメタデータを取得できません: {exc}。FFmpegのプレビュー変換は別途試せます。")
    if video is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("撮影動画FPS（メタデータ）", f"{video.fps:.3f}")
        c2.metric("解像度", f"{video.width} × {video.height}")
        c3.metric("フレーム数（推定）", f"{video.frame_count:,}" if video.frame_count else "不明")
        c4.metric("動画時間（推定）", f"{video.duration_s:.2f} 秒" if video.duration_s is not None else "不明")
    else:
        st.info("このファイルのFPS等をOpenCVから取得できません。FFmpeg変換が成功した場合でも撮影FPSは別途確認してください。")

player_tab, image_tab, edge_tab, fps_tab, flow_tab = st.tabs(["その場で再生・コマ送り", "画像計測・画像処理", "エッジ解析（Edge Counter）", "再生FPS変更・DL", "オプティカルフロー解析"])
with player_tab:
    st.subheader("その場で再生速度変更・フレーム送り")
    st.write("MP4（H.264）等は直接再生。AVIなどは**元FPSを維持**した互換MP4に変換して表示し、再生速度はプレーヤー内で変更できます。")
    preview_result = st.session_state.results.get("browser_preview")
    if source_path is not None:
        with st.expander("AVI・独自コーデック向け：ブラウザ再生用MP4を作る", expanded=source_path.suffix.lower() == ".avi"):
            st.caption("元動画のFPS・再生時間・フレーム順序を原則維持したまま、H.264 MP4へ変換します。元動画は変更しません。")
            st.info(
                f"元動画の記録FPS：{video.fps:g} FPS（OpenCVで取得） → 変換後も原則同じFPS"
                if video else "元動画のFPSを変更せず、元タイムスタンプを引き継ぎます。"
            )
            frame_limit = st.number_input(
                "先頭から変換するフレーム数（0で全フレーム・初期値）",
                min_value=0, max_value=2000000, value=0, step=100,
                help="例えば600を指定した場合、元動画の先頭600フレームだけを取り出します。FPSは変えません。",
            )
            st.caption("※ 一部フレームだけ変換する場合は動画時間が短くなります。映像はH.264で再圧縮され、画素値やコーデックは変わります。")
            if st.button("互換MP4を作成してプレーヤーで開く", type="primary", key="preview_button"):
                output = workdir / "browser_preview.mp4"
                try:
                    with st.spinner("FFmpegで動画を変換中…"):
                        result = make_browser_preview(source_path, output, max_frames=int(frame_limit))
                    st.session_state.results["browser_preview"] = {
                        "path": str(result.path), "fps": result.playback_fps, "frames": result.frame_count,
                    }
                    preview_result = st.session_state.results["browser_preview"]
                    st.success(f"変換成功：{result.frame_count:,}フレーム / {result.playback_fps:g} FPS。元動画を保持しています。")
                    if video is not None:
                        st.caption(
                            f"変換前 {video.fps:g} FPS / {video.frame_count:,}フレーム / "
                            f"推定{video.duration_s:.4f}秒 → 変換後 {result.playback_fps:g} FPS / "
                            f"{result.frame_count:,}フレーム / "
                            f"推定{result.frame_count / result.playback_fps:.4f}秒"
                        )
                except Exception as exc:
                    st.session_state.results.pop("browser_preview", None)
                    preview_result = None
                    st.error(f"変換失敗: {exc}")
            if preview_result and Path(preview_result["path"]).is_file():
                st.caption(f"準備済み：{preview_result['frames']:,} フレーム、{preview_result['fps']:g} FPS")
                st.download_button(
                    "プレビューMP4を保存", data=Path(preview_result["path"]).read_bytes(),
                    file_name="browser_preview.mp4", mime="video/mp4",
                )

    playback_path = Path(preview_result["path"]) if preview_result and Path(preview_result["path"]).is_file() else source_path
    playback_fps = preview_result["fps"] if preview_result and playback_path is not None and playback_path != source_path else (video.fps if video else None)
    capture_fps = video.fps if video and preview_result and playback_path != source_path else None
    if playback_path is not None and playback_path.stat().st_size > MAX_EMBED_BYTES:
        st.info("この動画は埋め込み上限を超えるため自動表示しません。プレーヤー内の『動画を開く』で互換MP4を指定してください。")
    elif playback_path is not None:
        st.caption("上で選択した動画を自動表示します。AVI等が再生できない場合は互換MP4を作成してください。")
    components.html(
        build_player_html(file_path=playback_path, source_fps=playback_fps, capture_fps=capture_fps),
        height=800, scrolling=True,
    )
    st.caption("※ 元AVIの直接再生はブラウザのコーデックに依存します。MP4に変換してもコマ送り精度はブラウザのシーク精度に依存します。厳密なフレーム計測には元AVIをPythonで直接デコードしてください。")

with image_tab:
    st.subheader("既知寸法からの校正・簡易画像処理・画像比較")
    st.write("定規以外の既知寸法も使えます。画像A/B、必要に応じて別撮りの校正画像を読み込んでください。")
    components.html(
        (Path(__file__).parent / "assets" / "image_lab.html").read_text(encoding="utf-8"),
        height=1130, scrolling=True,
    )
    st.caption("画像計測はブラウザ内で実行します。校正値は画像の撮影倍率・解像度・計測面が同一の場合のみ他画像へ適用できます。")


with edge_tab:
    render_edge_counter_tab(source_path, video)


with edge_tab:
    render_edge_counter_tab(source_path, video)


def make_progress():
    bar = st.progress(0.0, text="処理を開始します…")

    def update(n: int, total: int) -> None:
        if total > 0:
            bar.progress(min(1.0, n / total), text=f"処理中 {n:,} / 約{total:,} フレーム")
        else:
            bar.progress(0.0, text=f"処理中 {n:,} フレーム")

    return bar, update


if video is None or source_path is None:
    with fps_tab:
        st.info("FPS変更・保存するには画面上部から動画を選んでください。")
    with flow_tab:
        st.info("解析するには画面上部から動画を選んでください。")
    st.stop()


# The FPS export selector keeps a canonical half-open frame range in state.
# The slider and numeric inputs are alternative views of the same interval.
def _set_export_selection(start: int, stop: int) -> None:
    total = st.session_state["fps_export_total"]
    source_fps = st.session_state["fps_export_source_fps"]
    start = min(max(0, int(start)), total - 1)
    stop = min(max(start + 1, int(stop)), total)
    st.session_state["fps_export_start"] = start
    st.session_state["fps_export_stop"] = stop
    st.session_state["fps_export_frame_slider"] = (start, stop - 1)
    st.session_state["fps_export_frame_from"] = start
    st.session_state["fps_export_frame_through"] = stop - 1
    st.session_state["fps_export_sec_slider"] = (start / source_fps, stop / source_fps)
    st.session_state["fps_export_sec_from"] = start / source_fps
    st.session_state["fps_export_sec_to"] = stop / source_fps


def _fps_frame_slider_changed() -> None:
    first, last = st.session_state["fps_export_frame_slider"]
    _set_export_selection(first, last + 1)


def _fps_frame_from_changed() -> None:
    start = int(st.session_state["fps_export_frame_from"])
    stop = max(start + 1, st.session_state["fps_export_stop"])
    _set_export_selection(start, stop)


def _fps_frame_through_changed() -> None:
    stop = int(st.session_state["fps_export_frame_through"]) + 1
    start = min(st.session_state["fps_export_start"], stop - 1)
    _set_export_selection(start, stop)


def _fps_apply_seconds(start_s: float, stop_s: float) -> None:
    try:
        result = seconds_to_interval(
            start_s, stop_s,
            st.session_state["fps_export_source_fps"],
            st.session_state["fps_export_total"],
        )
    except ValueError as exc:
        st.session_state["fps_export_notice"] = str(exc)
        _set_export_selection(st.session_state["fps_export_start"], st.session_state["fps_export_stop"])
    else:
        st.session_state["fps_export_notice"] = ""
        _set_export_selection(result.start, result.stop)


def _fps_sec_slider_changed() -> None:
    _fps_apply_seconds(*st.session_state["fps_export_sec_slider"])


def _fps_sec_from_changed() -> None:
    start_s = float(st.session_state["fps_export_sec_from"])
    stop_s = max(float(st.session_state["fps_export_sec_to"]),
                 min(st.session_state["fps_export_total"] / st.session_state["fps_export_source_fps"],
                     start_s + 1 / st.session_state["fps_export_source_fps"]))
    _fps_apply_seconds(start_s, stop_s)


def _fps_sec_to_changed() -> None:
    stop_s = float(st.session_state["fps_export_sec_to"])
    start_s = min(float(st.session_state["fps_export_sec_from"]),
                  max(0.0, stop_s - 1 / st.session_state["fps_export_source_fps"]))
    _fps_apply_seconds(start_s, stop_s)


def _init_fps_export_selection(signature: object, total: int, source_fps: float) -> None:
    movie_token = (repr(signature), total, source_fps)
    if st.session_state.get("fps_export_movie") != movie_token:
        st.session_state["fps_export_movie"] = movie_token
        st.session_state["fps_export_total"] = total
        st.session_state["fps_export_source_fps"] = source_fps
        st.session_state["fps_export_notice"] = ""
        st.session_state["fps_export_target"] = float(source_fps)
        _set_export_selection(0, total)

with fps_tab:
    st.subheader("指定区間の再生FPSを変更してMP4保存")
    st.write("**元の記録FPSを維持したMP4変換**は『その場で再生・コマ送り』タブで行います。"
             "このタブでは、選んだ区間だけを取り出し、**全フレームを保持して再生時間を変更**します。")

    if video.frame_count <= 0:
        st.warning("元動画のフレーム数を取得できませんでした。区間指定には固定FPSと正しいフレーム数が必要です。")
    else:
        _init_fps_export_selection(st.session_state.get("signature"), video.frame_count, video.fps)
        target_fps = st.number_input(
            "出力再生FPS（上限設定なし）", min_value=0.001,
            step=1.0, format="%.6f", key="fps_export_target",
            help="2000 FPSなどのアプリ独自上限はありません。極端なFPSはH.264/MP4のエンコーダー・再生ソフトで扱えない場合があります。",
        )
        speed_ratio = target_fps / video.fps
        if speed_ratio < 1:
            speed_summary = f"{speed_ratio:.6g} 倍速（再生時間は {1 / speed_ratio:.6g} 倍）"
        elif speed_ratio > 1:
            speed_summary = f"{speed_ratio:.6g} 倍速"
        else:
            speed_summary = "1 倍速（等速）"
        st.info(
            f"元動画に対する再生速度：**{speed_summary}** "
            f"（元 {video.fps:g} FPS → 出力 {target_fps:g} FPS）"
        )
        st.caption("記録FPSと出力FPSは独立です。音声は出力しません。元動画自体は変更されません。")
        st.markdown("#### 出力範囲を指定")
        mode = st.radio(
            "指定単位", ["フレーム番号", "動画の秒数"], horizontal=True,
            key="fps_export_mode", help="どちらで指定しても、最終的には元動画のフレーム境界に合わせます。",
        )
        last = video.frame_count - 1
        duration = video.frame_count / video.fps
        if mode == "フレーム番号":
            if last:
                st.slider(
                    "抽出範囲（0始まり、両端を含むフレーム番号）", 0, last,
                    key="fps_export_frame_slider", on_change=_fps_frame_slider_changed,
                )
            else:
                st.caption("この動画は1フレームのため、抽出区間は0〜0です。")
            c0, c1 = st.columns(2)
            with c0:
                st.number_input(
                    "開始フレーム（含む）", min_value=0, max_value=last, step=1,
                    key="fps_export_frame_from", on_change=_fps_frame_from_changed,
                )
            with c1:
                st.number_input(
                    "終了フレーム（含む）", min_value=0, max_value=last, step=1,
                    key="fps_export_frame_through", on_change=_fps_frame_through_changed,
                )
        else:
            seconds_step = min(0.001, 1 / video.fps)
            st.slider(
                "抽出範囲（元動画の秒数、終了時刻は含まない）",
                0.0, float(duration), step=float(seconds_step), format="%.6f 秒",
                key="fps_export_sec_slider", on_change=_fps_sec_slider_changed,
            )
            c0, c1 = st.columns(2)
            with c0:
                st.number_input(
                    "開始時刻（秒）", min_value=0.0, max_value=float(duration),
                    step=float(seconds_step), format="%.6f", key="fps_export_sec_from",
                    on_change=_fps_sec_from_changed,
                )
            with c1:
                st.number_input(
                    "終了時刻（秒、境界は含まない）", min_value=0.0, max_value=float(duration),
                    step=float(seconds_step), format="%.6f", key="fps_export_sec_to",
                    on_change=_fps_sec_to_changed,
                )
            st.caption("秒数は元動画の先頭を0秒として指定します。時刻をフレーム境界に丸めて出力するため、入力値と実際の選択区間には最大1フレーム程度の差が出ます。")
        if st.session_state["fps_export_notice"]:
            st.warning(st.session_state["fps_export_notice"])

        first = st.session_state["fps_export_start"]
        stop = st.session_state["fps_export_stop"]
        selected = stop - first
        out_seconds = selected / target_fps
        ca, cb, cc = st.columns(3)
        ca.metric("出力するフレーム数", f"{selected:,}")
        cb.metric("元動画での区間", f"{selected / video.fps:.4f} 秒")
        cc.metric("出力動画の再生時間", f"{out_seconds:.4f} 秒")
        st.caption(
            f"出力対象：元動画の **{first:,}〜{stop - 1:,} フレーム**（両端を含む） / "
            f"元動画 **{first / video.fps:.6f}〜{stop / video.fps:.6f} 秒** "
            f"→ {target_fps:g} FPSで出力。区間外のフレームは出力しません。"
        )
        if st.button("開始・終了フレームの画像を確認（必要な場合）", key="fps_show_boundary_frames"):
            st.session_state["fps_boundary_preview"] = (repr(st.session_state.get("signature")), first, stop)
        if st.session_state.get("fps_boundary_preview") == (
            repr(st.session_state.get("signature")), first, stop
        ):
            try:
                with st.spinner("指定フレームを元動画から読み込んでいます…"):
                    first_image = load_selection_frame(str(source_path), repr(st.session_state.get("signature")), first)
                    last_image = load_selection_frame(str(source_path), repr(st.session_state.get("signature")), stop - 1)
                p0, p1 = st.columns(2)
                with p0:
                    st.image(cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB), caption=f"開始フレーム {first:,}", use_container_width=True)
                with p1:
                    st.image(cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB), caption=f"終了フレーム {stop - 1:,}", use_container_width=True)
            except Exception as exc:
                st.warning(f"フレーム画像の表示に失敗しました: {exc}")
        st.caption("画像確認は任意です。後半フレームの読み出しには時間がかかる場合があります。")
        if st.button("選択区間のFPSを変更してMP4を作成", type="primary", key="fps_button"):
            out = workdir / "playback_changed.mp4"
            bar, on_progress = make_progress()
            try:
                with st.spinner("選択したフレームを順番に読み取り、H.264へ変換中…"):
                    count = change_playback_fps(
                        source_path, out, float(target_fps), on_progress,
                        start_frame=int(first), end_frame=int(stop),
                    )
                bar.empty()
                st.session_state.results["fps"] = {
                    "path": str(out), "fps": target_fps, "count": count,
                    "start": first, "stop": stop,
                }
                st.success(f"完了：元 {first:,}〜{stop-1:,} フレーム / "
                           f"{count:,}フレーム / {target_fps:g} FPS / {count / target_fps:.4f}秒")
            except Exception as exc:
                bar.empty()
                st.session_state.results.pop("fps", None)
                st.error(f"処理失敗: {exc}")
        result = st.session_state.results.get("fps")
        if result and Path(result["path"]).is_file():
            st.write(f"作成済み動画：**元 {result['start']:,}〜{result['stop'] - 1:,}フレーム / "
                     f"{result['fps']:g} FPS**")
            output_path = Path(result["path"])
            if output_path.stat().st_size <= MAX_EMBED_BYTES:
                st.video(str(output_path))
            else:
                st.info("出力ファイルが大容量のため、画面への自動埋め込み再生は省略しました。MP4を保存して確認してください。")
            st.download_button(
                "選択区間のMP4を保存", data=output_path.read_bytes(),
                file_name=f"segment_{result['start']}-{result['stop']-1}_{result['fps']:g}fps.mp4",
                mime="video/mp4",
            )
    st.caption("計測用の元動画FPS・各フレームの位置は変更しません。大きな開始フレーム番号では、正確に読み取るため先頭からのデコードに時間がかかります。")

with flow_tab:
    st.subheader("オプティカルフロー：追跡点の指定")
    st.write("**手動選択がデフォルト**です。画像をクリックして追跡点を登録します。自動補正をOFFにするとクリック座標そのままから追跡します。")

    # A different input movie must not inherit the previous movie's pixel coordinates.
    signature_token = repr(st.session_state.get("signature"))
    if st.session_state.get("flow_movie_signature") != signature_token:
        st.session_state.flow_movie_signature = signature_token
        st.session_state.flow_points = []
        st.session_state.flow_click_event = None
        st.session_state.flow_points_frame = 0
        st.session_state.pop("flow_start_frame", None)

    tracking_mode = st.radio(
        "追跡点の決め方",
        ["手動で選択（デフォルト）", "画面全体から自動検出"],
        index=0, horizontal=True, key="flow_tracking_mode",
    )
    manual = tracking_mode.startswith("手動")

    col_a, col_b = st.columns(2)
    with col_a:
        last_frame = max(0, video.frame_count - 1)
        start_frame = int(st.number_input(
            "追跡を開始するフレーム（0始まり）", min_value=0,
            max_value=last_frame if video.frame_count > 0 else 1_000_000,
            value=0, step=1, key="flow_start_frame",
        ))
    with col_b:
        flow_playback_fps = st.number_input(
            "結果動画の再生FPS", min_value=0.1,
            value=float(round(min(video.fps, 30.0), 2)), step=1.0,
            format="%.2f", key="flow_output_fps",
            help="CSVの計測時間・速度には影響しません。",
        )

    if st.session_state.flow_points_frame != start_frame:
        st.session_state.flow_points = []
        st.session_state.flow_click_event = None
        st.session_state.flow_points_frame = start_frame
        st.session_state.results.pop("flow", None)

    frame_bgr = None
    try:
        frame_bgr = load_selection_frame(str(source_path), signature_token, start_frame)
    except Exception as exc:
        st.error(f"点指定用フレームを読み込めませんでした: {exc}")

    if manual and frame_bgr is not None:
        st.markdown("#### 動画の画像上で点を選択")
        st.caption("黄色い番号が登録済みの追跡点です。手動モードでは追跡喪失後に別の点へ自動で切り替わりません（下の追加設定で変更可能）。")
        control, settings = st.columns([3, 2], gap="large")
        with settings:
            auto_snap = st.checkbox(
                "クリック付近の特徴点へ自動補正", value=False, key="flow_auto_snap",
                help="ON: 指定位置の近傍から追跡に適したShi–Tomasi特徴点を1点選択。OFF: 指定座標を変更しません。",
            )
            search_radius = 15
            if auto_snap:
                search_radius = int(st.slider("近傍探索半径（元画像px）", 3, 100, 15, key="flow_snap_radius"))
            st.metric("現在の追跡点", f"{len(st.session_state.flow_points)} 点")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("直前の点を取り消す", disabled=not st.session_state.flow_points, key="flow_undo"):
                    st.session_state.flow_points.pop()
                    st.session_state.results.pop("flow", None)
                    st.rerun()
            with b2:
                if st.button("すべての点を削除", disabled=not st.session_state.flow_points, key="flow_clear"):
                    st.session_state.flow_points = []
                    st.session_state.results.pop("flow", None)
                    st.rerun()
            if st.session_state.flow_points:
                st.dataframe(
                    [{"ID": i, "x [px]": round(p[0], 2), "y [px]": round(p[1], 2)}
                     for i, p in enumerate(st.session_state.flow_points, 1)],
                    hide_index=True, use_container_width=True, height=210,
                )
        with control:
            rendered, rendered_width = make_point_picker_image(frame_bgr, st.session_state.flow_points)
            click = streamlit_image_coordinates(
                rendered, width=rendered_width,
                key=f"flow_image_{hashlib.sha256(signature_token.encode()).hexdigest()[:12]}_{start_frame}",
                cursor="crosshair",
            )
            if click:
                click_event_id = (click.get("unix_time"), click.get("x"), click.get("y"))
                if click_event_id != st.session_state.flow_click_event:
                    st.session_state.flow_click_event = click_event_id
                    try:
                        xy = map_click_to_frame(click, video.width, video.height)
                        if auto_snap:
                            snapped = snap_to_corner(frame_bgr, xy, radius_px=search_radius)
                            if snapped is None:
                                st.warning("近傍に有効な特徴点がありません。別の場所をクリックするか、自動補正をOFFにしてください。")
                            else:
                                xy = snapped
                        if not auto_snap or snapped is not None:
                            if any((xy[0]-p[0])**2+(xy[1]-p[1])**2 < 2**2
                                   for p in st.session_state.flow_points):
                                st.warning("既存の追跡点と近すぎます。別の位置をクリックしてください。")
                            else:
                                st.session_state.flow_points.append(xy)
                                st.session_state.results.pop("flow", None)
                                st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
            st.caption(f"元画像: {video.width}×{video.height} px / 表示画像: {rendered.width}×{rendered.height} px。クリック座標は元画像の座標へ換算します。")
    elif frame_bgr is not None:
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="自動検出に使用する開始フレーム", width=750)
        st.info("このモードは従来どおり、画面内の特徴点を自動抽出します。")

    st.markdown("#### 追跡・出力設定")
    left, right = st.columns(2)
    with left:
        trail_length = st.number_input("軌跡長（フレーム）", 1, 200, 15, 1, key="flow_trail_length")
        show_ids = st.checkbox("特徴点IDを動画に表示", value=True, key="flow_show_ids")
        if manual:
            auto_replenish = st.checkbox(
                "追跡中に新しい点を自動追加（通常はOFF）", value=False, key="flow_auto_replenish",
                help="ONにすると指定した点とは別のIDをもつ新規特徴点が追加されます。指定点だけを追跡する場合はOFFです。",
            )
            max_points = int(st.number_input("総同時追跡点数の上限", 1, 1000, 150, 10, key="flow_manual_max_points")) if auto_replenish else max(1, len(st.session_state.flow_points))
        else:
            auto_replenish = True
            max_points = int(st.number_input("最大同時追跡点数", 1, 1000, 150, 10, key="flow_auto_max_points"))
    with right:
        min_distance = st.number_input("特徴点間の最小間隔（px）", 1, 100, 12, 1, key="flow_min_distance")
        quality = st.number_input("自動検出時の特徴点品質しきい値", 0.001, 0.5, 0.01, 0.005,
                                  format="%.3f", key="flow_quality")
        redetect = st.number_input("点の追加間隔（フレーム）", 1, 500, 30, 1,
                                    key="flow_redetect", disabled=not auto_replenish)

    use_roi = st.checkbox("解析範囲（ROI）を数値指定", value=False, key="flow_use_roi")
    roi = None
    if use_roi:
        a, b, c, d = st.columns(4)
        x0 = a.number_input("左 x0", min_value=0, max_value=max(0, video.width - 1), value=0, key="flow_roi_x0")
        y0 = b.number_input("上 y0", min_value=0, max_value=max(0, video.height - 1), value=0, key="flow_roi_y0")
        x1 = c.number_input("右 x1（外端）", min_value=1, max_value=video.width, value=video.width, key="flow_roi_x1")
        y1 = d.number_input("下 y1（外端）", min_value=1, max_value=video.height, value=video.height, key="flow_roi_y1")
        roi = (int(x0), int(y0), int(x1), int(y1))
    st.caption("CSVの原点は画像左上、Yは下向き正、速度はpx/s。追跡点IDは固定ですが、追跡に失敗した点はstatus=0で終了します。")
    can_run = frame_bgr is not None and (not manual or bool(st.session_state.flow_points))
    if st.button("解析して動画・CSVを作成", type="primary", disabled=not can_run, key="flow_button"):
        out_mp4 = workdir / "flow_overlay.mp4"
        out_csv = workdir / "flow_tracks.csv"
        bar, on_progress = make_progress()
        try:
            opts = FlowOptions(
                max_points=max_points, min_distance_px=int(min_distance),
                quality_level=float(quality), trail_length=int(trail_length),
                redetect_interval=int(redetect), show_ids=bool(show_ids),
                output_fps=float(flow_playback_fps), roi=roi,
                initial_points=tuple(st.session_state.flow_points) if manual else None,
                auto_replenish=bool(auto_replenish), start_frame=start_frame,
            )
            with st.spinner("元動画を順番にデコードして追跡中…"):
                count, n_ids = run_sparse_flow(source_path, out_mp4, out_csv, opts, on_progress)
            bar.empty()
            st.session_state.results["flow"] = {"mp4": str(out_mp4), "csv": str(out_csv), "count": count, "ids": n_ids}
            st.success(f"解析完了：{count:,}フレーム / 追跡点ID {n_ids:,}件")
        except Exception as exc:
            bar.empty()
            st.session_state.results.pop("flow", None)
            st.error(f"処理失敗: {exc}")
    if manual and not can_run:
        st.info("動画の画像上をクリックして、1点以上指定してください。")
    result = st.session_state.results.get("flow")
    if result and Path(result["mp4"]).is_file() and Path(result["csv"]).is_file():
        st.video(result["mp4"])
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("追跡可視化MP4を保存", data=Path(result["mp4"]).read_bytes(),
                               file_name="flow_overlay.mp4", mime="video/mp4")
        with c2:
            st.download_button("座標・速度CSVを保存", data=Path(result["csv"]).read_bytes(),
                               file_name="flow_tracks.csv", mime="text/csv")
        st.caption("status=1 は追跡中、status=0 は追跡喪失。開始フレームでの変位と速度は空欄です。")

st.divider()
st.caption("ローカルPCで起動した場合、動画は同じPC上のPythonプロセスで処理されます。外部公開する場合はアクセス制御を別途設定してください。")
