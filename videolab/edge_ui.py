"""Streamlit tab for Edge Counter-style image / selected-frame analysis."""
from __future__ import annotations

import csv
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from .edge_counter import EdgeOptions, analyze_edges
from .point_selection import read_exact_frame


def _csv_bytes(rows: list[dict], headers: list[str]) -> bytes:
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def _png(image_bgr_or_gray: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", image_bgr_or_gray)
    if not ok:
        raise ValueError("PNG encoding failed")
    return data.tobytes()


def _read_source(video_path: Path | None, video_info):
    choices = ["静止画像を読み込む", "PC上の画像パスを指定"]
    if video_path is not None and video_info is not None:
        choices.append("上で読み込んだ動画の1フレーム")
    mode = st.radio("解析する画像", choices, horizontal=True, key="edge_source_mode")
    if mode == "静止画像を読み込む":
        f = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"], key="edge_image")
        if f is None:
            return None, None
        data = np.frombuffer(f.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image, f.name
    if mode == "PC上の画像パスを指定":
        path = st.text_input("画像ファイルの絶対パス", placeholder=r"D:\\Images\\sample.tif", key="edge_image_path").strip().strip('"')
        if not path:
            return None, None
        p = Path(path).expanduser()
        if not p.is_file():
            st.warning("画像ファイルが見つかりません。")
            return None, None
        image = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        return image, p.name
    frame_count = int(video_info.frame_count or 1)
    idx = int(st.number_input("解析フレーム（0始まり）", min_value=0, max_value=max(0, frame_count - 1), value=0, step=1, key="edge_frame_index"))
    try:
        return read_exact_frame(video_path, idx), f"{video_path.name}: frame {idx}"
    except Exception as exc:
        st.error(f"指定フレームの読込に失敗しました: {exc}")
        return None, None


def _roi_from_click(image: np.ndarray) -> None:
    """Store ROI on source-image coordinates, using click coordinates to pick 2 corners."""
    h, w = image.shape[:2]
    if "edge_roi_corners" not in st.session_state:
        st.session_state.edge_roi_corners = []
    left, right = st.columns([1, 2])
    with left:
        st.caption("画像を2回クリックして、ROIの対角2点を指定します。")
        if st.button("ROI選択をリセット", key="edge_clear_roi_clicks"):
            st.session_state.edge_roi_corners = []
            st.session_state.edge_last_click = None
            st.rerun()
        if len(st.session_state.edge_roi_corners) >= 2:
            a, b = st.session_state.edge_roi_corners[-2:]
            st.caption(f"クリックROI：{a} / {b}")
        else:
            st.caption(f"クリック済み：{len(st.session_state.edge_roi_corners)} / 2")
    with right:
        display_width = min(850, w)
        display_height = max(1, round(h * display_width / w))
        display = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        click = streamlit_image_coordinates(display_rgb, width=display_width, key="edge_roi_selector", cursor="crosshair")
        if click:
            event = (click.get("unix_time"), click.get("x"), click.get("y"))
            if event != st.session_state.get("edge_last_click"):
                st.session_state.edge_last_click = event
                x = max(0, min(w - 1, int(round(float(click["x"]) * w / display_width))))
                y = max(0, min(h - 1, int(round(float(click["y"]) * h / display_height))))
                pts = st.session_state.edge_roi_corners
                st.session_state.edge_roi_corners = (pts + [(x, y)])[-2:]
                st.rerun()


def render_edge_counter_tab(video_path: Path | None, video_info) -> None:
    st.subheader("Edge Counter | エッジ・輪郭解析")
    st.caption("旧Tkinter版の最終仕様を基に再実装。元画像・元動画を変更せず、輪郭マスクと座標CSVを出力します。")
    image, source_name = _read_source(video_path, video_info)
    if image is None:
        st.info("解析用の静止画像を読み込むか、動画からフレームを選択してください。")
        return
    h, w = image.shape[:2]
    st.caption(f"対象：{source_name}　解像度：{w} × {h} px")

    with st.expander("ROI（解析対象範囲）", expanded=False):
        use_roi = st.checkbox("ROIを使用", value=False, key="edge_use_roi")
        # Numerical coordinates take priority after manual editing. Click selector
        # can be used to initially determine the two corner coordinates.
        if use_roi:
            roi_method = st.radio("ROI指定方法", ["数値指定", "画像を2点クリック"], horizontal=True, key="edge_roi_method")
            if roi_method == "画像を2点クリック":
                _roi_from_click(image)
                pts = st.session_state.get("edge_roi_corners", [])
                if len(pts) >= 2:
                    (a, b) = pts[-2:]
                    roi = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]) + 1, max(a[1], b[1]) + 1)
                else:
                    roi = None
                    st.warning("ROIの対角2点を指定してください。")
            else:
                c1, c2, c3, c4 = st.columns(4)
                x0 = int(c1.number_input("左 x0", min_value=0, max_value=w - 1, value=0, key="edge_x0"))
                y0 = int(c2.number_input("上 y0", min_value=0, max_value=h - 1, value=0, key="edge_y0"))
                x1 = int(c3.number_input("右 x1（外端）", min_value=1, max_value=w, value=w, key="edge_x1"))
                y1 = int(c4.number_input("下 y1（外端）", min_value=1, max_value=h, value=h, key="edge_y1"))
                roi = (x0, y0, x1, y1)
        else:
            roi = None

    left, right = st.columns(2)
    with left:
        st.markdown("#### 前処理・輪郭入力")
        alpha = st.number_input("α（コントラスト）", min_value=0.0, value=1.0, step=0.1, key="edge_alpha")
        beta = st.number_input("β（明るさ）", value=0.0, step=5.0, key="edge_beta")
        blur_mode = st.selectbox("ぼかし", ["OFF", "Gaussian", "Median"], key="edge_blur")
        blur_kernel = int(st.select_slider("ぼかしカーネル", [1, 3, 5, 7, 9, 11], value=3, disabled=blur_mode == "OFF", key="edge_blur_k"))
        input_mode = st.radio("輪郭抽出入力", ["Binary", "Canny"], horizontal=True, key="edge_input_mode")
        if input_mode == "Binary":
            threshold = int(st.slider("二値化しきい値", 0, 255, 127, key="edge_threshold"))
            invert = st.checkbox("二値化を反転", value=False, key="edge_invert")
            canny_low, canny_high = 50, 150
        else:
            canny_low = int(st.slider("Canny 下限", 0, 255, 50, key="edge_canny_low"))
            canny_high = int(st.slider("Canny 上限", 0, 255, 150, key="edge_canny_high"))
            threshold, invert = 127, False
    with right:
        st.markdown("#### 形態学処理・対象輪郭")
        morph_mode = st.selectbox("morphology", ["None", "Open", "Close", "Dilate", "Erode"], key="edge_morph")
        morph_kernel = int(st.select_slider("形態学カーネル", [1, 3, 5, 7, 9], value=3, disabled=morph_mode == "None", key="edge_morph_k"))
        morph_iter = int(st.number_input("処理回数", min_value=1, max_value=20, value=1, disabled=morph_mode == "None", key="edge_morph_iter"))
        min_area = float(st.number_input("最小面積（px²）", min_value=0.0, value=0.0, key="edge_min_area"))
        max_area = float(st.number_input("最大面積フィルタ（px²、0で無制限）", min_value=0.0, value=0.0, key="edge_max_area"))
        target = st.selectbox("対象輪郭", ["All", "Max", "Top N"], key="edge_target")
        top_n = int(st.number_input("上位N輪郭", min_value=1, max_value=10000, value=5, disabled=target != "Top N", key="edge_top_n"))
        points_mode = st.selectbox("点列出力", ["None", "Edge Pixels", "Left Envelope", "Right Envelope"], key="edge_point_mode")
        largest_box = st.checkbox("最大輪郭の外接矩形を表示", value=False, key="edge_box")
        remove_border = st.checkbox("輪郭後に枠を消す（ROI・画像端）", value=True, key="edge_remove_border")
        border_margin = int(st.number_input("除去幅（px）", min_value=0, max_value=1000, value=1, disabled=not remove_border, key="edge_border_margin"))

    try:
        opts = EdgeOptions(alpha=float(alpha), beta=float(beta), blur_mode=blur_mode, blur_kernel=blur_kernel,
                           input_mode=input_mode, threshold=threshold, invert=invert,
                           canny_low=canny_low, canny_high=canny_high, morph_mode=morph_mode,
                           morph_kernel=morph_kernel, morph_iterations=morph_iter,
                           min_area=min_area, max_area=max_area, target=target, top_n=top_n,
                           contour_points=points_mode, roi=roi, remove_border=remove_border,
                           border_margin=border_margin, draw_largest_bbox=largest_box)
        result = analyze_edges(image, opts)
    except Exception as exc:
        st.error(f"解析設定を確認してください: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("抽出輪郭数", f"{len(result.stats):,}")
    c2.metric("輪郭画素数", f"{int(np.count_nonzero(result.contour_mask)):,}")
    c3.metric("出力点数", f"{len(result.point_rows):,}")
    view_choice = st.radio("表示画像", ["輪郭重畳", "Contour Input", "Contour Mask", "α/β調整後"],
                           horizontal=True, key="edge_view")
    views = {"輪郭重畳": result.overlay, "Contour Input": result.contour_input,
             "Contour Mask": result.contour_mask, "α/β調整後": result.adjusted}
    shown = views[view_choice]
    show_rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB) if shown.ndim == 3 else shown
    st.image(show_rgb, caption=view_choice, use_container_width=True)
    if result.stats:
        st.dataframe(result.stats, hide_index=True, use_container_width=True)

    st.markdown("#### 結果のダウンロード")
    btns = st.columns(4)
    btns[0].download_button("重畳PNG", data=_png(result.overlay), file_name="edge_overlay.png", mime="image/png", key="edge_dl_overlay")
    btns[1].download_button("輪郭Mask PNG", data=_png(result.contour_mask), file_name="edge_contour_mask.png", mime="image/png", key="edge_dl_mask")
    btns[2].download_button("輪郭統計CSV", data=_csv_bytes(result.stats, ["contour_id", "area_px2", "perimeter_px", "x_px", "y_px", "width_px", "height_px", "visible_edge_pixels"]),
                            file_name="edge_contours.csv", mime="text/csv", key="edge_dl_stats")
    btns[3].download_button("点列CSV", data=_csv_bytes(result.point_rows, ["contour_id", "x_px", "y_px"]),
                            file_name="edge_points.csv", mime="text/csv", key="edge_dl_points")
    st.download_button("解析条件JSONを保存", data=json.dumps({"source": source_name, "width": w, "height": h,
                                                               "options": asdict(opts)}, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="edge_config.json", mime="application/json", key="edge_dl_config")
    st.caption("輪郭面積はOpenCVのcv2.contourAreaによる幾何学的面積です。枠除去後の見える線の本数・長さとは別です。点列座標は元画像の左上原点(px)です。")
