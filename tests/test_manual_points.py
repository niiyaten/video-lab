import csv

import cv2
import numpy as np
import pytest

from videolab.optical_flow import FlowOptions, run_sparse_flow
from videolab.point_selection import (
    make_point_picker_image, map_click_to_frame, read_exact_frame, snap_to_corner,
)
from videolab.video_io import read_info


def create_video(path, frame_count=14, fps=40.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (128, 96))
    assert writer.isOpened()
    rng = np.random.default_rng(2026)
    pattern = rng.integers(40, 220, (31, 31), dtype=np.uint8)
    for n in range(frame_count):
        frame = np.zeros((96, 128, 3), np.uint8)
        frame[26:57, 30+n:61+n, :] = pattern[:, :, None]
        writer.write(frame)
    writer.release()


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def test_manual_mode_tracks_only_selected_points_with_stable_ids(tmp_path):
    src, movie, csv_path = (tmp_path / x for x in ['test.avi', 'manual.mp4', 'data.csv'])
    create_video(src)
    opt = FlowOptions(initial_points=((35.0, 32.0), (49.0, 47.0)),
                      auto_replenish=False, max_points=2, output_fps=20.0)
    frames, ids = run_sparse_flow(src, movie, csv_path, opt)
    assert frames == 14
    assert ids == 2
    assert read_info(movie).frame_count == 14
    rows = read_rows(csv_path)
    assert set(row['point_id'] for row in rows) == {'1', '2'}
    initial = [row for row in rows if row['frame'] == '0']
    assert len(initial) == 2
    assert {(float(r['x_px']), float(r['y_px'])) for r in initial} == {(35.0, 32.0), (49.0, 47.0)}
    assert all(r['speed_px_s'] == '' for r in initial)
    assert any(r['frame'] != '0' and r['speed_px_s'] != '' for r in rows)


def test_start_frame_is_original_absolute_frame(tmp_path):
    src, movie, csv_path = (tmp_path / x for x in ['test.avi', 'manual.mp4', 'data.csv'])
    create_video(src, frame_count=14)
    selected = read_exact_frame(src, 5)
    assert selected.shape == (96, 128, 3)
    opts = FlowOptions(initial_points=((45.0, 36.0),), start_frame=5,
                       auto_replenish=False, max_points=1, output_fps=10.0)
    frames, ids = run_sparse_flow(src, movie, csv_path, opts)
    assert frames == 9
    assert ids == 1
    assert read_info(movie).frame_count == 9
    rows = read_rows(csv_path)
    assert rows[0]['frame'] == '5'
    assert abs(float(rows[0]['time_s']) - 5/40) < 1e-7
    assert rows[0]['x_px'] == '45.0'


def test_no_manual_points_rejected(tmp_path):
    src = tmp_path/'test.avi'
    create_video(src)
    with pytest.raises(ValueError, match='手動モード'):
        run_sparse_flow(src, tmp_path/'out.mp4', tmp_path/'out.csv',
                        FlowOptions(initial_points=(), auto_replenish=False))


def test_outside_roi_manual_point_rejected(tmp_path):
    src = tmp_path/'test.avi'
    create_video(src)
    with pytest.raises(ValueError, match='ROI'):
        run_sparse_flow(src, tmp_path/'out.mp4', tmp_path/'out.csv',
                        FlowOptions(initial_points=((55.0, 55.0),),
                                    roi=(0,0,40,40), auto_replenish=False))


def test_snap_corner_and_no_corner_fallback():
    frame = np.zeros((96, 128, 3), np.uint8)
    cv2.rectangle(frame, (40, 30), (70, 60), (255, 255, 255), -1)
    found = snap_to_corner(frame, (42.0, 31.0), radius_px=10)
    assert found is not None
    assert np.hypot(found[0]-42, found[1]-31) < 11
    blank = np.zeros_like(frame)
    assert snap_to_corner(blank, (42, 31), radius_px=10) is None


def test_scale_click_and_marker_render():
    click = {'x': 99.5, 'y': 49.5, 'width':200, 'height':100}
    assert map_click_to_frame(click, 800, 400) == (399.5, 199.5)
    frame = np.zeros((400, 800, 3), np.uint8)
    rendered, width = make_point_picker_image(frame, [(399.5,199.5)], max_width=400)
    assert width == 400
    assert rendered.size == (400,200)
    assert np.any(np.asarray(rendered) != 0)
    with pytest.raises(ValueError):
        map_click_to_frame({'x': 200, 'y': 0, 'width':200, 'height':100}, 800,400)


def test_out_of_range_frame(tmp_path):
    src=tmp_path/'test.avi'
    create_video(src, frame_count=8)
    with pytest.raises(ValueError, match='フレーム'):
        read_exact_frame(src, 30)
