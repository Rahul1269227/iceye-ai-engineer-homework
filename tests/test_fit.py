"""Blob labelling and IoU-maximising box fits."""

import numpy as np
import pytest

from iceye_homework.blobs import RotatedBox, box_mask, fit_blob, fit_blob_boxes
from iceye_homework.blobs.fit import label_blobs, min_area_rect


SHAPE = (100, 140)


def nearest(fits, x, y):
    return min(fits, key=lambda f: (f.box.center_x - x) ** 2 + (f.box.center_y - y) ** 2)


def test_empty_mask_returns_no_fits():
    assert fit_blob_boxes(np.zeros(SHAPE, dtype=bool), workers=1) == []


def test_rejects_non_2d_mask():
    with pytest.raises(ValueError, match="2D"):
        fit_blob_boxes(np.zeros((4, 4, 3), dtype=bool))


def test_single_pixel():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[40, 55] = True
    fits = fit_blob_boxes(mask, objective="center", workers=1, restarts=0)
    assert len(fits) == 1
    assert fits[0].area == 1
    assert fits[0].blob_id == 1
    assert fits[0].iou == pytest.approx(1.0)


def test_line_segment_iou_one():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[50, 20:40] = True
    fits = fit_blob_boxes(mask, objective="center", workers=1, restarts=0)
    assert len(fits) == 1
    assert fits[0].area == 20
    assert fits[0].iou == pytest.approx(1.0)


def test_recovers_rotated_rectangles():
    rects = [
        RotatedBox(35.0, 30.0, 28.0, 12.0, 25.0),
        RotatedBox(100.0, 70.0, 18.0, 10.0, -40.0),
    ]
    mask = np.zeros(SHAPE, dtype=bool)
    for rect in rects:
        mask |= box_mask(rect, SHAPE, raster="round")

    fits = fit_blob_boxes(mask, objective="round", workers=1, restarts=2)
    assert len(fits) == 2
    for rect in rects:
        fit = nearest(fits, rect.center_x, rect.center_y)
        assert fit.iou == pytest.approx(1.0)
        assert (fit.box.center_x, fit.box.center_y) == pytest.approx(
            (rect.center_x, rect.center_y), abs=0.6
        )
        assert (fit.box.width, fit.box.height) == pytest.approx(
            (rect.width, rect.height), abs=1.0
        )
        assert fit.box.angle == pytest.approx(rect.angle, abs=1.5)


def test_spur_beats_min_area_rect():
    rect = RotatedBox(50.0, 50.0, 30.0, 12.0, 0.0)
    mask = box_mask(rect, SHAPE, raster="round")
    mask[50, 65:90] = True  # spur attached to the right edge

    fits = fit_blob_boxes(mask, objective="round", workers=1, restarts=2)
    assert len(fits) == 1
    assert fits[0].iou > fits[0].baseline_iou
    assert fits[0].iou > 0.9


def test_min_area_filters_small_blobs():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[10, 10] = True
    mask[60:70, 80:90] = True
    fits = fit_blob_boxes(mask, min_area=5, workers=1, restarts=0)
    assert len(fits) == 1
    assert fits[0].area >= 5


def test_connectivity_4_splits_diagonal():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 5] = True
    mask[6, 6] = True
    count8, _, _ = label_blobs(mask, connectivity=8)
    count4, _, _ = label_blobs(mask, connectivity=4)
    assert count8 == 1
    assert count4 == 2


def test_blob_ids_follow_raster_order():
    mask = np.zeros((30, 30), dtype=bool)
    mask[20, 20] = True
    mask[5, 5] = True
    fits = fit_blob_boxes(mask, workers=1, restarts=0)
    assert [f.blob_id for f in fits] == [1, 2]
    assert fits[0].box.center_y < fits[1].box.center_y


def test_fit_blob_baseline_is_min_area_rect():
    xs = np.array([10.0, 11.0, 12.0, 10.0, 11.0, 12.0])
    ys = np.array([20.0, 20.0, 20.0, 21.0, 21.0, 21.0])
    box, iou, baseline = fit_blob(xs, ys, objective="center", shape=(40, 40), restarts=0)
    mar = min_area_rect(xs, ys)
    assert baseline >= 0.0
    assert iou >= baseline - 1e-9
    assert box.width >= box.height
    assert -90.0 <= box.angle < 90.0
    assert mar.width > 0 or mar.height > 0


def test_workers_one_and_many_agree():
    mask = np.zeros(SHAPE, dtype=bool)
    for cx, cy in ((30, 25), (90, 60), (50, 80), (110, 30)):
        mask |= box_mask(RotatedBox(float(cx), float(cy), 14.0, 8.0, 10.0), SHAPE, "round")
    a = fit_blob_boxes(mask, objective="round", workers=1, restarts=0)
    b = fit_blob_boxes(mask, objective="round", workers=2, restarts=0)
    assert len(a) == len(b) == 4
    for fa, fb in zip(a, b, strict=True):
        assert fa.blob_id == fb.blob_id
        assert fa.iou == pytest.approx(fb.iou)
        assert (fa.box.center_x, fa.box.center_y) == pytest.approx(
            (fb.box.center_x, fb.box.center_y)
        )
