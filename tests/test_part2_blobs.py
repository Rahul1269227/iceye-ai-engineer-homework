"""Part 2: the rotated box that maximises IoU with each blob of a binary mask."""

import csv

import cv2
import numpy as np
import pytest

from iceye_homework.blobs import BlobFit, RotatedBox, box_mask, fit_blob_boxes
from iceye_homework.blobs.cli import main

SHAPE = (120, 200)
RECTANGLES = [
    RotatedBox(40.0, 35.0, 36.0, 14.0, 30.0),
    RotatedBox(130.0, 30.0, 20.0, 12.0, 0.0),
    RotatedBox(70.0, 90.0, 50.0, 9.0, -60.0),
]


def nearest(fits: list[BlobFit], x: float, y: float) -> BlobFit:
    return min(fits, key=lambda fit: (fit.box.center_x - x) ** 2 + (fit.box.center_y - y) ** 2)


def test_boxes_recover_rectangles_and_beat_min_area_rect(tmp_path):
    mask = np.zeros(SHAPE, dtype=bool)
    for rect in RECTANGLES:
        # cv2.fillPoly of the rounded cv2.boxPoints corners, as OpenCV users draw boxes.
        mask |= box_mask(rect, SHAPE, raster="round")
    mask[80:100, 150:170] = True
    mask[89, 170:190] = True  # a one-pixel spur that cv2.minAreaRect has to stretch over

    fits = fit_blob_boxes(mask, objective="round", workers=1)

    assert len(fits) == 4
    for rect in RECTANGLES:
        fit = nearest(fits, rect.center_x, rect.center_y)
        assert fit.iou == pytest.approx(1.0)
        assert (fit.box.center_x, fit.box.center_y) == pytest.approx(
            (rect.center_x, rect.center_y), abs=0.5
        )
        assert (fit.box.width, fit.box.height) == pytest.approx((rect.width, rect.height), abs=1.0)
        assert fit.box.angle == pytest.approx(rect.angle, abs=1.0)
    spur = nearest(fits, 160.0, 90.0)
    assert spur.iou > 0.95
    assert spur.baseline_iou < 0.6

    mask_path, csv_path = tmp_path / "mask.png", tmp_path / "boxes.csv"
    cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    assert main([str(mask_path), "--out", str(csv_path), "--workers", "1"]) == 0
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["blob_id", "center_x", "center_y", "width", "height", "angle"]
    assert len(rows) == 5
    assert all(-90.0 <= float(row[5]) < 90.0 for row in rows[1:])
