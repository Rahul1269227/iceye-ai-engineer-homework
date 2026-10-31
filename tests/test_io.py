"""Mask loading, CSV and overlay output."""

import csv

import cv2
import numpy as np
import pytest

from iceye_homework.blobs import BlobFit, RotatedBox, load_mask, write_csv
from iceye_homework.blobs.io import CSV_COLUMNS, write_overlay


def test_load_npy(tmp_path):
    path = tmp_path / "mask.npy"
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:6, 4:8] = 1
    np.save(path, mask)
    loaded = load_mask(path)
    assert loaded.dtype == bool
    assert loaded.sum() == 12


def test_load_npy_rejects_3d(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="2D"):
        load_mask(path)


def test_load_png_binary(tmp_path):
    path = tmp_path / "mask.png"
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 8:15] = 255
    cv2.imwrite(str(path), mask)
    loaded = load_mask(path)
    assert loaded[7, 10]
    assert not loaded[0, 0]


def test_load_png_zero_one(tmp_path):
    path = tmp_path / "01.png"
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:5, 2:5] = 1
    cv2.imwrite(str(path), img)
    assert load_mask(path).sum() == 9


def test_load_threshold(tmp_path):
    path = tmp_path / "grey.png"
    img = np.full((8, 8), 100, dtype=np.uint8)
    img[3, 3] = 200
    cv2.imwrite(str(path), img)
    assert load_mask(path, threshold=127).sum() == 1
    assert load_mask(path, threshold=50).sum() == 64


def test_load_missing_image(tmp_path):
    with pytest.raises(ValueError, match="not a readable"):
        load_mask(tmp_path / "missing.png")


def test_write_csv_header_and_rounding(tmp_path):
    path = tmp_path / "boxes.csv"
    fits = [
        BlobFit(
            1,
            RotatedBox(1.23456, 2.34567, 3.45678, 1.11119, -12.34567),
            area=10,
            iou=0.9,
            baseline_iou=0.8,
        )
    ]
    write_csv(fits, path)
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[1][0] == "1"
    assert rows[1][1:] == ["1.2346", "2.3457", "3.4568", "1.1112", "-12.3457"]


def test_write_csv_empty(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv([], path)
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [list(CSV_COLUMNS)]


def test_write_overlay(tmp_path):
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:20, 15:30] = True
    fits = [
        BlobFit(1, RotatedBox(22.0, 15.0, 16.0, 10.0, 0.0), 150, 0.9, 0.8),
    ]
    out = tmp_path / "overlay.png"
    write_overlay(mask, fits, out, scale=2)
    image = cv2.imread(str(out))
    assert image is not None
    assert image.shape == (80, 100, 3)
