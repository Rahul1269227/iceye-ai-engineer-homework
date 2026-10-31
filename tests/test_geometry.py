"""Rotated box geometry and pixel IoU."""

import numpy as np
import pytest

from iceye_homework.blobs.geometry import (
    RASTERS,
    BlobScorer,
    RotatedBox,
    box_mask,
    box_points,
    canonical_params,
    contains,
    objective_rasters,
    pixel_iou,
)


def test_canonical_swaps_width_height_and_folds_angle():
    assert canonical_params((10.0, 20.0, 5.0, 12.0, 10.0)) == (10.0, 20.0, 12.0, 5.0, -80.0)
    cx, cy, w, h, angle = canonical_params((0.0, 0.0, 8.0, 8.0, 80.0))
    assert (cx, cy, w, h) == (0.0, 0.0, 8.0, 8.0)
    assert -45.0 <= angle < 45.0


def test_rotated_box_corners_match_box_points():
    box = RotatedBox(50.0, 40.0, 20.0, 10.0, 30.0)
    assert box.corners() == pytest.approx(box_points(box), abs=1e-5)


def test_contains_axis_aligned():
    box = RotatedBox(5.0, 5.0, 4.0, 2.0, 0.0)
    xs = np.array([5.0, 7.0, 3.0, 5.0])
    ys = np.array([5.0, 5.0, 5.0, 7.0])
    assert contains(xs, ys, box).tolist() == [True, True, True, False]


@pytest.mark.parametrize("raster", RASTERS)
def test_box_mask_covers_solid_rectangle(raster):
    shape = (40, 50)
    box = RotatedBox(20.0, 15.0, 10.0, 6.0, 0.0)
    mask = box_mask(box, shape, raster=raster)
    assert mask.dtype == bool
    assert mask.any()
    # centre pixel is inside for every rasterisation
    assert mask[15, 20]


@pytest.mark.parametrize("raster", RASTERS)
def test_pixel_iou_perfect_on_own_mask(raster):
    shape = (60, 80)
    box = RotatedBox(30.0, 25.0, 16.0, 8.0, 20.0)
    mask = box_mask(box, shape, raster=raster)
    ys, xs = np.nonzero(mask)
    iou = pixel_iou(xs.astype(np.float64), ys.astype(np.float64), box, raster, shape)
    assert iou == pytest.approx(1.0)


def test_pixel_iou_robust_is_mean_over_rasters():
    shape = (60, 80)
    box = RotatedBox(30.0, 25.0, 16.0, 8.0, 20.0)
    mask = box_mask(box, shape, raster="center")
    ys, xs = np.nonzero(mask)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    robust = pixel_iou(xs, ys, box, "robust", shape)
    mean = sum(pixel_iou(xs, ys, box, r, shape) for r in RASTERS) / len(RASTERS)
    assert robust == pytest.approx(mean)


def test_pixel_iou_zero_on_disjoint():
    box = RotatedBox(10.0, 10.0, 4.0, 4.0, 0.0)
    xs = np.array([50.0, 51.0])
    ys = np.array([50.0, 50.0])
    assert pixel_iou(xs, ys, box, "center", (80, 80)) == 0.0


def test_blob_scorer_matches_pixel_iou():
    shape = (50, 50)
    box = RotatedBox(20.0, 20.0, 8.0, 4.0, 15.0)
    mask = box_mask(box, shape, raster="round")
    ys, xs = np.nonzero(mask)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    scorer = BlobScorer(xs, ys, "round", shape)
    direct = pixel_iou(xs, ys, box, "round", shape)
    assert scorer.iou(box) == pytest.approx(direct)
    # cached second call
    assert scorer.iou(box) == pytest.approx(direct)


def test_objective_rasters():
    assert objective_rasters("robust") == RASTERS
    assert objective_rasters("center") == ("center",)
