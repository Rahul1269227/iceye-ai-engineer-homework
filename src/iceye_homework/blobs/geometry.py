"""Rotated-rectangle geometry on the pixel grid.

Conventions (shared with OpenCV's ``RotatedRect``):

* ``x`` is the column index, ``y`` is the row index, and pixel centres sit at integer
  coordinates, so pixel ``(row, col)`` covers the unit square centred on ``(col, row)``.
* ``angle`` is in degrees; the ``width`` side points along ``(cos(angle), sin(angle))``
  in image coordinates (y grows downwards), exactly as ``cv2.boxPoints`` interprets it.

"The mask defined by the rectangle" depends on how the rectangle is rasterised, and for
blobs only a few pixels wide that choice moves IoU a lot. Two exact models are supported:

* ``"opencv"``: the pixels ``cv2.fillPoly`` paints for the ``cv2.boxPoints`` polygon
  (sub-pixel precision). This is what filling the RotatedRect with OpenCV produces;
  it includes every pixel the outline touches.
* ``"center"``: pixels whose centre lies inside the closed rectangle (the point-sampling
  rule used by GDAL/rasterio and scikit-image).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, assert_never

import cv2
import numpy as np

Raster = Literal["opencv", "center"]
RASTERS: tuple[Raster, ...] = ("opencv", "center")

# Pixel centres that lie exactly on an edge (common for axis-aligned boxes) count as inside.
_EDGE_EPS = 1e-9
# cv2.fillPoly fixed-point precision: coordinates are given in 1/16 pixel.
_FILL_SHIFT = 4


@dataclass(frozen=True, slots=True)
class RotatedBox:
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    def canonical(self) -> RotatedBox:
        """Same rectangle with ``width >= height`` and ``angle`` in ``[-90, 90)``."""
        width, height, angle = abs(self.width), abs(self.height), self.angle
        if width < height:
            width, height, angle = height, width, angle + 90.0
        # A square repeats every 90 degrees, so it folds into [-45, 45).
        period = 90.0 if width == height else 180.0
        angle = (angle + period / 2.0) % period - period / 2.0
        return RotatedBox(self.center_x, self.center_y, width, height, angle)

    def corners(self) -> np.ndarray:
        """Corner points as a ``(4, 2)`` array of ``(x, y)``, in ``cv2.boxPoints`` order."""
        return np.array(self.corner_list(), dtype=np.float64)

    def corner_list(self) -> list[tuple[float, float]]:
        """Same as ``corners`` as plain floats (much faster than numpy for four points)."""
        theta = math.radians(self.angle)
        b, a = math.cos(theta) * 0.5, math.sin(theta) * 0.5
        p0 = (
            self.center_x - a * self.height - b * self.width,
            self.center_y + b * self.height - a * self.width,
        )
        p1 = (
            self.center_x + a * self.height - b * self.width,
            self.center_y - b * self.height - a * self.width,
        )
        p2 = (2 * self.center_x - p0[0], 2 * self.center_y - p0[1])
        p3 = (2 * self.center_x - p1[0], 2 * self.center_y - p1[1])
        return [p0, p1, p2, p3]


def contains(xs: np.ndarray, ys: np.ndarray, box: RotatedBox) -> np.ndarray:
    """Boolean mask of which points ``(xs, ys)`` lie inside the closed rectangle."""
    theta = math.radians(box.angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx = xs - box.center_x
    dy = ys - box.center_y
    u = dx * cos_t + dy * sin_t
    v = dy * cos_t - dx * sin_t
    return (np.abs(u) <= abs(box.width) / 2 + _EDGE_EPS) & (
        np.abs(v) <= abs(box.height) / 2 + _EDGE_EPS
    )


def pixel_iou(
    xs: np.ndarray,
    ys: np.ndarray,
    box: RotatedBox,
    raster: Raster = "opencv",
    shape: tuple[int, int] | None = None,
) -> float:
    """Exact IoU between blob pixels ``(xs, ys)`` (integer-valued) and the box's pixel mask.

    With ``shape`` (rows, cols) the box mask is clipped to the image exactly as rasterising
    it into an image of that shape would; without it the box is unbounded.
    """
    if box.width <= 0 or box.height <= 0:
        return 0.0
    x_lo, y_lo, x_hi, y_hi = _window(xs, ys, box, shape)
    match raster:
        case "center":
            inter = int(contains(xs, ys, box).sum())
            gx, gy = np.meshgrid(
                np.arange(x_lo, x_hi + 1, dtype=np.float64),
                np.arange(y_lo, y_hi + 1, dtype=np.float64),
            )
            rect = int(contains(gx, gy, box).sum())
        case "opencv":
            canvas = np.zeros((y_hi - y_lo + 1, x_hi - x_lo + 1), dtype=np.uint8)
            _fill(canvas, box.corners() - (x_lo, y_lo))
            inter = int(canvas[(ys - y_lo).astype(np.intp), (xs - x_lo).astype(np.intp)].sum())
            rect = int(np.count_nonzero(canvas))
        case _:
            assert_never(raster)
    union = xs.size + rect - inter
    return inter / union if union else 0.0


class BlobScorer:
    """Exact ``pixel_iou`` of many candidate boxes against one fixed blob.

    Boxes that stay inside a small window around the blob are scored on that window
    (one fill or one vectorised test over a few hundred pixels); anything larger falls
    back to ``pixel_iou``. Scores are memoised because local searches revisit boxes.
    """

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        raster: Raster,
        shape: tuple[int, int] | None,
        margin: int = 4,
    ) -> None:
        self._xs, self._ys, self._raster, self._shape = xs, ys, raster, shape
        x_lo, y_lo = int(xs.min()) - margin, int(ys.min()) - margin
        x_hi, y_hi = int(xs.max()) + margin, int(ys.max()) + margin
        # On a side where the window meets the image border, clipping is the image's own.
        self._open = [False, False, False, False]
        if shape is not None:
            rows, cols = shape
            self._open = [x_lo <= 0, y_lo <= 0, x_hi >= cols - 1, y_hi >= rows - 1]
            x_lo, y_lo = max(x_lo, 0), max(y_lo, 0)
            x_hi, y_hi = min(x_hi, cols - 1), min(y_hi, rows - 1)
        self._bounds = (x_lo, y_lo, x_hi, y_hi)
        self._blob = np.zeros((y_hi - y_lo + 1, x_hi - x_lo + 1), dtype=bool)
        self._blob[(ys - y_lo).astype(np.intp), (xs - x_lo).astype(np.intp)] = True
        self._n = xs.size
        if raster == "center":
            self._gx, self._gy = np.meshgrid(
                np.arange(x_lo, x_hi + 1, dtype=np.float64),
                np.arange(y_lo, y_hi + 1, dtype=np.float64),
            )
        self._cache: dict[tuple[float, ...], float] = {}

    def iou(self, box: RotatedBox) -> float:
        key = (box.center_x, box.center_y, box.width, box.height, box.angle)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._score(box)
            self._cache[key] = cached
        return cached

    def _score(self, box: RotatedBox) -> float:
        if box.width <= 0 or box.height <= 0:
            return 0.0
        corners = box.corner_list()
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        x_lo, y_lo, x_hi, y_hi = self._bounds
        # One pixel of slack covers fillPoly painting slightly beyond the outline.
        fits = (
            (self._open[0] or min(xs) - 1 >= x_lo)
            and (self._open[1] or min(ys) - 1 >= y_lo)
            and (self._open[2] or max(xs) + 1 <= x_hi)
            and (self._open[3] or max(ys) + 1 <= y_hi)
        )
        if not fits:
            return pixel_iou(self._xs, self._ys, box, self._raster, self._shape)
        match self._raster:
            case "opencv":
                canvas = np.zeros(self._blob.shape, dtype=np.uint8)
                scale = 1 << _FILL_SHIFT
                points = np.array(
                    [
                        (
                            math.floor((x - x_lo) * scale + 0.5),
                            math.floor((y - y_lo) * scale + 0.5),
                        )
                        for x, y in corners
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(canvas, [points], 1, cv2.LINE_8, _FILL_SHIFT)
                inside = canvas.view(bool)
            case "center":
                inside = contains(self._gx, self._gy, box)
            case _:
                assert_never(self._raster)
        inter = int(np.count_nonzero(inside & self._blob))
        union = self._n + int(np.count_nonzero(inside)) - inter
        return inter / union if union else 0.0


def _window(
    xs: np.ndarray, ys: np.ndarray, box: RotatedBox, shape: tuple[int, int] | None
) -> tuple[int, int, int, int]:
    """Inclusive pixel window covering the blob and the box, clipped to the image if given."""
    corners = box.corners()
    x_lo = math.floor(min(corners[:, 0].min(), xs.min())) - 1
    y_lo = math.floor(min(corners[:, 1].min(), ys.min())) - 1
    x_hi = math.ceil(max(corners[:, 0].max(), xs.max())) + 1
    y_hi = math.ceil(max(corners[:, 1].max(), ys.max())) + 1
    if shape is not None:
        rows, cols = shape
        x_lo, y_lo = max(x_lo, 0), max(y_lo, 0)
        x_hi, y_hi = min(x_hi, cols - 1), min(y_hi, rows - 1)
    return x_lo, y_lo, x_hi, y_hi


def _fill(canvas: np.ndarray, corners: np.ndarray) -> None:
    # Round half up (not numpy's half-to-even) so the fill is invariant to integer shifts.
    points = np.floor(corners * (1 << _FILL_SHIFT) + 0.5).astype(np.int32)
    cv2.fillPoly(canvas, [points], 1, cv2.LINE_8, _FILL_SHIFT)


def box_mask(box: RotatedBox, shape: tuple[int, int], raster: Raster = "opencv") -> np.ndarray:
    """Rasterise the rectangle into a boolean mask of ``shape`` (rows, cols)."""
    match raster:
        case "opencv":
            canvas = np.zeros(shape, dtype=np.uint8)
            _fill(canvas, box.corners())
            return canvas.astype(bool)
        case "center":
            return _center_mask(box, shape)
        case _:
            assert_never(raster)


def _center_mask(box: RotatedBox, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    corners = box.corners()
    x_lo, y_lo = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    x_hi = min(int(np.ceil(corners[:, 0].max())), cols - 1)
    y_hi = min(int(np.ceil(corners[:, 1].max())), rows - 1)
    if x_lo > x_hi or y_lo > y_hi:
        return mask
    gx, gy = np.meshgrid(
        np.arange(x_lo, x_hi + 1, dtype=np.float64),
        np.arange(y_lo, y_hi + 1, dtype=np.float64),
    )
    mask[y_lo : y_hi + 1, x_lo : x_hi + 1] = contains(gx, gy, box)
    return mask
