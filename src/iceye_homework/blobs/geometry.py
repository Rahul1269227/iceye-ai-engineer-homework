"""Rotated-rectangle geometry on the pixel grid.

Conventions (shared with OpenCV's ``RotatedRect``):

* ``x`` is the column index, ``y`` is the row index, and pixel centres sit at integer
  coordinates, so pixel ``(row, col)`` covers the unit square centred on ``(col, row)``.
* ``angle`` is in degrees; the ``width`` side points along ``(cos(angle), sin(angle))``
  in image coordinates (y grows downwards), exactly as ``cv2.boxPoints`` interprets it.

"The mask defined by the rectangle" depends on how the rectangle is rasterised, and for
blobs only a few pixels wide that choice moves IoU a lot. Four exact models are supported:

* ``"center"``: pixels whose centre lies inside the closed rectangle (the point-sampling
  rule used by GDAL/rasterio and scikit-image).
* ``"subpixel"``: ``cv2.fillPoly`` of the ``cv2.boxPoints`` polygon at 1/16 pixel
  precision. ``fillPoly`` paints every pixel the outline touches.
* ``"round"``: ``cv2.fillPoly`` of the corners rounded to integers (``np.round``).
* ``"truncate"``: ``cv2.fillPoly`` of the corners truncated to integers, as ``np.intp`` or
  ``astype(int)`` do (the OpenCV tutorial's ``np.int0(box)``).

The models disagree by about half a pixel per side, so no box is best under all of them.
The ``"robust"`` objective scores a box by its mean IoU over the four.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

import cv2
import numpy as np

Raster = Literal["center", "subpixel", "round", "truncate"]
RASTERS: tuple[Raster, ...] = ("center", "subpixel", "round", "truncate")
Objective = Literal["robust", "center", "subpixel", "round", "truncate"]
OBJECTIVES: tuple[Objective, ...] = ("robust", *RASTERS)


def objective_rasters(objective: Objective) -> tuple[Raster, ...]:
    """The raster models whose IoUs an objective averages."""
    return RASTERS if objective == "robust" else (objective,)


# Pixel centres that lie exactly on an edge (common for axis-aligned boxes) count as inside.
_EDGE_EPS = 1e-9
# cv2.fillPoly fixed-point precision for "subpixel": coordinates are given in 1/16 pixel.
_FILL_SHIFT = 4
# Window pixels scored per batch, which bounds BlobScorer's temporary arrays (~50 MB).
_BATCH_PIXELS = 1 << 20

# ``(center_x, center_y, width, height, angle)``: the search's hot loops use plain tuples.
BoxParams = tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class RotatedBox:
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float

    def params(self) -> BoxParams:
        return (self.center_x, self.center_y, self.width, self.height, self.angle)

    def canonical(self) -> RotatedBox:
        """Same rectangle with ``width >= height`` and ``angle`` in ``[-90, 90)``."""
        return RotatedBox(*canonical_params(self.params()))

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


def canonical_params(params: BoxParams) -> BoxParams:
    """``RotatedBox.canonical`` on a parameter tuple."""
    center_x, center_y, width, height, angle = params
    width, height = abs(width), abs(height)
    if width < height:
        width, height, angle = height, width, angle + 90.0
    # A square repeats every 90 degrees, so it folds into [-45, 45).
    period = 90.0 if width == height else 180.0
    angle = (angle + period / 2.0) % period - period / 2.0
    return (center_x, center_y, width, height, angle)


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
    objective: Objective = "robust",
    shape: tuple[int, int] | None = None,
) -> float:
    """Exact IoU between blob pixels ``(xs, ys)`` (integer-valued) and the box's pixel mask.

    With ``shape`` (rows, cols) the box mask is clipped to the image exactly as rasterising
    it into an image of that shape would; without it the box is unbounded. ``"robust"``
    averages the IoU over all raster models.
    """
    rasters = objective_rasters(objective)
    return sum(_raster_iou(xs, ys, box, raster, shape) for raster in rasters) / len(rasters)


def _raster_iou(
    xs: np.ndarray,
    ys: np.ndarray,
    box: RotatedBox,
    raster: Raster,
    shape: tuple[int, int] | None,
) -> float:
    x_lo, y_lo, x_hi, y_hi = _window(xs, ys, box, shape)
    match raster:
        case "center":
            inter = int(contains(xs, ys, box).sum())
            gx, gy = np.meshgrid(
                np.arange(x_lo, x_hi + 1, dtype=np.float64),
                np.arange(y_lo, y_hi + 1, dtype=np.float64),
            )
            rect = int(contains(gx, gy, box).sum())
        case "subpixel" | "round" | "truncate":
            canvas = np.zeros((y_hi - y_lo + 1, x_hi - x_lo + 1), dtype=np.uint8)
            _fill(canvas, box_points(box), raster, x_lo, y_lo)
            inter = int(canvas[(ys - y_lo).astype(np.intp), (xs - x_lo).astype(np.intp)].sum())
            rect = int(np.count_nonzero(canvas))
        case _:
            assert_never(raster)
    union = xs.size + rect - inter
    return inter / union if union else 0.0


class BlobScorer:
    """Exact ``pixel_iou`` of many candidate boxes against one fixed blob.

    Boxes that stay inside a small window around the blob are scored on that window, a
    whole batch at a time: one vectorised pixel-centre test, and one ``cv2.fillPoly`` call
    per fill model with every box's polygon in its own band of a stacked canvas. Anything
    larger falls back to ``pixel_iou``. Scores are memoised because local searches revisit
    boxes. The result is exactly ``pixel_iou``, box by box.
    """

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        objective: Objective,
        shape: tuple[int, int] | None,
        margin: int = 4,
    ) -> None:
        self._xs, self._ys, self._shape = xs, ys, shape
        self._objective, self._rasters = objective, objective_rasters(objective)
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
        self._origin = np.array([x_lo, y_lo])
        self._cols = np.arange(x_lo, x_hi + 1, dtype=np.float64)
        self._rows = np.arange(y_lo, y_hi + 1, dtype=np.float64)
        self._cache: dict[BoxParams, float] = {}

    def iou(self, box: RotatedBox) -> float:
        return self.ious([box.params()])[0]

    def ious(self, boxes: Sequence[BoxParams]) -> list[float]:
        """Exact IoU of each box, scoring the new ones that fit the window in one batch."""
        cache = self._cache
        fresh = list(dict.fromkeys(box for box in boxes if box not in cache))
        if fresh:
            fits = self._fits(fresh)
            batch = [box for box, ok in zip(fresh, fits, strict=True) if ok]
            size = max(1, _BATCH_PIXELS // self._blob.size)
            for start in range(0, len(batch), size):
                chunk = batch[start : start + size]
                cache.update(zip(chunk, self._score_batch(chunk), strict=True))
            for box, ok in zip(fresh, fits, strict=True):
                if not ok:
                    cache[box] = pixel_iou(
                        self._xs, self._ys, RotatedBox(*box), self._objective, self._shape
                    )
        return [cache[box] for box in boxes]

    def _fits(self, boxes: list[BoxParams]) -> list[bool]:
        cx, cy, w, h, angle = np.array(boxes).T
        theta = np.radians(angle)
        cos_t, sin_t = np.abs(np.cos(theta)), np.abs(np.sin(theta))
        half_x = (np.abs(w) * cos_t + np.abs(h) * sin_t) / 2.0
        half_y = (np.abs(w) * sin_t + np.abs(h) * cos_t) / 2.0
        x_lo, y_lo, x_hi, y_hi = self._bounds
        # One pixel of slack covers fillPoly painting beyond the outline and integer snapping.
        return (
            (self._open[0] | (cx - half_x - 1 >= x_lo))
            & (self._open[1] | (cy - half_y - 1 >= y_lo))
            & (self._open[2] | (cx + half_x + 1 <= x_hi))
            & (self._open[3] | (cy + half_y + 1 <= y_hi))
        ).tolist()

    def _score_batch(self, boxes: list[BoxParams]) -> list[float]:
        total = np.zeros(len(boxes))
        points: np.ndarray | None = None
        for raster in self._rasters:
            match raster:
                case "center":
                    inside = self._contains_batch(boxes)
                case "subpixel" | "round" | "truncate":
                    if points is None:
                        points = np.stack([_box_points(box) for box in boxes])
                    inside = self._fill_batch(points, raster)
                case _:
                    assert_never(raster)
            inter = np.count_nonzero(inside & self._blob, axis=(1, 2))
            union = self._n + np.count_nonzero(inside, axis=(1, 2)) - inter
            total += inter / union
        return (total / len(self._rasters)).tolist()

    def _contains_batch(self, boxes: list[BoxParams]) -> np.ndarray:
        """``contains`` over the window for every box, element for element the same sums."""
        thetas = [math.radians(box[4]) for box in boxes]
        cos_t = np.array([math.cos(theta) for theta in thetas])[:, None, None]
        sin_t = np.array([math.sin(theta) for theta in thetas])[:, None, None]
        half_w = np.array([abs(box[2]) / 2 + _EDGE_EPS for box in boxes])[:, None, None]
        half_h = np.array([abs(box[3]) / 2 + _EDGE_EPS for box in boxes])[:, None, None]
        centers = np.array([box[:2] for box in boxes])
        dx = (self._cols[None, :] - centers[:, :1])[:, None, :]
        dy = (self._rows[None, :] - centers[:, 1:])[:, :, None]
        u = dx * cos_t + dy * sin_t
        v = dy * cos_t - dx * sin_t
        return (np.abs(u) <= half_w) & (np.abs(v) <= half_h)

    def _fill_batch(
        self, points: np.ndarray, raster: Literal["subpixel", "round", "truncate"]
    ) -> np.ndarray:
        """``_fill`` of every box into its own window-sized band of one canvas."""
        count, (rows, cols) = len(points), self._blob.shape
        polygons, shift = _fill_points(points, raster, self._origin)
        canvas = np.zeros((count * rows, cols), dtype=np.uint8)
        if self._open[1] or self._open[3]:
            # Boxes may run past the top or bottom of the window, i.e. into another band.
            for k in range(count):
                band = canvas[k * rows : (k + 1) * rows]
                cv2.fillPoly(band, [polygons[k]], 1, cv2.LINE_8, shift)
        else:
            polygons[:, :, 1] += (np.arange(count, dtype=np.int32) * (rows << shift))[:, None]
            cv2.fillPoly(canvas, list(polygons), 1, cv2.LINE_8, shift)
        return canvas.reshape(count, rows, cols).view(bool)


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


def box_points(box: RotatedBox) -> np.ndarray:
    """``cv2.boxPoints`` corners (float32), exactly as a caller rasterising a CSV row gets them."""
    return _box_points(box.params())


def _box_points(params: BoxParams) -> np.ndarray:
    center_x, center_y, width, height, angle = params
    return cv2.boxPoints(((center_x, center_y), (width, height), angle))


def _fill(
    canvas: np.ndarray,
    corners: np.ndarray,
    raster: Literal["subpixel", "round", "truncate"],
    x0: int = 0,
    y0: int = 0,
) -> None:
    """``cv2.fillPoly`` ``box_points`` corners onto ``canvas``, whose top-left is ``(x0, y0)``."""
    points, shift = _fill_points(corners, raster, np.array([x0, y0]))
    cv2.fillPoly(canvas, [points], 1, cv2.LINE_8, shift)


def _fill_points(
    corners: np.ndarray, raster: Literal["subpixel", "round", "truncate"], origin: np.ndarray
) -> tuple[np.ndarray, int]:
    """``cv2.fillPoly`` vertices (int32, relative to ``origin``) and shift for ``corners``.

    Works on any ``(..., 4, 2)`` stack of corners. The integer snapping happens in absolute
    coordinates, so the fill does not depend on the origin.
    """
    match raster:
        case "subpixel":
            # Round half up (not half-to-even) so the fill is invariant to integer shifts.
            scaled = (corners.astype(np.float64) - origin) * (1 << _FILL_SHIFT)
            return np.floor(scaled + 0.5).astype(np.int32), _FILL_SHIFT
        case "round":
            return (np.round(corners).astype(np.int64) - origin).astype(np.int32), 0
        case "truncate":
            return (corners.astype(np.int64) - origin).astype(np.int32), 0
        case _:
            assert_never(raster)


def box_mask(box: RotatedBox, shape: tuple[int, int], raster: Raster = "subpixel") -> np.ndarray:
    """Rasterise the rectangle into a boolean mask of ``shape`` (rows, cols)."""
    match raster:
        case "center":
            return _center_mask(box, shape)
        case "subpixel" | "round" | "truncate":
            canvas = np.zeros(shape, dtype=np.uint8)
            _fill(canvas, box_points(box), raster)
            return canvas.astype(bool)
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
