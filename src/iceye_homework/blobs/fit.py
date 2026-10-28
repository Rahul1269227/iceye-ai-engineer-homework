"""Fit, for every blob in a binary mask, the rotated rectangle that maximises pixel IoU.

The IoU-optimal rectangle is generally *not* ``cv2.minAreaRect``: that one must contain
every pixel, so a single stray pixel inflates it. The search here is free to cut
outliers off. Per blob it runs:

1. Seeds: ``cv2.minAreaRect`` (through the outer pixel centres, and grown to the outer
   pixel edges) and a moments rectangle whose sides come from the covariance eigenvalues
   (``side = sqrt(12 * eigenvalue)``, exact for a solid rectangle, robust to spurs).
2. Global sweep over orientations. For a fixed angle, the best axis-aligned rectangle
   in the rotated frame is found on a coarse histogram grid by Dinkelbach's method:
   maximising ``I / (|B| + A - I)`` reduces to repeatedly finding the maximum-sum
   subrectangle of the weights ``(1 + lam) * count - lam * cell_area``.
3. Local refinement with Nelder-Mead on a smooth area-overlap IoU, whose cost only
   touches the blob's own pixels.
4. Exact polish: pattern search (per-parameter, single-side and pivot-rotation moves) on
   the exact pixel IoU of the chosen rasterisation, from the best few candidates, then
   seeded random restarts around the incumbent (iterated local search). The exact
   objective is piecewise constant, so ties are broken by a smooth area-overlap IoU of
   the region the raster model paints; this keeps the search moving across plateaus and
   returns the box in the middle of its plateau, the one least sensitive to rounding.

Every exact score is taken on the box exactly as it is reported (absolute coordinates,
canonical form, rounded to ``DECIMALS``), so the reported IoU is reproducible from the
CSV. Both minAreaRect seeds stay among the final candidates, so the result is never worse
than that baseline.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import assert_never

import cv2
import numpy as np
from scipy.optimize import minimize

from .geometry import BlobScorer, Raster, RotatedBox

DECIMALS = 4

# (center_x, center_y, width, height, angle)
Params = tuple[float, float, float, float, float]

# Degenerate (zero-width) boxes still rasterise to their outline with cv2.fillPoly.
_MIN_SIDE = 1e-3
_SWEEP_STEP_DEG = 5.0
_GRID_CAP = 64
_SUPERSAMPLE = 2
_REFINE_TOP_K = 2
_SIMPLEX_STEPS = np.array([0.5, 0.5, 1.0, 1.0, 2.0])
_POLISH_TOP_K = 3
_POLISH_START_STEP = 1.0
_POLISH_MIN_STEP = 1.0 / 16.0
_POLISH_MAX_ITER = 500
# Rotation step per unit of polish step; pivot moves rotate about either end of the box.
_POLISH_DEG_PER_STEP = 4.0
_RESTARTS = 8
_RESTART_SCALE = np.array([1.0, 1.0, 1.5, 1.5, 15.0])
# Tie-score gains below this do not count as progress on an exact-IoU plateau.
_TIE_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class BlobFit:
    blob_id: int
    box: RotatedBox
    area: int
    iou: float
    baseline_iou: float


@dataclass(frozen=True, slots=True)
class _Blob:
    """One blob's pixels relative to an integer anchor near its centre, plus its scorer.

    The integer anchor keeps relative pixel centres on the integer lattice and keeps the
    continuous optimisation well conditioned.
    """

    dx: np.ndarray
    dy: np.ndarray
    anchor_x: float
    anchor_y: float
    scorer: BlobScorer
    tie_grow: float

    @classmethod
    def from_pixels(
        cls, xs: np.ndarray, ys: np.ndarray, raster: Raster, shape: tuple[int, int] | None
    ) -> _Blob:
        ax, ay = float(np.round(xs.mean())), float(np.round(ys.mean()))
        match raster:
            case "opencv":
                # fillPoly paints every pixel the outline touches: ~half a pixel per side.
                grow = 1.0
            case "center":
                grow = 0.0
            case _:
                assert_never(raster)
        return cls(xs - ax, ys - ay, ax, ay, BlobScorer(xs, ys, raster, shape), grow)

    def tie_score(self, params: Params) -> float:
        """Smooth IoU of the region the raster model paints, for breaking exact-IoU ties."""
        cx, cy, w, h, angle = params
        g = self.tie_grow
        return _soft_iou_params((cx, cy, abs(w) + g, abs(h) + g, angle), self.dx, self.dy)

    def finalize(self, box: RotatedBox) -> RotatedBox:
        """Relative box -> the box as reported: absolute, canonical, rounded."""
        absolute = RotatedBox(
            box.center_x + self.anchor_x, box.center_y + self.anchor_y,
            box.width, box.height, box.angle,
        ).canonical()  # fmt: skip
        return RotatedBox(
            round(absolute.center_x, DECIMALS),
            round(absolute.center_y, DECIMALS),
            round(absolute.width, DECIMALS),
            round(absolute.height, DECIMALS),
            round(absolute.angle, DECIMALS),
        )

    def exact_iou(self, box: RotatedBox) -> float:
        return self.scorer.iou(self.finalize(box))

    def soft_iou(self, box: RotatedBox) -> float:
        return _soft_iou_params(_box_params(box), self.dx, self.dy)


def label_blobs(mask: np.ndarray, connectivity: int = 8) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected components of a boolean mask: ``(count, labels, stats)`` as in OpenCV."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity
    )
    return count - 1, labels, stats


def fit_blob_boxes(
    mask: np.ndarray,
    connectivity: int = 8,
    min_area: int = 1,
    raster: Raster = "opencv",
    workers: int = 1,
) -> list[BlobFit]:
    """Fit one IoU-maximising rotated box per connected component of ``mask``.

    Blob ids are assigned 1..N in raster order of each blob's first pixel. IoU is measured
    against the box mask clipped to the image, as an image-sized evaluation would see it.
    Blobs are independent, so ``workers > 1`` fits them in that many processes; the result
    is identical either way.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    count, labels, stats = label_blobs(mask, connectivity)
    pixels: list[tuple[np.ndarray, np.ndarray]] = []
    areas: list[int] = []
    for label in range(1, count + 1):
        x0, y0, w, h, area = (int(v) for v in stats[label])
        if area < min_area:
            continue
        rows, cols = np.nonzero(labels[y0 : y0 + h, x0 : x0 + w] == label)
        pixels.append(((cols + x0).astype(np.float64), (rows + y0).astype(np.float64)))
        areas.append(area)

    xs_all, ys_all = [xs for xs, _ in pixels], [ys for _, ys in pixels]
    args = (xs_all, ys_all, repeat(raster), repeat(mask.shape))
    # Process start-up costs more than fitting a handful of blobs.
    if workers > 1 and len(pixels) >= 4 * workers:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunksize = max(1, len(pixels) // (4 * workers))
            results = list(pool.map(fit_blob, *args, chunksize=chunksize))
    else:
        results = list(map(fit_blob, *args))
    return [
        BlobFit(blob_id, box, area, iou, baseline_iou)
        for blob_id, (area, (box, iou, baseline_iou)) in enumerate(
            zip(areas, results, strict=True), start=1
        )
    ]


def fit_blob(
    xs: np.ndarray,
    ys: np.ndarray,
    raster: Raster = "opencv",
    shape: tuple[int, int] | None = None,
) -> tuple[RotatedBox, float, float]:
    """Best rotated box for one blob given its integer pixel centres ``(xs, ys)``.

    Returns ``(box, iou, baseline_iou)``. The baseline is the better of ``cv2.minAreaRect``
    as returned (through the outer pixel centres) and grown by one pixel (to the outer
    pixel edges).
    """
    blob = _Blob.from_pixels(xs, ys, raster, shape)

    tight = _min_area_rect(blob.dx, blob.dy)
    enclosing = _inset(tight, -0.5)
    moments = _moments_seed(blob.dx, blob.dy)
    sweep = _sweep_boxes(blob.dx, blob.dy, [tight.angle, moments.angle])
    seeds = [tight, enclosing, moments, *sweep]

    candidates = list(seeds)
    by_soft = sorted(seeds, key=blob.soft_iou, reverse=True)
    candidates += [_refine(box, blob.dx, blob.dy) for box in by_soft[:_REFINE_TOP_K]]
    if blob.tie_grow:
        # Area-optimal boxes, inset by however far the raster model paints beyond them.
        candidates += [_inset(box, blob.tie_grow / 2.0) for box in candidates]

    ranked = sorted(
        candidates,
        key=lambda box: (blob.exact_iou(box), blob.tie_score(_box_params(box))),
        reverse=True,
    )
    best, best_iou, best_tie = max(
        (_polish(blob, box) for box in ranked[:_POLISH_TOP_K]), key=lambda r: (r[1], r[2])
    )
    rng = np.random.default_rng([xs.size, int(xs.min()), int(ys.min())])
    for _ in range(_RESTARTS):
        start = _params_box(_box_params(best) + rng.normal(0.0, _RESTART_SCALE))
        box, iou, tie = _polish(blob, start)
        if (iou, tie) > (best_iou, best_tie + _TIE_TOL):
            best, best_iou, best_tie = box, iou, tie

    baseline_iou = max(blob.exact_iou(tight), blob.exact_iou(enclosing))
    return blob.finalize(best), best_iou, baseline_iou


def _min_area_rect(dx: np.ndarray, dy: np.ndarray) -> RotatedBox:
    points = np.column_stack([dx, dy]).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    return _params_box((cx, cy, w, h, angle))


def _inset(box: RotatedBox, margin: float) -> RotatedBox:
    """Move every side inwards by ``margin`` pixels (outwards when negative)."""
    return RotatedBox(
        box.center_x,
        box.center_y,
        max(box.width - 2.0 * margin, _MIN_SIDE),
        max(box.height - 2.0 * margin, _MIN_SIDE),
        box.angle,
    )


def _moments_seed(dx: np.ndarray, dy: np.ndarray) -> RotatedBox:
    # +1/12 per axis accounts for each pixel being a unit square rather than a point.
    cxx = float(np.mean(dx * dx)) + 1.0 / 12.0
    cyy = float(np.mean(dy * dy)) + 1.0 / 12.0
    cxy = float(np.mean(dx * dy))
    radius = math.hypot((cxx - cyy) / 2.0, cxy)
    mean = (cxx + cyy) / 2.0
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    width = math.sqrt(12.0 * (mean + radius))
    height = math.sqrt(12.0 * max(mean - radius, 1.0 / 12.0))
    return RotatedBox(float(dx.mean()), float(dy.mean()), width, height, math.degrees(theta))


def _sweep_boxes(dx: np.ndarray, dy: np.ndarray, extra_angles: list[float]) -> list[RotatedBox]:
    """For each orientation in a sweep over ``[0, 90)``, the best box at that angle."""
    offsets = (np.arange(_SUPERSAMPLE) + 0.5) / _SUPERSAMPLE - 0.5
    sub_x = (dx[:, None, None] + offsets[None, :, None]).repeat(_SUPERSAMPLE, axis=2).ravel()
    sub_y = (dy[:, None, None] + offsets[None, None, :]).repeat(_SUPERSAMPLE, axis=1).ravel()
    weight = 1.0 / _SUPERSAMPLE**2
    half_spacing = 0.5 / _SUPERSAMPLE

    angles = np.unique(
        np.round(
            np.concatenate(
                [np.arange(0.0, 90.0, _SWEEP_STEP_DEG), np.mod(np.asarray(extra_angles), 90.0)]
            ),
            6,
        )
    )
    boxes: list[RotatedBox] = []
    for angle in angles:
        theta = math.radians(float(angle))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        u = sub_x * cos_t + sub_y * sin_t
        v = sub_y * cos_t - sub_x * sin_t
        u0, v0 = u.min() - half_spacing, v.min() - half_spacing
        extent = max(u.max() - u0, v.max() - v0) + half_spacing
        cell = max(1.0, extent / _GRID_CAP)
        cu = np.floor((u - u0) / cell).astype(np.int64)
        cv = np.floor((v - v0) / cell).astype(np.int64)
        n_u, n_v = int(cu.max()) + 1, int(cv.max()) + 1
        grid = np.bincount(cv * n_u + cu, minlength=n_v * n_u).reshape(n_v, n_u) * weight

        r0, r1, c0, c1 = _dinkelbach(grid, cell * cell, float(dx.size))
        u_lo, u_hi = u0 + c0 * cell, u0 + (c1 + 1) * cell
        v_lo, v_hi = v0 + r0 * cell, v0 + (r1 + 1) * cell
        uc, vc = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0
        boxes.append(
            RotatedBox(
                uc * cos_t - vc * sin_t,
                uc * sin_t + vc * cos_t,
                u_hi - u_lo,
                v_hi - v_lo,
                float(angle),
            )
        )
    return boxes


def _dinkelbach(
    grid: np.ndarray, cell_area: float, total: float, max_iter: int = 30
) -> tuple[int, int, int, int]:
    """Axis-aligned cell rectangle maximising ``I / (total + A - I)`` on a count grid."""
    lam = 0.0
    best = (0, grid.shape[0] - 1, 0, grid.shape[1] - 1)
    for _ in range(max_iter):
        r0, r1, c0, c1 = _max_sum_subrect((1.0 + lam) * grid - lam * cell_area)
        inter = float(grid[r0 : r1 + 1, c0 : c1 + 1].sum())
        area = (r1 - r0 + 1) * (c1 - c0 + 1) * cell_area
        iou = inter / (total + area - inter)
        if iou <= lam + 1e-12:
            break
        lam = iou
        best = (r0, r1, c0, c1)
    return best


def _max_sum_subrect(weights: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive ``(r0, r1, c0, c1)`` of the maximum-sum subrectangle, O(R^2 C) vectorised."""
    transposed = weights.shape[0] > weights.shape[1]
    w = weights.T if transposed else weights
    n_rows, n_cols = w.shape
    row_prefix = np.vstack([np.zeros((1, n_cols)), np.cumsum(w, axis=0)])
    # band[t, b] = sum of rows t..b (meaningful for b >= t)
    band = row_prefix[None, 1:, :] - row_prefix[:-1, None, :]
    col_prefix = np.concatenate([np.zeros((n_rows, n_rows, 1)), np.cumsum(band, axis=2)], axis=2)
    running_min = np.minimum.accumulate(col_prefix[..., :-1], axis=2)
    gain = col_prefix[..., 1:] - running_min
    gain[np.tril(np.ones((n_rows, n_rows), dtype=bool), k=-1)] = -np.inf
    top, bottom, end = np.unravel_index(int(np.argmax(gain)), gain.shape)
    start = int(np.argmin(col_prefix[top, bottom, : end + 1]))
    if transposed:
        return start, int(end), int(top), int(bottom)
    return int(top), int(bottom), start, int(end)


def _soft_iou_params(params: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> float:
    """IoU with each pixel weighted by its (separable) area overlap with the rectangle."""
    cx, cy, w, h, angle = params
    w, h = abs(w), abs(h)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    ddx, ddy = dx - cx, dy - cy
    u = ddx * cos_t + ddy * sin_t
    v = ddy * cos_t - ddx * sin_t
    cover_u = np.minimum(np.maximum(w / 2.0 + 0.5 - np.abs(u), 0.0), 1.0)
    cover_v = np.minimum(np.maximum(h / 2.0 + 0.5 - np.abs(v), 0.0), 1.0)
    inter = float(cover_u @ cover_v)
    return inter / (dx.size + w * h - inter)


def _box_params(box: RotatedBox) -> np.ndarray:
    return np.array([box.center_x, box.center_y, box.width, box.height, box.angle])


def _params_box(params: Params | np.ndarray) -> RotatedBox:
    cx, cy, w, h, angle = (float(p) for p in params)
    return RotatedBox(cx, cy, max(abs(w), _MIN_SIDE), max(abs(h), _MIN_SIDE), angle)


def _refine(seed: RotatedBox, dx: np.ndarray, dy: np.ndarray) -> RotatedBox:
    x0 = _box_params(seed)
    simplex = np.vstack([x0, x0 + np.diag(_SIMPLEX_STEPS)])
    result = minimize(
        lambda p: -_soft_iou_params(p, dx, dy),
        x0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 2e-2, "fatol": 1e-5, "maxfev": 300},
    )
    return _params_box(result.x)


def _polish(blob: _Blob, box: RotatedBox) -> tuple[RotatedBox, float, float]:
    """Pattern search on ``(exact IoU, tie score)``, compared lexicographically.

    The exact IoU is piecewise constant; on its plateaus the smooth tie score decides,
    which guides the search across flat regions and settles in the middle of the plateau,
    where small perturbations (e.g. rounding the corners) are least likely to change the
    painted pixels. Returns ``(box, exact_iou, tie_score)``.
    """
    params = (box.center_x, box.center_y, box.width, box.height, box.angle)
    iou, tie = blob.exact_iou(box), blob.tie_score(params)
    step = _POLISH_START_STEP
    for _ in range(_POLISH_MAX_ITER):
        if step < _POLISH_MIN_STEP:
            break
        best_iou, best_tie, best_params = iou, tie, None
        for trial in _moves(params, step):
            trial_iou = blob.exact_iou(_params_box(trial))
            if trial_iou < best_iou:
                continue
            trial_tie = blob.tie_score(trial)
            if trial_iou > best_iou or trial_tie > best_tie + _TIE_TOL:
                best_iou, best_tie, best_params = trial_iou, trial_tie, trial
        if best_params is None:
            step /= 2.0
        else:
            iou, tie, params = best_iou, best_tie, best_params
    return _params_box(params), iou, tie


def _moves(params: Params, step: float) -> list[Params]:
    """Neighbouring boxes: each parameter +-step, one side out/in, rotation about an end.

    Moving a single side (resize plus half-step shift) adds or drops one row of pixels on
    that side only, which the plain per-parameter moves cannot do in one step.
    """
    cx, cy, w, h, angle = params
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    moves: list[Params] = []
    for sign in (1.0, -1.0):
        delta = sign * step
        moves += [
            (cx + delta, cy, w, h, angle),
            (cx, cy + delta, w, h, angle),
            (cx, cy, w + delta, h, angle),
            (cx, cy, w, h + delta, angle),
            (cx, cy, w, h, angle + delta),
        ]
        for side in (1.0, -1.0):
            shift = side * delta / 2.0
            moves.append((cx + shift * cos_t, cy + shift * sin_t, w + delta, h, angle))
            moves.append((cx - shift * sin_t, cy + shift * cos_t, w, h + delta, angle))
    turn = step * _POLISH_DEG_PER_STEP
    for sign in (1.0, -1.0):
        rot = math.radians(sign * turn)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        for end in (1.0, -1.0):
            px, py = cx + end * w / 2.0 * cos_t, cy + end * w / 2.0 * sin_t
            nx = px + (cx - px) * cos_r - (cy - py) * sin_r
            ny = py + (cx - px) * sin_r + (cy - py) * cos_r
            moves.append((nx, ny, w, h, angle + sign * turn))
    return moves
