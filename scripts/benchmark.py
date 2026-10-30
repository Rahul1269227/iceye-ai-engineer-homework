"""Reproduce the Part 2 numbers and the figure quoted in the README.

    uv run python scripts/benchmark.py [--mask data/example_mask.png] [--figure docs/blobs_zoom.png]

Prints markdown tables: results on the mask per objective, every objective's boxes scored
under every raster model next to cv2.minAreaRect, IoU against the number of random
restarts, and recovery of synthetic rectangles whose ground truth is known. Every IoU is
recomputed independently from the box as written to the CSV, with ``box_mask`` on the
full image. Times are the best of three runs.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from iceye_homework.blobs import (
    OBJECTIVES,
    RASTERS,
    BlobFit,
    Objective,
    RotatedBox,
    box_mask,
    fit_blob_boxes,
    load_mask,
)
from iceye_homework.blobs.fit import RESTARTS, label_blobs, min_area_rect

CPUS = os.cpu_count() or 1


def raster_ious(blobs: Sequence[np.ndarray], boxes: Sequence[RotatedBox]) -> np.ndarray:
    """``(n_blobs, n_rasters)`` IoU of each box under each raster model, on the full image."""
    out = np.zeros((len(blobs), len(RASTERS)))
    for i, (blob, box) in enumerate(zip(blobs, boxes, strict=True)):
        for j, raster in enumerate(RASTERS):
            rect = box_mask(box, blob.shape, raster)
            out[i, j] = np.count_nonzero(blob & rect) / np.count_nonzero(blob | rect)
    return out


def objective_iou(per_raster: np.ndarray, objective: Objective) -> np.ndarray:
    if objective == "robust":
        return per_raster.mean(axis=1)
    return per_raster[:, RASTERS.index(objective)]


def min_area_rects(blobs: Sequence[np.ndarray]) -> list[RotatedBox]:
    boxes = []
    for blob in blobs:
        ys, xs = np.nonzero(blob)
        boxes.append(min_area_rect(xs, ys))
    return boxes


def timed(fn: Callable[[], list[BlobFit]], repeats: int = 3) -> tuple[list[BlobFit], float]:
    """Result and best-of-``repeats`` wall time; the fits are deterministic."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best


def mask_report(mask: np.ndarray) -> list[BlobFit]:
    count, labels, _ = label_blobs(mask)
    blobs = [labels == blob_id for blob_id in range(1, count + 1)]
    baseline = raster_ious(blobs, min_area_rects(blobs))
    print(f"## Example mask: {mask.shape[1]}x{mask.shape[0]}, {count} blobs\n")
    print(
        "| objective | mean IoU | median | min | cv2.minAreaRect | better than it "
        f"| time, 1 process | time, {CPUS} processes |"
    )
    print("|---|---|---|---|---|---|---|---|")
    per_objective: dict[Objective, np.ndarray] = {}
    fits_by_objective: dict[Objective, list[BlobFit]] = {}
    for objective in OBJECTIVES:
        fits, serial = timed(partial(fit_blob_boxes, mask, objective=objective))
        _, parallel = timed(partial(fit_blob_boxes, mask, objective=objective, workers=CPUS))
        scores = raster_ious(blobs, [fit.box for fit in fits])
        ious = objective_iou(scores, objective)
        reported = np.array([fit.iou for fit in fits])
        if not np.allclose(ious, reported, rtol=0, atol=1e-12):
            raise AssertionError(f"{objective}: reported IoU does not match the CSV boxes")
        base = objective_iou(baseline, objective)
        if not np.allclose(base, [fit.baseline_iou for fit in fits], rtol=0, atol=1e-12):
            raise AssertionError(f"{objective}: fit baseline differs from cv2.minAreaRect")
        print(
            f"| `{objective}` | {ious.mean():.4f} | {np.median(ious):.4f} | {ious.min():.4f} "
            f"| {base.mean():.4f} | {np.count_nonzero(ious > base + 1e-9)}/{count} "
            f"| {serial:.1f} s | {parallel:.1f} s |"
        )
        per_objective[objective] = scores
        fits_by_objective[objective] = fits

    print("\n## Every objective's boxes under every raster model (mean IoU)\n")
    print("| boxes | " + " | ".join(f"`{r}`" for r in RASTERS) + " | mean | worst |")
    print("|---|" + "---|" * (len(RASTERS) + 2))
    rows = {f"ours, `{o}`": s for o, s in per_objective.items()}
    rows["cv2.minAreaRect"] = baseline
    for name, scores in rows.items():
        means = scores.mean(axis=0)
        cells = " | ".join(f"{v:.4f}" for v in means)
        print(f"| {name} | {cells} | {means.mean():.4f} | {means.min():.4f} |")
    return fits_by_objective["robust"]


def restarts_report(mask: np.ndarray, counts: Sequence[int] = (0, 8, 16, 32, 64)) -> None:
    """Mean ``robust`` IoU and run time against the number of random restarts per blob."""
    runs = {
        restarts: timed(partial(fit_blob_boxes, mask, workers=CPUS, restarts=restarts))
        for restarts in sorted({*counts, RESTARTS})
    }
    default = np.array([fit.iou for fit in runs[RESTARTS][0]])
    print(f"\n## Random restarts per blob (`robust`, {CPUS} processes)\n")
    print(f"| restarts | mean IoU | better than {RESTARTS} | worse than {RESTARTS} | time |")
    print("|---|---|---|---|---|")
    for restarts, (fits, elapsed) in runs.items():
        ious = np.array([fit.iou for fit in fits])
        print(
            f"| {restarts} | {ious.mean():.4f} | {np.count_nonzero(ious > default + 1e-12)} "
            f"| {np.count_nonzero(ious < default - 1e-12)} | {elapsed:.1f} s |"
        )


def synthetic_report(n: int = 200, seed: int = 7) -> None:
    """Rectangles painted with one raster model must come back with IoU 1 under it."""
    rng = np.random.default_rng(seed)
    cell, per_row = 64, 20
    shape = (cell * (n // per_row + 1), cell * per_row)
    truth = []
    for i in range(n):
        width = rng.uniform(2.0, 40.0)
        height = rng.uniform(1.0, min(width, 12.0))
        row, col = divmod(i, per_row)
        cx = col * cell + cell / 2 + rng.uniform(-0.5, 0.5)
        cy = row * cell + cell / 2 + rng.uniform(-0.5, 0.5)
        truth.append(RotatedBox(cx, cy, width, height, rng.uniform(-90.0, 90.0)))

    print(f"\n## Synthetic rectangles ({n}; 2-40 px by 1-12 px, any angle)\n")
    print("| painted with, and optimised for | mean IoU | min IoU | recovered exactly |")
    print("|---|---|---|---|")
    for raster in RASTERS:
        mask = np.zeros(shape, dtype=bool)
        for box in truth:
            mask |= box_mask(box, shape, raster)
        fits = fit_blob_boxes(mask, objective=raster, workers=CPUS)
        ious = [fit.iou for fit in fits]
        exact = sum(iou == 1.0 for iou in ious)
        print(
            f"| `{raster}` | {statistics.fmean(ious):.4f} | {min(ious):.4f} | {exact}/{len(fits)} |"
        )


def zoom_figure(
    mask: np.ndarray, fits: list[BlobFit], path: Path, size: int = 96, scale: int = 8
) -> None:
    """Blob pixels, our boxes (green) and minAreaRect (red) in the ``size``-pixel window
    where the fit gains most over minAreaRect (IoU gain weighted by blob area)."""
    count, labels, stats = label_blobs(mask)
    lo, hi = stats[1:, :2], stats[1:, :2] + stats[1:, 2:4]
    gain = np.array([fit.area * max(fit.iou - fit.baseline_iou, 0.0) for fit in fits])
    step = size // 8
    windows = [
        (x0, y0)
        for y0 in range(0, max(mask.shape[0] - size, 0) + 1, step)
        for x0 in range(0, max(mask.shape[1] - size, 0) + 1, step)
    ]

    def gain_in(window: tuple[int, int]) -> float:
        margin = 3
        inside = np.all((lo >= np.add(window, margin)) & (hi <= np.add(window, size - margin)), 1)
        return float(gain[inside].sum())

    x0, y0 = max(windows, key=gain_in)
    crop = mask[y0 : y0 + size, x0 : x0 + size].astype(np.uint8) * 90 + 25
    canvas = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    shift = 4

    def draw(box: RotatedBox, colour: tuple[int, int, int]) -> None:
        corners = (box.corners() - (x0, y0)) * scale + (scale - 1) / 2.0
        points = np.round(corners * (1 << shift)).astype(np.int32)
        cv2.polylines(canvas, [points], True, colour, 2, cv2.LINE_AA, shift)

    blobs = [labels == blob_id for blob_id in range(1, count + 1)]
    for box in min_area_rects(blobs):
        draw(box, (60, 60, 230))
    for fit in fits:
        draw(fit.box, (80, 220, 80))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "ours (robust)", (10, 26), font, 0.7, (80, 220, 80), 2)
    cv2.putText(canvas, "cv2.minAreaRect", (190, 26), font, 0.7, (60, 60, 230), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    print(f"\nFigure: {path} (window x={x0}..{x0 + size}, y={y0}..{y0 + size}, x{scale})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mask", type=Path, default=Path("data/example_mask.png"))
    parser.add_argument("--figure", type=Path, default=Path("docs/blobs_zoom.png"))
    args = parser.parse_args()
    mask = load_mask(args.mask)
    fits = mask_report(mask)
    restarts_report(mask)
    synthetic_report()
    zoom_figure(mask, fits, args.figure)


if __name__ == "__main__":
    main()
