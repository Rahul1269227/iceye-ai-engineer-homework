"""``blob-boxes``: fit IoU-maximising rotated boxes to every blob in a binary mask."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from .fit import fit_blob_boxes
from .geometry import OBJECTIVES
from .io import load_mask, write_csv, write_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blob-boxes",
        description="Detect blobs in a binary mask and write their IoU-maximising rotated "
        "bounding boxes to a CSV file.",
    )
    parser.add_argument("mask", type=Path, help="mask as .npy (2D array) or image (png/jpg/...)")
    parser.add_argument(
        "--out", type=Path, default=Path("blob_bounding_boxes.csv"), help="output CSV path"
    )
    parser.add_argument("--overlay", type=Path, help="optional PNG with the boxes drawn on top")
    parser.add_argument(
        "--threshold", type=int, default=127, help="foreground is gray > threshold (images only)"
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="robust",
        help="how a box becomes a pixel mask for IoU: 'center' = pixel centres inside the "
        "box; cv2.fillPoly of cv2.boxPoints with corners at 1/16 px ('subpixel'), rounded "
        "('round') or truncated ('truncate', like np.intp); 'robust' (default) = mean IoU "
        "over all four",
    )
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-area", type=int, default=1, help="ignore blobs smaller than this")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes to fit blobs in parallel (default: all CPUs; 1 = in-process)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mask = load_mask(args.mask, args.threshold)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    start = time.perf_counter()
    fits = fit_blob_boxes(
        mask,
        connectivity=args.connectivity,
        min_area=args.min_area,
        objective=args.objective,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - start

    write_csv(fits, args.out)
    if args.overlay:
        write_overlay(mask, fits, args.overlay)

    summary = f"{len(fits)} blobs in {elapsed:.2f}s -> {args.out}"
    if fits:
        ious = [fit.iou for fit in fits]
        baseline = [fit.baseline_iou for fit in fits]
        summary += (
            f"\nIoU mean {statistics.fmean(ious):.4f} / median {statistics.median(ious):.4f}"
            f"  (minAreaRect baseline mean {statistics.fmean(baseline):.4f},"
            f" improved on {sum(i > b + 1e-9 for i, b in zip(ious, baseline, strict=True))}"
            f" of {len(fits)} blobs)"
        )
    print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
