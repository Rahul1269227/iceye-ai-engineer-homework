"""Mask loading, CSV output and a visual overlay for the fitted boxes."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .fit import DECIMALS, BlobFit

CSV_COLUMNS = ("blob_id", "center_x", "center_y", "width", "height", "angle")


def load_mask(path: Path, threshold: int = 127) -> np.ndarray:
    """Load a binary mask as a 2D boolean array.

    ``.npy`` files are read as-is (non-zero is foreground). Images are read as grayscale
    and thresholded (``> threshold``), which also removes JPEG compression noise.
    """
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"{path}: expected a 2D array, got shape {array.shape}")
        return array != 0
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"{path}: not a readable image")
    return image > threshold


def write_csv(fits: Iterable[BlobFit], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for fit in fits:
            box = fit.box
            writer.writerow(
                [
                    fit.blob_id,
                    *(
                        round(value, DECIMALS)
                        for value in (box.center_x, box.center_y, box.width, box.height, box.angle)
                    ),
                ]
            )


def write_overlay(mask: np.ndarray, fits: Iterable[BlobFit], path: Path, scale: int = 2) -> None:
    """Save the mask (grey) with each fitted box (red) and its id, upscaled for legibility."""
    canvas = cv2.cvtColor((mask.astype(np.uint8) * 140), cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    shift = 4
    for fit in fits:
        corners = fit.box.corners() * scale + (scale - 1) / 2.0
        points = np.round(corners * (1 << shift)).astype(np.int32)
        cv2.polylines(canvas, [points], True, (40, 40, 255), 1, cv2.LINE_AA, shift)
        anchor = tuple(int(v) for v in corners.min(axis=0) - (0, 2))
        cv2.putText(
            canvas, str(fit.blob_id), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 220, 80), 1
        )
    cv2.imwrite(str(path), canvas)
