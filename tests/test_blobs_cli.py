"""blob-boxes CLI."""

import csv

import cv2
import numpy as np

from iceye_homework.blobs.cli import main


def _mask_png(tmp_path, mask):
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return path


def test_cli_writes_csv(tmp_path, capsys):
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:35, 30:50] = True
    mask_path = _mask_png(tmp_path, mask)
    out = tmp_path / "boxes.csv"

    assert main([str(mask_path), "--out", str(out), "--workers", "1", "--restarts", "0"]) == 0
    err = capsys.readouterr().err
    assert "1 blobs" in err
    with out.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "blob_id"
    assert len(rows) == 2


def test_cli_empty_mask_warns(tmp_path, capsys):
    mask_path = _mask_png(tmp_path, np.zeros((20, 20), dtype=bool))
    out = tmp_path / "boxes.csv"
    assert main([str(mask_path), "--out", str(out), "--workers", "1"]) == 0
    err = capsys.readouterr().err
    assert "no foreground" in err
    with out.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1


def test_cli_bad_mask(tmp_path, capsys):
    assert main([str(tmp_path / "nope.png"), "--out", str(tmp_path / "o.csv")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_overlay_and_objective(tmp_path):
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:25] = True
    mask_path = _mask_png(tmp_path, mask)
    out = tmp_path / "boxes.csv"
    overlay = tmp_path / "overlay.png"
    assert (
        main(
            [
                str(mask_path),
                "--out",
                str(out),
                "--overlay",
                str(overlay),
                "--objective",
                "center",
                "--workers",
                "1",
                "--restarts",
                "0",
            ]
        )
        == 0
    )
    assert overlay.is_file()


def test_cli_min_area(tmp_path):
    mask = np.zeros((40, 40), dtype=bool)
    mask[5, 5] = True
    mask[20:30, 20:30] = True
    mask_path = _mask_png(tmp_path, mask)
    out = tmp_path / "boxes.csv"
    assert (
        main(
            [
                str(mask_path),
                "--out",
                str(out),
                "--min-area",
                "5",
                "--workers",
                "1",
                "--restarts",
                "0",
            ]
        )
        == 0
    )
    with out.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2
