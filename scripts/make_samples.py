# /// script
# requires-python = "==3.12.*"
# dependencies = ["pymupdf==2.0", "numpy==2.5.4", "opencv-python-headless==5.0.0.93"]
# ///
"""Generate the Part 1 sample documents (already committed under data/).

    uv run scripts/make_samples.py

``sample_report.pdf`` has a text layer; ``sample_scan.png`` is the same page rasterised,
slightly rotated and blurred, so extracting it exercises the OCR path. The report is
fictional.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf

TITLE = "Flood Response Field Report (fictional sample)"
BODY = """\
On 14 March 2026, the Nordic Flood Coordination Centre in Helsinki asked Kestrel Orbital \
to task its radar satellites over the Kokemaki river basin in western Finland. Kestrel \
Orbital delivered the first SAR images to the centre within six hours.

Dr. Aino Virtanen, who leads the Helsinki team, compared the radar images with optical \
data from Sentinel-2. Cloud cover over Pori made the optical images unusable, so the team \
relied on Kestrel Orbital radar data for the first three days.

By 17 March 2026, Aino Virtanen reported that the flooded area near Pori had grown to 42 \
square kilometres. The Finnish Red Cross used the maps to plan evacuations in Pori and \
Ulvila. Marcus Lindqvist of the Finnish Red Cross said the maps arrived faster than in any \
previous flood.

Kestrel Orbital will present the response at the Living Planet Symposium in Vienna in \
June 2026. The Nordic Flood Coordination Centre plans to extend its contract with Kestrel \
Orbital through 2027.
"""

DATA = Path(__file__).resolve().parent.parent / "data"


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    page.insert_text((60, 80), TITLE, fontsize=15, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(60, 105, 535, 780), BODY, fontsize=11.5, lineheight=1.4)
    doc.set_metadata({"title": TITLE, "author": "ICEYE homework sample"})
    doc.save(path, garbage=4, deflate=True)


def make_scan(pdf: Path, path: Path, dpi: int = 150) -> None:
    page = pymupdf.open(pdf)[0]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    rows, cols = image.shape
    rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0.8, 1.0)
    rotated = cv2.warpAffine(image, rotation, (cols, rows), borderValue=255)
    scan = cv2.GaussianBlur(rotated[: rows // 2], (3, 3), 0)  # the text fills the top half
    cv2.imwrite(str(path), scan, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    pdf = DATA / "sample_report.pdf"
    make_pdf(pdf)
    make_scan(pdf, DATA / "sample_scan.png")
    print(f"wrote {pdf} and {DATA / 'sample_scan.png'}")


if __name__ == "__main__":
    main()
