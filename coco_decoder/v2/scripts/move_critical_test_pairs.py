#!/usr/bin/env python3
"""
Copy ONLY test_image files into per-model folders.

- Source folder contains all images
- Output root will contain one subfolder per unique model
- Each model folder gets the test_image files used by that model
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from collections import defaultdict



# =========================
# ===== EDIT THESE ========
# =========================
CSV_PATH = Path("/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/v1/identification/geogaze_model_predictions_identification_cornetIDEN_01_29_26.csv")
SOURCE_IMAGE_DIR = Path("/zpool/vladlab/active_drive/omaltz/scripts/geogaze/decoder_model/stimuli/out/test_stimuli/test_pairs")
OUTPUT_ROOT = Path("/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/v2/identification/test_images")
# =========================

MODEL_COL = "model"
TEST_IMAGE_COL = "test_image"


def clean_cell(s: str | None) -> str:
    return (s or "").strip()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not SOURCE_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Source image folder not found: {SOURCE_IMAGE_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    copied_per_model: dict[str, set[str]] = defaultdict(set)

    n_rows = 0
    n_copied = 0
    n_missing = 0
    missing_files: list[tuple[str, str]] = []  # (model, filename)

    with CSV_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if MODEL_COL not in reader.fieldnames or TEST_IMAGE_COL not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{MODEL_COL}' and '{TEST_IMAGE_COL}'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            n_rows += 1

            model = clean_cell(row.get(MODEL_COL))
            test_image = clean_cell(row.get(TEST_IMAGE_COL))

            if not model or not test_image:
                continue

            model_dir = OUTPUT_ROOT / model
            model_dir.mkdir(parents=True, exist_ok=True)

            if test_image in copied_per_model[model]:
                continue

            src = SOURCE_IMAGE_DIR / test_image
            if not src.exists():
                n_missing += 1
                missing_files.append((model, test_image))
                continue

            shutil.copy2(src, model_dir / test_image)
            copied_per_model[model].add(test_image)
            n_copied += 1

    print("Done.")
    print(f"Rows read: {n_rows}")
    print(f"Models found: {len(copied_per_model)}")
    print(f"Images copied: {n_copied}")
    print(f"Missing images: {n_missing}")

    if missing_files:
        report = OUTPUT_ROOT / "missing_test_images.csv"
        with report.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "test_image"])
            w.writerows(missing_files)
        print(f"Missing-image report written to: {report}")


if __name__ == "__main__":
    main()
