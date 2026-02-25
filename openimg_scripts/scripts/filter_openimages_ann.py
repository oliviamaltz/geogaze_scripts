#!/usr/bin/env python3
"""
Filter Open Images bounding-box annotations before downloading images.

Per split (train/validation/test), this script:

  1. Loads the original *-annotations-bbox.csv
  2. Removes any IMAGE that has at least one box with IsGroupOf == 1
  3. Removes any BOX whose area < AREA_MIN_FRAC (fraction of image area)
  4. Removes any IMAGE that has more than MAX_BOXES_PER_IMAGE boxes
  5. Drops any image that ends up with zero boxes (implicit)
  6. Saves:
       - filtered CSV
       - text file of "split/ImageID" lines for downloader.py
"""

import pandas as pd
from pathlib import Path

# ===== EDIT THESE PATHS / CONSTANTS =====
ANNOT_DIR = Path("/zpool/vladlab/data_drive/stimulus_sets/geogaze_open_images_stim/openimages_orig/annotations")

SPLITS = {
    "train":      "train-annotations-bbox.csv",
    "validation": "validation-annotations-bbox.csv",
    "test":       "test-annotations-bbox.csv",
}

# area threshold as a fraction of image area (1/16)
AREA_MIN_FRAC = 1.0 / 16.0   # 0.0625

# maximum number of boxes allowed per image (after other filters)
MAX_BOXES_PER_IMAGE = 6


def process_split(split_name: str, csv_name: str):
    in_path = ANNOT_DIR / csv_name
    print(f"\n=== Processing {split_name} from {in_path} ===")

    df = pd.read_csv(in_path)

    # Ensure numeric types
    for col in ["XMin", "XMax", "YMin", "YMax"]:
        df[col] = df[col].astype(float)

    df["IsGroupOf"] = df["IsGroupOf"].astype(int)

    # 1) Remove any IMAGE that has at least one IsGroupOf == 1
    bad_images = df.loc[df["IsGroupOf"] == 1, "ImageID"].unique()
    print(f"  Images with IsGroupOf=1: {len(bad_images):,}")

    df = df[~df["ImageID"].isin(bad_images)]
    print(f"  Remaining rows after removing IsGroupOf images: {len(df):,}")

    # 2) Remove boxes that are smaller than AREA_MIN_FRAC of the image
    box_area = (df["XMax"] - df["XMin"]) * (df["YMax"] - df["YMin"])
    df = df[box_area >= AREA_MIN_FRAC]
    print(f"  Remaining rows after area filter (>= {AREA_MIN_FRAC}): {len(df):,}")

    # Optional sanity check for valid boxes
    df = df[(df["XMax"] > df["XMin"]) & (df["YMax"] > df["YMin"])]

    # 3) NEW: remove images that have more than MAX_BOXES_PER_IMAGE boxes
    box_counts = df["ImageID"].value_counts()
    too_many_images = box_counts[box_counts > MAX_BOXES_PER_IMAGE].index
    print(f"  Images with > {MAX_BOXES_PER_IMAGE} boxes: {len(too_many_images):,}")

    df = df[~df["ImageID"].isin(too_many_images)]
    print(f"  Remaining rows after max-boxes filter: {len(df):,}")

    # 4) Drop images that no longer have any boxes (images with 0 boxes are gone)
    keep_images = df["ImageID"].unique()
    print(f"  Images with at least one kept box: {len(keep_images):,}")

    # 5) Save filtered CSV
    out_csv = ANNOT_DIR / f"{split_name}-annotations-bbox.filtered.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved filtered annotations to: {out_csv}")

    # 6) Save image list for downloader.py  (format: split/ImageID)
    img_list_path = ANNOT_DIR / f"{split_name}_images_filtered.txt"
    with open(img_list_path, "w") as f:
        for img_id in keep_images:
            f.write(f"{split_name}/{img_id}\n")
    print(f"  Saved image list for downloader.py to: {img_list_path}")


def main():
    for split, fname in SPLITS.items():
        process_split(split, fname)


if __name__ == "__main__":
    main()
