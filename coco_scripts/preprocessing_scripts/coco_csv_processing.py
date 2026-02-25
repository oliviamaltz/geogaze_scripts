#!/usr/bin/env python3
"""
COCO JSON -> CSV

Writes one row per annotation (unique annotations.id):
- image_file_name (images.file_name)
- image_id        (annotations.image_id)
- annotation_id   (annotations.id)  <-- this is the "segmentation instance id" in COCO
- category_id     (annotations.category_id == categories.id)
- category_name   (categories.name)
"""

import json
import csv
import argparse
from pathlib import Path


def coco_to_csv(json_path: Path, out_csv: Path) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    # Lookups
    image_id_to_filename = {
        im.get("id"): (im.get("file_name") or im.get("filename") or "")
        for im in images
        if im.get("id") is not None
    }

    cat_id_to_name = {
        c.get("id"): (c.get("name") or "")
        for c in categories
        if c.get("id") is not None
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_file_name",
        "image_id",
        "annotation_id",
        "category_id",
        "category_name",
        "segmentation_len",
    ]

    seen = set()
    n_rows = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ann in annotations:
            ann_id = ann.get("id")
            if ann_id is None or ann_id in seen:
                continue
            seen.add(ann_id)

            image_id = ann.get("image_id")
            category_id = ann.get("category_id")

            seg = ann.get("segmentation", None)
            # segmentation is usually: list[list[float]] (polygons) OR RLE dict for crowds
            if isinstance(seg, list):
                seg_len = sum(len(poly) for poly in seg if isinstance(poly, list))
            elif isinstance(seg, dict):
                seg_len = 1  # RLE present (not polygon list)
            else:
                seg_len = 0

            writer.writerow({
                "image_file_name": image_id_to_filename.get(image_id, ""),
                "image_id": image_id if image_id is not None else "",
                "annotation_id": ann_id,
                "category_id": category_id if category_id is not None else "",
                "category_name": cat_id_to_name.get(category_id, ""),
                "segmentation_len": seg_len,
            })
            n_rows += 1

    print(f"Wrote {n_rows} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to COCO annotations JSON")
    ap.add_argument("--out", required=True, help="Path to output CSV")
    args = ap.parse_args()

    coco_to_csv(Path(args.json), Path(args.out))


if __name__ == "__main__":
    main()
