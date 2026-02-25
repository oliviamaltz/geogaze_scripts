#!/usr/bin/env python3
"""
Reduce a COCO instances JSON file to just the fields needed for:
- all images
- all bounding boxes
- what each bounding box contains (category)

Keeps:
  images:     id, file_name, width, height
  annotations: id, image_id, category_id, bbox, iscrowd, area
  categories: id, name, supercategory

Usage:
  python coco_minify_instances.py \
      /path/to/instances_train2017.json \
      /path/to/instances_train2017_min.json
"""

import json
import argparse
from pathlib import Path


def minify_coco(input_path: Path, output_path: Path):
    print(f"[info] Loading COCO file: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # ---- images ----
    images = []
    for img in data.get("images", []):
        images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })

    # ---- annotations ----
    annotations = []
    for ann in data.get("annotations", []):
        annotations.append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],          # [x, y, w, h]
            "iscrowd": ann.get("iscrowd", 0),
            "area": ann.get("area", None),
        })

    # ---- categories ----
    categories = []
    for cat in data.get("categories", []):
        categories.append({
            "id": cat["id"],
            "name": cat["name"],
            "supercategory": cat.get("supercategory"),
        })

    minimal = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    print(f"[info] Writing minimized COCO file: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(minimal, f, ensure_ascii=False)

    print("[done] Minified COCO saved.")


def main():
    parser = argparse.ArgumentParser(description="Minify COCO instances JSON.")
    parser.add_argument("input_json", type=Path,
                        help="Path to original instances_*.json")
    parser.add_argument("output_json", type=Path,
                        help="Path to write minimized JSON")

    args = parser.parse_args()
    minify_coco(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
