#!/usr/bin/env python3
"""
Filter COCO instances JSON based on:
  1) Remove annotations where iscrowd == 1
  2) Remove annotations whose bounding box area is < 1/8 of the image area
  3) Remove annotations whose category *name* is in an exclusion list
  4) After the above, remove any image that still has > MAX_BBOX_PER_IMAGE annotations
     (and remove those annotations too)

Keeps the original 'categories' unchanged.

Usage:
  python json_filtering.py \
      /path/to/instances_train2017.json \
      /path/to/instances_train2017_filtered.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# -----------------------------
# CONFIG: you will edit this part
# -----------------------------

# Minimum fraction of the *image* area a bounding box must cover to be kept.
MIN_FRACTION_OF_IMAGE = 1.0 / 8.0   # "less than an eighth" will be removed

# Maximum number of bounding boxes allowed per image
MAX_BBOX_PER_IMAGE = 6

# Category names whose bounding boxes you want to remove entirely
# (edit this list as you like)
EXCLUDED_CATEGORY_NAMES = [
    # Example:
    # "person",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def build_image_lookup(images):
    """Return dict: image_id -> {'width': ..., 'height': ..., ...}"""
    return {img["id"]: img for img in images}


def build_category_lookup(categories):
    """Return dict: category_id -> category_dict"""
    return {cat["id"]: cat for cat in categories}


def filter_annotations(data):
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    images_by_id = build_image_lookup(images)
    cats_by_id = build_category_lookup(categories)

    # Map category_id -> name for exclusion checks
    cat_id_to_name = {
        cid: cdict.get("name", "")
        for cid, cdict in cats_by_id.items()
    }
    excluded_names_set = set(name.lower() for name in EXCLUDED_CATEGORY_NAMES)

    filtered_annotations = []

    # -------- 1–3: apply per-annotation filters --------
    for ann in annotations:
        iscrowd = ann.get("iscrowd", 0)
        if iscrowd == 1:
            # Skip crowd annotations
            continue

        img_id = ann["image_id"]
        img_info = images_by_id.get(img_id)
        if img_info is None:
            # Unknown image_id; skip
            continue

        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            # Missing dimensions; skip this annotation
            continue

        # Bounding box: [x, y, width, height]
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            # No bbox, skip
            continue

        _, _, bw, bh = bbox
        bbox_area = bw * bh
        img_area = img_w * img_h

        # Remove if bbox is smaller than the fraction threshold
        if bbox_area < img_area * MIN_FRACTION_OF_IMAGE:
            continue

        # Remove if category name is in the excluded list
        cat_id = ann.get("category_id")
        cat_name = cat_id_to_name.get(cat_id, "").lower()

        if cat_name in excluded_names_set:
            continue

        # Passed all filters -> keep it
        filtered_annotations.append(ann)

    # -------- 4: remove images with too many boxes --------
    # Count annotations per image
    ann_count_by_image = defaultdict(int)
    for ann in filtered_annotations:
        ann_count_by_image[ann["image_id"]] += 1

    # Images to remove: > MAX_BBOX_PER_IMAGE remaining annotations
    images_to_remove = {
        img_id
        for img_id, count in ann_count_by_image.items()
        if count > MAX_BBOX_PER_IMAGE
    }

    # Keep only annotations whose image_id is not in images_to_remove
    final_annotations = [
        ann for ann in filtered_annotations
        if ann["image_id"] not in images_to_remove
    ]

    # Keep only images that are not removed AND still have at least one annotation
    # (i.e., appear in final_annotations)
    remaining_image_ids = {ann["image_id"] for ann in final_annotations}
    final_images = [
        img for img in images
        if img["id"] in remaining_image_ids
        and img["id"] not in images_to_remove
    ]

    # Categories are unchanged
    final_categories = categories

    return {
        "images": final_images,
        "annotations": final_annotations,
        "categories": final_categories,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter COCO instances JSON by iscrowd, bbox size, "
                    "excluded category names, and max bboxes per image."
    )
    parser.add_argument("input_json", type=Path,
                        help="Path to original instances_*.json")
    parser.add_argument("output_json", type=Path,
                        help="Path to write filtered JSON")

    args = parser.parse_args()

    print(f"[info] Loading: {args.input_json}")
    data = load_json(args.input_json)

    print("[info] Filtering annotations/images...")
    filtered = filter_annotations(data)

    # Some simple stats
    n_img_before = len(data.get("images", []))
    n_ann_before = len(data.get("annotations", []))
    n_img_after = len(filtered["images"])
    n_ann_after = len(filtered["annotations"])

    print(f"[stats] images: {n_img_before} -> {n_img_after}")
    print(f"[stats] annotations: {n_ann_before} -> {n_ann_after}")

    print(f"[info] Saving filtered JSON to: {args.output_json}")
    save_json(filtered, args.output_json)
    print("[done] Finished.")


if __name__ == "__main__":
    main()
