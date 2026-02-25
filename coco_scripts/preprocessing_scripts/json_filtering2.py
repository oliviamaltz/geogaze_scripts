#!/usr/bin/env python3
"""
Filter COCO instances JSON based on:
  1) Remove *images* that have ANY annotation with iscrowd == 1,
     and remove ALL annotations belonging to those images.
     (Example: if image 3 has 5 boxes and any one has iscrowd=1,
      drop image 3 and all 5 boxes.)
  2) Remove annotations whose bbox area is < 1/16 of the image area
  3) Remove annotations whose category *name* is in an exclusion list
  4) After the above, remove any image that still has > MAX_BBOX_PER_IMAGE annotations
     (and remove those annotations too)

Keeps the original 'categories' unchanged.

Usage:
  python json_filtering2.py \
      /path/to/instances_train2017.json \
      /path/to/instances_train2017_filtered.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------

# Minimum fraction of the *image* area a bounding box must cover to be kept.
MIN_FRACTION_OF_IMAGE = 1.0 / 16.0   # "less than an eighth" will be removed

# Maximum number of bounding boxes allowed per image
MAX_BBOX_PER_IMAGE = 5

# Category names whose bounding boxes you want to remove entirely
EXCLUDED_CATEGORY_NAMES = [
    # Example:
    # "person",
    "tie",
    "skis",
    "snowboard",
    "sink",
    "train",
    "apple",
    "banana",
    "broccoli",
    "orange"
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def build_image_lookup(images):
    """Return dict: image_id -> image_dict"""
    return {img["id"]: img for img in images}


def build_category_lookup(categories):
    """Return dict: category_id -> category_dict"""
    return {cat["id"]: cat for cat in categories}


def filter_coco(data):
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    images_by_id = build_image_lookup(images)
    cats_by_id = build_category_lookup(categories)

    # Map category_id -> name for exclusion checks
    cat_id_to_name = {cid: cdict.get("name", "") for cid, cdict in cats_by_id.items()}
    excluded_names_set = {name.lower() for name in EXCLUDED_CATEGORY_NAMES}

    # -------- 1) Find images that have ANY crowd annotation --------
    crowd_image_ids = {
        ann.get("image_id")
        for ann in annotations
        if ann.get("iscrowd", 0) == 1 and "image_id" in ann
    }

    # Filter out any annotation that belongs to an image that has a crowd box
    annotations_no_crowd_images = [
        ann for ann in annotations
        if ann.get("image_id") not in crowd_image_ids
    ]

    # Also filter out those images from the image catalog
    images_no_crowd_images = [
        img for img in images
        if img.get("id") not in crowd_image_ids
    ]

    # Rebuild lookup in case we removed images
    images_by_id = build_image_lookup(images_no_crowd_images)

    # -------- 2–3) Per-annotation filters --------
    filtered_annotations = []
    for ann in annotations_no_crowd_images:
        img_id = ann.get("image_id")
        if img_id is None:
            continue

        img_info = images_by_id.get(img_id)
        if img_info is None:
            # Image entry missing (should be rare), skip
            continue

        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        _, _, bw, bh = bbox
        bbox_area = bw * bh
        img_area = img_w * img_h

        # 2) Remove if bbox is smaller than the fraction threshold
        if bbox_area < img_area * MIN_FRACTION_OF_IMAGE:
            continue

        # 3) Remove if category name is excluded
        cat_id = ann.get("category_id")
        cat_name = cat_id_to_name.get(cat_id, "").lower()
        if cat_name in excluded_names_set:
            continue

        filtered_annotations.append(ann)

    # -------- 4) Remove images with too many remaining boxes --------
    ann_count_by_image = defaultdict(int)
    for ann in filtered_annotations:
        ann_count_by_image[ann["image_id"]] += 1

    images_too_many = {
        img_id for img_id, count in ann_count_by_image.items()
        if count > MAX_BBOX_PER_IMAGE
    }

    final_annotations = [
        ann for ann in filtered_annotations
        if ann["image_id"] not in images_too_many
    ]

    # Keep only images that:
    #  - are not crowd images (already removed)
    #  - are not too-many-box images
    #  - and still have at least one remaining annotation
    remaining_image_ids = {ann["image_id"] for ann in final_annotations}

    final_images = [
        img for img in images_no_crowd_images
        if img["id"] in remaining_image_ids and img["id"] not in images_too_many
    ]

    return {
        "images": final_images,
        "annotations": final_annotations,
        "categories": categories,  # unchanged
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter COCO instances JSON by removing crowd images, "
                    "bbox size, excluded category names, and max bboxes per image."
    )
    parser.add_argument("input_json", type=Path, help="Path to original instances_*.json")
    parser.add_argument("output_json", type=Path, help="Path to write filtered JSON")
    args = parser.parse_args()

    print(f"[info] Loading: {args.input_json}")
    data = load_json(args.input_json)

    n_img_before = len(data.get("images", []))
    n_ann_before = len(data.get("annotations", []))

    print("[info] Filtering...")
    filtered = filter_coco(data)

    n_img_after = len(filtered.get("images", []))
    n_ann_after = len(filtered.get("annotations", []))

    # Useful stats: how many crowd images were removed?
    # (Compute again here for reporting only.)
    crowd_image_ids = {
        ann.get("image_id")
        for ann in data.get("annotations", [])
        if ann.get("iscrowd", 0) == 1 and "image_id" in ann
        # this is the point where you need to remove all the images with no bounding boxes
    }

    print(f"[stats] images:      {n_img_before} -> {n_img_after}")
    print(f"[stats] annotations: {n_ann_before} -> {n_ann_after}")
    print(f"[stats] crowd-images removed (had any iscrowd=1): {len(crowd_image_ids)}")

    print(f"[info] Saving filtered JSON to: {args.output_json}")
    save_json(filtered, args.output_json)
    print("[done] Finished.")


if __name__ == "__main__":
    main()
