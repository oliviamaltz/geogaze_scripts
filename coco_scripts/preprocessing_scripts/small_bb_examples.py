#!/usr/bin/env python3

import json
import random
import argparse
from pathlib import Path

MIN_FRACTION_OF_IMAGE = 1.0 / 16.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="List random examples of annotations removed due to small bbox area"
    )
    parser.add_argument("input_json", type=Path, help="Original COCO instances JSON")
    parser.add_argument("--n", type=int, default=10, help="Number of examples to show")
    args = parser.parse_args()

    data = load_json(args.input_json)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    images_by_id = {img["id"]: img for img in images}
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    small_bbox_examples = []

    for ann in annotations:
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")

        if img_id is None or not bbox or len(bbox) != 4:
            continue

        img = images_by_id.get(img_id)
        if img is None:
            continue

        img_w = img.get("width")
        img_h = img.get("height")
        if not img_w or not img_h:
            continue

        _, _, bw, bh = bbox
        bbox_area = bw * bh
        img_area = img_w * img_h
        frac = bbox_area / img_area

        if frac < MIN_FRACTION_OF_IMAGE:
            small_bbox_examples.append({
                "annotation_id": ann.get("id"),
                "image_id": img_id,
                "file_name": img.get("file_name"),
                "category": cat_id_to_name.get(ann.get("category_id"), "UNKNOWN"),
                "bbox": bbox,
                "bbox_fraction": frac
            })

    if not small_bbox_examples:
        print("No small-bbox annotations found.")
        return

    sample = random.sample(
        small_bbox_examples,
        min(args.n, len(small_bbox_examples))
    )

    print(f"\nShowing {len(sample)} random annotations removed due to small bbox:\n")

    for ex in sample:
        print(
            f"- ann_id={ex['annotation_id']}, "
            f"img_id={ex['image_id']}, "
            f"file={ex['file_name']}, "
            f"cat={ex['category']}, "
            f"bbox={ex['bbox']}, "
            f"frac={ex['bbox_fraction']:.5f}"
        )


if __name__ == "__main__":
    main()
