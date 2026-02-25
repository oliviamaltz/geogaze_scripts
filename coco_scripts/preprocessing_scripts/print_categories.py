#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

def main(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build category_id -> name mapping
    categories = {
        cat["id"]: cat["name"]
        for cat in data.get("categories", [])
        if "id" in cat and "name" in cat
    }

    # Collect category_ids that appear in annotations
    annotation_category_ids = {
        ann["category_id"]
        for ann in data.get("annotations", [])
        if "category_id" in ann
    }

    # Get unique category names that actually exist
    present_category_names = sorted({
        categories[cat_id]
        for cat_id in annotation_category_ids
        if cat_id in categories
    })

    # Print results
    print("Categories present in annotations:")
    for name in present_category_names:
        print(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print unique category names appearing in COCO annotations"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to COCO-style JSON annotation file"
    )

    args = parser.parse_args()
    main(args.json_file)
