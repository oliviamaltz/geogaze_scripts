#!/usr/bin/env python3
"""
List all unique object names present in Open Images bounding-box annotations.

Usage examples:
  python list_openimages_bbox_labels.py \
    --ann /path/to/train-annotations-bbox.filtered.csv \
    --classmap /path/to/class-descriptions-boxable.csv

  # Multiple CSVs at once (train/val/test)
  python list_openimages_bbox_labels.py \
    --ann /path/to/train-annotations-bbox.filtered.csv \
    --ann /path/to/validation-annotations-bbox.filtered.csv \
    --ann /path/to/test-annotations-bbox.filtered.csv \
    --classmap /path/to/class-descriptions-boxable.csv \
    --save /path/to/unique_labels.txt
"""

import argparse
from pathlib import Path
import pandas as pd


def load_classmap(classmap_path: Path) -> dict:
    """
    Open Images class descriptions file is typically 2 columns:
      LabelName, DisplayName
    No header.
    """
    df = pd.read_csv(classmap_path, header=None, names=["LabelName", "DisplayName"])
    df["LabelName"] = df["LabelName"].astype(str)
    df["DisplayName"] = df["DisplayName"].astype(str)
    return dict(zip(df["LabelName"], df["DisplayName"]))


def list_labels(annotation_paths: list[Path], classmap: dict) -> list[str]:
    all_label_ids = set()

    for p in annotation_paths:
        if not p.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {p}")

        df = pd.read_csv(p)

        # Common Open Images bbox column name:
        if "LabelName" not in df.columns:
            raise ValueError(
                f"{p} does not have a 'LabelName' column. "
                f"Found columns: {list(df.columns)[:30]}"
            )

        all_label_ids.update(df["LabelName"].dropna().astype(str).unique().tolist())

    # Map to human-readable names, skipping unknown IDs
    names = sorted({classmap[lid] for lid in all_label_ids if lid in classmap})
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ann",
        action="append",
        required=True,
        help="Path to an Open Images bbox annotations CSV. Can be provided multiple times.",
    )
    ap.add_argument(
        "--classmap",
        required=True,
        help="Path to class-descriptions-boxable.csv (LabelName,DisplayName).",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="Optional path to save the unique labels (one per line).",
    )
    args = ap.parse_args()

    ann_paths = [Path(x) for x in args.ann]
    classmap_path = Path(args.classmap)

    if not classmap_path.exists():
        raise FileNotFoundError(f"Classmap file not found: {classmap_path}")

    classmap = load_classmap(classmap_path)
    names = list_labels(ann_paths, classmap)

    print(f"Unique box labels found: {len(names)}\n")
    for n in names:
        print(n)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
