#!/usr/bin/env python3
"""
Downsample images with EXACTLY ONE bounding box.

Outputs:
  1) New bounding-box CSV with all boxes for dropped images removed
  2) New txt image list with dropped images removed
  3) (Optional) txt file listing dropped ImageIDs
"""

import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd


def load_txt(ids_txt: Path):
    """
    Returns:
      raw_lines: original txt lines (e.g., 'train/000abc...')
      image_ids: extracted ImageIDs
      has_prefix: whether lines include 'split/' prefix
    """
    raw_lines = []
    image_ids = []
    has_prefix = None

    with ids_txt.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            raw_lines.append(s)
            parts = s.split("/")
            if len(parts) > 1:
                image_ids.append(parts[-1])
                has_prefix = True
            else:
                image_ids.append(s)
                has_prefix = False

    return raw_lines, image_ids, bool(has_prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-csv", required=True, type=Path)
    ap.add_argument("--ids-txt", required=True, type=Path)

    ap.add_argument("--out-bbox-csv", required=True, type=Path)
    ap.add_argument("--out-ids-txt", required=True, type=Path)

    ap.add_argument("--drop-n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--image-id-col", default="ImageID")

    ap.add_argument("--dropped-ids-out", type=Path, default=None)

    args = ap.parse_args()

    # -------------------------
    # Load txt list
    # -------------------------
    raw_lines, image_ids, has_prefix = load_txt(args.ids_txt)
    valid_ids = set(image_ids)

    print(f"[info] Images in txt file: {len(valid_ids):,}")

    # -------------------------
    # PASS 1: count boxes per image
    # -------------------------
    counts = Counter()

    for chunk in pd.read_csv(args.bbox_csv, chunksize=args.chunksize):
        if args.image_id_col not in chunk.columns:
            raise SystemExit(f"CSV missing column '{args.image_id_col}'")

        chunk = chunk[chunk[args.image_id_col].isin(valid_ids)]
        if chunk.empty:
            continue

        counts.update(chunk[args.image_id_col])

    one_box_ids = [img for img, c in counts.items() if c == 1]
    print(f"[info] Images with exactly 1 bbox: {len(one_box_ids):,}")

    if len(one_box_ids) == 0:
        raise SystemExit("No 1-box images found.")

    drop_n = min(args.drop_n, len(one_box_ids))
    rng = np.random.default_rng(args.seed)
    drop_ids = set(rng.choice(one_box_ids, size=drop_n, replace=False))

    print(f"[info] Dropping {len(drop_ids):,} 1-box images")

    # -------------------------
    # PASS 2: write new bbox CSV
    # -------------------------
    args.out_bbox_csv.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    kept_rows = 0
    dropped_rows = 0

    for chunk in pd.read_csv(args.bbox_csv, chunksize=args.chunksize):
        chunk = chunk[chunk[args.image_id_col].isin(valid_ids)]
        if chunk.empty:
            continue

        mask_drop = chunk[args.image_id_col].isin(drop_ids)
        dropped_rows += int(mask_drop.sum())

        kept = chunk[~mask_drop]
        kept_rows += len(kept)

        if not kept.empty:
            kept.to_csv(
                args.out_bbox_csv,
                index=False,
                mode="w" if not wrote_header else "a",
                header=not wrote_header
            )
            wrote_header = True

    print(f"[done] New bbox CSV written: {args.out_bbox_csv}")
    print(f"[done] Kept bbox rows: {kept_rows:,}")
    print(f"[done] Dropped bbox rows: {dropped_rows:,}")

    # -------------------------
    # Write new txt list
    # -------------------------
    args.out_ids_txt.parent.mkdir(parents=True, exist_ok=True)

    with args.out_ids_txt.open("w") as f:
        for line, img_id in zip(raw_lines, image_ids):
            if img_id not in drop_ids:
                f.write(line + "\n")

    print(f"[done] New txt file written: {args.out_ids_txt}")

    # -------------------------
    # Optional: save dropped IDs
    # -------------------------
    if args.dropped_ids_out is not None:
        args.dropped_ids_out.parent.mkdir(parents=True, exist_ok=True)
        with args.dropped_ids_out.open("w") as f:
            for img_id in sorted(drop_ids):
                f.write(img_id + "\n")
        print(f"[done] Dropped ImageIDs saved to: {args.dropped_ids_out}")


if __name__ == "__main__":
    main()
