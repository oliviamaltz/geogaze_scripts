#!/usr/bin/env python3
"""
Quick sanity check for pair stimuli + masks.

- Computes foreground coverage (percent of image that is mask) per MID.
- Saves a contact sheet of a few random (image, mask) pairs.

Usage example:

  python check_pairs_and_masks.py \
      --pair_mids gs_bs,gs_gc,bs_gs \
      --mask_side L \
      --num_samples 12 \
      --out_png debug_L_gs_pairs.png
"""

import os
import re
import argparse
import random
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---- EDIT THESE IF NEEDED (match your training script) ----
PAIRS_DIR_BASE = "/zpool/vladlab/data_drive/geogaze_data/pairs"
LEFT_MASKS_DIR = "/zpool/vladlab/data_drive/geogaze_data/left_masks"
RIGHT_MASKS_DIR = "/zpool/vladlab/data_drive/geogaze_data/right_masks"


def parse_args():
    p = argparse.ArgumentParser("Check pair+mask stimuli.")
    p.add_argument("--pair_mids", type=str, required=True,
                   help="CSV of MIDs to include, e.g. 'gs_bs,gs_gc,bs_gs'")
    p.add_argument("--mask_side", type=str, choices=["L", "R"], default="L")
    p.add_argument("--num_samples", type=int, default=12,
                   help="Number of random pairs to visualize")
    p.add_argument("--out_png", type=str, default="debug_pairs_masks.png",
                   help="Output PNG for contact sheet")
    return p.parse_args()


def get_mask_dir(mask_side: str) -> str:
    return LEFT_MASKS_DIR if mask_side == "L" else RIGHT_MASKS_DIR


def collect_items(pair_mids, mask_side):
    """Return list of (img_path, mask_path, mid, id_)."""
    masks_dir = get_mask_dir(mask_side)

    allowed_mids = [m.strip() for m in pair_mids.split(",") if m.strip()]
    mid_union = "|".join(re.escape(m) for m in allowed_mids)
    pair_re = re.compile(rf"^pair_({mid_union})_(\d+)\.png$")

    items = []
    skipped = 0

    for fn in os.listdir(PAIRS_DIR_BASE):
        m = pair_re.match(fn)
        if not m:
            continue
        mid, id_ = m.group(1), m.group(2)
        img_p = os.path.join(PAIRS_DIR_BASE, fn)
        mask_name = f"mask{mask_side}_{mid}_{id_}.png"
        msk_p = os.path.join(masks_dir, mask_name)
        if not os.path.isfile(msk_p):
            skipped += 1
            continue
        items.append((img_p, msk_p, mid, id_))

    items.sort(key=lambda x: (x[2], int(x[3])))
    print(f"[INFO] Found {len(items)} (image, mask{mask_side}) pairs across MIDs: {allowed_mids}")
    if skipped:
        print(f"[WARN] Skipped {skipped} image(s) with no matching mask{mask_side}.")
    return items


def compute_mask_stats(items):
    """
    Compute per-MID stats: mean/min/max foreground % and #zero masks.
    Foreground is pixels == 0 (black), matching your training code.
    """
    stats = defaultdict(list)

    for _, msk_p, mid, _ in items:
        m = np.asarray(Image.open(msk_p).convert("L"), dtype=np.uint8)
        total = m.size
        # In your training code, foreground = (m == 0)
        fg = np.sum(m == 0)
        fg_pct = 100.0 * fg / max(1, total)
        stats[mid].append((fg_pct, fg))

    print("\n=== Mask coverage stats (foreground == black pixels) ===")
    for mid, vals in stats.items():
        pcts = [v[0] for v in vals]
        fgs = [v[1] for v in vals]
        zero_masks = sum(1 for f in fgs if f == 0)
        print(f"MID {mid}:")
        print(f"  n          = {len(vals)}")
        print(f"  mean fg %  = {np.mean(pcts):.3f}")
        print(f"  min fg %   = {np.min(pcts):.3f}")
        print(f"  max fg %   = {np.max(pcts):.3f}")
        print(f"  #zero masks (no black pixels) = {zero_masks}")
    print("========================================================\n")


def make_contact_sheet(items, num_samples, out_png):
    if len(items) == 0:
        print("[WARN] No items to visualize.")
        return

    num_samples = min(num_samples, len(items))
    subset = random.sample(items, num_samples)

    # 2 columns: left = image, right = mask
    ncols = 2
    nrows = num_samples

    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 3 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)  # make it 2D for consistency

    for row_idx, (img_p, msk_p, mid, id_) in enumerate(subset):
        img = Image.open(img_p).convert("RGB")
        msk = Image.open(msk_p).convert("L")

        ax_img = axes[row_idx, 0]
        ax_msk = axes[row_idx, 1]

        ax_img.imshow(img)
        ax_img.set_title(f"pair_{mid}_{id_}.png")
        ax_img.axis("off")

        ax_msk.imshow(msk, cmap="gray")
        ax_msk.set_title(f"mask_{mid}_{id_}.png")
        ax_msk.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved contact sheet to: {out_png}")


def main():
    args = parse_args()
    items = collect_items(args.pair_mids, args.mask_side)
    if not items:
        print("[ERROR] No (image, mask) pairs found for given arguments.")
        return
    compute_mask_stats(items)
    make_contact_sheet(items, args.num_samples, args.out_png)


if __name__ == "__main__":
    main()
