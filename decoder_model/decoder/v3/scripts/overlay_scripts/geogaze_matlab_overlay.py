#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay each test image with its predicted mask; save a grid PNG.

Matching rule (flexible):
For each image like  test_bc_bc_LR.png  in IMGS_DIR,
find a mask PNG in MASKS_DIR whose filename STARTS WITH the image stem,
e.g., test_bc_bc_LR__model=..._mask.png  (any extra suffix is OK).
If multiple matches exist, prefer one that contains "_mask" then the shortest name.
"""

import os, glob
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---- DEFAULT PATHS (can be overridden via CLI) ----
DEFAULT_IMGS_DIR  = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test_stimuli/test_pairs"
DEFAULT_MASKS_DIR = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/v3/resnet/test_masks/resnet50.a1_in1k_maskL_bc_bs_best"

# ---- OVERLAY STYLE ----
OVERLAY_COLOR   = (255, 0, 0)  # RGB for overlay tint
OVERLAY_ALPHA   = 0.35         # 0..1


def parse_args():
    p = argparse.ArgumentParser("Overlay test images with predicted masks.")
    p.add_argument("--imgs_dir",  type=str, default=DEFAULT_IMGS_DIR,
                   help="Directory with test images (test_*.png).")
    p.add_argument("--masks_dir", type=str, default=DEFAULT_MASKS_DIR,
                   help="Directory with mask PNGs for one model.")
    p.add_argument("--out_dir",   type=str, default=None,
                   help="Where to save the overlay PNG (default = masks_dir).")
    p.add_argument("--out_png",   type=str, default=None,
                   help="Output filename (default = <masks_dir_name>_overlays.png).")
    p.add_argument("--object_is_black", action="store_true",
                   help="If set, treat black pixels as object (arr == 0).")
    return p.parse_args()


def load_image_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask_binary(path, object_is_black=False):
    """
    Return uint8 binary mask (0 or 1).
    If object_is_black=True, treat black (0) as object; else any >0 as object.
    """
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    if object_is_black:
        bin01 = (arr == 0).astype(np.uint8)
    else:
        bin01 = (arr > 0).astype(np.uint8)
    return bin01


def style_axes_with_border(ax, spine_color="black", spine_width=1.5):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(spine_width)
        spine.set_edgecolor(spine_color)


def find_mask_for_stem(stem, masks_dir):
    """
    Find a mask png in masks_dir whose filename starts with <stem>.
    Accepts names like: <stem>__model=..._mask.png, <stem>_anything.png, etc.
    Selection policy if multiple:
      1) Prefer names containing '_mask'
      2) Then prefer the shortest filename
      3) Fall back to lexicographic first
    """
    patterns = [
        f"{stem}__*.png",
        f"{stem}_*.png",
        f"{stem}*.png",   # broad catch-all
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(str(Path(masks_dir) / pat)))

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        name = Path(p).name
        if p not in seen and name != f"{stem}.png":  # exclude bare image name
            seen.add(p)
            uniq.append(p)

    if not uniq:
        return None

    # ranking: contains '_mask' first, then shortest name, then lexicographic
    def rank_key(p):
        name = Path(p).name
        has_mask = "_mask" in name.lower()
        return (0 if has_mask else 1, len(name), name)

    uniq.sort(key=rank_key)
    return uniq[0]


def main():
    args = parse_args()

    imgs_dir = args.imgs_dir
    masks_dir = args.masks_dir
    out_dir = args.out_dir or masks_dir
    out_png = args.out_png or (Path(masks_dir).name + "_overlays.png")
    object_is_black = args.object_is_black

    os.makedirs(out_dir, exist_ok=True)

    # gather image files
    img_paths = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))
    if not img_paths:
        raise SystemExit(f"No PNGs found in {imgs_dir}")

    # build (image, mask) pairs using flexible prefix matching
    items = []
    for img_path in img_paths:
        stem = Path(img_path).stem  # e.g., 'test_bc_bc_LR'
        mask_path = find_mask_for_stem(stem, masks_dir)
        if mask_path is None or not os.path.isfile(mask_path):
            print(f"[WARN] Missing mask for {stem} in {masks_dir} — skipping")
            continue

        I = load_image_rgb(img_path)
        M = load_mask_binary(mask_path, object_is_black=object_is_black)

        # resize mask to image size if needed
        if M.shape[::-1] != I.size:
            M = np.array(
                Image.fromarray(M * 255, mode="L").resize(I.size, resample=Image.NEAREST)
            ) // 255

        items.append((Path(img_path).name, I, M, Path(mask_path).name))

    if not items:
        raise SystemExit("No valid (image, mask) pairs found.")

    n = len(items)
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(6, 3 * n))
    if n == 1:
        axs = np.array([axs])

    overlay_rgb = np.array(OVERLAY_COLOR, dtype=np.float32) / 255.0

    for r, (img_name, I_pil, M_bin, mask_name) in enumerate(items):
        ax = axs[r, 0] if axs.ndim == 2 else axs[r]
        ax.imshow(I_pil)
        H, W = M_bin.shape
        color_img = np.tile(overlay_rgb.reshape(1, 1, 3), (H, W, 1))
        ax.imshow(color_img, alpha=(M_bin * OVERLAY_ALPHA))
        style_axes_with_border(ax)
        if r == 0:
            ax.set_title("Image + Mask Overlay", pad=6, fontsize=11)
        # left margin label = image name; small right note = chosen mask name
        ax.text(-0.02, 0.5, img_name, transform=ax.transAxes,
                va="center", ha="right", fontsize=9)
        ax.text(1.002, 0.02, mask_name, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, out_png)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved overlay grid to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
