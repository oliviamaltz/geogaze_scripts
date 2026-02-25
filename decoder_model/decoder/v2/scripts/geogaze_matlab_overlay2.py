#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay each test image with its predicted mask; save a grid PNG.
Matches masks named: <image_stem>_maskL_bcgc_md.png
"""

import os, glob
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---- EDIT THESE PATHS IF NEEDED ----
IMGS_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/v2/test_stimuli/test_pairs'  # where your test PNGs live
MASKS_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/v2/maskL_bc_gc_model/model_masks'  # where model saved masks
OUT_DIR   = MASKS_DIR
OUT_PNG   = 'all_overlays_grid_bordered.png'

# ---- OVERLAY STYLE ----
OBJECT_IS_BLACK = False         # your saved masks have white foreground (255) on black (0)
OVERLAY_COLOR   = (255, 0, 0)   # red tint
OVERLAY_ALPHA   = 0.35          # 0..1

os.makedirs(OUT_DIR, exist_ok=True)

def load_image_rgb(path):
    return Image.open(path).convert('RGB')

def load_mask_binary(path, object_is_black=False):
    """
    Return uint8 binary mask (0 or 1).
    If object_is_black=True, treat black (0) as object; else any >0 as object.
    """
    m = Image.open(path).convert('L')
    arr = np.asarray(m, dtype=np.uint8)
    if object_is_black:
        bin01 = (arr == 0).astype(np.uint8)
    else:
        bin01 = (arr > 0).astype(np.uint8)
    return bin01

def style_axes_with_border(ax, spine_color='black', spine_width=1.5):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(True)
    for side, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(spine_width)
        spine.set_edgecolor(spine_color)

def main():
    # gather image files
    img_paths = sorted(glob.glob(os.path.join(IMGS_DIR, '*.png')))
    if not img_paths:
        raise SystemExit(f'No PNGs found in {IMGS_DIR}')

    # build (image, mask) pairs using the new mask naming
    items = []
    for img_path in img_paths:
        stem = Path(img_path).stem
        mask_path = str(Path(MASKS_DIR) / f'{stem}_maskL_bcgc_md.png')
        if not os.path.isfile(mask_path):
            print(f'[WARN] Missing mask for {stem}: {mask_path} — skipping')
            continue

        I = load_image_rgb(img_path)
        M = load_mask_binary(mask_path, object_is_black=OBJECT_IS_BLACK)

        # resize mask to image size if needed
        if M.shape[::-1] != I.size:
            M = np.array(Image.fromarray(M * 255, mode='L').resize(I.size, resample=Image.NEAREST)) // 255

        items.append((Path(img_path).name, I, M))

    if not items:
        raise SystemExit('No valid (image, mask) pairs found.')

    n = len(items)
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(6, 3*n))
    if n == 1:
        axs = np.array([axs])

    overlay_rgb = np.array(OVERLAY_COLOR, dtype=np.float32) / 255.0

    for r, (name, I_pil, M_bin) in enumerate(items):
        ax = axs[r, 0] if axs.ndim == 2 else axs[r]
        ax.imshow(I_pil)
        H, W = M_bin.shape
        color_img = np.tile(overlay_rgb.reshape(1, 1, 3), (H, W, 1))
        ax.imshow(color_img, alpha=(M_bin * OVERLAY_ALPHA))
        style_axes_with_border(ax)
        if r == 0:
            ax.set_title('Image + Mask Overlay', pad=6, fontsize=11)
        ax.text(-0.02, 0.5, name, transform=ax.transAxes, va='center', ha='right', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_PNG)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'[OK] Saved overlay grid to: {out_path}')
    plt.close(fig)

if __name__ == '__main__':
    main()
