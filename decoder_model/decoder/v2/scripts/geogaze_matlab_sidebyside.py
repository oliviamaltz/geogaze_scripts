#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show all test images + masks in one figure (8 rows x 2 cols) with borders, save as PNG.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---- EDIT THESE PATHS IF NEEDED ----
IMGS_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test stimuli'
MASKS_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test_outputs_2'
OUT_DIR   = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test_outputs_2'
OUT_PNG   = 'all_pairs_grid_bordered.png'   # output filename

os.makedirs(OUT_DIR, exist_ok=True)

def load_image_rgb(path):
    return Image.open(path).convert('RGB')

def load_mask_binary(path, object_is_black=True):
    """
    Load mask and return uint8 0/255 grayscale for display.
    Set object_is_black=True if your masks have black foreground on white background.
    """
    m = Image.open(path).convert('L')
    arr = np.asarray(m, dtype=np.uint8)
    if object_is_black:
        bin01 = (arr == 0).astype(np.uint8)  # black=object -> 1
    else:
        bin01 = (arr > 0).astype(np.uint8)   # nonzero=object -> 1
    return Image.fromarray(bin01 * 255, mode='L')

def style_axes_with_border(ax, spine_color='black', spine_width=1.5):
    """Hide ticks, keep frame, and style a visible border around the image."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    for side, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(spine_width)
        spine.set_edgecolor(spine_color)

def main():
    ids = list(range(1, 9))  # test1..test8

    # Preload all pairs
    pairs = []
    for i in ids:
        img_path  = os.path.join(IMGS_DIR,  f'test{i}.png')
        mask_path = os.path.join(MASKS_DIR, f'test{i}_mask.png')
        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            print(f'[WARN] Missing one of: {img_path} or {mask_path} — skipping this index')
            continue
        I = load_image_rgb(img_path)
        M = load_mask_binary(mask_path, object_is_black=True)  # flip to False if needed

        # resize mask to image size (if needed)
        if M.size != I.size:
            M = M.resize(I.size, resample=Image.NEAREST)

        pairs.append((f'test{i}.png', I, M))

    if not pairs:
        raise SystemExit('No valid (image, mask) pairs found.')

    # Build a single figure grid: rows = N, cols = 2 (image | mask)
    n = len(pairs)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(8, 3*n))  # ~3 inches per row
    if n == 1:
        axs = np.array([axs])  # normalize shape for single row

    for r, (name, I, M) in enumerate(pairs):
        axs[r, 0].imshow(I)
        style_axes_with_border(axs[r, 0])
        if r == 0:
            axs[r, 0].set_title('Image', pad=6, fontsize=10)
        axs[r, 0].text(-0.02, 0.5, name, transform=axs[r, 0].transAxes,
                       va='center', ha='right', fontsize=9)

        axs[r, 1].imshow(M, cmap='gray', vmin=0, vmax=255)
        style_axes_with_border(axs[r, 1])
        if r == 0:
            axs[r, 1].set_title('Mask', pad=6, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_PNG)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'[OK] Saved grid to: {out_path}')
    plt.close(fig)

if __name__ == '__main__':
    main()
