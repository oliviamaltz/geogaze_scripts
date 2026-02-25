#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay each test image with its mask (N rows x 1 col), with borders, save as PNG.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---- EDIT THESE PATHS IF NEEDED ----
IMGS_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/v2/test_stimuli'
MASKS_DIR = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test_output_5'
OUT_DIR   = '/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test_output_5'
OUT_PNG   = 'all_overlays_grid_bordered.png'   # output filename

# ---- OVERLAY STYLE ----
OBJECT_IS_BLACK = True      # set False if your masks have white/positive object
OVERLAY_COLOR   = (255, 0, 0)  # RGB for overlay tint (red)
OVERLAY_ALPHA   = 0.35         # 0..1 transparency (higher = more visible)

os.makedirs(OUT_DIR, exist_ok=True)

def load_image_rgb(path):
    return Image.open(path).convert('RGB')

def load_mask_binary(path, object_is_black=True):
    """
    Return uint8 binary mask (0 or 1) as a numpy array.
    Set object_is_black=True if masks use black foreground on white background.
    """
    m = Image.open(path).convert('L')
    arr = np.asarray(m, dtype=np.uint8)
    if object_is_black:
        bin01 = (arr == 0).astype(np.uint8)  # black=object -> 1
    else:
        bin01 = (arr > 0).astype(np.uint8)   # nonzero=object -> 1
    return bin01

def style_axes_with_border(ax, spine_color='black', spine_width=1.5):
    """Hide ticks, keep frame, and style a visible border around the image."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(True)
    for side, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(spine_width)
        spine.set_edgecolor(spine_color)

def main():
    ids = list(range(1, 9))  # test1..test8

    # Preload valid pairs
    items = []
    for i in ids:
        img_path  = os.path.join(IMGS_DIR,  f'test{i}.png')
        mask_path = os.path.join(MASKS_DIR, f'test{i}_mask.png')
        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            print(f'[WARN] Missing one of: {img_path} or {mask_path} — skipping this index')
            continue

        I = load_image_rgb(img_path)
        M = load_mask_binary(mask_path, object_is_black=OBJECT_IS_BLACK)

        # resize mask to image size (if needed)
        if M.shape[::-1] != I.size:
            M = np.array(Image.fromarray(M*255, mode='L').resize(I.size, resample=Image.NEAREST)) // 255

        items.append((f'test{i}.png', I, M))

    if not items:
        raise SystemExit('No valid (image, mask) pairs found.')

    n = len(items)
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(6, 3*n))  # ~3" height per row
    if n == 1:
        axs = np.array([axs])

    # Normalize overlay color to [0,1]
    overlay_rgb = np.array(OVERLAY_COLOR, dtype=np.float32) / 255.0

    for r, (name, I_pil, M_bin) in enumerate(items):
        ax = axs[r, 0] if axs.ndim == 2 else axs[r]
        # Base image
        ax.imshow(I_pil)
        # Overlay: show a solid color where mask==1, with alpha
        # We pass a full-color image and a per-pixel alpha mask
        H, W = M_bin.shape
        color_img = np.tile(overlay_rgb.reshape(1, 1, 3), (H, W, 1))
        ax.imshow(color_img, alpha=(M_bin * OVERLAY_ALPHA))

        style_axes_with_border(ax)
        if r == 0:
            ax.set_title('Image + Mask Overlay', pad=6, fontsize=11)
        # Row label on the left margin
        ax.text(-0.02, 0.5, name, transform=ax.transAxes,
                va='center', ha='right', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_PNG)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'[OK] Saved overlay grid to: {out_path}')
    plt.close(fig)

if __name__ == '__main__':
    main()
