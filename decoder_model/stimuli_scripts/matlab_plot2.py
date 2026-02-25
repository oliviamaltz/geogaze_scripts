#!/usr/bin/env python3
"""
random_16_grid.py
Display a random sample of 16 images from a folder in a 4x4 grid.

Usage:
  python random_16_grid.py /path/to/folder
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random
import math
import sys

def show_random_grid(img_dir: Path, N=16, cols=4, save_prefix="random_grid"):
    # Collect all image files
    exts = {".png", ".jpg", ".jpeg"}
    all_images = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    if len(all_images) == 0:
        print(f"❌ No images found in {img_dir}")
        return

    # Randomly pick N images (without replacement)
    random.shuffle(all_images)
    files_to_show = all_images[:min(N, len(all_images))]

    rows = math.ceil(len(files_to_show) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.ravel()

    for ax, img_path in zip(axes, files_to_show):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=7)
        ax.axis("off")

        # Add border around each image
        border = patches.Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=2
        )
        ax.add_patch(border)

    # Hide unused axes
    for ax in axes[len(files_to_show):]:
        ax.axis("off")

    plt.tight_layout(pad=0.4)

    # Save outputs
    png_out = img_dir / f"{save_prefix}.png"
    pdf_out = img_dir / f"{save_prefix}.pdf"
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.show()

    print(f"✅ Saved grid to:\n   {png_out}\n   {pdf_out}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python random_16_grid.py /path/to/folder")
        sys.exit(1)

    img_dir = Path(sys.argv[1]).expanduser().resolve()
    show_random_grid(img_dir)
