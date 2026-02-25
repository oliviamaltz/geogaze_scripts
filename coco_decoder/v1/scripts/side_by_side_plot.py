#!/usr/bin/env python3
"""
Hard-coded image comparison script.

What it does:
- Takes a LIST of image name keys (e.g., "cat", "dog")
- Looks for matching images in TWO folders
- Displays them side-by-side (Folder A | Folder B)
- Optionally adds extra panels (diff / absdiff)
- Saves figures to an output directory
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# ===================== HARD-CODE HERE =======================
# ============================================================

# Folder A (left images)
FOLDER_A = Path("/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/identification/maskL_bc_gc_model_overlay_pos10_epoch200")

# Folder B (right images)
FOLDER_B = Path("/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/individuation/maskL_bc_gc_model_overlay_pos10_epoch200")

# Output folder for saved figures
OUTPUT_DIR = Path("/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder")

# List of image *keys* to plot (first part of filename)
# Example filenames it will match:
IMAGE_KEYS = [
    "test_bc_gc_LR",
    "test_bc_bc_LR",
    "test_gc_gc_LR",
    "test_bc_gc_UD",
    "test_gc_bc_UD",
    "test_gc_bc_LR"
]

# Panels shown left → right
# Options: "a", "b", "diff", "absdiff"
PANELS = ["identification", "individuation"]

OUTPUT_NAME = "maskL_bc_gc_pos10_epoch200.png"

# ============================================================
# ============================================================

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image(folder: Path, key: str) -> Path | None:
    """Find the first image whose filename starts with key."""
    for ext in IMG_EXTS:
        matches = list(folder.glob(f"{key}*{ext}"))
        if matches:
            return sorted(matches)[0]
    return None


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_to_match(a: np.ndarray, b: np.ndarray):
    if a.shape == b.shape:
        return a, b
    h, w = a.shape[:2]
    b_resized = Image.fromarray(b).resize((w, h), Image.BILINEAR)
    return a, np.asarray(b_resized)


def add_subplot_border(ax, color="black", lw=1.5):
    """Draw a border around a matplotlib subplot."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(lw)


def main():
    pairs = []

    for key in IMAGE_KEYS:
        img_a = find_image(FOLDER_A, key)
        img_b = find_image(FOLDER_B, key)

        if img_a is None or img_b is None:
            print(f"[skip] Missing image for key '{key}'")
            continue

        pairs.append((key, img_a, img_b))

    if not pairs:
        raise RuntimeError("No matching image pairs found.")

    nrows = len(pairs)
    ncols = len(PANELS)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 2.8 * nrows),
    )

    if nrows == 1:
        axes = [axes]

    fig.suptitle("Image comparison", fontsize=16)

    for r, (key, path_a, path_b) in enumerate(pairs):
        img_a = load_rgb(path_a)
        img_b = load_rgb(path_b)
        img_a, img_b = resize_to_match(img_a, img_b)

        a_f = img_a.astype(np.float32) / 255.0
        b_f = img_b.astype(np.float32) / 255.0
        diff = a_f - b_f
        absdiff = np.abs(diff)

        for c, panel in enumerate(PANELS):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])

            if panel in ("a", "identification"):
                ax.imshow(img_a)
                ax.set_title(f"identification: {path_a.name}", fontsize=9)

            elif panel in ("b", "individuation"):
                ax.imshow(img_b)
                ax.set_title(f"individuation: {path_b.name}", fontsize=9)

            elif panel == "diff":
                ax.imshow(np.clip(diff * 0.5 + 0.5, 0, 1))
                ax.set_title("diff (A − B)", fontsize=9)

            elif panel == "absdiff":
                ax.imshow(np.clip(absdiff, 0, 1))
                ax.set_title("|A − B|", fontsize=9)

            else:
                ax.text(0.5, 0.5, f"Unknown panel: {panel}", ha="center", va="center")
                ax.set_title(panel, fontsize=9)


            # 🔲 ADD BORDER
            add_subplot_border(ax)

        # Row label
        axes[r][0].set_ylabel(key, rotation=0, labelpad=40, fontsize=11, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_NAME
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
