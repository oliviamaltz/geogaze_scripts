#!/usr/bin/env python3
"""
CSV-driven IDEN vs IDIV overlay comparison.

For each unique `model` in a CSV:
  - Take the 6 (or N) filenames in the `prediction` column (e.g., *_mask.png)
  - Convert each to an overlay filename by replacing "_mask" -> "_overlay"
  - Find that overlay image inside:
        identification/<model>/<model>_overlays/
    and the matching image inside:
        individuation/<model_with_IDEN_replaced_by_IDIV>/<that_model>_overlays/
  - Make ONE figure per model, with rows = images, cols = [IDEN | IDIV]
  - Save all figures to an output folder.

Usage:
  python compare_iden_idiv_from_csv.py /path/to/results.csv \
      --root /zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder \
      --out /zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/comparisons

Notes:
- Assumes the CSV `model` values contain "IDEN" somewhere that can be replaced with "IDIV".
- If your overlay filenames use a different suffix, change MASK_TOKEN / OVERLAY_TOKEN below.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Customize if needed
# ----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MASK_TOKEN = "_mask"
OVERLAY_TOKEN = "_overlay"

# These are subfolders under --root
IDEN_TOP = "identification"
IDIV_TOP = "individuation"


# ----------------------------
# Helpers
# ----------------------------
def find_image_by_prefix(folder: Path, prefix: str) -> Optional[Path]:
    """
    Find the first file in `folder` whose name starts with `prefix` and has a supported extension.
    Returns the alphabetically first match, or None if not found.
    """
    if not folder.exists():
        return None

    for ext in IMG_EXTS:
        matches = list(folder.glob(f"{prefix}*{ext}"))
        if matches:
            return sorted(matches)[0]
    return None


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h, w = a.shape[:2]
    b_resized = Image.fromarray(b).resize((w, h), Image.BILINEAR)
    return a, np.asarray(b_resized)


def add_subplot_border(ax, color="black", lw=1.5):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(lw)


def model_iden_to_idiv(model: str) -> str:
    """
    Map IDEN model folder name -> IDIV model folder name.
    Example: cornetIDEN_maskL_bc_bs -> cornetIDIV_maskL_bc_bs
    """
    if "IDEN" not in model:
        # still return something deterministic; user can fix if needed
        return model.replace("iden", "idiv").replace("IDen", "IDiv")
    return model.replace("IDEN", "IDIV")


def prediction_to_overlay_prefix(prediction_name: str) -> str:
    """
    Convert CSV prediction filename (often ends with _mask.png) to overlay *prefix*.

    Example:
      test_bc_bs_LR_mask.png -> test_bc_bs_LR_overlay   (prefix; extension removed)
    """
    p = Path(prediction_name).name  # drop any accidental paths
    stem = Path(p).stem             # remove extension
    # Replace only the token part, not the extension
    overlay_stem = stem.replace(MASK_TOKEN, OVERLAY_TOKEN)
    return overlay_stem


def read_csv_grouped(csv_path: Path) -> Dict[str, List[str]]:
    """
    Read CSV and return dict: model -> list of prediction filenames (as in CSV).
    Requires columns: 'model' and 'prediction'
    """
    groups: Dict[str, List[str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("model", "prediction") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            model = (row.get("model") or "").strip()
            pred = (row.get("prediction") or "").strip()
            if not model or not pred:
                continue
            groups.setdefault(model, []).append(pred)

    return groups


def build_overlay_folder(root: Path, top: str, model: str) -> Path:
    """
    Build:
      root/top/model/model_overlays
    """
    return root / top / model / f"{model}_overlays"


# ----------------------------
# Main per-model plotting
# ----------------------------
def plot_one_model(
    model: str,
    preds: List[str],
    root: Path,
    out_dir: Path,
    show_titles: bool = True,
    dpi: int = 250,
) -> Path:
    idiv_model = model_iden_to_idiv(model)

    iden_overlay_dir = build_overlay_folder(root, IDEN_TOP, model)
    idiv_overlay_dir = build_overlay_folder(root, IDIV_TOP, idiv_model)

    # Build list of (row_label, iden_path, idiv_path)
    rows: List[Tuple[str, Optional[Path], Optional[Path]]] = []
    for pred in preds:
        overlay_prefix = prediction_to_overlay_prefix(pred)  # no extension
        # We search by prefix so it can match .png/.jpg etc
        iden_path = find_image_by_prefix(iden_overlay_dir, overlay_prefix)
        idiv_path = find_image_by_prefix(idiv_overlay_dir, overlay_prefix)

        rows.append((overlay_prefix, iden_path, idiv_path))

    # Filter down to rows where at least one exists, but keep placeholders
    if all(r[1] is None and r[2] is None for r in rows):
        raise RuntimeError(
            f"No overlay images found for model '{model}'.\n"
            f"Looked in:\n  {iden_overlay_dir}\n  {idiv_overlay_dir}\n"
            f"Using overlay prefixes derived from CSV predictions."
        )

    nrows = len(rows)
    ncols = 2  # [IDEN | IDIV]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.5 * ncols, 3.1 * nrows),
    )

    # Normalize axes indexing for single-row case
    if nrows == 1:
        axes = np.array([axes])

    fig.suptitle(f"IDEN vs IDIV overlays — {model}", fontsize=16)

    for r, (overlay_prefix, iden_path, idiv_path) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            add_subplot_border(ax)

        # Load images if present
        img_iden = load_rgb(iden_path) if iden_path else None
        img_idiv = load_rgb(idiv_path) if idiv_path else None

        # If both exist but shapes differ, resize IDIV to IDEN (or vice versa if IDEN missing)
        if img_iden is not None and img_idiv is not None:
            img_iden, img_idiv = resize_to_match(img_iden, img_idiv)

        # Column 0: IDEN
        ax0 = axes[r][0]
        if img_iden is not None:
            ax0.imshow(img_iden)
            if show_titles:
                ax0.set_title(f"IDEN: {iden_path.name}", fontsize=9)
        else:
            ax0.text(0.5, 0.5, "Missing IDEN", ha="center", va="center", fontsize=12)
            if show_titles:
                ax0.set_title("IDEN: (not found)", fontsize=9)

        # Column 1: IDIV
        ax1 = axes[r][1]
        if img_idiv is not None:
            ax1.imshow(img_idiv)
            if show_titles:
                ax1.set_title(f"IDIV: {idiv_path.name}", fontsize=9)
        else:
            ax1.text(0.5, 0.5, "Missing IDIV", ha="center", va="center", fontsize=12)
            if show_titles:
                ax1.set_title("IDIV: (not found)", fontsize=9)

        # Row label (left side)
        axes[r][0].set_ylabel(overlay_prefix, rotation=0, labelpad=55, fontsize=11, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model.replace("/", "_")
    out_path = out_dir / f"{safe_name}_IDEN_vs_IDIV.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path, help="CSV with columns: model, prediction")
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder that contains identification/ and individuation/",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder where per-model comparison PNGs will be saved.",
    )
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--no-titles", action="store_true", help="Disable per-panel filename titles.")
    args = ap.parse_args()

    groups = read_csv_grouped(args.csv_path)
    if not groups:
        raise RuntimeError("No rows found (or missing model/prediction values).")

    saved = 0
    for model, preds in groups.items():
        try:
            out_path = plot_one_model(
                model=model,
                preds=preds,
                root=args.root,
                out_dir=args.out,
                show_titles=(not args.no_titles),
                dpi=args.dpi,
            )
            print(f"[saved] {out_path}")
            saved += 1
        except Exception as e:
            print(f"[skip] model={model}  reason={e}")

    print(f"[done] saved {saved} figure(s) to {args.out}")


if __name__ == "__main__":
    main()
