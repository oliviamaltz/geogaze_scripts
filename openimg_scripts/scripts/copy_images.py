#!/usr/bin/env python3
"""
Copy only the images listed in a text file to another folder.

IMPORTANT BEHAVIOR (per your request):
- If a line is "validation/ed52f03ad1a8ef84", this script uses ONLY the basename:
    "ed52f03ad1a8ef84"
  and searches for:
    <SRC_ROOT>/ed52f03ad1a8ef84.jpg (and other extensions)

Usage:
  python copy_images_basename.py \
    --src-root /path/to/all_images \
    --list-file /path/to/validation_images_filtered.v2.txt \
    --dst-root /path/to/output_subset

Optional:
  --exts jpg jpeg png webp
  --overwrite
  --dry-run
  --log-missing /path/to/missing.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_EXTS = ("jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp")


def iter_list_lines(list_file: Path) -> Iterable[str]:
    # Stream file line-by-line (handles very large lists)
    with list_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield s


def find_existing_in_root_by_basename(
    src_root: Path, line: str, exts: tuple[str, ...]
) -> tuple[Path | None, str]:
    """
    Convert 'validation/abc123' -> basename 'abc123' and try:
      src_root/abc123.jpg, .jpeg, ...
    If line already has an extension, we still only use its basename:
      'validation/abc123.png' -> tries src_root/abc123.png first, then other exts if needed.

    Returns: (found_path or None, base_stem_used_for_search)
    """
    p = Path(line)
    base_name = p.name               # strips any folders (validation/, train/, etc.)
    base = Path(base_name)

    # If line includes an extension, try that exact basename-with-ext first
    if base.suffix:
        candidate = src_root / base
        if candidate.is_file():
            return candidate, base.stem
        # If not found, fall through and try other extensions using stem

    stem = base.stem if base.suffix else base_name  # if no suffix, whole basename is stem
    for ext in exts:
        candidate = src_root / f"{stem}.{ext}"
        if candidate.is_file():
            return candidate, stem

    return None, stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, type=Path, help="Folder containing ALL images (flat in this mode)")
    ap.add_argument("--list-file", required=True, type=Path, help="Text file listing images (one per line)")
    ap.add_argument("--dst-root", required=True, type=Path, help="Folder to copy matching images into")
    ap.add_argument("--exts", nargs="*", default=list(DEFAULT_EXTS),
                    help="Extensions to try if list lines have no extension")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in dst")
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen, don't actually copy")
    ap.add_argument("--log-missing", type=Path, default=None,
                    help="Optional path to write missing entries (one per line, original line from list-file)")
    args = ap.parse_args()

    src_root: Path = args.src_root
    list_file: Path = args.list_file
    dst_root: Path = args.dst_root
    exts = tuple(e.lstrip(".").lower() for e in args.exts)

    if not src_root.is_dir():
        print(f"ERROR: --src-root is not a directory: {src_root}", file=sys.stderr)
        return 2
    if not list_file.is_file():
        print(f"ERROR: --list-file not found: {list_file}", file=sys.stderr)
        return 2

    dst_root.mkdir(parents=True, exist_ok=True)

    missing_out = None
    if args.log_missing is not None:
        args.log_missing.parent.mkdir(parents=True, exist_ok=True)
        missing_out = args.log_missing.open("w", encoding="utf-8")

    copied = 0
    skipped_existing = 0
    missing = 0

    try:
        for i, line in enumerate(iter_list_lines(list_file), start=1):
            src_path, stem = find_existing_in_root_by_basename(src_root, line, exts)

            if src_path is None:
                missing += 1
                if missing_out:
                    missing_out.write(line + "\n")
                print(f"[MISSING] {line}  (basename tried: {Path(line).name})")
                continue

            # Destination is always flat: dst_root/<actual filename>
            dst_path = dst_root / src_path.name

            if dst_path.exists() and not args.overwrite:
                skipped_existing += 1
                print(f"[SKIP exists] {dst_path}")
                continue

            print(f"[COPY] {src_path.name}")

            if not args.dry_run:
                shutil.copy2(src_path, dst_path)

            copied += 1

            if i % 5000 == 0:
                print(f"--- progress: lines={i:,} copied={copied:,} missing={missing:,} skipped={skipped_existing:,}")

    finally:
        if missing_out:
            missing_out.close()

    print("\nDone.")
    print(f"Copied:  {copied:,}")
    print(f"Missing: {missing:,}")
    print(f"Skipped: {skipped_existing:,} (already existed; use --overwrite to replace)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
