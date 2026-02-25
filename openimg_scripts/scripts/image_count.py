#!/usr/bin/env python3
"""
Count .jpg/.jpeg images in a folder efficiently (works for millions of files).

Examples:
  python count_jpgs.py /path/to/folder
  python count_jpgs.py /path/to/folder --recursive
"""

import argparse
import os
from pathlib import Path


def is_jpg(name: str) -> bool:
    n = name.lower()
    return n.endswith(".jpg") or n.endswith(".jpeg")


def count_jpgs_nonrecursive(folder: Path) -> int:
    count = 0
    with os.scandir(folder) as it:
        for entry in it:
            # entry.is_file() uses stat calls; follow_symlinks=False avoids surprises
            if entry.is_file(follow_symlinks=False) and is_jpg(entry.name):
                count += 1
    return count


def count_jpgs_recursive(folder: Path) -> int:
    count = 0
    # os.walk is a generator; good for huge trees
    for root, dirs, files in os.walk(folder):
        for fn in files:
            if is_jpg(fn):
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder to count .jpg/.jpeg files in")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Count .jpg/.jpeg files in all subfolders too",
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"[error] Not a folder: {folder}")

    if args.recursive:
        n = count_jpgs_recursive(folder)
        print(f"JPG/JPEG files (recursive): {n:,}")
    else:
        n = count_jpgs_nonrecursive(folder)
        print(f"JPG/JPEG files (non-recursive): {n:,}")


if __name__ == "__main__":
    main()
