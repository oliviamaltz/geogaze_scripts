#!/usr/bin/env python3
import os
import re
import argparse

PATTERN = re.compile(
    r'^pair_(?P<middle>.+)_(?P<idx>\d+)\.(?P<ext>png|jpg|jpeg|bmp|tif|tiff)$',
    re.IGNORECASE
)

def main():
    ap = argparse.ArgumentParser(description="List unique middle tokens from pair_<MIDDLE>_<INDEX> image filenames.")
    ap.add_argument("folder", help="Path to folder with images (e.g., /path/to/dir)")
    ap.add_argument("-o", "--out", help="Optional output text file to write unique values", default=None)
    args = ap.parse_args()

    uniques = set()

    for fname in os.listdir(args.folder):
        m = PATTERN.match(fname)
        if m:
            uniques.add(m.group("middle"))

    uniques = sorted(uniques)
    if not uniques:
        print("No matching files found with pattern: pair_<MIDDLE>_<INDEX>.(png|jpg|jpeg|bmp|tif|tiff)")
        return

    print("Unique <MIDDLE> values:")
    for u in uniques:
        print(u)

    if args.out:
        out_path = os.path.abspath(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            for u in uniques:
                f.write(u + "\n")
        print(f"\nWrote {len(uniques)} unique values to: {out_path}")

if __name__ == "__main__":
    main()
