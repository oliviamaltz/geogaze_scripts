#!/usr/bin/env python3
"""
resize_images.py
Resize all images in a folder from 2000x2000 to 200x200 pixels.

Usage:
  python resize_images.py /path/to/folder
"""

import sys
from pathlib import Path
from PIL import Image

def resize_images(folder_path, new_size=(25, 25)):
    folder = Path(folder_path)
    if not folder.is_dir():
        sys.exit(f"❌ Error: {folder_path} is not a valid folder.")

    output_folder = folder / "resized"
    output_folder.mkdir(exist_ok=True)

    for img_path in folder.glob("*.*"):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
            continue  # skip non-image files

        try:
            with Image.open(img_path) as img:
                resized_img = img.resize(new_size, Image.LANCZOS)
                resized_img.save(output_folder / img_path.name)
                print(f"✅ Resized: {img_path.name}")
        except Exception as e:
            print(f"⚠️ Could not process {img_path.name}: {e}")

    print(f"\n🎉 All done! Resized images saved in: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resize_images.py /path/to/folder")
        sys.exit(1)
    resize_images(sys.argv[1])
