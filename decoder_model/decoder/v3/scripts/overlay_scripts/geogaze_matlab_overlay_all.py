#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

# path to the overlay script you just edited
OVERLAY_SCRIPT = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/v3/scripts/overlay_scripts/geogaze_matlab_overlay.py"

# root folder that contains all the per-model subfolders (your screenshot)
TEST_MASKS_ROOT = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/v3/resnet100/test_masks"

# common test images directory
IMGS_DIR = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/stimuli/out/test_stimuli/test_pairs"

def main():
    root = Path(TEST_MASKS_ROOT)

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        # skip debug/weights/etc
        if subdir.name.startswith("_") or subdir.name == "weights":
            continue

        out_png = f"{subdir.name}_overlays.png"

        cmd = [
            sys.executable, OVERLAY_SCRIPT,
            "--imgs_dir", IMGS_DIR,
            "--masks_dir", str(subdir),
            "--out_dir", str(subdir),
            "--out_png", out_png,
        ]

        print("\n==> Running overlay for", subdir.name)
        print("    Cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\n[OK] Finished overlays for all subfolders.")

if __name__ == "__main__":
    main()
