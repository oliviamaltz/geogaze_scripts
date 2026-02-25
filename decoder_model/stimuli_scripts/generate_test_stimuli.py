#!/usr/bin/env python3
"""
generate_test_stimuli.py

Create centered 224x224 composites for:
  shapes: circle, square
  layouts: left-right (LR), up-down (UD)
  color orders: blue-green and green-blue

Outputs (8 total):
  out/
    center_circle_LR_blue-green.png
    center_circle_LR_green-blue.png
    center_circle_UD_blue-green.png
    center_circle_UD_green-blue.png
    center_square_LR_blue-green.png
    center_square_LR_green-blue.png
    center_square_UD_blue-green.png
    center_square_UD_green-blue.png
"""

from pathlib import Path
from PIL import Image, ImageDraw

# Canvas and sprite config (matching your previous script)
W, H = 224, 224
SHAPE_SZ = 25
GAP = 8

# Colors (RGBA) to match previous look
BLUE  = (60, 150, 255, 255)
GREEN = (120, 210,  60, 255)

def make_sprite(shape: str, color_rgba: tuple[int,int,int,int]) -> Image.Image:
    """Return a 25x25 RGBA sprite of a filled circle or square."""
    img = Image.new("RGBA", (SHAPE_SZ, SHAPE_SZ), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([0, 0, SHAPE_SZ-1, SHAPE_SZ-1], fill=color_rgba)
    elif shape == "square":
        draw.rectangle([0, 0, SHAPE_SZ-1, SHAPE_SZ-1], fill=color_rgba)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return img

def paste_lr_centered(canvas: Image.Image, left_sprite: Image.Image, right_sprite: Image.Image) -> None:
    """
    Paste two sprites side-by-side, centered both horizontally and vertically.
    """
    total_w = SHAPE_SZ * 2 + GAP
    left_x = (W - total_w) // 2
    y = (H - SHAPE_SZ) // 2
    right_x = left_x + SHAPE_SZ + GAP

    canvas.paste(left_sprite,  (left_x,  y), left_sprite)
    canvas.paste(right_sprite, (right_x, y), right_sprite)

def paste_ud_centered(canvas: Image.Image, top_sprite: Image.Image, bottom_sprite: Image.Image) -> None:
    """
    Paste two sprites stacked vertically, centered both horizontally and vertically.
    """
    total_h = SHAPE_SZ * 2 + GAP
    x = (W - SHAPE_SZ) // 2
    top_y = (H - total_h) // 2
    bot_y = top_y + SHAPE_SZ + GAP

    canvas.paste(top_sprite,    (x, top_y), top_sprite)
    canvas.paste(bottom_sprite, (x,  bot_y), bottom_sprite)

def main(outdir: str = "out"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for shape in ("circle", "square"):
        # Build sprites once per shape
        blue_sprite  = make_sprite(shape, BLUE)
        green_sprite = make_sprite(shape, GREEN)

        # LR layouts (centered)
        # 1) blue on left, green on right
        img = Image.new("RGB", (W, H), (255, 255, 255))
        paste_lr_centered(img, blue_sprite, green_sprite)
        img.save(out / f"center_{shape}_LR_blue-green.png")

        # 2) green on left, blue on right
        img = Image.new("RGB", (W, H), (255, 255, 255))
        paste_lr_centered(img, green_sprite, blue_sprite)
        img.save(out / f"center_{shape}_LR_green-blue.png")

        # UD layouts (centered)
        # 3) blue on top, green on bottom
        img = Image.new("RGB", (W, H), (255, 255, 255))
        paste_ud_centered(img, blue_sprite, green_sprite)
        img.save(out / f"center_{shape}_UD_blue-green.png")

        # 4) green on top, blue on bottom
        img = Image.new("RGB", (W, H), (255, 255, 255))
        paste_ud_centered(img, green_sprite, blue_sprite)
        img.save(out / f"center_{shape}_UD_green-blue.png")

if __name__ == "__main__":
    # Optionally accept an output directory as the first CLI arg
    import sys
    outdir = sys.argv[1] if len(sys.argv) >= 2 else "out"
    main(outdir)
