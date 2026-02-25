#!/usr/bin/env python3
"""
generate_pairs_224_500loc.py

Generates composite pair images + left/right masks with these rules:
  - Canvas: 224 x 224
  - Shape sprite: 25 x 25
  - Restricted center square: 37.5 x 37.5 (no overlap by either sprite)
  - Placement: Two shapes side-by-side (same y), separated by GAP
  - Pairs: All 16 minus identical tokens (no bc-bc, gc-gc, bs-bs, gs-gs) => 12 pairs/location
  - Locations: Precompute N valid random locations (default 500), then place every allowed pair at each

Output structure:
  output_dir/
    pairs/         pair_<left>_<right>_####.png
    left_masks/    maskL_<left>_<right>_####.png
    right_masks/   maskR_<left>_<right>_####.png
    manifest.csv

CLI:
  python generate_pairs_224_500loc.py /path/to/output_dir [num_locations=500] [num_total_pairs_cap]
Examples:
  python generate_pairs_224_500loc.py ./out/pairs_224               # 500 locations, full set
  python generate_pairs_224_500loc.py ./out/pairs_224 10 48         # 10 locations, stop after 48 composites
"""

from pathlib import Path
from PIL import Image, ImageDraw
import random, csv, sys
from dataclasses import dataclass

# -------------------- Config --------------------
W, H = 224, 224            # canvas
SHAPE_SZ = 25              # sprite size
GAP = 8                    # horizontal gap between sprites
SEED = 1234                # RNG seed for reproducibility

# Restricted center square (37.5 x 37.5), float coords
R_SIDE = 37.5
R_HALF = R_SIDE / 2.0
R_CX, R_CY = W / 2.0, H / 2.0
R_LEFT, R_TOP = R_CX - R_HALF, R_CY - R_HALF
R_RIGHT, R_BOTTOM = R_CX + R_HALF, R_CY + R_HALF

# Tokens: code -> (shape, color)
CODES = {
    "bc": ("circle", "blue"),
    "gc": ("circle", "green"),
    "bs": ("square", "blue"),
    "gs": ("square", "green"),
}
ALL_PAIRS = [(l, r) for l in CODES for r in CODES]  # 16 total

# -------------------- Geometry helpers --------------------
@dataclass
class Box:
    x: int; y: int; w: int; h: int
    def intersects(self, other) -> bool:
        return not (self.x + self.w <= other.x or other.x + other.w <= self.x or
                    self.y + self.h <= other.y or other.y + other.h <= self.y)

def rect_from_floats(x, y, w, h) -> Box:
    return Box(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

RESTRICTED = rect_from_floats(R_LEFT, R_TOP, R_SIDE, R_SIDE)

def box_at(x, y) -> Box:
    return Box(x, y, SHAPE_SZ, SHAPE_SZ)

def fits_in_canvas(b: Box) -> bool:
    return 0 <= b.x and 0 <= b.y and (b.x + b.w) <= W and (b.y + b.h) <= H

def overlaps_restricted(b: Box) -> bool:
    return b.intersects(RESTRICTED)

# -------------------- Sprites --------------------
def make_sprite(shape: str, color: str) -> Image.Image:
    """RGBA sprite of the colored shape."""
    img = Image.new("RGBA", (SHAPE_SZ, SHAPE_SZ), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = {"blue": (60,150,255,255), "green": (120,210,60,255)}[color]
    if shape == "circle":
        draw.ellipse([0, 0, SHAPE_SZ-1, SHAPE_SZ-1], fill=fill)
    elif shape == "square":
        draw.rectangle([0, 0, SHAPE_SZ-1, SHAPE_SZ-1], fill=fill)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return img

def make_mask_sprite(shape: str) -> Image.Image:
    """RGBA sprite (black shape, transparent elsewhere) for masks."""
    img = Image.new("RGBA", (SHAPE_SZ, SHAPE_SZ), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (0, 0, 0, 255)
    if shape == "circle":
        draw.ellipse([0, 0, SHAPE_SZ-1, SHAPE_SZ-1], fill=fill)
    elif shape == "square":
        draw.rectangle([0, 0, SHAPE_SZ-1, SHAPE_SZ-1], fill=fill)
    else:
        raise ValueError(f"Unknown shape for mask: {shape}")
    return img

# -------------------- Pair filter --------------------
def disallow_pair(lc: str, rc: str) -> bool:
    """Exclude identical left/right tokens (e.g., bc-bc)."""
    return lc == rc

# If you ever need to forbid same SHAPE regardless of color, uncomment:
# def disallow_pair(lc: str, rc: str) -> bool:
#     if lc == rc: return True
#     return CODES[lc][0] == CODES[rc][0]

# -------------------- Location generation --------------------
def valid_location(x: int, y: int) -> bool:
    """Check if left & right sprites fit and avoid the restricted square."""
    L = box_at(x, y)
    R = box_at(x + SHAPE_SZ + GAP, y)
    return (fits_in_canvas(L) and fits_in_canvas(R) and
            not overlaps_restricted(L) and not overlaps_restricted(R))

def generate_locations(rng: random.Random, n: int) -> list[tuple[int, int]]:
    """
    Precompute n unique valid (x,y) locations.
    Ensures two 25x25 sprites at x and x+SHAPE_SZ+GAP do not overlap restricted zone.
    """
    coords: set[tuple[int, int]] = set()
    max_x = W - (2 * SHAPE_SZ + GAP)
    max_y = H - SHAPE_SZ
    attempts = 0
    max_attempts = n * 10000  # generous upper bound

    while len(coords) < n and attempts < max_attempts:
        attempts += 1
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        if (x, y) in coords:
            continue
        if valid_location(x, y):
            coords.add((x, y))

    if len(coords) < n:
        raise RuntimeError(f"Only found {len(coords)} valid locations out of requested {n}. "
                           f"Try reducing n or loosening constraints.")
    return list(coords)

# -------------------- Main generation --------------------
def generate(output_dir: Path, num_locations: int = 500, max_pairs: int | None = None):
    rng = random.Random(SEED)

    # Allowed pairs (12 after filtering)
    pairs = [(l, r) for (l, r) in ALL_PAIRS if not disallow_pair(l, r)]
    if len(pairs) != 12:
        raise AssertionError(f"Expected 12 allowed pairs, got {len(pairs)}")

    # Prepare output folders
    pairs_dir   = output_dir / "pairs"
    left_dir    = output_dir / "left_masks"
    right_dir   = output_dir / "right_masks"
    for d in (pairs_dir, left_dir, right_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Prebuild sprites
    color_sprites = {code: make_sprite(*CODES[code]) for code in CODES}
    mask_sprites  = {code: make_mask_sprite(CODES[code][0]) for code in CODES}

    # Precompute locations
    print(f"[info] Generating {num_locations} random valid locations...")
    locations = generate_locations(rng, num_locations)
    print(f"[info] Got {len(locations)} locations.")

    # Manifest
    man_path = output_dir / "manifest.csv"
    with man_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "idx","location_idx","left_code","right_code",
            "left_shape","left_color","right_shape","right_color",
            "left_x","left_y","right_x","right_y",
            "gap","width","height","seed","restricted_center",
            "pair_path","maskL_path","maskR_path"
        ])

        global_idx = 1
        generated = 0

        for loc_idx, (x, y) in enumerate(locations, start=1):
            for lc, rc in pairs:
                if max_pairs is not None and generated >= max_pairs:
                    print(f"✂️ Reached cap of {max_pairs} composites; stopping.")
                    print(f"✅ Manifest: {man_path}")
                    return

                left_shape, left_color   = CODES[lc]
                right_shape, right_color = CODES[rc]

                # Canvases
                comp  = Image.new("RGB", (W, H), (255, 255, 255))
                maskL = Image.new("RGB", (W, H), (255, 255, 255))
                maskR = Image.new("RGB", (W, H), (255, 255, 255))

                # Positions
                lx, ly = x, y
                rx, ry = x + SHAPE_SZ + GAP, y

                # Paste shapes
                comp.paste(color_sprites[lc], (lx, ly), color_sprites[lc])
                comp.paste(color_sprites[rc], (rx, ry), color_sprites[rc])
                maskL.paste(mask_sprites[lc], (lx, ly), mask_sprites[lc])
                maskR.paste(mask_sprites[rc], (rx, ry), mask_sprites[rc])

                # File names
                idx_str   = f"{global_idx:04d}"
                pair_name = f"pair_{lc}_{rc}_{idx_str}.png"
                ml_name   = f"maskL_{lc}_{rc}_{idx_str}.png"
                mr_name   = f"maskR_{lc}_{rc}_{idx_str}.png"

                # Save
                comp.save(pairs_dir / pair_name)
                maskL.save(left_dir / ml_name)
                maskR.save(right_dir / mr_name)

                # Log
                wcsv.writerow([
                    idx_str, loc_idx, lc, rc,
                    left_shape, left_color, right_shape, right_color,
                    lx, ly, rx, ry,
                    GAP, W, H, SEED, 1,
                    str(pairs_dir / pair_name),
                    str(left_dir / ml_name),
                    str(right_dir / mr_name),
                ])

                global_idx += 1
                generated  += 1

    print(f"✅ Wrote {generated} composites (+ masks) to {output_dir}")
    print(f"   • pairs:       {pairs_dir}")
    print(f"   • left_masks:  {left_dir}")
    print(f"   • right_masks: {right_dir}")
    print(f"   • manifest:    {man_path}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pairs_224_500loc.py /path/to/output_dir [num_locations=500] [num_total_pairs_cap]")
        sys.exit(1)
    outdir = Path(sys.argv[1]).expanduser().resolve()
    num_loc = int(sys.argv[2]) if len(sys.argv) >= 3 else 500
    cap = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    outdir.mkdir(parents=True, exist_ok=True)
    generate(outdir, num_locations=num_loc, max_pairs=cap)
