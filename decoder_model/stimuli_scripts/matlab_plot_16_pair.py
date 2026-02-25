from pathlib import Path
import re
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# >>> CHANGE THIS to your actual folder
img_dir = Path("/Users/oliviamaltz/Downloads/Temple Items/geogaze/stimuli/out/pairs").expanduser().resolve()

# Collect images recursively and accept common extensions
exts = {".png", ".jpg", ".jpeg"}
all_imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])

# Helper: extract a trailing 4‐digit idx from the stem
idx_re = re.compile(r"(\d{4})$")   # matches last 4 digits of the stem

def get_idx(p: Path):
    m = idx_re.search(p.stem)
    return int(m.group(1)) if m else None

# Keep ONLY 0001–0016
wanted_range = set(range(1, 17))
files_to_show = []
for p in all_imgs:
    idx = get_idx(p)
    if idx in wanted_range:
        files_to_show.append((idx, p))

# Sort by idx ascending
files_to_show.sort(key=lambda t: t[0])
files_only = [p for _, p in files_to_show]

print(f"[info] Found {len(files_only)} images in idx 0001–0016")
if len(files_only) != 16:
    print("⚠️ Warning: expected 16 images (0001–0016). Missing indices:")
    found = {get_idx(p) for p in files_only}
    missing = sorted(wanted_range - found)
    print("   ", missing)

# --- Plot 4×4 grid and save as PNG ---
N = len(files_only)
cols = 4
rows = math.ceil(N / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
axes = axes.ravel()

for ax, img_path in zip(axes, files_only):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(img_path.stem, fontsize=8)
    ax.axis("off")

    # Add a border around each tile
    ax.add_patch(
        patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="black", linewidth=2)
    )

# Hide any unused axes (in case some are missing)
for ax in axes[len(files_only):]:
    ax.axis("off")

plt.tight_layout(pad=0.4)
out_png = img_dir / "location1_pairs.png"
plt.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Saved 4×4 grid for 0001–0016 to: {out_png}")
