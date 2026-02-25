from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import math

# >>> CHANGE THIS to your actual folder
img_dir = Path("/Users/oliviamaltz/Downloads/Temple Items/geogaze/stimuli/out/pairs").expanduser().resolve()

# Collect images recursively and accept common extensions
exts = {".png", ".jpg", ".jpeg"}
image_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
image_files.sort()

print(f"[info] Looking in: {img_dir}")
print(f"[info] Found {len(image_files)} images")
if len(image_files) == 0:
    print("[hint] Check the folder path, verify files exist, and confirm extensions.")
    for p in img_dir.glob("*"):
        print("   -", p.name)
    raise SystemExit(1)

# Show all images (or as many as you want)
N = min(len(image_files), 64)   # show 64 if you have 64 images
files_to_show = image_files[:N]

# Compute grid size automatically (rows x cols)
cols = 8                       # 8x8 grid for 64 images
rows = math.ceil(N / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
axes = axes.ravel()

for ax, img_path in zip(axes, files_to_show):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(img_path.stem, fontsize=6)
    ax.axis("off")

    # Add border around each image
    border = patches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor="black",   # change to another color if you want
        linewidth=2
    )
    ax.add_patch(border)

# Hide any unused axes
for ax in axes[len(files_to_show):]:
    ax.axis("off")

plt.tight_layout(pad=0.3)
plt.savefig("my_grid2.pdf", format="pdf", bbox_inches="tight")
plt.show()
