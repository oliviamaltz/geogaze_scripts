#!/usr/bin/env python3
"""
Filter OpenImages-style bounding boxes by removing any image that contains
at least one "banned" object label, and removing ALL boxes for those images.

Does NOT delete files on disk — it writes new filtered outputs.

Inputs:
  1) image_list.txt: one image id per line (optionally with split prefix like "validation/XXXX")
  2) bboxes.csv: OpenImages bbox annotations with at least columns: ImageID, LabelName
  3) class-descriptions-boxable.csv (optional): columns LabelName,DisplayName

Outputs:
  - filtered_image_list.txt
  - filtered_bboxes.csv
  - removed_images.txt
  - summary printed to stdout

Edit the CONFIG section below.
"""

from pathlib import Path
import pandas as pd

# ===================== CONFIG (EDIT THESE) =====================
IMAGE_LIST_TXT = Path("/zpool/vladlab/data_drive/geogaze_data/annotations/v2/filtered_image_index/test_images_filtered.v2.txt")          # input txt
BBOX_CSV        = Path("/zpool/vladlab/data_drive/geogaze_data/annotations/v2/filtered_bb_annotations/test-annotations-bbox.filtered.v2.csv")        # input bbox csv
CLASS_MAP_CSV   = Path("/zpool/vladlab/data_drive/stimulus_sets/geogaze_open_images_stim/openimages_working/class-descriptions-boxable.csv")         # optional

OUT_IMAGE_LIST_TXT = Path("/zpool/vladlab/data_drive/geogaze_data/annotations/v5/filtered_image_index/test_images_filtered.v5.txt")
OUT_BBOX_CSV        = Path("/zpool/vladlab/data_drive/geogaze_data/annotations/v5/filtered_bb_annotations/test-annotations-bbox.filtered.v5.csv")
OUT_REMOVED_TXT     = Path("/zpool/vladlab/data_drive/stimulus_sets/geogaze_open_images_stim/openimages_working/v4/train_removed_image.txt")

OUT_IMAGE_LIST_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_BBOX_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_REMOVED_TXT.parent.mkdir(parents=True, exist_ok=True)

# --- Provide ONE of these ban lists ---

# Option A: ban by human-readable class names (DisplayName), e.g. "Apple"
BANNED_DISPLAYNAMES = [
    "Human arm", "Human body", "Human ear", "Human eye", "Human face", "Human foot", "Human hair", "Human hand", "Human head", "Human leg", "Human mouth", "Human nose", "Bathroom accessory", 
    "Fashion accessory", "Animal", "Auto part", "Baked goods", "Beard", "Bowling equipment", "Hiking equipment", "Medical equipment", "Sports equipment", "Sports uniform", "Building", "Cabinetry", 
    "Carnivore", "Cat furniture", "Furniture", "Closet", "Clothing", "Common fig", "Common sunflower", "Cosmetics", "Countertop", "Dairy Product", "Door handle", "Drinking straw", "Face powder", 
    "Facial tissue holder", "Fast food", "Fixed-wing aircraft", "Flying disc", "Footwear", "Fruit", "Grinder", "Home appliance", "House", "Insect", "Invertebrate", "Ipod", "Isopod", 
    "Kitchen & dining room table", "Kitchen appliance", "Luggage and bags", "Loveseat", "Marine invertebrates", "Marine mammal", "Musical instrument", "Musical keyboard", "Office building",
    "Office supplies", "Personal care", "Personal flotation device", "Plant", "Plumbing fixture", "Seafood", "Skyscraper", "Tableware", "Tool", "Tower", "Toy", "Vegetable", "Watercraft", "Weapon", 
    "Winter melon", "Apple", "Artichoke", "Bagel", "Banana", "Beer", "Bell pepper", "Bread", "Broccoli", "Cabbage", "Candy", "Cantaloupe", "Carrot", "Cheese", "Chicken", "Coconut", "Cookie", "Cooking spray", 
    "Crab", "Cream", "Croissant", "Cucumber", "Dessert", "Doughnut", "Drink", "Duck", "Egg", "Fish", "Food", "French fries", "Garden Asparagus", "Grape", "Grapefruit", "Guacamole", "Hamburger", "Honeycomb", 
    "Hot dog", "Ice cream", "Juice", "Lemon (plant)", "Lobster", "Mango", "Milk", "Muffin", "Mushroom", "Orange (fruit)", "Oyster", "Pancake", "Pasta", "Pastry", "Peach", "Pear", "Pineapple", "Pizza", 
    "Pomegranate", "Popcorn", "Potato", "Pretzel", "Pumpkin", "Radish", "Salad", "Sandwich", "Shellfish", "Shrimp", "Snack", "Squash (Plant)", "Strawberry", "Submarine sandwich", "Sushi", "Taco", "Tart", 
    "Tea", "Tomato", "Turkey", "Waffle", "Watermelon", "Wine", "Zucchini", "Ball (Object)", "Convenience store", "Kitchen utensil", "Land vehicle", "Mammal", "Reptile", "Vehicle", "Person", "Man", "Woman", "Tree", "Suit","Dress","Car"

]

# Option B: ban by LabelName ids directly, e.g. "/m/014j1m"
BANNED_LABELNAMES = [
    # "/m/014j1m",
]

# Matching behavior for DisplayName (OpenImages uses Title Case, but be safe)
DISPLAYNAME_CASE_INSENSITIVE = True
# ===============================================================


def load_image_ids(txt_path: Path) -> list[str]:
    """Read lines, strip whitespace, drop empties. Keep as-is (may include split prefix)."""
    lines = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def normalize_image_id_for_bbox(image_id_from_txt: str) -> str:
    """
    OpenImages bbox CSV ImageID is usually just the hex id without "validation/" prefix.
    Your txt lines look like "validation/<id>".
    This returns the raw ImageID part (last path component).
    """
    return image_id_from_txt.split("/")[-1]


def resolve_banned_labelnames(
    class_map_csv: Path,
    banned_displaynames: list[str],
    banned_labelnames: list[str],
    case_insensitive: bool = True
) -> set[str]:
    """
    Returns a set of LabelName strings to ban.
    If banned_labelnames is non-empty, includes those directly.
    If banned_displaynames is non-empty, maps DisplayName -> LabelName using class_map_csv.
    """
    banned = set([x.strip() for x in banned_labelnames if x and str(x).strip()])

    if banned_displaynames:
        if not class_map_csv.exists():
            raise FileNotFoundError(
                f"You provided BANNED_DISPLAYNAMES but CLASS_MAP_CSV not found: {class_map_csv}"
            )
        m = pd.read_csv(class_map_csv)  # expects columns LabelName, DisplayName
        if "LabelName" not in m.columns or "DisplayName" not in m.columns:
            raise ValueError(f"{class_map_csv} must have columns: LabelName, DisplayName")

        # build lookup
        if case_insensitive:
            m["_dn_norm"] = m["DisplayName"].astype(str).str.strip().str.casefold()
            wanted = {str(x).strip().casefold() for x in banned_displaynames if str(x).strip()}
            matched = m[m["_dn_norm"].isin(wanted)]
        else:
            m["_dn_norm"] = m["DisplayName"].astype(str).str.strip()
            wanted = {str(x).strip() for x in banned_displaynames if str(x).strip()}
            matched = m[m["_dn_norm"].isin(wanted)]

        found_dns = set(matched["DisplayName"].astype(str))
        missing = [x for x in banned_displaynames if (x.strip() and (x not in found_dns))]
        # Note: if case_insensitive, missing detection above may be slightly off for casing,
        # but it's still useful; we print a clearer check below.
        banned_from_names = set(matched["LabelName"].astype(str).str.strip())
        banned |= banned_from_names

        # Better missing detection (casefold aware)
        if case_insensitive:
            found_casefold = set(found_dns_i.casefold() for found_dns_i in found_dns)
            missing = [x for x in banned_displaynames if x.strip().casefold() not in found_casefold]
        else:
            missing = [x for x in banned_displaynames if x.strip() not in found_dns]

        if missing:
            print("WARNING: These DisplayNames were not found in class map and will be ignored:")
            for x in missing:
                print(f"  - {x}")

    return banned


def main():
    # 1) Load image list
    img_lines = load_image_ids(IMAGE_LIST_TXT)
    img_ids_for_bbox = [normalize_image_id_for_bbox(x) for x in img_lines]
    img_id_set = set(img_ids_for_bbox)

    # 2) Load bbox CSV (only needed cols + keep all to write filtered full CSV)
    bboxes = pd.read_csv(BBOX_CSV)
    required = {"ImageID", "LabelName"}
    missing_cols = required - set(bboxes.columns)
    if missing_cols:
        raise ValueError(f"BBOX_CSV missing required columns: {missing_cols}")

    # 3) Restrict bbox rows to only images in your txt list (so we only operate on your subset)
    bboxes_in_subset = bboxes[bboxes["ImageID"].astype(str).isin(img_id_set)].copy()

    # 4) Resolve banned labels
    banned_label_set = resolve_banned_labelnames(
        CLASS_MAP_CSV, BANNED_DISPLAYNAMES, BANNED_LABELNAMES, DISPLAYNAME_CASE_INSENSITIVE
    )
    if not banned_label_set:
        raise ValueError("No banned labels provided. Fill BANNED_DISPLAYNAMES and/or BANNED_LABELNAMES.")

    # 5) Find images to remove: any image with at least one banned label
    mask_banned = bboxes_in_subset["LabelName"].astype(str).isin(banned_label_set)
    removed_image_ids = set(bboxes_in_subset.loc[mask_banned, "ImageID"].astype(str))

    # 6) Filter image list txt (remove those images)
    kept_img_lines = []
    removed_img_lines = []
    for line in img_lines:
        imgid = normalize_image_id_for_bbox(line)
        if imgid in removed_image_ids:
            removed_img_lines.append(line)
        else:
            kept_img_lines.append(line)

    # 7) Filter bbox CSV: remove ALL boxes for removed images (within your subset)
    kept_bboxes_subset = bboxes_in_subset[~bboxes_in_subset["ImageID"].astype(str).isin(removed_image_ids)].copy()

    # Optional: if you want the output bbox file to contain ONLY your subset (filtered),
    # keep as-is (this script does).
    # If instead you want to apply removals to the full original bbox CSV, you could also:
    # kept_bboxes_full = bboxes[~bboxes["ImageID"].astype(str).isin(removed_image_ids)].copy()

    # 8) Write outputs
    OUT_IMAGE_LIST_TXT.write_text("\n".join(kept_img_lines) + ("\n" if kept_img_lines else ""), encoding="utf-8")
    OUT_REMOVED_TXT.write_text("\n".join(removed_img_lines) + ("\n" if removed_img_lines else ""), encoding="utf-8")
    kept_bboxes_subset.to_csv(OUT_BBOX_CSV, index=False)

    # 9) Summary
    total_imgs = len(img_lines)
    total_boxes_subset = len(bboxes_in_subset)
    removed_imgs = len(removed_img_lines)
    removed_boxes = total_boxes_subset - len(kept_bboxes_subset)

    print("=== Filter summary ===")
    print(f"Input image list lines:           {total_imgs:,}")
    print(f"BBox rows in your image subset:   {total_boxes_subset:,}")
    print(f"Banned labels (LabelName) count:  {len(banned_label_set):,}")
    print(f"Images removed (contain banned):  {removed_imgs:,}")
    print(f"BBox rows removed (all for imgs): {removed_boxes:,}")
    print("")
    print("Wrote:")
    print(f"  - {OUT_IMAGE_LIST_TXT}")
    print(f"  - {OUT_BBOX_CSV}")
    print(f"  - {OUT_REMOVED_TXT}")


if __name__ == "__main__":
    main()
