import pandas as pd

INPUT_CSV = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/v2/pretrained/geogaze_model_predictions_pretrained_pred_avg_ratio.csv"
OUTPUT_CSV = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/v2/pretrained/geogaze_model_predictions_pretrained_pred_avg_ratio_stimuli_same_shape.csv"

# List of substrings to REMOVE
REMOVE_KEYS = [
    "bc_bs",
    "bs_bc",
    "gc_gs",
    "gs_bs",
]

# Column to check
MODEL_COL = "model"

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Build regex pattern like: bc_bs|gc_gs|bc_gc
pattern = "|".join(REMOVE_KEYS)

# Keep rows that do NOT contain any of the unwanted substrings
filtered_df = df[~df[MODEL_COL].str.contains(pattern, regex=True, na=False)]

# Save result
filtered_df.to_csv(OUTPUT_CSV, index=False)

print(f"Original rows: {len(df)}")
print(f"Remaining rows: {len(filtered_df)}")
print(f"Removed rows: {len(df) - len(filtered_df)}")


# # ================== CONFIG ==================
# INPUT_CSV = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/v1/identification/geogaze_model_predictions_identification_cornetIDEN_01_29_26.csv"
# OUTPUT_CSV = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/v1/identification/geogaze_model_predictions_identification_cornetIDEN_01_29_26_abovechance.csv"

# CONDITION_COL = "condition"
# MODEL_COL = "model"
# ACC_COL = "acc_continuous"

# TRAIN_LABEL = "train"
# THRESHOLD = 0.5
# # ============================================

# # Load CSV
# df = pd.read_csv(INPUT_CSV)

# # Ensure acc_continuous is numeric
# df[ACC_COL] = pd.to_numeric(df[ACC_COL], errors="coerce")

# # --------------------------------------------------
# # 1) Find models that FAIL on TRAIN
# # --------------------------------------------------
# bad_models = (
#     df.loc[
#         (df[CONDITION_COL] == TRAIN_LABEL) &
#         (df[ACC_COL] < THRESHOLD),
#         MODEL_COL
#     ]
#     .unique()
# )

# print(f"Found {len(bad_models)} models failing train threshold:")
# for m in bad_models:
#     print("  ", m)

# # --------------------------------------------------
# # 2) Remove ALL rows with those model names
# # --------------------------------------------------
# filtered_df = df[~df[MODEL_COL].isin(bad_models)]

# # Save result
# filtered_df.to_csv(OUTPUT_CSV, index=False)

# print("\nSummary:")
# print(f"Original rows: {len(df)}")
# print(f"Remaining rows: {len(filtered_df)}")
# print(f"Removed rows: {len(df) - len(filtered_df)}")