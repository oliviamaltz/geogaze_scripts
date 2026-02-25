import pandas as pd

CSV_PATH = "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/instances_train_filtered3.csv"
OUT_PATH = "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/instances_train_filtered3.csv"

df = pd.read_csv(CSV_PATH)

df["image_file_name"] = df["image_file_name"].str.replace(r"\.jpg$", "", regex=True)

df.to_csv(OUT_PATH, index=False)

print("Done.")
