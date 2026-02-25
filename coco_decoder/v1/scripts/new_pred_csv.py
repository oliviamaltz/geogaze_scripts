import pandas as pd
import re

IN_CSV  = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/geogaze_model_predictions.csv"
OUT_CSV = "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/geogaze_model_predictions_identification.csv"

df = pd.read_csv(IN_CSV)

def convert_row_to_cornet(row):
    # 1) Fix model name
    model = row["model"]
    if not model.startswith("cornetIDEN_"):
        model = "cornetIDEN_" + model

    # 2) Fix prediction filename
    # Example input:
    # test_bc_bs_LR__model=resnet50.a1_in1k_maskL_bc_bs_best_mask.png
    pred = row["prediction"]

    m = re.search(r"(test_[a-z]{2}_[a-z]{2}_(?:LR|UD))", pred)
    if m:
        pred = m.group(1) + "_mask.png"
    else:
        raise ValueError(f"Could not parse prediction filename: {pred}")

    return model, pred

df["model"], df["prediction"] = zip(*df.apply(convert_row_to_cornet, axis=1))

df.to_csv(OUT_CSV, index=False)
print("Saved new cornet CSV to:")
print(OUT_CSV)
