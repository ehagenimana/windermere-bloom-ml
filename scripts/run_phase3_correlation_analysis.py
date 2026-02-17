from pathlib import Path
import pandas as pd
import numpy as np

FEATURE_PATH = "data/features/features_raw_NW-88010013_ALLFULL_20260216T055951Z_lb30_FEAT_V1.parquet"

df = pd.read_parquet(FEATURE_PATH)

df["phenomenonTime"] = pd.to_datetime(df["phenomenonTime"], errors="coerce")
df["month"] = df["phenomenonTime"].dt.month

# Ensure numeric
df["chl_ugL"] = pd.to_numeric(df["chl_ugL"], errors="coerce")
df["det_348"] = pd.to_numeric(df["det_348"], errors="coerce")   # TP
df["det_9686"] = pd.to_numeric(df["det_9686"], errors="coerce") # TN
df["det_61"] = pd.to_numeric(df["det_61"], errors="coerce")     # pH

def corr_block(data, label):
    print(f"\n=== {label} ===")
    for col, name in [("det_348", "TP"),
                      ("det_9686", "TN"),
                      ("det_61", "pH")]:
        sub = data[["chl_ugL", col]].dropna()
        if len(sub) > 5:
            corr = sub["chl_ugL"].corr(sub[col])
            print(f"{name} correlation: {corr:.3f} (n={len(sub)})")
        else:
            print(f"{name} correlation: insufficient data (n={len(sub)})")

# Overall
corr_block(df, "Overall")

# Summer (Jun–Sep)
summer = df[df["month"].isin([6,7,8,9])]
corr_block(summer, "Summer (Jun–Sep)")

# Non-summer
nonsummer = df[~df["month"].isin([6,7,8,9])]
corr_block(nonsummer, "Non-Summer (Oct–May)")
