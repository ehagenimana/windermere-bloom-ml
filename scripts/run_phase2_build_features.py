import pandas as pd

from windermere_project.features.build_feature_matrix import (
    FeatureMatrixBuilder,
    FeatureMatrixConfig,
)

CLEAN_PATH = "data/clean/clean_raw_NW-88010013_ALLFULL_20260216T055951Z.parquet"

df_clean = pd.read_parquet(CLEAN_PATH)

cfg = FeatureMatrixConfig(
    snapshot_id="raw_NW-88010013_ALLFULL_20260216T055951Z",
    lookback_days=30,
    predictor_ids=("348", "9686", "61"),  # TP, TN, pH as strings for EA-style determinand
    determinand_cast="str",
)

builder = FeatureMatrixBuilder(df_clean=df_clean, config=cfg)
df_feat = builder.build()
out_path = builder.save(df_feat)

print("Saved:", out_path)
print(df_feat.head())
print("Rows:", len(df_feat), "Pos:", df_feat["y"].sum(), "PosRate:", df_feat["y"].mean())

