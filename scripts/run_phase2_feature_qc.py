import pandas as pd

PATH = "data/features/features_raw_NW-88010013_ALLFULL_20260216T055951Z_lb30_FEAT_V1.parquet"
df = pd.read_parquet(PATH)

cols = ["det_348", "det_9686", "det_61"]
missing = df[cols].isna().mean().sort_values(ascending=False)
print("Missingness rate after lookback:")
print(missing)

print("\nMissing-flag rates (if present):")
flag_cols = [c for c in df.columns if c.endswith("__is_missing")]
print(df[flag_cols].mean().sort_values(ascending=False))

print("\nAge-days summary (if present):")
age_cols = [c for c in df.columns if c.endswith("__age_days")]
print(df[age_cols].describe().T[["mean","50%","max"]])
