import pandas as pd

from windermere_project.features.feature_builder import FeatureBuilder, FeatureConfig

SRC = "data/clean/wide_features_base.parquet"
OUT = "data/features/features_v1.parquet"

def main():
    df = pd.read_parquet(SRC).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)

    cfg = FeatureConfig(
        chl_threshold=20.0,
        chl_col="chl",
        tn_col="tn",
        tp_col="tp",
        chl_lags=[1, 7],
        tn_lags=[7],
        tp_lags=[7],
        chl_rolls=[7, 30],
        tn_rolls=[30],
        tp_rolls=[30],
    )

    fb = FeatureBuilder(cfg)
    X = fb.build(df)

    # Optional: include raw context columns alongside engineered features
    X = X.join(df[["tp", "tn", "ph", "temp", "chl"]], how="left", rsuffix="_raw")

    # ensure output dir exists
    OUT_PATH = OUT
    X.to_parquet(OUT_PATH)
    print("Saved:", OUT_PATH, "shape:", X.shape)
    print(X.head(10).to_string())

if __name__ == "__main__":
    main()
