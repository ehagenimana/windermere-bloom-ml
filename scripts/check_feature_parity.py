import pandas as pd

from windermere_project.features.feature_builder import FeatureBuilder, FeatureConfig


def main():
    df = pd.read_parquet("data/clean/wide_features_base.parquet").sort_index()
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

    train = df.loc["2005-01-01":"2018-12-31"]
    test = df.loc["2019-01-01":"2025-12-31"]

    X_train = fb.build(train)
    X_test = fb.build(test)

    print("Train features:", X_train.shape, "Test features:", X_test.shape)
    print("Train NA rate (mean):", X_train.isna().mean().mean())
    print("Test  NA rate (mean):", X_test.isna().mean().mean())

    # Key parity sanity checks
    assert set(X_train.columns) == set(X_test.columns)
    assert X_train.columns.equals(X_test.columns)


if __name__ == "__main__":
    main()
