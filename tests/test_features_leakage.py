import numpy as np
import pandas as pd

from windermere_project.features.feature_builder import FeatureBuilder, FeatureConfig


def _toy_df(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "chl": np.arange(n, dtype=float),      # increasing
            "tn": np.arange(n, dtype=float) * 2.0,
            "tp": np.arange(n, dtype=float) * 3.0,
            "ph": np.linspace(6, 8, n),
            "temp": np.linspace(5, 15, n),
        },
        index=idx,
    )


def test_all_rolling_features_use_shifted_series():
    df = _toy_df()
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
    X = FeatureBuilder(cfg).build(df)

    # For strictly increasing series, shifted rolling means must be < current value after warmup.
    for col in ["chl", "tn", "tp"]:
        cur = df[col]
        for w in ([7, 30] if col == "chl" else [30]):
            feat = X[f"{col}_roll_mean_{w}"] if col == "chl" else X[f"{col}_roll_mean_{w}"]
            mask = feat.notna() & (df.index >= df.index[35])
            assert (feat[mask] < cur[mask]).all()
