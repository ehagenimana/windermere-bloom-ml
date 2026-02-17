import pandas as pd

from windermere_project.features.build_feature_matrix import (
    FeatureMatrixBuilder,
    FeatureMatrixConfig,
)


def test_feature_matrix_rows_equal_chl_anchors():
    cfg = FeatureMatrixConfig(
        datetime_col="phenomenonTime",
        site_col="samplingPoint.notation",
        determinand_col="determinand.notation",
        value_col="result",
        unit_col="unit",
        determinand_cast="str",
        target_determinand_id="7887",
        predictor_ids=("348",),
        lookback_days=30,
        snapshot_id="test_snap",
    )

    df = pd.DataFrame(
        {
            "samplingPoint.notation": ["A", "A", "A"],
            "phenomenonTime": ["2005-01-01", "2005-01-10", "2005-01-20"],
            "determinand.notation": ["7887", "348", "7887"],
            "result": [10.0, 1.0, 30.0],
            "unit": ["ug/L", "mg/L", "ug/L"],
            "_det_id_norm": ["7887", "348", "7887"],
        }
    )

    builder = FeatureMatrixBuilder(df_clean=df, config=cfg)
    feat = builder.build()

    # 2 chl anchors -> 2 rows
    assert len(feat) == 2
    assert set(feat["y"].tolist()) == {0, 1}


def test_lookback_enforced_stale_predictor_dropped():
    cfg = FeatureMatrixConfig(
        datetime_col="phenomenonTime",
        site_col="samplingPoint.notation",
        determinand_col="determinand.notation",
        value_col="result",
        unit_col="unit",
        determinand_cast="str",
        target_determinand_id="7887",
        predictor_ids=("348",),
        lookback_days=5,
        snapshot_id="test_snap",
        add_missing_flags=True,
    )

    df = pd.DataFrame(
        {
            "samplingPoint.notation": ["A", "A"],
            "phenomenonTime": ["2005-01-01", "2005-01-10"],
            "determinand.notation": ["348", "7887"],
            "result": [1.0, 25.0],
            "unit": ["mg/L", "ug/L"],
            "_det_id_norm": ["348", "7887"],
        }
    )

    builder = FeatureMatrixBuilder(df_clean=df, config=cfg)
    feat = builder.build()

    # predictor at Jan-01 is 9 days before Jan-10, but lookback is 5 => should be NaN
    assert pd.isna(feat.loc[0, "det_348"])
    assert feat.loc[0, "det_348__is_missing"] == 1

