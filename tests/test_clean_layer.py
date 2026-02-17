from __future__ import annotations

import pandas as pd
import pytest

# Adjust these imports after you create the module files.
# Recommended locations:
#   src/windermere_project/clean/builder.py  -> CleanDatasetBuilder
#   src/windermere_project/clean/config.py   -> CleanConfig
from windermere_project.clean.builder import CleanDatasetBuilder
from windermere_project.clean.config import CleanConfig


# ---- Test fixtures ----

@pytest.fixture()
def base_config() -> CleanConfig:
    # We keep these as simple names for the synthetic test dataframe.
    # When you integrate real EA snapshot schema, update these mappings in config
    # (not in the core logic).
    return CleanConfig(
        determinand_ids=(100, 200, 300),  # allowlist
        datetime_col="dt",
        site_col="site",
        determinand_col="det_id",
        unit_col="unit",
        value_col="value",
        assume_tz="UTC",
        output_tz="UTC",
        coerce_numeric_errors="raise",
        drop_non_numeric=True,
        drop_negative=True,
        # deterministic ordering + dedupe keys (event-level)
        sort_keys=("dt", "site", "det_id"),
        dedupe_keys=("dt", "site", "det_id"),
        dedupe_keep="last",
        snapshot_id="SNAP_TEST",
        clean_version="CLEAN_V1",
        require_units=True,
        allowed_units=None,
        censored_value_col=None,
        min_max_by_determinand=None,
        keep_cols_extra=(),
        output_long=True,
    )


@pytest.fixture()
def df_raw() -> pd.DataFrame:
    # Synthetic raw-like event-level table (long format)
    # Includes:
    # - an irrelevant determinand (999) that should be filtered out
    # - a non-numeric value that should raise or be dropped depending on config
    # - a negative value to test validity rules
    # - a duplicate key to test dedupe policy
    return pd.DataFrame(
        {
            "dt": [
                "2025-06-01T10:00:00",
                "2025-06-01T10:00:00",  # duplicate key (same dt, site, det_id)
                "2025-06-02T10:00:00",
                "2025-06-03T10:00:00",
                "2025-06-04T10:00:00",
            ],
            "site": ["A", "A", "A", "B", "B"],
            "det_id": [100, 100, 200, 300, 999],  # 999 should be filtered out
            "unit": ["mg/L", "mg/L", "mg/L", "mg/L", "mg/L"],
            "value": ["1.0", "2.0", "x", "-0.5", "5.0"],  # "x" non-numeric, -0.5 negative
        }
    )


# ---- Tests ----

def test_filter_determinands_is_deterministic(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    builder = CleanDatasetBuilder(base_config)
    df_filt = builder.filter_determinands(df_raw)

    assert set(df_filt[base_config.determinand_col].unique()).issubset(set(base_config.determinand_ids))
    assert 999 not in set(df_filt[base_config.determinand_col].unique())


def test_parse_datetime_produces_timezone_aware_utc(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    builder = CleanDatasetBuilder(base_config)
    df = builder.filter_determinands(df_raw)
    df = builder.parse_datetime(df)

    col = base_config.datetime_col
    assert pd.api.types.is_datetime64_any_dtype(df[col])
    # Must be tz-aware and in UTC (or output_tz if you choose different)
    assert getattr(df[col].dt, "tz", None) is not None
    assert str(df[col].dt.tz) == base_config.output_tz


def test_numeric_coercion_raises_when_configured(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    # With coerce_numeric_errors="raise", the "x" value should raise.
    builder = CleanDatasetBuilder(base_config)
    df = builder.filter_determinands(df_raw)
    df = builder.parse_datetime(df)

    with pytest.raises(Exception):
        builder.coerce_numeric(df)


def test_numeric_coercion_can_drop_non_numeric_when_coerce_enabled(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    # When configured to coerce, non-numeric becomes NaN, then dropped if drop_non_numeric=True.
    cfg = base_config.__class__(**{**base_config.__dict__, "coerce_numeric_errors": "coerce"})
    builder = CleanDatasetBuilder(cfg)

    df = builder.filter_determinands(df_raw)
    df = builder.parse_datetime(df)
    df2, dropped = builder.coerce_numeric(df)

    assert dropped >= 1
    assert df2[cfg.value_col].isna().sum() == 0
    assert pd.api.types.is_float_dtype(df2[cfg.value_col]) or pd.api.types.is_numeric_dtype(df2[cfg.value_col])


def test_validity_rules_drop_negatives_when_enabled(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    cfg = base_config.__class__(**{**base_config.__dict__, "coerce_numeric_errors": "coerce"})
    builder = CleanDatasetBuilder(cfg)

    df = builder.filter_determinands(df_raw)
    df = builder.parse_datetime(df)
    df, _ = builder.coerce_numeric(df)

    df2, dropped_invalid = builder.apply_validity_rules(df)

    assert dropped_invalid >= 1
    assert (df2[cfg.value_col] < 0).sum() == 0


def test_dedupe_and_sort_is_deterministic(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    cfg = base_config.__class__(**{**base_config.__dict__, "coerce_numeric_errors": "coerce"})
    builder = CleanDatasetBuilder(cfg)

    df = builder.filter_determinands(df_raw)
    df = builder.parse_datetime(df)
    df, _ = builder.coerce_numeric(df)
    df, _ = builder.apply_validity_rules(df)
    df_final = builder.dedupe_and_sort(df)

    # duplicates removed: only one row should remain for (dt, site, det_id) = (2025-06-01 10:00, A, 100)
    key_cols = list(cfg.dedupe_keys)
    assert df_final.duplicated(subset=key_cols).sum() == 0

    # sorted order must be consistent
    assert df_final[key_cols].equals(df_final.sort_values(list(cfg.sort_keys))[key_cols])


def test_clean_dataframe_is_pure_and_repeatable(base_config: CleanConfig, df_raw: pd.DataFrame) -> None:
    # End-to-end clean_dataframe should produce identical output on repeated calls
    cfg = base_config.__class__(**{**base_config.__dict__, "coerce_numeric_errors": "coerce"})
    builder = CleanDatasetBuilder(cfg)

    df1, rep1 = builder.clean_dataframe(df_raw)
    df2, rep2 = builder.clean_dataframe(df_raw)

    pd.testing.assert_frame_equal(df1, df2, check_like=False)
    assert rep1["config_fingerprint"] == rep2["config_fingerprint"]
    assert rep1["rows"]["in"] == rep2["rows"]["in"]
    assert rep1["rows"]["out"] == rep2["rows"]["out"]

