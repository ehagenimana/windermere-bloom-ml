from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import hashlib
import json

import numpy as np
import pandas as pd

DeterminandId = Union[int, str]


@dataclass(frozen=True)
class FeatureMatrixConfig:
    # Input / columns (aligned with your clean layer)
    datetime_col: str = "phenomenonTime"
    site_col: str = "samplingPoint.notation"
    determinand_col: str = "determinand.notation"
    value_col: str = "result"
    unit_col: str = "unit"

    # Determinand matching
    determinand_cast: str = "str"  # "str" | "int" | "none"

    # Window & target
    window_start_year: int = 2005
    window_end_year: int = 2025  # inclusive
    target_determinand_id: DeterminandId = 7887
    threshold_ugL: float = 20.0

    # Predictors
    predictor_ids: Tuple[DeterminandId, ...] = (348, 9686, 61)  # TP, TN, pH

    # Lookback logic
    lookback_days: int = 30

    # Feature options
    add_missing_flags: bool = True
    add_age_days: bool = True
    add_seasonality: bool = True  # month_sin/cos + doy_sin/cos
    add_days_since_last_chl: bool = True

    # Output / governance
    output_dir: str = "data/features"
    snapshot_id: Optional[str] = None  # used in filename
    feature_version: str = "FEAT_V1"


class FeatureMatrixBuilder:
    """
    Build modelling-ready (X, y) anchored on chlorophyll timestamps.

    Guarantees:
      - merge_asof BACKWARD only (no future leakage)
      - lookback window enforced; stale matches are dropped to NaN
      - deterministic output ordering and config fingerprinting
    """

    def __init__(self, df_clean: pd.DataFrame, config: FeatureMatrixConfig):
        self._df = df_clean.copy(deep=False)
        self.config = config
        self.artifacts: Dict[str, Any] = {}
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        cfg = self.config
        required = {cfg.datetime_col, cfg.site_col, cfg.determinand_col, cfg.value_col, cfg.unit_col}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"df_clean missing required columns: {sorted(missing)}")

        if cfg.determinand_cast not in {"str", "int", "none"}:
            raise ValueError("determinand_cast must be one of: 'str', 'int', 'none'")

        if cfg.window_end_year < cfg.window_start_year:
            raise ValueError("window_end_year must be >= window_start_year")

        if cfg.lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")

    def _normalize_det(self, series: pd.Series, target: DeterminandId) -> Tuple[pd.Series, Any]:
        cast = self.config.determinand_cast
        if cast == "str":
            return series.astype(str), str(target)
        if cast == "int":
            return pd.to_numeric(series, errors="coerce"), int(target)
        return series, target

    def _filter_window(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.copy()
        df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
        df = df.dropna(subset=[cfg.datetime_col])
        df["year"] = df[cfg.datetime_col].dt.year
        return df[(df["year"] >= cfg.window_start_year) & (df["year"] <= cfg.window_end_year)].copy()

    def _make_config_fingerprint(self) -> str:
        payload = json.dumps(asdict(self.config), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_target_anchor(self) -> pd.DataFrame:
        cfg = self.config
        dfw = self._filter_window(self._df)

        det_norm, target_norm = self._normalize_det(dfw[cfg.determinand_col], cfg.target_determinand_id)
        df_t = dfw[det_norm == target_norm].copy()

        df_t[cfg.value_col] = pd.to_numeric(df_t[cfg.value_col], errors="coerce")
        df_t = df_t.dropna(subset=[cfg.value_col])

        df_t = df_t.sort_values([cfg.site_col, cfg.datetime_col]).reset_index(drop=True)

        df_t["y"] = (df_t[cfg.value_col] > cfg.threshold_ugL).astype(int)
        df_t = df_t.rename(columns={cfg.value_col: "chl_ugL"})

        return df_t[[cfg.site_col, cfg.datetime_col, "chl_ugL", "y"]].copy()

    def _extract_predictor_series(self, determinand_id: DeterminandId, feature_name: str) -> pd.DataFrame:
        """
        Return a predictor dataframe already renamed to:
          - predictor timestamp column: f"{feature_name}__time"
          - predictor value column: feature_name
        so that a single merge_asof brings both across.
        """
        cfg = self.config
        dfw = self._filter_window(self._df)

        det_norm, pid_norm = self._normalize_det(dfw[cfg.determinand_col], determinand_id)
        d = dfw[det_norm == pid_norm].copy()

        d[cfg.value_col] = pd.to_numeric(d[cfg.value_col], errors="coerce")
        d = d.dropna(subset=[cfg.value_col])

        d = d[[cfg.site_col, cfg.datetime_col, cfg.value_col]].copy()
        d = d.sort_values([cfg.site_col, cfg.datetime_col]).reset_index(drop=True)

        d = d.rename(
            columns={
                cfg.datetime_col: f"{feature_name}__time",
                cfg.value_col: feature_name,
            }
        )
        return d

    def _merge_asof_backward(self, anchor: pd.DataFrame, pred: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Backward as-of merge that brings in:
          - feature_name
          - feature_name__time
        """
        cfg = self.config

        a = anchor.sort_values([cfg.site_col, cfg.datetime_col]).copy()
        p = pred.sort_values([cfg.site_col, f"{feature_name}__time"]).copy()

        merged = pd.merge_asof(
            a,
            p,
            left_on=cfg.datetime_col,
            right_on=f"{feature_name}__time",
            by=cfg.site_col,
            direction="backward",
            allow_exact_matches=True,
        )

        # Age calculation
        age = (merged[cfg.datetime_col] - merged[f"{feature_name}__time"]).dt.total_seconds() / 86400.0
        if cfg.add_age_days:
            merged[f"{feature_name}__age_days"] = age

        # Lookback enforcement (stale OR no match => NaN)
        stale = age > float(cfg.lookback_days)
        merged.loc[stale, feature_name] = np.nan
        merged.loc[stale, f"{feature_name}__time"] = pd.NaT
        if cfg.add_age_days:
            merged.loc[stale, f"{feature_name}__age_days"] = np.nan

        # Missingness flags
        if cfg.add_missing_flags:
            merged[f"{feature_name}__is_missing"] = merged[feature_name].isna().astype(int)

        return merged

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = df.copy()
        t = pd.to_datetime(out[cfg.datetime_col], errors="coerce")

        if cfg.add_seasonality:
            month = t.dt.month.astype(float)
            doy = t.dt.dayofyear.astype(float)

            out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
            out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
            out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
            out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        if cfg.add_days_since_last_chl:
            out = out.sort_values([cfg.site_col, cfg.datetime_col]).copy()
            out["days_since_last_chl"] = (
                out.groupby(cfg.site_col)[cfg.datetime_col].diff().dt.total_seconds().div(86400.0)
            )

        return out

    def build(self) -> pd.DataFrame:
        cfg = self.config
        fingerprint = self._make_config_fingerprint()

        anchor = self._build_target_anchor()
        df_feat = anchor.copy()

        # Merge predictors
        for pid in cfg.predictor_ids:
            feature_name = f"det_{pid}"
            pred = self._extract_predictor_series(pid, feature_name=feature_name)
            df_feat = self._merge_asof_backward(df_feat, pred, feature_name=feature_name)

        # Add time-derived features
        df_feat = self._add_time_features(df_feat)

        # Deterministic ordering
        df_feat = df_feat.sort_values([cfg.site_col, cfg.datetime_col]).reset_index(drop=True)

        # Governance columns
        df_feat["feature_version"] = cfg.feature_version
        df_feat["feature_config_fingerprint"] = fingerprint
        df_feat["snapshot_id"] = cfg.snapshot_id

        self.artifacts["feature_config_fingerprint"] = fingerprint
        self.artifacts["n_rows"] = int(len(df_feat))
        self.artifacts["n_pos"] = int(df_feat["y"].sum())
        self.artifacts["pos_rate"] = float(df_feat["y"].mean()) if len(df_feat) else float("nan")

        return df_feat

    def save(self, df_feat: pd.DataFrame) -> Path:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        snap = cfg.snapshot_id or "unknown_snapshot"
        fname = f"features_{snap}_lb{cfg.lookback_days}_{cfg.feature_version}.parquet"
        out_path = out_dir / fname

        df_feat.to_parquet(out_path, index=False)

        # Save JSON sidecar
        sidecar = out_dir / f"{fname}.meta.json"
        meta = {
            "snapshot_id": cfg.snapshot_id,
            "feature_version": cfg.feature_version,
            "lookback_days": cfg.lookback_days,
            "predictor_ids": [str(x) for x in cfg.predictor_ids],
            "target_determinand_id": str(cfg.target_determinand_id),
            "threshold_ugL": cfg.threshold_ugL,
            "window": f"{cfg.window_start_year}-{cfg.window_end_year}",
            "feature_config_fingerprint": self.artifacts.get("feature_config_fingerprint"),
            "n_rows": self.artifacts.get("n_rows"),
            "n_pos": self.artifacts.get("n_pos"),
            "pos_rate": self.artifacts.get("pos_rate"),
        }
        sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return out_path


def build_feature_matrix(df_clean: pd.DataFrame, config: Optional[FeatureMatrixConfig] = None) -> pd.DataFrame:
    cfg = config or FeatureMatrixConfig()
    builder = FeatureMatrixBuilder(df_clean=df_clean, config=cfg)
    return builder.build()

