from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    chl_threshold: float
    chl_col: str
    tn_col: Optional[str]
    tp_col: Optional[str]
    chl_lags: List[int]
    tn_lags: List[int]
    tp_lags: List[int]
    chl_rolls: List[int]
    tn_rolls: List[int]
    tp_rolls: List[int]
    add_missingness_flags: bool = True


class FeatureBuilder:
    """
    Leakage-safe feature builder.

    Key rule: any rolling statistic must be computed on a shifted series:
      x.shift(1).rolling(...)

    This ensures features at time t do not use x(t) or any future values.
    """

    def __init__(self, config: FeatureConfig):
        self.cfg = config

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("FeatureBuilder requires a DatetimeIndex. Set/convert before calling build().")

        df = df.sort_index().copy()

        # Validate required column exists
        if self.cfg.chl_col not in df.columns:
            raise KeyError(f"Missing chlorophyll column '{self.cfg.chl_col}' in dataframe.")

        out = pd.DataFrame(index=df.index)

        # 1) Temporal persistence (chl)
        out = out.join(self._lag_features(df, self.cfg.chl_col, self.cfg.chl_lags, prefix="chl_lag"))
        out = out.join(self._rolling_mean_features(df, self.cfg.chl_col, self.cfg.chl_rolls, prefix="chl_roll_mean"))

        # Previous exceedance flag
        out["prev_exceed_flag"] = (df[self.cfg.chl_col].shift(1) > self.cfg.chl_threshold).astype("Int64")

        # Days since previous sample
        out["days_since_prev"] = (df.index.to_series().diff().dt.total_seconds() / (24 * 3600)).astype(float)

        # 2) Seasonality
        out = out.join(self._seasonality_features(df.index))

        # 3) Nutrients (optional)
        if self.cfg.tn_col and self.cfg.tn_col in df.columns:
            out = out.join(self._lag_features(df, self.cfg.tn_col, self.cfg.tn_lags, prefix="tn_lag"))
            out = out.join(self._rolling_mean_features(df, self.cfg.tn_col, self.cfg.tn_rolls, prefix="tn_roll_mean"))

        if self.cfg.tp_col and self.cfg.tp_col in df.columns:
            out = out.join(self._lag_features(df, self.cfg.tp_col, self.cfg.tp_lags, prefix="tp_lag"))
            out = out.join(self._rolling_mean_features(df, self.cfg.tp_col, self.cfg.tp_rolls, prefix="tp_roll_mean"))

        # 4) Missingness flags
        if self.cfg.add_missingness_flags:
            for col in [self.cfg.chl_col, self.cfg.tn_col, self.cfg.tp_col]:
                if col and col in df.columns:
                    out[f"miss_{col}"] = df[col].isna().astype("int8")

        return out

    @staticmethod
    def _lag_features(df: pd.DataFrame, col: str, lags: List[int], prefix: str) -> pd.DataFrame:
        out = {}
        s = pd.to_numeric(df[col], errors="coerce")
        for k in lags:
            out[f"{prefix}_{k}"] = s.shift(k)
        return pd.DataFrame(out, index=df.index)

    @staticmethod
    def _rolling_mean_features(df: pd.DataFrame, col: str, windows: List[int], prefix: str) -> pd.DataFrame:
        out = {}
        s = pd.to_numeric(df[col], errors="coerce")

        # CRITICAL: shift before rolling => no leakage
        s_shifted = s.shift(1)

        for w in windows:
            out[f"{prefix}_{w}"] = s_shifted.rolling(window=w, min_periods=1).mean()
        return pd.DataFrame(out, index=df.index)

    @staticmethod
    def _seasonality_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
        doy = idx.dayofyear.astype(float)
        ang = 2.0 * np.pi * doy / 365.25
        out = pd.DataFrame(index=idx)
        out["month"] = idx.month.astype(int)
        out["doy_sin"] = np.sin(ang)
        out["doy_cos"] = np.cos(ang)
        return out
