from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import hashlib
import json
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    # Where to find chlorophyll values in a dataframe (features or clean-long)
    chl_value_col: str = "chl_ugL"   # in feature matrix
    threshold_ugL: float = 20.0

    # Output label column name
    label_col: str = "y"

    # Optional: enforce strict definition ">"
    strictly_greater: bool = True

    # Optional: used for governance/fingerprinting
    label_version: str = "LBL_V1"


class LabelBuilder:
    """
    Governing component for target construction.

    Phase 3 requirement: target definition must be standalone + unit-tested,
    not buried inside feature engineering.
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        self.config = config or LabelConfig()

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self.config), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build_from_feature_matrix(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if cfg.chl_value_col not in df_feat.columns:
            raise ValueError(f"Missing chlorophyll column '{cfg.chl_value_col}' in df_feat")

        out = df_feat.copy()
        chl = pd.to_numeric(out[cfg.chl_value_col], errors="coerce")

        if cfg.strictly_greater:
            out[cfg.label_col] = (chl > cfg.threshold_ugL).astype(int)
        else:
            out[cfg.label_col] = (chl >= cfg.threshold_ugL).astype(int)

        out["label_version"] = cfg.label_version
        out["label_config_fingerprint"] = self.fingerprint()
        return out

    def audit_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.config
        if cfg.label_col not in df.columns:
            raise ValueError(f"Missing label column '{cfg.label_col}'")

        n = int(len(df))
        n_pos = int(df[cfg.label_col].sum()) if n else 0
        pos_rate = float(n_pos / n) if n else float("nan")

        return {
            "n": n,
            "n_pos": n_pos,
            "pos_rate": pos_rate,
            "threshold_ugL": cfg.threshold_ugL,
            "strictly_greater": cfg.strictly_greater,
            "label_version": cfg.label_version,
            "label_config_fingerprint": self.fingerprint(),
        }
