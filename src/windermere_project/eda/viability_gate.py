from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

DeterminandId = Union[int, str]


@dataclass(frozen=True)
class ViabilityGateConfig:
    # Target definition
    target_determinand_id: DeterminandId = 7887
    threshold_ugL: float = 20.0

    # ✅ Defaults aligned with EA clean-layer columns (overrideable)
    datetime_col: str = "phenomenonTime"
    value_col: str = "result"
    determinand_col: str = "determinand.notation"

    # ✅ Determinand comparison policy for handling "7887" vs 7887
    # - "str": compare everything as string (recommended for EA notation fields)
    # - "int": compare everything as int (recommended if you store numeric IDs)
    # - "none": compare as-is (only if you are sure types match)
    determinand_cast: str = "str"  # "str" | "int" | "none"

    # Modelling window
    window_start_year: int = 2005
    window_end_year: int = 2025  # inclusive

    # Autocorrelation
    acf_nlags: int = 10

    # Output
    output_dir: str = "reports/phase2"
    figures_dirname: str = "figures"

    # Plot options
    y_scale_log: bool = True  # for chl seasonal plot


class ViabilityGate:
    """
    Phase 2 scientific viability gate.

    Purpose:
      - deterministic, auditable checks before feature engineering/modelling
      - produces structured outputs + saved figures

    This is NOT a generic EDA module.
    """

    def __init__(self, df_clean: pd.DataFrame, config: ViabilityGateConfig):
        self._df = df_clean.copy(deep=False)  # avoid mutation; no deep copy cost
        self.config = config
        self.results: Dict[str, Any] = {}

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        required = {self.config.datetime_col, self.config.value_col, self.config.determinand_col}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"df_clean missing required columns: {sorted(missing)}")

        if self.config.determinand_cast not in {"str", "int", "none"}:
            raise ValueError(
                "ViabilityGateConfig.determinand_cast must be one of: 'str', 'int', 'none'"
            )

    def _normalize_determinand_series(
        self, series: pd.Series, target: DeterminandId
    ) -> Tuple[pd.Series, Any]:
        """
        Normalizes determinand series + target to a comparable representation.
        Returns (normalized_series, normalized_target).
        """
        cast = self.config.determinand_cast

        if cast == "str":
            return series.astype(str), str(target)

        if cast == "int":
            # Convert to numeric, NaNs may appear if non-numeric; comparisons will be False there
            return pd.to_numeric(series, errors="coerce"), int(target)

        # cast == "none"
        return series, target

    def _prepare_target_series(self) -> pd.DataFrame:
        cfg = self.config

        det_series = self._df[cfg.determinand_col]
        det_norm, target_norm = self._normalize_determinand_series(det_series, cfg.target_determinand_id)

        df = self._df[det_norm == target_norm].copy()

        df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
        df = df.dropna(subset=[cfg.datetime_col, cfg.value_col])

        df["year"] = df[cfg.datetime_col].dt.year
        df = df[(df["year"] >= cfg.window_start_year) & (df["year"] <= cfg.window_end_year)]

        # enforce numeric
        df[cfg.value_col] = pd.to_numeric(df[cfg.value_col], errors="coerce")
        df = df.dropna(subset=[cfg.value_col])

        df = df.sort_values(cfg.datetime_col).reset_index(drop=True)

        # target label
        df["y_exceed"] = (df[cfg.value_col] > cfg.threshold_ugL).astype(int)
        df["month"] = df[cfg.datetime_col].dt.month

        return df

    def compute_class_balance(self) -> Dict[str, Any]:
        df_t = self._prepare_target_series()

        n_total = int(len(df_t))
        n_pos = int(df_t["y_exceed"].sum())
        n_neg = int(n_total - n_pos)
        pos_rate = float(n_pos / n_total) if n_total else float("nan")

        out = {
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "threshold_ugL": self.config.threshold_ugL,
            "target_determinand_id": self.config.target_determinand_id,
            "window": f"{self.config.window_start_year}-{self.config.window_end_year}",
            "determinand_cast": self.config.determinand_cast,
        }
        self.results["class_balance"] = out
        return out

    def assess_sampling_frequency_by_year(self) -> Dict[str, Any]:
        df_t = self._prepare_target_series()
        counts = df_t.groupby("year").size()

        # basic gap detection
        years = list(range(self.config.window_start_year, self.config.window_end_year + 1))
        missing_years = [y for y in years if y not in counts.index]

        out = {
            "counts_by_year": {int(k): int(v) for k, v in counts.to_dict().items()},
            "missing_years": missing_years,
            "min_per_year": int(counts.min()) if len(counts) else 0,
            "median_per_year": float(counts.median()) if len(counts) else float("nan"),
            "max_per_year": int(counts.max()) if len(counts) else 0,
        }
        self.results["sampling_frequency"] = out
        return out

    def assess_temporal_persistence_acf(self) -> Dict[str, Any]:
        df_t = self._prepare_target_series()
        series = df_t[self.config.value_col].astype(float).values

        acf_vals: Optional[np.ndarray] = None
        used_statsmodels = False

        try:
            from statsmodels.tsa.stattools import acf  # type: ignore

            # Using fft=False for stability on small/irregular series; missing handled earlier
            acf_vals = acf(series, nlags=self.config.acf_nlags, fft=False)
            used_statsmodels = True
        except Exception:
            acf_vals = None

        lag1 = float(np.corrcoef(series[1:], series[:-1])[0, 1]) if len(series) > 2 else float("nan")

        out = {
            "lag1_corr": lag1,
            "acf_nlags": self.config.acf_nlags,
            "acf_values": acf_vals.tolist() if acf_vals is not None else None,
            "statsmodels_used": used_statsmodels,
            "n_points": int(len(series)),
        }
        self.results["temporal_persistence"] = out
        return out

    def plot_seasonality(self) -> Dict[str, Any]:
        """
        Saves a seasonality plot (chl distribution by month).
        Deterministic output location and filename.
        """
        df_t = self._prepare_target_series()
        _, fig_dir = self._ensure_output_dirs()

        import matplotlib.pyplot as plt

        fig_path = fig_dir / "chl_by_month_boxplot.png"

        plt.figure()
        df_t.boxplot(column=self.config.value_col, by="month")
        if self.config.y_scale_log:
            plt.yscale("log")
        plt.title("Chlorophyll-a by Month")
        plt.suptitle("")
        plt.xlabel("Month")
        plt.ylabel("Chlorophyll-a (µg/L)")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        out = {"figure": str(fig_path)}
        self.results["seasonality"] = out
        return out

    def plot_sampling_frequency(self) -> Dict[str, Any]:
        df_t = self._prepare_target_series()
        _, fig_dir = self._ensure_output_dirs()

        import matplotlib.pyplot as plt

        fig_path = fig_dir / "chl_samples_by_year.png"
        counts = df_t.groupby("year").size()

        plt.figure()
        plt.plot(counts.index.values, counts.values, marker="o")
        plt.title("Chlorophyll-a Sampling Frequency by Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        out = {"figure": str(fig_path)}
        self.results["sampling_frequency_plot"] = out
        return out

    def missingness_profile_for_predictors(self, predictor_ids: list[DeterminandId]) -> Dict[str, Any]:
        """
        Simple availability counts in the modelling window (raw availability, not after lookback merge).
        """
        cfg = self.config
        df = self._df.copy(deep=False)

        df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
        df = df.dropna(subset=[cfg.datetime_col])

        df["year"] = df[cfg.datetime_col].dt.year
        df = df[(df["year"] >= cfg.window_start_year) & (df["year"] <= cfg.window_end_year)]

        det_series = df[cfg.determinand_col]

        out: Dict[str, Any] = {}
        for pid in predictor_ids:
            det_norm, pid_norm = self._normalize_determinand_series(det_series, pid)
            n = int((det_norm == pid_norm).sum())
            out[str(pid)] = {"n_observations_window": n}

        self.results["predictor_availability"] = out
        return out

    def run_all(self, predictor_ids: Optional[list[DeterminandId]] = None) -> Dict[str, Any]:
        self.results["config"] = asdict(self.config)

        self.compute_class_balance()
        self.assess_sampling_frequency_by_year()
        self.assess_temporal_persistence_acf()
        self.plot_seasonality()
        self.plot_sampling_frequency()

        if predictor_ids:
            self.missingness_profile_for_predictors(predictor_ids)

        self._write_outputs()
        return self.results

    def _ensure_output_dirs(self) -> Tuple[Path, Path]:
        out_dir = Path(self.config.output_dir)
        fig_dir = out_dir / self.config.figures_dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, fig_dir

    def _write_outputs(self) -> None:
        out_dir, _ = self._ensure_output_dirs()

        # JSON summary
        summary_path = out_dir / "viability_summary.json"
        pd.Series([self.results]).to_json(summary_path, orient="records", indent=2)

        # Markdown report (simple + deterministic)
        md_path = out_dir / "eda_report.md"
        cb = self.results.get("class_balance", {})
        sf = self.results.get("sampling_frequency", {})
        tp = self.results.get("temporal_persistence", {})

        lines = []
        lines.append("# Phase 2 — Viability Gate Report\n")
        lines.append("## Config\n")
        lines.append("```json\n")
        lines.append(pd.Series([self.results.get("config", {})]).to_json(orient="records", indent=2))
        lines.append("\n```\n")

        lines.append("## Class Balance\n")
        lines.append(f"- n_total: {cb.get('n_total')}\n")
        lines.append(f"- n_pos: {cb.get('n_pos')}\n")
        lines.append(f"- pos_rate: {cb.get('pos_rate')}\n")
        lines.append(f"- determinand_cast: {cb.get('determinand_cast')}\n")

        lines.append("\n## Sampling Frequency\n")
        lines.append(f"- missing_years: {sf.get('missing_years')}\n")
        lines.append(
            f"- min/median/max per year: "
            f"{sf.get('min_per_year')}/{sf.get('median_per_year')}/{sf.get('max_per_year')}\n"
        )

        lines.append("\n## Temporal Persistence\n")
        lines.append(f"- lag1_corr: {tp.get('lag1_corr')}\n")
        lines.append(f"- statsmodels_used: {tp.get('statsmodels_used')}\n")

        lines.append("\n## Figures\n")
        season_fig = self.results.get("seasonality", {}).get("figure")
        freq_fig = self.results.get("sampling_frequency_plot", {}).get("figure")
        if season_fig:
            lines.append(f"- {season_fig}\n")
        if freq_fig:
            lines.append(f"- {freq_fig}\n")

        md_path.write_text("".join(lines), encoding="utf-8")


def run_viability_gate(
    df_clean: pd.DataFrame,
    config: Optional[ViabilityGateConfig] = None,
    predictor_ids: Optional[list[DeterminandId]] = None,
) -> Dict[str, Any]:
    cfg = config or ViabilityGateConfig()
    gate = ViabilityGate(df_clean=df_clean, config=cfg)
    return gate.run_all(predictor_ids=predictor_ids)

