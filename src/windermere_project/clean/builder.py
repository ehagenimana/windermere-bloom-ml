from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd


@dataclass(frozen=True)
class CleanConfig:
    # What to keep
    determinand_ids: tuple[int, ...]
    require_units: bool = True
    allowed_units: Optional[dict[int, tuple[str, ...]]] = None

    # Time handling
    datetime_col: str = "sample_datetime"
    assume_tz: str = "UTC"
    output_tz: str = "UTC"

    # Numeric handling
    value_col: str = "result"
    censored_value_col: Optional[str] = None
    coerce_numeric_errors: str = "raise"  # "raise" | "coerce"
    drop_non_numeric: bool = True

    # Data quality rules (clean layer only)
    drop_negative: bool = True
    min_max_by_determinand: Optional[dict[int, tuple[float, float]]] = None

    # Identity + provenance columns
    site_col: str = "sampling_point"
    determinand_col: str = "determinand_id"
    unit_col: str = "unit"
    keep_cols_extra: tuple[str, ...] = ()

    # Output format
    output_long: bool = True
    sort_keys: tuple[str, ...] = ("sample_datetime", "sampling_point", "determinand_id")
    dedupe_keys: tuple[str, ...] = ("sample_datetime", "sampling_point", "determinand_id")
    dedupe_keep: str = "last"  # "first" | "last"

    # Governance
    snapshot_id: Optional[str] = None
    clean_version: str = "CLEAN_V1"


@dataclass(frozen=True)
class CleanBuildResult:
    clean_path: Path
    report_path: Path
    n_rows_in: int
    n_rows_out: int
    n_dropped_invalid: int
    n_dropped_non_numeric: int
    n_dropped_missing_unit: int
    unit_normalisations: Dict[str, int]
    config_fingerprint: str


class CleanDatasetBuilder:
    """
    Build harmonised, typed, filtered event-level clean dataset from a single raw snapshot.
    Clean layer does NOT aggregate/pivot; it only standardises structure, types, and basic QA rules.
    """

    def __init__(self, config: CleanConfig):
        self.config = config

    def build(
        self,
        raw_snapshot_path: Path,
        output_dir: Path,
        *,
        run_id: Optional[str] = None,
    ) -> CleanBuildResult:
        df_raw = self.load_raw(raw_snapshot_path)
        df_clean, report = self.clean_dataframe(df_raw)

        clean_path = self.persist(df_clean, output_dir, run_id=run_id)
        report_path = self.persist_report(report, output_dir, run_id=run_id)

        return CleanBuildResult(
            clean_path=clean_path,
            report_path=report_path,
            n_rows_in=int(len(df_raw)),
            n_rows_out=int(len(df_clean)),
            n_dropped_invalid=int(report["drops"]["invalid_values"]),
            n_dropped_non_numeric=int(report["drops"]["non_numeric"]),
            n_dropped_missing_unit=int(report["drops"]["missing_or_bad_unit"]),
            unit_normalisations=report.get("unit_normalisations", {}),
            config_fingerprint=str(report["config_fingerprint"]),
        )

    # ----- pure-ish steps (testable independently) -----

    def load_raw(self, raw_snapshot_path: Path) -> pd.DataFrame:
        return pd.read_parquet(raw_snapshot_path)

    def clean_dataframe(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = df_raw.copy()

        df = self.filter_determinands(df)
        df = self.parse_datetime(df)

        df, non_numeric_drops = self.coerce_numeric(df)
        df, unit_report = self.harmonise_units(df)
        df, invalid_drops = self.apply_validity_rules(df)
        df = self.dedupe_and_sort(df)

        report = self.build_report(
            df_raw=df_raw,
            df_clean=df,
            non_numeric_drops=non_numeric_drops,
            invalid_drops=invalid_drops,
            unit_report=unit_report,
        )
        return df, report

    def filter_determinands(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        out = df.copy()
    
        # Normalize determinand IDs deterministically: "0061" -> "61"
        det_norm = (
            out[c.determinand_col]
            .astype(str)
            .str.strip()
            .str.lstrip("0")
            .replace("", "0")
        )
    
        # Normalize config IDs the same way
        wanted = {str(x).strip().lstrip("0") or "0" for x in c.determinand_ids}
    
        out = out.loc[det_norm.isin(wanted)].copy()
    
        # Optional: persist the normalized determinand id for downstream debugging/QA
        out["_det_id_norm"] = det_norm.loc[out.index]
    
        return out


    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        out = df.copy()

        out[c.datetime_col] = pd.to_datetime(out[c.datetime_col], errors="raise", utc=False)

        # If tz-naive, localize to assume_tz
        if getattr(out[c.datetime_col].dt, "tz", None) is None:
            out[c.datetime_col] = out[c.datetime_col].dt.tz_localize(c.assume_tz)

        # Convert to output_tz (keeps tz-aware, deterministic)
        out[c.datetime_col] = out[c.datetime_col].dt.tz_convert(c.output_tz)
        return out

    def coerce_numeric(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        c = self.config
        out = df.copy()

        if c.coerce_numeric_errors not in ("raise", "coerce"):
            raise ValueError("coerce_numeric_errors must be 'raise' or 'coerce'")

        if c.coerce_numeric_errors == "raise":
            # strict: any non-numeric should raise
            out[c.value_col] = out[c.value_col].astype(float)
            return out, 0

        # permissive: coerce to NaN, then drop if configured
        out[c.value_col] = pd.to_numeric(out[c.value_col], errors="coerce")
        non_numeric = int(out[c.value_col].isna().sum())

        if c.drop_non_numeric:
            out = out.dropna(subset=[c.value_col]).copy()

        return out, non_numeric

    def harmonise_units(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Phase-2 minimal version:
        - Enforce unit presence if require_units=True
        - Optionally enforce allowed_units if provided
        - Do not perform conversions yet (we'll add later based on observed units)
        """
        c = self.config
        out = df.copy()
        missing_or_bad = 0

        if c.require_units:
            missing = out[c.unit_col].isna()
            missing_or_bad += int(missing.sum())
            out = out.loc[~missing].copy()

        if c.allowed_units is not None:
            def _ok(row) -> bool:
                det = row[c.determinand_col]
                unit = row[c.unit_col]
                allowed = c.allowed_units.get(det)
                return True if allowed is None else unit in allowed

            ok_mask = out.apply(_ok, axis=1)
            missing_or_bad += int((~ok_mask).sum())
            out = out.loc[ok_mask].copy()

        report = {
            "missing_or_bad_unit": int(missing_or_bad),
            "unit_normalisations": {},  # placeholder for later unit normalisation mappings
        }
        return out, report

    def apply_validity_rules(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        c = self.config
        out = df.copy()
        dropped = 0

        if c.drop_negative:
            mask = out[c.value_col] < 0
            dropped += int(mask.sum())
            out = out.loc[~mask].copy()

        if c.min_max_by_determinand:
            for det_id, (mn, mx) in c.min_max_by_determinand.items():
                det_mask = out[c.determinand_col] == det_id
                val = out[c.value_col]
                bad = det_mask & ((val < mn) | (val > mx))
                dropped += int(bad.sum())
                out = out.loc[~bad].copy()

        return out, dropped

    def dedupe_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        out = df.copy()

        if c.dedupe_keys:
            out = out.drop_duplicates(subset=list(c.dedupe_keys), keep=c.dedupe_keep).copy()

        if c.sort_keys:
            # mergesort is stable → helps determinism
            out = out.sort_values(list(c.sort_keys), kind="mergesort").reset_index(drop=True)

        return out

    # ----- persistence + governance -----

    def persist(self, df_clean: pd.DataFrame, output_dir: Path, *, run_id: Optional[str]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = run_id or self.config.snapshot_id or "clean"
        path = output_dir / f"clean_{stem}.parquet"
        df_clean.to_parquet(path, index=False)
        return path

    def build_report(
        self,
        *,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        non_numeric_drops: int,
        invalid_drops: int,
        unit_report: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Deterministic QA report describing what happened in the clean layer.
        Must be stable across reruns given same input + config.
        """
        # Deterministic fingerprint of config (stable ordering)
        cfg_items = sorted(self.config.__dict__.items())
        cfg_bytes = repr(cfg_items).encode("utf-8")
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()

        missing_or_bad_unit = int(unit_report.get("missing_or_bad_unit", 0))
        unit_norms = unit_report.get("unit_normalisations", {})

        return {
            "config_fingerprint": cfg_hash,
            "snapshot_id": self.config.snapshot_id,
            "clean_version": self.config.clean_version,
            "rows": {"in": int(len(df_raw)), "out": int(len(df_clean))},
            "drops": {
                "non_numeric": int(non_numeric_drops),
                "invalid_values": int(invalid_drops),
                "missing_or_bad_unit": missing_or_bad_unit,
            },
            "unit_normalisations": unit_norms,
        }

    def persist_report(self, report: dict[str, Any], output_dir: Path, *, run_id: Optional[str]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = run_id or self.config.snapshot_id or "clean"
        path = output_dir / f"clean_{stem}_report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return path

