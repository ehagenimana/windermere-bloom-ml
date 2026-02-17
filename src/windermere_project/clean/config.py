from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CleanConfig:
    # Determinands to retain in the clean layer (integers, not strings)
    determinand_ids: tuple[int, ...]

    # Column names in the raw snapshot (or synthetic tests)
    datetime_col: str
    site_col: str
    determinand_col: str
    unit_col: str
    value_col: str

    # Timezone policy
    assume_tz: str = "UTC"
    output_tz: str = "UTC"

    # Numeric handling
    censored_value_col: Optional[str] = None
    coerce_numeric_errors: str = "raise"  # "raise" | "coerce"
    drop_non_numeric: bool = True

    # Basic validity rules (clean-layer only)
    drop_negative: bool = True
    min_max_by_determinand: Optional[dict[int, tuple[float, float]]] = None

    # Units
    require_units: bool = True
    allowed_units: Optional[dict[int, tuple[str, ...]]] = None

    # Determinism
    sort_keys: tuple[str, ...] = ()
    dedupe_keys: tuple[str, ...] = ()
    dedupe_keep: str = "last"  # "first" | "last"

    # Governance
    snapshot_id: Optional[str] = None
    clean_version: str = "CLEAN_V1"

    # Output control
    keep_cols_extra: tuple[str, ...] = ()
    output_long: bool = True


# ✅ Default config instance MUST be defined outside the class body
DEFAULT_CLEAN_CONFIG = CleanConfig(
    determinand_ids=(7887, 348, 9686, 111, 117, 9901, 76, 61),
    datetime_col="phenomenonTime",
    site_col="samplingPoint.notation",
    determinand_col="determinand.notation",
    unit_col="unit",
    value_col="result",
    snapshot_id="raw_NW-88010013_ALLFULL_20260216T055951Z",
    coerce_numeric_errors="coerce",
    drop_non_numeric=True,
)

# Optional: keep backward-compatible name if other modules import `config`
config = DEFAULT_CLEAN_CONFIG


