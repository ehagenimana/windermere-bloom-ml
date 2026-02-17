# Feature Registry — Windermere Bloom Prediction (V1)

## Scope
This registry documents the engineered features used for Phase 4 (Feature Engineering Framework).
All features are designed to be leakage-safe and reproducible, with train/score parity enforced.

**Base table:** `data/clean/wide_features_base.parquet`  
**Feature table output:** `data/features/features_v1.parquet`

## Inputs (wide base columns)
- `chl` — Chlorophyll : Acetone Extract (µg/L) [EA determinand 7887]
- `tp` — Phosphorus, Total as P (mg/L) [EA determinand 348]
- `tn` — Nitrogen, Total as N (mg/L) [EA determinand 9686]
- `ph` — pH [EA determinand 61]
- `temp` — Temperature of Water (°C) [EA determinand 76]

## Anti-leakage rules (hard)
1. Any rolling statistic MUST be computed on a shifted series:
   - `x.shift(1).rolling(...).mean()`
2. No features may use future timestamps, forward-looking windows, or global statistics across the full dataset.
3. Train/test boundary must not be crossed by imputations or scaling operations.
4. Feature generation must be deterministic and config-driven.

## Feature families and definitions

### A) Persistence / memory (chlorophyll)
| Feature | Definition |
|---|---|
| `chl_lag_1` | `chl.shift(1)` |
| `chl_lag_7` | `chl.shift(7)` |
| `chl_roll_mean_7` | `chl.shift(1).rolling(7, min_periods=1).mean()` |
| `chl_roll_mean_30` | `chl.shift(1).rolling(30, min_periods=1).mean()` |
| `prev_exceed_flag` | `(chl.shift(1) > 20).astype(int)` |
| `days_since_prev` | `(t - t.shift(1))` in days |

### B) Seasonality
| Feature | Definition |
|---|---|
| `month` | `index.month` |
| `doy_sin` | `sin(2π * doy/365.25)` |
| `doy_cos` | `cos(2π * doy/365.25)` |

### C) Nutrients (V1 minimal)
| Feature | Definition |
|---|---|
| `tn_lag_7` | `tn.shift(7)` |
| `tn_roll_mean_30` | `tn.shift(1).rolling(30, min_periods=1).mean()` |
| `tp_lag_7` | `tp.shift(7)` |
| `tp_roll_mean_30` | `tp.shift(1).rolling(30, min_periods=1).mean()` |

### D) Context (already available, optional to use in models)
| Feature | Definition |
|---|---|
| `ph` | raw pH value (no transformation) |
| `temp` | raw temperature value (no transformation) |

### E) Missingness flags
| Feature | Definition |
|---|---|
| `miss_chl` | `chl.isna().astype(int)` |
| `miss_tn` | `tn.isna().astype(int)` |
| `miss_tp` | `tp.isna().astype(int)` |
| `miss_ph` | `ph.isna().astype(int)` |
| `miss_temp` | `temp.isna().astype(int)` |

## Notes / rationale
- Persistence features are mandatory because Phase 3 showed persistence as the strongest baseline.
- Seasonality features are mandatory due to strong June–September structure.
- Nutrient features are included but constrained to avoid feature creep and reduce overfitting risk in small data.
