\# Decision Log



| Date | Decision | Options considered | Rationale | Consequences / follow-ups |

|---|---|---|---|---|

| 2026-02-11 | Forecast horizon set to 14 days | 7 vs 14 days | 14 days provides earlier operational warning and aligns with weekly-to-biweekly monitoring actions | Target generation will use t+1..t+14 window |

| 2026-02-11 | Alert budget set to <10% of days | alerts/month vs alerts/week vs %days | %days is stable across seasons and supports thresholding policies | Evaluation will report recall at alert budget |

| 2026-02-11 | Bloom threshold deferred until after ingestion + QC | fixed guideline vs percentile vs data-driven + guidance | Avoids arbitrary threshold; supports defensible governance | Must document chosen threshold + freeze it per cycle |



| 2026-02-11 |

Treat Windermere as a single lake system for MVP |

single-site vs multi-site/hierarchical |

Reduces complexity; avoids site-specific confounding in early iteration |

Keep site metadata for diagnostics; revisit if strong site heterogeneity emerges |



| 2026-02-11 | Prioritise Environment Agency in-situ data before remote sensing | in-situ vs satellite-first | In-situ data is regulator-grade and avoids atmospheric correction uncertainty | Remote sensing considered only if coverage insufficient |

## Phase 2 — Feature Matrix Decisions

- Target: chlorophyll > 20 µg/L

- Window: 2005–2025

 # Predictors V1:

- TP (348)

- TN (9686)

- pH (61)

- Lookback: 30 days

# Missingness flags: enabled

- Sparse DO% and Temp dropped for V1

- Deterministic merge_asof backward (no leakage)

