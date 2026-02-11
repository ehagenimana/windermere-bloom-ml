\# Decision Log



| Date | Decision | Options considered | Rationale | Consequences / follow-ups |

|---|---|---|---|---|

| 2026-02-11 | Forecast horizon set to 14 days | 7 vs 14 days | 14 days provides earlier operational warning and aligns with weekly-to-biweekly monitoring actions | Target generation will use t+1..t+14 window |

| 2026-02-11 | Alert budget set to <10% of days | alerts/month vs alerts/week vs %days | %days is stable across seasons and supports thresholding policies | Evaluation will report recall at alert budget |

| 2026-02-11 | Bloom threshold deferred until after ingestion + QC | fixed guideline vs percentile vs data-driven + guidance | Avoids arbitrary threshold; supports defensible governance | Must document chosen threshold + freeze it per cycle |



