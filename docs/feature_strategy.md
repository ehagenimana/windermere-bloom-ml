\# Feature Strategy (Conceptual)



\## Objective

Define hypothesised drivers of bloom risk and the feature domains required before data acquisition.



\## Feature domains (MVP)

1\) Lake chemistry (mandatory)

2\) Physical lake conditions (mandatory)

3\) Meteorology \& catchment loading proxies (strongly recommended)

4\) Temporal/seasonal structure (mandatory)

5\) Sampling structure (mandatory)



\## Anti-leakage rules

\- No future aggregation

\- No target-aware transformations

\- No global statistics computed across the full dataset



\## System scope (MVP)

Lake Windermere is treated as a single system. If multiple sampling sites are available, they will be harmonised into one modelling dataset, with site handled as:

\- optional metadata (for diagnostics), and/or

\- a categorical feature if it improves performance without introducing leakage.



\## Complexity boundary (MVP)

Maximum lag horizon: 30 days (may be revised after EDA).



