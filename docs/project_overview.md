## Phase 0.1 — Decision Framing (MVP)

### Decision the system supports
Predict the probability of a bloom-risk event in Lake Windermere and raise an alert when risk exceeds a threshold, enabling proactive monitoring and response.

### Primary user / actor
Environmental analyst / regulator / lake monitoring team.

### Operational output
- A daily risk score (0–1)
- A binary alert (ALERT / NO ALERT) derived from the risk score threshold

### Lead time (forecast horizon)
[TBD — choose 7 days or 14 days for MVP]

### Scoring cadence
Daily (MVP)

### Alert budget (false-alert constraint)
[TBD — define maximum acceptable alert frequency, e.g., ≤ 4 alerts per month]

### Success criteria (MVP)
- Model beats a persistence baseline under PR-AUC and “recall at alert budget”
- Alert rate respects the alert budget
- Results are reproducible from config + locked dependencies
### Lead time (forecast horizon)
14 days

### Alert budget (false-alert constraint)
Alerts on < 10% of scoring days
Rationale: keeps the system operationally usable by limiting false-alarm burden while still allowing high sensitivity during elevated-risk periods.
