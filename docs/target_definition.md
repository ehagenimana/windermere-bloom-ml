\## Target Definition (Conceptual)



\### Target event

A “bloom-risk event” will be defined as chlorophyll-a exceeding a threshold that is selected \*\*after data ingestion and QC\*\*, based on:

1\) any available regulatory guidance / monitoring best practice, and

2\) the observed distribution and seasonality of chlorophyll-a in the Windermere dataset.



\### Prediction task

Binary classification:

\- 1 = a bloom-risk event occurs within the next 14 days (t+1 … t+14)

\- 0 = no bloom-risk event occurs within the next 14 days



\### Governance / selection rule (MVP)

Threshold selection must be documented and justified, and the chosen threshold must be “frozen” for a model-training cycle (to avoid post-hoc tuning).



