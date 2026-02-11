\# EA Water Quality API — Endpoints Used (Windermere)



\## Core endpoints

1\) Observations

\- /water-quality/data/observation

\- Purpose: download measurements for pointNotation + determinand over a date range



2\) Sampling points

\- /water-quality/data/sampling-point

\- Purpose: discover/validate monitoring sites (pointNotation + metadata)



3\) Determinands codelist

\- /water-quality/codelist/determinand

\- Purpose: map determinand.notation to determinand.prefLabel and select MVP determinands



\## Query parameters we will standardise

\- pointNotation = samplingPoint.notation (from Windermere\_data.csv)

\- determinand = determinand.notation (from Windermere\_data.csv)

\- dateFrom, dateTo = time window filters

\- limit, skip = pagination



