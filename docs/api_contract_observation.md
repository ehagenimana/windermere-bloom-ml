\# API Contract — /data/observation (Windermere)



\## Request template

GET /water-quality/data/observation

&nbsp; ?pointNotation={samplingPoint.notation}

&nbsp; \&determinand={determinand.notation}

&nbsp; \&dateFrom=YYYY-MM-DD

&nbsp; \&dateTo=YYYY-MM-DD

&nbsp; \&limit={N}

&nbsp; \&skip={K}



\## Response fields expected (minimum)

\- phenomenonTime (ISO datetime)

\- result (string/number; may include censored values)

\- unit

\- samplingPurpose

\- sampleMaterialType

\- samplingPoint.notation, samplingPoint.prefLabel, samplingPoint.lat/long, samplingPoint.region/area/subArea

\- determinand.notation, determinand.prefLabel



