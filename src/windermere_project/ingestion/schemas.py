# src/windermere_project/ingestion/schemas.py

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# --- Contract metadata (Phase 1.3) ---
SCHEMA_NAME = "ea_wqe_observations_schema"
SCHEMA_VERSION = 1

REQUIRED_COLUMNS = [
    "id",
    "samplingPoint.notation",
    "samplingPoint.prefLabel",
    "samplingPoint.longitude",
    "samplingPoint.latitude",
    "samplingPoint.region",
    "samplingPoint.area",
    "samplingPoint.subArea",
    "samplingPoint.samplingPointStatus",
    "samplingPoint.samplingPointType",
    "phenomenonTime",
    "samplingPurpose",
    "sampleMaterialType",
    "determinand.notation",
    "determinand.prefLabel",
    "result",
    "unit",
]


def ea_wqe_observations_schema() -> DataFrameSchema:
    """
    Phase 1 Pandera schema for EA WQE observation CSV pages.

    - Enforces presence + basic types for the columns you observed (17 columns).
    - Allows extra columns (EA may add fields) via strict=False.
    - Coerces numeric types where appropriate.
    """

    return DataFrameSchema(
        {
            "id": Column(str, nullable=True, coerce=True),

            "samplingPoint.notation": Column(str, nullable=False, coerce=True),
            "samplingPoint.prefLabel": Column(str, nullable=True, coerce=True),
            "samplingPoint.longitude": Column(float, nullable=True, coerce=True),
            "samplingPoint.latitude": Column(float, nullable=True, coerce=True),
            "samplingPoint.region": Column(str, nullable=True, coerce=True),
            "samplingPoint.area": Column(str, nullable=True, coerce=True),
            "samplingPoint.subArea": Column(str, nullable=True, coerce=True),
            "samplingPoint.samplingPointStatus": Column(str, nullable=True, coerce=True),
            "samplingPoint.samplingPointType": Column(str, nullable=True, coerce=True),

            # Still "raw" (no feature engineering): parsing catches drift early.
            "phenomenonTime": Column(pa.DateTime, nullable=True, coerce=True),

            "samplingPurpose": Column(str, nullable=True, coerce=True),
            "sampleMaterialType": Column(str, nullable=True, coerce=True),

            "determinand.notation": Column(str, nullable=False, coerce=True),
            "determinand.prefLabel": Column(str, nullable=True, coerce=True),

            "result": Column(float, nullable=True, coerce=True),
            "unit": Column(str, nullable=True, coerce=True),
        },
        checks=[
            Check(lambda df: df["samplingPoint.notation"].str.len().fillna(0).gt(0).all()),
            Check(lambda df: df["determinand.notation"].str.len().fillna(0).gt(0).all()),
        ],
        strict=False,
        coerce=True,
    )

