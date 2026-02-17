import pandas as pd
import pytest
import pandera.pandas as pa

from windermere_project.ingestion.schemas import ea_wqe_observations_schema


def _minimal_valid_row():
    return {
        "id": "1",
        "samplingPoint.notation": "NW-88010013",
        "samplingPoint.prefLabel": "Windermere",
        "samplingPoint.longitude": -2.95,
        "samplingPoint.latitude": 54.36,
        "samplingPoint.region": "North West",
        "samplingPoint.area": "Cumbria",
        "samplingPoint.subArea": "Windermere",
        "samplingPoint.samplingPointStatus": "Active",
        "samplingPoint.samplingPointType": "Lake",
        "phenomenonTime": "2020-01-01T00:00:00Z",
        "samplingPurpose": "Routine",
        "sampleMaterialType": "Water",
        "determinand.notation": "9999",
        "determinand.prefLabel": "Dummy",
        "result": 1.23,
        "unit": "mg/L",
    }


def test_schema_passes_valid_df():
    df = pd.DataFrame([_minimal_valid_row()])
    out = ea_wqe_observations_schema().validate(df)
    assert len(out) == 1


def test_schema_fails_missing_required_column():
    row = _minimal_valid_row()
    row.pop("samplingPoint.notation")
    df = pd.DataFrame([row])

    with pytest.raises(pa.errors.SchemaError):
        ea_wqe_observations_schema().validate(df)
