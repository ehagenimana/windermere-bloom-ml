import pandas as pd

from windermere_project.eda.viability_gate import ViabilityGate, ViabilityGateConfig


def test_class_balance_threshold_logic():
    cfg = ViabilityGateConfig(
        target_determinand_id=7887,
        threshold_ugL=20.0,
        window_start_year=2005,
        window_end_year=2025,
        datetime_col="sample_datetime",
        value_col="result",
        determinand_col="determinand_id",
        output_dir="reports/phase2_test"  # safe test dir
    )

    df = pd.DataFrame(
        {
            "sample_datetime": [
                "2005-01-01T00:00:00",
                "2005-02-01T00:00:00",
                "2005-03-01T00:00:00",
            ],
            "determinand_id": [7887, 7887, 7887],
            "result": [19.9, 20.0, 20.1],
        }
    )

    gate = ViabilityGate(df_clean=df, config=cfg)
    out = gate.compute_class_balance()

    assert out["n_total"] == 3
    # strictly ">" 20
    assert out["n_pos"] == 1
    assert out["n_neg"] == 2
    assert abs(out["pos_rate"] - (1 / 3)) < 1e-9


def test_does_not_require_extra_columns():
    cfg = ViabilityGateConfig(
        output_dir="reports/phase2_test",
        datetime_col="sample_datetime",
        determinand_col="determinand_id",
        value_col="result",
        determinand_cast="int",
        target_determinand_id=7887,
    )

    df = pd.DataFrame(
        {
            "sample_datetime": ["2006-01-01T00:00:00"],
            "determinand_id": [7887],
            "result": [10.0],
        }
    )

    gate = ViabilityGate(df_clean=df, config=cfg)
    out = gate.compute_class_balance()
    assert out["n_total"] == 1

