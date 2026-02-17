import pandas as pd

from windermere_project.eda import run_viability_gate, ViabilityGateConfig

CLEAN_PATH = "data/clean/clean_raw_NW-88010013_ALLFULL_20260216T055951Z.parquet"

df_clean = pd.read_parquet(CLEAN_PATH)

cfg = ViabilityGateConfig(
    output_dir="reports/phase2",
    window_start_year=2005,
    window_end_year=2025,
    threshold_ugL=20.0,
)

results = run_viability_gate(
    df_clean=df_clean,
    config=cfg,
    predictor_ids=[348, 9686, 61, 9901, 76],  # TP, TN, pH, DO%, Temp
)

print("CLASS BALANCE:", results["class_balance"])
print("SAMPLING FREQ:", results["sampling_frequency"])
print("PERSISTENCE:", results["temporal_persistence"])
print("PRED AVAIL:", results.get("predictor_availability"))
print("FIGURES:", results["seasonality"], results["sampling_frequency_plot"])
