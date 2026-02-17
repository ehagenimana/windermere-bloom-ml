import pandas as pd

SRC = "data/clean/clean_raw_NW-88010013_ALLFULL_20260216T055951Z.parquet"
OUT = "data/clean/wide_features_base.parquet"

# Determinand IDs (confirmed)
DET_CHL = "7887"   # Chlorophyll : Acetone Extract (µg/L)
DET_TP  = "348"    # Phosphorus, Total as P (mg/L)
DET_TN  = "9686"   # Nitrogen, Total as N (mg/L)

# Optional extras (often useful and already present)
DET_PH  = "61"     # pH
DET_TEMP = "76"    # Temperature of Water (°C)

DETS = [DET_CHL, DET_TP, DET_TN, DET_PH, DET_TEMP]

RENAME = {
    DET_CHL: "chl",
    DET_TP: "tp",
    DET_TN: "tn",
    DET_PH: "ph",
    DET_TEMP: "temp",
}

def main():
    df = pd.read_parquet(SRC).copy()

    # Time
    df["phenomenonTime"] = pd.to_datetime(df["phenomenonTime"], utc=True)

    # Numeric values
    df["result_num"] = pd.to_numeric(df["result"], errors="coerce")

    # Filter determinands
    df["_det_id_norm"] = df["_det_id_norm"].astype(str)
    df = df[df["_det_id_norm"].isin(DETS)].copy()

    # Pivot to wide
    wide = (
        df.pivot_table(
            index="phenomenonTime",
            columns="_det_id_norm",
            values="result_num",
            aggfunc="mean",
        )
        .sort_index()
        .rename(columns=RENAME)
    )

    wide.to_parquet(OUT)
    print("Saved:", OUT)
    print("Shape:", wide.shape)
    print("Columns:", list(wide.columns))
    print(wide.head(10).to_string())

if __name__ == "__main__":
    main()
