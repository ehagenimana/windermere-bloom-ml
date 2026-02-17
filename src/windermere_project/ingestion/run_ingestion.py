from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from windermere_project.ingestion.data_ingestor import DataIngestor


def main() -> None:
    # TODO: set these to your Windermere sampling point and determinand IDs
    point_notation = "NW-88010013"  # example you used earlier
    determinands = [9686]          # placeholder: replace with list of determinand IDs you want

    ingestor = DataIngestor(api_config_path="config/api.yaml")

    frames: list[pd.DataFrame] = []
    for det in determinands:
        df_det = ingestor.fetch_observations(determinand=det, point_notation=point_notation)
        frames.append(df_det)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    snapshot_id = datetime.now(timezone.utc).strftime(f"raw_{point_notation}_%Y%m%dT%H%M%SZ")
    snap_path = ingestor.snapshot_to_parquet(
        df=df,
        output_dir=Path("data/raw"),
        snapshot_id=snapshot_id,
    )

    print(f"Wrote snapshot: {snap_path}")
    print(f"Rows: {len(df):,} | Cols: {df.shape[1]}")


if __name__ == "__main__":
    main()
