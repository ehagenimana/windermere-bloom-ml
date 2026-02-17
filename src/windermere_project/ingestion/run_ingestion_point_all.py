from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from windermere_project.ingestion.data_ingestor import DataIngestor


def main() -> None:
    point_notation = "NW-88010013"

    print("Starting ingestion for point:", point_notation)

    ingestor = DataIngestor(api_config_path="config/api.yaml")

    df = ingestor.fetch_observations(
        point_notation=point_notation,
        determinand=None,
        date_from=None,
        date_to=None,
        compliance_only=False,
        limit_pages=2,   # smoke test first
    )

    print("Fetched rows:", len(df))
    print("Columns:", list(df.columns) if not df.empty else "EMPTY")

    if df.empty:
        print("No data returned. Aborting snapshot.")
        return

    snapshot_id = datetime.now(timezone.utc).strftime(
        f"raw_{point_notation}_ALL_%Y%m%dT%H%M%SZ"
    )

    snap_path = ingestor.snapshot_to_parquet(
        df=df,
        output_dir=Path("data/raw"),
        snapshot_id=snapshot_id,
    )

    print("Wrote snapshot:", snap_path)


if __name__ == "__main__":
    main()
