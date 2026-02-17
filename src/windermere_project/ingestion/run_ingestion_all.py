from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from windermere_project.ingestion.data_ingestor import DataIngestor
from windermere_project.ingestion.determinands_discovery import fetch_determinands_catalogue


def main() -> None:
    point_notation = "NW-88010013"

    # 1) Load full determinands catalogue
    cat = fetch_determinands_catalogue(page_size=1000, limit_pages=None)

    if "notation" not in cat.columns:
        raise ValueError("Determinands catalogue missing 'notation' column; cannot derive IDs.")

    # Normalize: "0076" -> 76, etc.
    cat["det_id"] = (
        cat["notation"].astype(str).str.strip().str.lstrip("0").replace("", "0").astype(int)
    )
    determinands = cat["det_id"].dropna().astype(int).unique().tolist()

    # 2) Prepare output folders
    run_id = datetime.now(timezone.utc).strftime("raw_all_%Y%m%dT%H%M%SZ")
    out_root = Path("data/raw") / point_notation / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    # 3) Fetch observations determinand-by-determinand (phenomenon time = all)
    ingestor = DataIngestor(api_config_path="config/api.yaml")

    manifest_rows = []
    for i, det in enumerate(determinands, start=1):
        det_out = out_root / f"det_{det}.parquet"
        det_hash = out_root / f"det_{det}.sha256"

        # Resume: skip if already written
        if det_out.exists() and det_hash.exists():
            manifest_rows.append({"determinant_id": det, "status": "skipped_exists", "rows": None})
            continue

        try:
            df_det = ingestor.fetch_observations(
                determinand=det,
                point_notation=point_notation,
                date_from=None,
                date_to=None,
                compliance_only=False,
                limit_pages=None,
            )

            # Write per-determinand snapshot (governed)
            snap_path = ingestor.snapshot_to_parquet(
                df=df_det,
                output_dir=out_root,
                snapshot_id=f"det_{det}",
            )

            manifest_rows.append({"determinant_id": det, "status": "ok", "rows": int(len(df_det)), "path": str(snap_path)})

            # Lightweight progress output
            if i % 25 == 0:
                print(f"[{i}/{len(determinands)}] done...")

        except Exception as e:
            manifest_rows.append({"determinant_id": det, "status": "error", "error": repr(e)})
            print(f"[ERROR] det={det}: {e}")

    # 4) Persist manifest (what succeeded/failed)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    print(f"Finished. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

