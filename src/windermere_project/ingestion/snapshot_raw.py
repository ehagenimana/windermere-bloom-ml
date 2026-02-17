from __future__ import annotations

from pathlib import Path

from windermere_project.clean.builder import CleanDatasetBuilder, CleanConfig


def main() -> None:
    raw_snapshot_path = Path("data/raw/raw_NW-88010013_ALLFULL_20260216T055951Z.parquet")

    config = CleanConfig(
        determinand_ids=("61", "76", "111", "117", "348", "7887", "9686", "9901"),
        datetime_col="phenomenonTime",
        site_col="samplingPoint.notation",
        determinand_col="determinand.notation",
        unit_col="unit",
        value_col="result",
        coerce_numeric_errors="coerce",
        drop_non_numeric=True,
        snapshot_id=raw_snapshot_path.stem,
    
        # ✅ IMPORTANT: align keys to real columns
        sort_keys=("phenomenonTime", "samplingPoint.notation", "determinand.notation"),
        dedupe_keys=("phenomenonTime", "samplingPoint.notation", "determinand.notation"),
    )


    builder = CleanDatasetBuilder(config)

    result = builder.build(
        raw_snapshot_path=raw_snapshot_path,
        output_dir=Path("data/clean"),
    )

    print(f"Raw snapshot: {raw_snapshot_path}")
    print(result)


if __name__ == "__main__":
    main()

