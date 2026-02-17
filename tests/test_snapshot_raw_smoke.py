import sys
import pytest
import subprocess


@pytest.mark.network
def test_snapshot_raw_runs():
    cmd = [
        sys.executable,
        "-m",
        "windermere_project.ingestion.snapshot_raw",
        "--date-from", "2020-01-01",
        "--date-to", "2020-12-31",
        "--page-limit", "500",
    ]
    subprocess.check_call(cmd)
