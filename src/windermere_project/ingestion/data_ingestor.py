from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
from pandas.errors import EmptyDataError

import pandas as pd
import requests
import yaml


@dataclass(frozen=True)
class EAConfig:
    base_url: str
    page_size: int = 250
    timeout_s: int = 60
    max_retries: int = 5
    backoff_s: int = 2
    user_agent: str = "windermere-bloom-ml/0.1"


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_query(url: str) -> str:
    """Ensure base_url has no embedded query string (prevents duplicate params)."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class DataIngestor:
    """
    Environment Agency Water Quality Archive ingestor (observations).

    Endpoint style used in your project:
      {base_url}/sampling-point/{point_notation}/observation

    Query parameters:
      - determinand (optional)
      - dateFrom / dateTo (optional)
      - skip / limit
      - complianceOnly (optional)
    """

    def __init__(self, api_config_path: str = "config/api.yaml") -> None:
        cfg = _load_yaml(api_config_path)
        ea = cfg["ea_water_quality"]

        base_url = _strip_query(str(ea["base_url"]))

        self.cfg = EAConfig(
            base_url=base_url,
            page_size=int(ea.get("page_size", 250)),
            timeout_s=int(ea.get("timeout_s", 60)),
            max_retries=int(ea.get("max_retries", 5)),
            backoff_s=int(ea.get("backoff_s", 2)),
            user_agent=str(ea.get("user_agent", "windermere-bloom-ml/0.1")),
        )

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": self.cfg.user_agent, "Accept": "application/json"}
        )




    def _get_csv_df_with_retries(self, url: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch one page as CSV -> DataFrame. Retries on transient failures.
        Handles occasional empty-body responses gracefully.
        """
        last_err: Optional[Exception] = None
    
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = self.session.get(
                    url,
                    params=params,
                    timeout=self.cfg.timeout_s,
                    headers={"Accept": "text/csv"},
                )
                r.raise_for_status()
    
                text = (r.text or "").strip()
    
                # If the API returns an empty body, treat as empty page (end of pagination)
                # rather than crashing the whole run.
                if text == "":
                    return pd.DataFrame()
    
                # Occasionally services return HTML error pages; avoid parsing as CSV.
                if text.startswith("<!DOCTYPE") or text.startswith("<html"):
                    raise RuntimeError(
                        f"Non-CSV response received (HTML). status={r.status_code} "
                        f"content_type={r.headers.get('Content-Type')} "
                        f"snippet={text[:200]!r}"
                    )
    
                try:
                    return pd.read_csv(io.StringIO(text))
                except EmptyDataError:
                    # EmptyDataError can happen even with non-empty strings (rare formatting edge)
                    # Retry a couple of times; if persistent, treat as empty page.
                    last_err = EmptyDataError("No columns to parse from CSV page.")
                    time.sleep(self.cfg.backoff_s * attempt)
                    continue
    
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff_s * attempt)
    
        raise RuntimeError(f"EA API request failed after retries: {last_err}") from last_err


    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def snapshot_to_parquet(self, df: pd.DataFrame, output_dir: Path, snapshot_id: str) -> Path:
        """
        Persist an immutable raw snapshot (Parquet) plus SHA256 sidecar.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        snap_path = output_dir / f"{snapshot_id}.parquet"
        df.to_parquet(snap_path, index=False)

        digest = self._sha256_file(snap_path)
        (output_dir / f"{snapshot_id}.sha256").write_text(digest, encoding="utf-8")

        return snap_path

    def fetch_observations(
        self,
        point_notation: str,
        determinand: int | str | None = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        compliance_only: bool = False,
        limit_pages: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch observations for a sampling point, optionally filtered by determinand.

        IMPORTANT:
        - If determinand is None, we OMIT the parameter entirely (do not send determinand=None),
          otherwise the API returns 422.
        """
        if not point_notation:
            raise ValueError("point_notation is required (e.g., 'NW-88010013').")

        # DSP docs recommend max 250 for /data/observation; keep <= 250.
        page_size = min(int(self.cfg.page_size), 250)

        url = f"{self.cfg.base_url.rstrip('/')}/sampling-point/{point_notation}/observation"

        params: Dict[str, Any] = {
            "limit": page_size,
            "skip": 0,
            "complianceOnly": str(compliance_only).lower(),
        }

        # ✅ Only include determinand if provided
        if determinand is not None:
            params["determinand"] = str(determinand)

        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        dfs: List[pd.DataFrame] = []
        page = 0

        while True:
            page += 1
            df_page = self._get_csv_df_with_retries(url, params=params)

            if df_page.empty:
                # Could be end-of-data OR transient empty page.
                # If it happens once, stop; if you want more robustness, you can tolerate 1 empty.
                break

            dfs.append(df_page)
            params["skip"] += params["limit"]

            if limit_pages is not None and page >= limit_pages:
                break

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
