from __future__ import annotations

from typing import Optional, List
import pandas as pd
import requests


def fetch_determinands_catalogue(
    page_size: int = 2000,
    limit_pages: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch determinands catalogue using EA linked-data paging conventions.

    Why this shape:
    - Many EA list endpoints default to 100 rows unless you set _pageSize.
    - They page with _page (0-indexed) and _pageSize.
    - You can request fields with _properties to avoid minimal views.
    """
    base_url = "https://environment.data.gov.uk/water-quality/codelist/determinand.json"

    frames: List[pd.DataFrame] = []
    page = 0

    while True:
        params = {
            "_pageSize": page_size,
            "_page": page,
            "_properties": "notation,label,altLabel",
        }

        r = requests.get(base_url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Common EA linked-data pattern: results under "result" (list)
        # Sometimes nested; handle a few variants safely.
        if isinstance(data, dict):
            items = data.get("result") or data.get("results") or data.get("items") or []
        else:
            items = data

        if not items:
            break

        df_page = pd.DataFrame(items)

        # Keep requested columns if present
        keep = [c for c in ["notation", "label", "altLabel"] if c in df_page.columns]
        df_page = df_page[keep].copy() if keep else df_page

        frames.append(df_page)

        page += 1
        if limit_pages is not None and page >= limit_pages:
            break

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return df



