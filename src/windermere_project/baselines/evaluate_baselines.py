from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss


@dataclass(frozen=True)
class BaselineMetrics:
    name: str
    pr_auc: float
    brier: float
    # recall at fixed alert rate = top k% highest risk scores
    recall_at_alert_rate: float
    precision_at_alert_rate: float
    alert_rate: float
    n: int
    n_pos: int


def _topk_metrics(y_true: np.ndarray, y_score: np.ndarray, alert_rate: float) -> Tuple[float, float]:
    """
    Alert rate means we allow alerts on top K% of samples by score.
    Returns (recall, precision) at that operating point.
    """
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")

    k = max(1, int(np.ceil(alert_rate * n)))
    idx = np.argsort(-y_score)[:k]
    y_alert = y_true[idx]

    tp = int(y_alert.sum())
    pos = int(y_true.sum())
    recall = float(tp / pos) if pos > 0 else float("nan")
    precision = float(tp / k) if k > 0 else float("nan")
    return recall, precision


def evaluate_baselines(
    results: List[Dict[str, Any]],
    *,
    alert_rate: float = 0.10,
) -> pd.DataFrame:
    """
    Takes a list of dicts that each have:
      - name
      - y_true (np.ndarray)
      - y_score (np.ndarray)

    Returns a tidy metrics dataframe.
    """
    rows = []
    for r in results:
        name = str(r["name"])
        y_true = np.asarray(r["y_true"]).astype(int)
        y_score = np.asarray(r["y_score"]).astype(float)

        pr_auc = float(average_precision_score(y_true, y_score)) if len(y_true) else float("nan")
        brier = float(brier_score_loss(y_true, y_score)) if len(y_true) else float("nan")

        rec, prec = _topk_metrics(y_true, y_score, alert_rate)

        rows.append(
            BaselineMetrics(
                name=name,
                pr_auc=pr_auc,
                brier=brier,
                recall_at_alert_rate=rec,
                precision_at_alert_rate=prec,
                alert_rate=float(alert_rate),
                n=int(len(y_true)),
                n_pos=int(y_true.sum()),
            ).__dict__
        )

    dfm = pd.DataFrame(rows).sort_values(["pr_auc"], ascending=False).reset_index(drop=True)
    return dfm
