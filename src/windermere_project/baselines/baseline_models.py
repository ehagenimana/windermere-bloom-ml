from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


@dataclass(frozen=True)
class BaselineResult:
    name: str
    y_true: np.ndarray
    y_score: np.ndarray  # probability-like score for PR curve
    y_pred: np.ndarray   # hard predictions (0/1)
    meta: Dict[str, object]


class PersistenceBaseline:
    """
    Predict y(t) = y(t-1) per site (chronological).
    Produces probability-like score as {0.0, 1.0}.
    """
    name = "persistence"

    def fit(self, df_train: pd.DataFrame, *, y_col: str, site_col: str, time_col: str) -> "PersistenceBaseline":
        # No parameters to learn
        return self

    def predict(self, df: pd.DataFrame, *, y_col: str, site_col: str, time_col: str) -> BaselineResult:
        d = df.sort_values([site_col, time_col]).copy()
        y_true = d[y_col].astype(int).values

        prev = d.groupby(site_col)[y_col].shift(1)
        # If no previous label, default to majority class of available labels in df (safe fallback)
        fallback = int(d[y_col].mode().iloc[0]) if len(d) else 0
        y_pred = prev.fillna(fallback).astype(int).values
        y_score = y_pred.astype(float)

        return BaselineResult(
            name=self.name,
            y_true=y_true,
            y_score=y_score,
            y_pred=y_pred,
            meta={"fallback_class": fallback},
        )


class SeasonalMeanBaseline:
    """
    Predict P(y=1 | month) learned from TRAIN only.
    Requires month_col already in df (e.g., derived from phenomenonTime).
    """
    name = "seasonal_mean"

    def __init__(self, month_col: str = "month"):
        self.month_col = month_col
        self.p_by_month: Dict[int, float] = {}
        self.global_p: float = 0.0

    def fit(self, df_train: pd.DataFrame, *, y_col: str) -> "SeasonalMeanBaseline":
        d = df_train.copy()
        if self.month_col not in d.columns:
            raise ValueError(f"Missing '{self.month_col}' column for SeasonalMeanBaseline")

        self.global_p = float(d[y_col].mean()) if len(d) else 0.0
        grp = d.groupby(self.month_col)[y_col].mean()
        self.p_by_month = {int(k): float(v) for k, v in grp.to_dict().items()}
        return self

    def predict(self, df: pd.DataFrame, *, y_col: str) -> BaselineResult:
        d = df.copy()
        if self.month_col not in d.columns:
            raise ValueError(f"Missing '{self.month_col}' column for SeasonalMeanBaseline")

        y_true = d[y_col].astype(int).values
        months = d[self.month_col].astype(int).values
        y_score = np.array([self.p_by_month.get(int(m), self.global_p) for m in months], dtype=float)
        y_pred = (y_score >= 0.5).astype(int)

        return BaselineResult(
            name=self.name,
            y_true=y_true,
            y_score=y_score,
            y_pred=y_pred,
            meta={"global_p": self.global_p, "p_by_month": self.p_by_month},
        )


class DummyBaseline:
    """
    sklearn DummyClassifier baseline (stratified by default).
    """
    name = "dummy_stratified"

    def __init__(self, strategy: str = "stratified", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state
        self.model: Optional[DummyClassifier] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "DummyBaseline":
        self.model = DummyClassifier(strategy=self.strategy, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame, y_true: pd.Series) -> BaselineResult:
        if self.model is None:
            raise RuntimeError("DummyBaseline must be fit() before predict().")

        y_true_arr = y_true.astype(int).values
        proba = self.model.predict_proba(X)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        return BaselineResult(
            name=self.name,
            y_true=y_true_arr,
            y_score=proba.astype(float),
            y_pred=y_pred,
            meta={"strategy": self.strategy, "random_state": self.random_state},
        )
