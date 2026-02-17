from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from windermere_project.labels import LabelBuilder, LabelConfig
from windermere_project.baselines import (
    PersistenceBaseline,
    SeasonalMeanBaseline,
    DummyBaseline,
)
from windermere_project.baselines.evaluate_baselines import evaluate_baselines


FEATURE_PATH = "data/features/features_raw_NW-88010013_ALLFULL_20260216T055951Z_lb30_FEAT_V1.parquet"
OUT_DIR = Path("reports/phase3")
FIG_DIR = OUT_DIR / "figures"


def add_month_col(df: pd.DataFrame) -> pd.DataFrame:
    # Your time column in feature matrix is phenomenonTime by config
    t = pd.to_datetime(df["phenomenonTime"], errors="coerce")
    out = df.copy()
    out["month"] = t.dt.month.astype(int)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURE_PATH)
    df = add_month_col(df)

    # Ensure labels are governed by LabelBuilder (even if y already exists)
    lb = LabelBuilder(LabelConfig(chl_value_col="chl_ugL", threshold_ugL=20.0, label_col="y"))
    df = lb.build_from_feature_matrix(df)
    label_summary = lb.audit_summary(df)

    # Time-aware split: train 2005-2018, test 2019-2025 (simple baseline evaluation)
    t = pd.to_datetime(df["phenomenonTime"], errors="coerce")
    df["year"] = t.dt.year

    train = df[df["year"] <= 2018].copy()
    test = df[df["year"] >= 2019].copy()

    # Minimal X for DummyBaseline (it ignores features, but sklearn requires X)
    # We'll feed month_sin/month_cos if present, else month.
    X_cols = [c for c in ["month_sin", "month_cos", "doy_sin", "doy_cos"] if c in df.columns]
    if not X_cols:
        X_cols = ["month"]

    X_train = train[X_cols].fillna(0.0)
    y_train = train["y"].astype(int)

    X_test = test[X_cols].fillna(0.0)
    y_test = test["y"].astype(int)

    # Baselines
    res = []

    pers = PersistenceBaseline().fit(train, y_col="y", site_col="samplingPoint.notation", time_col="phenomenonTime")
    r1 = pers.predict(test, y_col="y", site_col="samplingPoint.notation", time_col="phenomenonTime")
    res.append({"name": r1.name, "y_true": r1.y_true, "y_score": r1.y_score, "y_pred": r1.y_pred})

    seas = SeasonalMeanBaseline(month_col="month").fit(train, y_col="y")
    r2 = seas.predict(test, y_col="y")
    res.append({"name": r2.name, "y_true": r2.y_true, "y_score": r2.y_score, "y_pred": r2.y_pred})

    dummy = DummyBaseline(strategy="stratified", random_state=42).fit(X_train, y_train)
    r3 = dummy.predict(X_test, y_test)
    res.append({"name": r3.name, "y_true": r3.y_true, "y_score": r3.y_score, "y_pred": r3.y_pred})

    # Evaluate
    metrics = evaluate_baselines(res, alert_rate=0.10)
    metrics_path = OUT_DIR / "baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # PR curves + simple calibration plots
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    # PR curve
    plt.figure()
    for r in res:
        p, rcl, _ = precision_recall_curve(r["y_true"], r["y_score"])
        plt.plot(rcl, p, label=r["name"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Baseline Precision–Recall Curves (Test: 2019–2025)")
    plt.legend()
    plt.tight_layout()
    pr_path = FIG_DIR / "baseline_pr_curves.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # Calibration (binned)
    def calib_plot(y_true, y_score, name):
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(y_score, bins) - 1
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() == 0:
                continue
            xs.append(float(y_score[mask].mean()))
            ys.append(float(y_true[mask].mean()))
        return xs, ys

    plt.figure()
    for r in res:
        xs, ys = calib_plot(np.asarray(r["y_true"]), np.asarray(r["y_score"]), r["name"])
        plt.plot(xs, ys, marker="o", label=r["name"])
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Observed frequency")
    plt.title("Baseline Calibration (Test: 2019–2025)")
    plt.legend()
    plt.tight_layout()
    cal_path = FIG_DIR / "baseline_calibration.png"
    plt.savefig(cal_path, dpi=200)
    plt.close()

    # Markdown report
    md_path = OUT_DIR / "baseline_report.md"
    lines = []
    lines.append("# Phase 3 — Baseline Benchmarking Report\n\n")
    lines.append("## Label Definition\n")
    lines.append(f"- Threshold: 20 µg/L (strictly >)\n")
    lines.append(f"- Label fingerprint: {label_summary['label_config_fingerprint']}\n")
    lines.append(f"- n: {label_summary['n']} | n_pos: {label_summary['n_pos']} | pos_rate: {label_summary['pos_rate']:.4f}\n\n")

    lines.append("## Split\n")
    lines.append("- Train: 2005–2018\n")
    lines.append("- Test: 2019–2025\n\n")

    lines.append("## Metrics (alert_rate = 10%)\n\n")
    lines.append(metrics.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Figures\n")
    lines.append(f"- {pr_path.as_posix()}\n")
    lines.append(f"- {cal_path.as_posix()}\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("Saved metrics:", metrics_path)
    print("Saved report:", md_path)
    print("Saved figures:", pr_path, cal_path)
    print(metrics)


if __name__ == "__main__":
    main()
