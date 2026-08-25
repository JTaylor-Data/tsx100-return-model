"""Walk-forward validation with purge/embargo folds.

Headline metrics, in trust order (Section 7):
  1. rank_ic       -- Spearman correlation between predicted and actual return
  2. decile_spread -- mean actual return of predicted top decile minus bottom decile
  3. rmse / r2     -- reported for completeness, not the main story (Section 4)

Compares the LightGBM model against an Elastic Net baseline and a naive
"rank by trailing 12mo momentum" rule, so the dashboard can show the GBM
actually beating dumb baselines rather than its number in isolation.

Output: docs/backtest_history.json
"""
import json
import warnings
from datetime import datetime, timezone

# early folds have zero fundamental-statement coverage (first filing year hasn't
# happened yet), so some fundamental columns are all-NaN within that fold --
# SimpleImputer warns and skips them, which is expected, not a bug.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from config import DATA_PROCESSED, DOCS
from labels import purge_embargo_folds
from model import (
    TARGET,
    get_feature_columns,
    predict_elastic_net,
    predict_lightgbm,
    predict_naive_momentum,
    train_elastic_net,
    train_lightgbm,
)

MODEL_NAMES = ["lightgbm", "elastic_net", "naive_momentum"]


def decile_spread(actual: np.ndarray, predicted: np.ndarray, n: int = 10) -> float:
    order = np.argsort(-predicted)
    k = max(len(order) // n, 1)
    top = actual[order[:k]]
    bottom = actual[order[-k:]]
    return float(np.mean(top) - np.mean(bottom))


def evaluate(actual: np.ndarray, predicted: np.ndarray):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) < 10:
        return None
    rho, _ = spearmanr(predicted, actual)
    return {
        "rank_ic": float(rho),
        "decile_spread": decile_spread(actual, predicted),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "n": int(len(actual)),
    }


def run_backtest(min_train_dates: int = 12, step_months: int = 6):
    df = pd.read_parquet(DATA_PROCESSED / "labeled_panel.parquet")
    feature_cols = get_feature_columns(df)
    realized = df[df[TARGET].notna()].copy()

    fold_metrics = {name: [] for name in MODEL_NAMES}
    periods = []

    for train_dates, test_date in purge_embargo_folds(realized["date"], min_train_dates, step_months):
        train_df = realized[realized["date"].isin(train_dates)]
        test_df = realized[realized["date"] == test_date]
        if len(train_df) < 50 or len(test_df) < 10:
            continue

        actual = test_df[TARGET].values
        preds = {
            "lightgbm": predict_lightgbm(train_lightgbm(train_df, feature_cols), test_df, feature_cols),
            "elastic_net": predict_elastic_net(train_elastic_net(train_df, feature_cols), test_df, feature_cols),
            "naive_momentum": predict_naive_momentum(test_df),
        }

        for name, pred in preds.items():
            m = evaluate(actual, np.asarray(pred, dtype=float))
            if m:
                m["test_date"] = str(test_date.date())
                fold_metrics[name].append(m)

        periods.append({
            "test_date": str(test_date.date()),
            "ticker": test_df["ticker"].tolist(),
            "actual_return": [None if pd.isna(v) else round(float(v), 4) for v in actual],
            "predicted_return": [round(float(v), 4) for v in preds["lightgbm"]],
        })

        print(f"fold test={test_date.date()} train_n={len(train_df)} test_n={len(test_df)} "
              f"lgb_ic={fold_metrics['lightgbm'][-1]['rank_ic']:.3f}" if fold_metrics["lightgbm"] else "")

    return fold_metrics, periods, feature_cols


def summarize(fold_metrics: dict) -> dict:
    summary = {}
    for name, folds in fold_metrics.items():
        if not folds:
            summary[name] = None
            continue
        summary[name] = {
            "n_folds": len(folds),
            "mean_rank_ic": float(np.mean([f["rank_ic"] for f in folds])),
            "mean_decile_spread": float(np.mean([f["decile_spread"] for f in folds])),
            "mean_rmse": float(np.mean([f["rmse"] for f in folds])),
            "mean_r2": float(np.mean([f["r2"] for f in folds])),
        }
    return summary


def main():
    fold_metrics, periods, feature_cols = run_backtest()
    summary = summarize(fold_metrics)

    print("\n=== Backtest summary (mean across walk-forward folds) ===")
    for name, s in summary.items():
        if s is None:
            print(f"{name}: no folds evaluated")
            continue
        print(f"{name}: rank_ic={s['mean_rank_ic']:.3f}  decile_spread={s['mean_decile_spread']:.3f}  "
              f"rmse={s['mean_rmse']:.3f}  r2={s['mean_r2']:.3f}  (n_folds={s['n_folds']})")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "fold_metrics": fold_metrics,
        "periods": periods,
    }
    out_path = DOCS / "backtest_history.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
