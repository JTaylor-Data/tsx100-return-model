"""SHAP explanations for the production LightGBM model (Section 6).

shap.TreeExplainer is fast and effectively exact for tree ensembles, so
computing per-stock SHAP for the full latest cross-section is cheap. Stores,
per stock, the top ~8 features by |SHAP| with signed contribution, plus the
global mean-|SHAP| feature ranking across the universe -- enough to drive both
a global "what the model cares about" chart and a per-stock waterfall bar
chart in the static dashboard (no matplotlib SHAP plots needed).

Output: docs/shap_values.json
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap

from config import DATA_PROCESSED, DOCS
from model import TARGET, get_feature_columns, train_lightgbm

TOP_K = 8


def _safe_val(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.floating, float, np.integer, int)):
        return round(float(v), 4)
    return str(v)


def main():
    df = pd.read_parquet(DATA_PROCESSED / "labeled_panel.parquet")
    feature_cols = get_feature_columns(df)
    realized = df[df[TARGET].notna()]

    model = train_lightgbm(realized, feature_cols)

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].reset_index(drop=True)
    X = latest[feature_cols].copy()
    X["sector"] = X["sector"].astype("category")

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    global_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
    )

    per_stock = []
    for i, row in latest.iterrows():
        contribs = pd.Series(shap_values[i], index=feature_cols)
        top_idx = contribs.abs().sort_values(ascending=False).index[:TOP_K]
        top = contribs.reindex(top_idx)
        per_stock.append({
            "ticker": row["ticker"],
            "predicted_return": round(float(base_value + contribs.sum()), 4),
            "top_features": [
                {"feature": f, "value": _safe_val(row[f]), "shap": round(float(v), 5)}
                for f, v in top.items()
            ],
        })

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(latest_date.date()),
        "base_value": round(base_value, 4),
        "global_importance": [
            {"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in global_importance.items()
        ],
        "stocks": per_stock,
    }
    out_path = DOCS / "shap_values.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} ({len(per_stock)} stocks)")
    print("\nTop 10 global features by mean |SHAP|:")
    print(global_importance.head(10))


if __name__ == "__main__":
    main()
