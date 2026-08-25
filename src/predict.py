"""Assemble docs/rankings.json: rank all 100 stocks by predicted 12-month
forward return, using the production LightGBM model trained on all realized
history (Section 8 dashboard needs a full ranked table + per-stock snapshot).
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DOCS
from model import TARGET, get_feature_columns, predict_lightgbm, train_lightgbm
from universe import load_universe

SNAPSHOT_FIELDS = [
    "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "payout_ratio",
    "roe", "roa", "gross_margin", "operating_margin", "revenue_growth_yoy",
    "debt_equity", "mom_12m", "vol_252d", "beta_252d",
]


def _safe(v):
    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isinf(v))):
        return None
    return round(float(v), 4)


def main():
    df = pd.read_parquet(DATA_PROCESSED / "labeled_panel.parquet")
    universe = load_universe().set_index("yf_ticker")
    feature_cols = get_feature_columns(df)
    realized = df[df[TARGET].notna()]

    model = train_lightgbm(realized, feature_cols)

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    latest["predicted_return_12m"] = predict_lightgbm(model, latest, feature_cols)
    latest = latest.sort_values("predicted_return_12m", ascending=False).reset_index(drop=True)
    latest["rank"] = latest.index + 1

    stocks = []
    for _, row in latest.iterrows():
        u = universe.loc[row["yf_ticker"]] if row["yf_ticker"] in universe.index else None
        stocks.append({
            "rank": int(row["rank"]),
            "ticker": row["ticker"],
            "yf_ticker": row["yf_ticker"],
            "name": u["name"] if u is not None else None,
            "sector": row["sector"],
            "predicted_return_12m": _safe(row["predicted_return_12m"]),
            "current_price": _safe(row["close"]),
            "snapshot": {f: _safe(row[f]) for f in SNAPSHOT_FIELDS},
        })

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(latest_date.date()),
        "n_stocks": len(stocks),
        "model": "lightgbm",
        "n_features": len(feature_cols),
        "n_training_rows": int(len(realized)),
        "stocks": stocks,
    }
    out_path = DOCS / "rankings.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} ({len(stocks)} stocks, as of {out['as_of_date']})")
    print("\nTop 5 predicted:")
    for s in stocks[:5]:
        print(f"  {s['rank']:>3} {s['ticker']:<8} {s['predicted_return_12m']:+.1%}  {s['sector']}")
    print("Bottom 5 predicted:")
    for s in stocks[-5:]:
        print(f"  {s['rank']:>3} {s['ticker']:<8} {s['predicted_return_12m']:+.1%}  {s['sector']}")


if __name__ == "__main__":
    main()
