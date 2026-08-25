"""Recent price history for the dashboard's per-stock chart, plus a projected
forward path from each stock's predicted 12mo return (Section 8 detail view).

The "future" leg isn't a second model -- it's a straight-line glide path from
today's close to close * (1 + predicted_return_12m) at the 12-month mark, so
the dashboard can plot it as a visually distinct continuation of the actual
price line. It's a visualization of the prediction already in rankings.json,
not a new forecast.

Output: docs/price_history.json
"""
import json
from datetime import datetime, timezone

import pandas as pd

from config import DATA_RAW, DOCS, FORWARD_RETURN_MONTHS

HISTORY_MONTHS = 24
PROJECTION_POINTS = 12  # one point per month out to the 12mo horizon


def build_history(prices: pd.DataFrame, rankings: dict) -> dict:
    month_start = pd.to_datetime(rankings["as_of_date"])
    cutoff = month_start - pd.DateOffset(months=HISTORY_MONTHS)
    stocks_by_yft = {s["yf_ticker"]: s for s in rankings["stocks"]}

    out = {}
    for yft, g in prices.groupby("yf_ticker"):
        stock = stocks_by_yft.get(yft)
        if stock is None or stock["current_price"] is None:
            continue
        g = g.sort_values("Date")

        # the feature panel snapshots the first trading day ON OR AFTER month_start
        # (features._monthly_snapshots), which may be a few days later than the
        # as_of_date label itself -- find that same actual trading date here so the
        # chart's last "actual" point lines up exactly with the current_price card.
        at_or_after = g[g["Date"] >= month_start]
        if at_or_after.empty:
            continue
        ref_date = at_or_after.iloc[0]["Date"]
        ref_close = stock["current_price"]

        hist_before_ref = g[(g["Date"] >= cutoff) & (g["Date"] < ref_date)]
        weekly = hist_before_ref.set_index("Date")["Close"].resample("W-FRI").last().dropna()
        weekly = pd.concat([weekly, pd.Series([ref_close], index=[ref_date])])

        last_date, last_close = ref_date, float(ref_close)
        pred_return = stock["predicted_return_12m"]
        target_close = last_close * (1 + pred_return) if pred_return is not None else None

        projected_dates, projected_close = [], []
        if target_close is not None:
            for i in range(1, PROJECTION_POINTS + 1):
                frac = i / PROJECTION_POINTS
                d = last_date + pd.DateOffset(months=FORWARD_RETURN_MONTHS * frac)
                projected_dates.append(d.strftime("%Y-%m-%d"))
                projected_close.append(round(last_close + (target_close - last_close) * frac, 4))

        out[stock["ticker"]] = {
            "history_dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
            "history_close": [round(float(v), 4) for v in weekly.values],
            "projected_dates": projected_dates,
            "projected_close": projected_close,
        }
    return out


def main():
    prices = pd.read_parquet(DATA_RAW / "prices.parquet")
    with open(DOCS / "rankings.json") as f:
        rankings = json.load(f)

    history = build_history(prices, rankings)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": rankings["as_of_date"],
        "history_months": HISTORY_MONTHS,
        "stocks": history,
    }
    out_path = DOCS / "price_history.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {out_path} ({len(history)} tickers)")


if __name__ == "__main__":
    main()
