"""Forward 12-month total-return labels + purge/embargo walk-forward fold logic.

Labels are built from `Close` in prices.parquet, which ingest.py pulled with
auto_adjust=True -- that series is already dividend+split adjusted, so a plain
close-to-close pct change over the 12-month window IS the total return
(Section 4 of the project plan). No separate dividend addback needed here.

Purge/embargo (Section 7): 12-month forward labels overlap heavily between
adjacent monthly rebalance dates, so a naive time split leaks. A training date
is only eligible against a given test date if its label window doesn't reach
into the test period: train_date <= test_date - EMBARGO_MONTHS. This module
owns that logic so every walk-forward fold (in backtest.py and model.py)
applies it identically, per the build order.
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, EMBARGO_MONTHS, FORWARD_RETURN_MONTHS


def build_labels(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    fwd_rows = []
    for yft, g in prices.groupby("yf_ticker"):
        close = g.sort_values("Date").set_index("Date")["Close"]
        last_date = close.index.max()
        dates = panel.loc[panel["yf_ticker"] == yft, "date"]
        for d in dates:
            end_date = d + pd.DateOffset(months=FORWARD_RETURN_MONTHS)
            p0 = close.asof(d)
            p1 = np.nan if end_date > last_date else close.asof(end_date)
            ret = (p1 / p0 - 1) if pd.notna(p0) and pd.notna(p1) and p0 != 0 else np.nan
            fwd_rows.append({
                "yf_ticker": yft, "date": d,
                "label_end_date": end_date, "forward_return_12m": ret,
            })
    fwd_df = pd.DataFrame(fwd_rows)
    return panel.merge(fwd_df, on=["yf_ticker", "date"], how="left")


def purge_embargo_folds(dates: pd.Series, min_train_dates: int = 12, step_months: int = 6):
    """Yield (train_dates, test_date) walk-forward folds with the label-horizon embargo applied."""
    uniq = sorted(pd.to_datetime(pd.Series(dates).unique()))
    for i in range(min_train_dates, len(uniq), max(step_months, 1)):
        test_date = uniq[i]
        cutoff = test_date - pd.DateOffset(months=EMBARGO_MONTHS)
        train_dates = [d for d in uniq if d <= cutoff]
        if len(train_dates) < min_train_dates:
            continue
        yield train_dates, test_date


def main():
    panel = pd.read_parquet(DATA_PROCESSED / "feature_panel.parquet")
    prices = pd.read_parquet(DATA_RAW / "prices.parquet")
    labeled = build_labels(panel, prices)

    out_path = DATA_PROCESSED / "labeled_panel.parquet"
    labeled.to_parquet(out_path, index=False)

    n_realized = labeled["forward_return_12m"].notna().sum()
    print(f"Wrote {len(labeled)} rows to {out_path}")
    print(f"Realized labels (12mo forward window already elapsed): "
          f"{n_realized} ({n_realized / len(labeled):.1%})")


if __name__ == "__main__":
    main()
