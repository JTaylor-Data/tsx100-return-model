"""Pull price history, annual fundamentals, and macro series for the universe.

Usage:
    python ingest.py            # full 100-stock universe
    python ingest.py 10         # small test slice (first N tickers) for plumbing checks

Outputs (data/raw/):
    prices.parquet        daily OHLC-ish [yf_ticker, Date, Close, Volume, Dividends]
    fundamentals.parquet  annual statement line items [yf_ticker, fiscal_date, ...]
    macro.parquet         daily macro series shared across all tickers
"""
import sys
import time

import pandas as pd
import requests
import yfinance as yf

from config import (
    BENCHMARK_TICKER,
    BOC_SERIES,
    BOC_VALET_URL,
    DATA_RAW,
    HTTP_HEADERS,
    OIL_TICKER,
    PRICE_HISTORY_YEARS,
    USDCAD_TICKER,
)
from universe import load_universe

# Annual statement line items pulled per ticker. Not every sector reports every
# item (e.g. banks have no "Gross Profit") -- missing ones are left as NaN and
# LightGBM's native NaN handling deals with it downstream.
INCOME_KEYS = [
    "Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income",
    "Diluted EPS", "Diluted Average Shares",
]
BALANCE_KEYS = [
    "Total Assets", "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents",
    "Ordinary Shares Number",
]


def fetch_price_history(yf_ticker: str, years: int = PRICE_HISTORY_YEARS):
    hist = yf.Ticker(yf_ticker).history(period=f"{years}y", auto_adjust=True, actions=True)
    if hist.empty:
        return None
    hist = hist.reset_index()[["Date", "Close", "Volume", "Dividends"]]
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist["yf_ticker"] = yf_ticker
    return hist


def fetch_annual_fundamentals(yf_ticker: str):
    t = yf.Ticker(yf_ticker)
    inc, bs = t.financials, t.balance_sheet
    if inc is None or inc.empty or bs is None or bs.empty:
        return None
    dates = sorted(set(inc.columns) & set(bs.columns))
    rows = []
    for d in dates:
        row = {"yf_ticker": yf_ticker, "fiscal_date": pd.Timestamp(d)}
        for k in INCOME_KEYS:
            row[k] = inc.loc[k, d] if k in inc.index else None
        for k in BALANCE_KEYS:
            row[k] = bs.loc[k, d] if k in bs.index else None
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_macro():
    series_str = ",".join(BOC_SERIES.values())
    url = BOC_VALET_URL.format(series=series_str)
    resp = requests.get(url, params={"start_date": "2005-01-01"}, headers=HTTP_HEADERS, timeout=60)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    raw = pd.DataFrame(obs)
    code_to_name = {v: k for k, v in BOC_SERIES.items()}

    out = pd.DataFrame({"date": pd.to_datetime(raw["d"])})
    for code, name in code_to_name.items():
        out[name] = raw[code].apply(lambda x: float(x["v"]) if isinstance(x, dict) and "v" in x else None)

    def yf_daily_close(ticker, colname):
        h = yf.Ticker(ticker).history(period="max", auto_adjust=True)[["Close"]].reset_index()
        h.columns = ["date", colname]
        h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
        return h

    oil = yf_daily_close(OIL_TICKER, "wti_oil")
    bench = yf_daily_close(BENCHMARK_TICKER, "tsx_close")
    usdcad = yf_daily_close(USDCAD_TICKER, "usdcad")

    out = out.merge(oil, on="date", how="outer").merge(bench, on="date", how="outer").merge(usdcad, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main(limit=None):
    universe = load_universe()
    if limit:
        universe = universe.head(limit)

    price_frames, fund_frames, failed = [], [], []
    for i, row in universe.iterrows():
        yft = row["yf_ticker"]
        try:
            p = fetch_price_history(yft)
            if p is not None:
                price_frames.append(p)
            f = fetch_annual_fundamentals(yft)
            if f is not None:
                fund_frames.append(f)
            print(f"[{i + 1}/{len(universe)}] {yft} ok "
                  f"(price rows={len(p) if p is not None else 0}, "
                  f"fund rows={len(f) if f is not None else 0})")
        except Exception as e:
            print(f"[{i + 1}/{len(universe)}] {yft} FAILED: {e}")
            failed.append(yft)
        time.sleep(0.3)

    prices = pd.concat(price_frames, ignore_index=True)
    prices.to_parquet(DATA_RAW / "prices.parquet", index=False)

    funds = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()
    funds.to_parquet(DATA_RAW / "fundamentals.parquet", index=False)

    macro = fetch_macro()
    macro.to_parquet(DATA_RAW / "macro.parquet", index=False)

    print(f"\nDone. Prices: {len(prices)} rows / {prices['yf_ticker'].nunique()} tickers.")
    print(f"Fundamentals: {len(funds)} rows / {funds['yf_ticker'].nunique() if not funds.empty else 0} tickers.")
    print(f"Macro: {len(macro)} rows.")
    if failed:
        print(f"Failed tickers ({len(failed)}): {failed}")


if __name__ == "__main__":
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit_arg)
