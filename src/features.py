"""Build the monthly feature panel from raw prices, fundamentals, and macro data.

Point-in-time discipline:
- Fundamentals are joined via merge_asof with `available_date = fiscal_date +
  REPORTING_LAG_DAYS`, so a stock's Q4/annual filing only becomes visible to the
  model after it would realistically have been public. yfinance only exposes
  ~5 years of annual statements for most TSX names, so fundamental columns are
  NaN before a ticker's first available filing -- LightGBM handles that
  natively, it isn't imputed.
- Price-dependent ratios (P/E, P/B, EV/EBITDA, div yield, payout ratio) use the
  *current* monthly close against the latest lagged fundamentals, not the
  historical filing-date price, so they move with price between filings.

Output: data/processed/feature_panel.parquet
    [date, yf_ticker, ticker, sector, close, <~27 feature columns>, fwd flag cols]
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, REBALANCE_FREQ, REPORTING_LAG_DAYS
from universe import load_universe

FUND_RAW_COLS = [
    "Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income",
    "Diluted EPS", "Diluted Average Shares", "Total Assets", "Stockholders Equity",
    "Total Debt", "Cash And Cash Equivalents", "Ordinary Shares Number",
]


def _load_raw():
    prices = pd.read_parquet(DATA_RAW / "prices.parquet")
    funds = pd.read_parquet(DATA_RAW / "fundamentals.parquet")
    macro = pd.read_parquet(DATA_RAW / "macro.parquet")
    universe = load_universe()
    return prices, funds, macro, universe


def _monthly_snapshots(prices: pd.DataFrame) -> pd.DataFrame:
    """First trading day per ticker per calendar month -> the panel's rebalance dates."""
    out = []
    for yft, g in prices.groupby("yf_ticker"):
        g = g.sort_values("Date").set_index("Date")
        snap = g.resample(REBALANCE_FREQ).first().dropna(subset=["Close"])
        snap = snap.reset_index().rename(columns={"Date": "date"})
        snap["yf_ticker"] = yft
        out.append(snap[["date", "yf_ticker", "Close"]].rename(columns={"Close": "close"}))
    return pd.concat(out, ignore_index=True)


def _technical_features(prices: pd.DataFrame, monthly: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    bench_close = benchmark.set_index("date")["tsx_close"].sort_index()

    rows = []
    for yft, g in prices.groupby("yf_ticker"):
        g = g.sort_values("Date")
        close = g.set_index("Date")["Close"]
        ret = close.pct_change()
        vol63 = (ret.rolling("91D").std() * np.sqrt(252)).reindex(close.index)
        vol252 = (ret.rolling("365D").std() * np.sqrt(252)).reindex(close.index)

        # rolling beta vs TSX Composite over a ~252 trading day / 365 calendar day window
        bench_ret = bench_close.pct_change()
        joined = pd.DataFrame({"stock": ret, "bench": bench_ret}).dropna()
        cov = joined["stock"].rolling("365D").cov(joined["bench"])
        var = joined["bench"].rolling("365D").var()
        beta = (cov / var)

        snap_dates = monthly.loc[monthly["yf_ticker"] == yft, "date"]
        for d in snap_dates:
            def mom(months):
                p0 = close.asof(d - pd.DateOffset(months=months))
                p1 = close.asof(d)
                if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                    return np.nan
                return p1 / p0 - 1

            rows.append({
                "yf_ticker": yft,
                "date": d,
                "mom_1m": mom(1), "mom_3m": mom(3), "mom_6m": mom(6), "mom_12m": mom(12),
                "vol_63d": vol63.asof(d) if d in vol63.index or vol63.index.min() <= d else np.nan,
                "vol_252d": vol252.asof(d) if len(vol252) else np.nan,
                "beta_252d": beta.asof(d) if len(beta) else np.nan,
            })
    return pd.DataFrame(rows)


def _fundamental_features(funds: pd.DataFrame) -> pd.DataFrame:
    f = funds.sort_values(["yf_ticker", "fiscal_date"]).copy()
    f["available_date"] = f["fiscal_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    f["revenue_growth_yoy"] = f.groupby("yf_ticker")["Total Revenue"].pct_change()
    return f[["yf_ticker", "available_date"] + FUND_RAW_COLS + ["revenue_growth_yoy"]]


def _trailing_dividend_yield(prices: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yft, g in prices.groupby("yf_ticker"):
        g = g.sort_values("Date").set_index("Date")
        trailing_div = g["Dividends"].rolling("365D").sum()
        snap_dates = monthly.loc[monthly["yf_ticker"] == yft, "date"]
        for d in snap_dates:
            rows.append({"yf_ticker": yft, "date": d,
                         "trailing_div_per_share": trailing_div.asof(d) if len(trailing_div) else np.nan})
    return pd.DataFrame(rows)


def _macro_features(macro: pd.DataFrame, monthly_dates: pd.Series) -> pd.DataFrame:
    m = macro.sort_values("date").ffill()
    m = m.set_index("date")
    rows = []
    for d in sorted(monthly_dates.unique()):
        d = pd.Timestamp(d)
        boc = m["boc_overnight_rate"].asof(d)
        y2 = m["yield_2y"].asof(d)
        y10 = m["yield_10y"].asof(d)
        usdcad = m["usdcad"].asof(d)
        usdcad_3m = m["usdcad"].asof(d - pd.DateOffset(months=3))
        oil = m["wti_oil"].asof(d)
        oil_3m = m["wti_oil"].asof(d - pd.DateOffset(months=3))
        rows.append({
            "date": d,
            "boc_rate": boc,
            "yield_curve_2s10s": (y10 - y2) if pd.notna(y10) and pd.notna(y2) else np.nan,
            "usdcad_chg_3m": (usdcad / usdcad_3m - 1) if usdcad_3m else np.nan,
            "oil_chg_3m": (oil / oil_3m - 1) if oil_3m else np.nan,
        })
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    prices, funds, macro, universe = _load_raw()
    benchmark = macro[["date", "tsx_close"]].dropna()

    monthly = _monthly_snapshots(prices)
    tech = _technical_features(prices, monthly, benchmark)
    div = _trailing_dividend_yield(prices, monthly)
    macro_feat = _macro_features(macro, monthly["date"])
    fund_pit = _fundamental_features(funds)

    panel = monthly.merge(tech, on=["yf_ticker", "date"], how="left")
    panel = panel.merge(div, on=["yf_ticker", "date"], how="left")
    panel = panel.merge(universe[["yf_ticker", "ticker", "sector"]], on="yf_ticker", how="left")
    panel = panel.merge(macro_feat, on="date", how="left")

    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    fund_pit = fund_pit.copy()
    fund_pit["available_date"] = pd.to_datetime(fund_pit["available_date"]).astype("datetime64[ns]")

    panel = panel.sort_values("date")
    fund_pit = fund_pit.sort_values("available_date")
    panel = pd.merge_asof(panel, fund_pit, left_on="date", right_on="available_date",
                           by="yf_ticker", direction="backward")

    shares = panel["Ordinary Shares Number"]
    market_cap = panel["close"] * shares
    eps = panel["Diluted EPS"].where(panel["Diluted EPS"] > 0)
    equity = panel["Stockholders Equity"].where(panel["Stockholders Equity"] > 0)
    ebitda = panel["EBITDA"].where(panel["EBITDA"] > 0)
    revenue = panel["Total Revenue"].where(panel["Total Revenue"] > 0)

    panel["pe_ratio"] = panel["close"] / eps
    panel["pb_ratio"] = market_cap / equity
    panel["ev_ebitda"] = (market_cap + panel["Total Debt"] - panel["Cash And Cash Equivalents"]) / ebitda
    panel["roe"] = panel["Net Income"] / equity
    panel["roa"] = panel["Net Income"] / panel["Total Assets"].where(panel["Total Assets"] > 0)
    panel["gross_margin"] = panel["Gross Profit"] / revenue
    panel["operating_margin"] = panel["Operating Income"] / revenue
    panel["debt_equity"] = panel["Total Debt"] / equity
    panel["div_yield"] = panel["trailing_div_per_share"] / panel["close"]
    panel["payout_ratio"] = panel["trailing_div_per_share"] / eps

    # cross-sectional z-scores, computed within each rebalance date across the whole universe
    def zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    panel["value_z_universe"] = panel.groupby("date")["pe_ratio"].transform(
        lambda s: zscore(-s))  # cheaper (lower P/E) => higher z
    panel["momentum_z_universe"] = panel.groupby("date")["mom_12m"].transform(zscore)

    # sector-relative momentum
    sector_mom = panel.groupby(["date", "sector"])["mom_3m"].transform("mean")
    panel["rel_strength_sector"] = panel["mom_3m"] - sector_mom

    bench_close = benchmark.set_index("date")["tsx_close"].sort_index()

    def bench_mom_3m(d):
        p0 = bench_close.asof(d - pd.DateOffset(months=3))
        p1 = bench_close.asof(d)
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            return np.nan
        return p1 / p0 - 1

    tsx_mom_by_date = {d: bench_mom_3m(d) for d in panel["date"].unique()}
    panel["rel_strength_tsx"] = panel["mom_3m"] - panel["date"].map(tsx_mom_by_date)

    feature_cols = [
        "mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_63d", "vol_252d", "beta_252d",
        "rel_strength_tsx", "rel_strength_sector",
        "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield", "payout_ratio", "roe", "roa",
        "gross_margin", "operating_margin", "revenue_growth_yoy", "debt_equity",
        "boc_rate", "yield_curve_2s10s", "usdcad_chg_3m", "oil_chg_3m",
        "value_z_universe", "momentum_z_universe", "sector",
    ]
    keep = ["date", "yf_ticker", "ticker", "close"] + feature_cols
    panel = panel[keep].copy()

    # a handful of ratios divide by a prior value that was ~0 (e.g. revenue_growth_yoy
    # off a near-zero prior-year base) -- treat as missing rather than +/-inf.
    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    panel[numeric_cols] = panel[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return panel


def main():
    panel = build_panel()
    out_path = DATA_PROCESSED / "feature_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {len(panel)} rows x {panel.shape[1]} cols to {out_path}")
    print(f"Date range: {panel['date'].min()} -> {panel['date'].max()}")
    print(f"Tickers: {panel['yf_ticker'].nunique()}")
    print("\nNaN rate per feature column:")
    print((panel.isna().mean() * 100).round(1).sort_values(ascending=False))


if __name__ == "__main__":
    main()
