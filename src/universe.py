"""Build the top-100 TSX universe from iShares XIC (S&P/TSX Capped Composite) holdings.

Re-run periodically (quarterly is plenty -- constituent weights don't move fast).
Output: data/raw/universe.csv with columns [ticker, yf_ticker, name, sector, weight]
"""
import io

import pandas as pd
import requests

from config import HTTP_HEADERS, UNIVERSE_FILE, UNIVERSE_SIZE, XIC_HOLDINGS_URL


def _to_yf_ticker(ishares_ticker: str) -> str:
    """iShares uses e.g. 'CRT.UN', 'BAM.A'; Yahoo Finance wants 'CRT-UN.TO', 'BAM-A.TO'."""
    return ishares_ticker.replace(".", "-") + ".TO"


def fetch_holdings() -> pd.DataFrame:
    params = {"fileType": "csv", "fileName": "XIC_holdings", "dataType": "fund"}
    resp = requests.get(XIC_HOLDINGS_URL, params=params, headers=HTTP_HEADERS, timeout=30)
    resp.raise_for_status()

    # First two lines are an "as of" date + blank line before the real header row.
    text = resp.content.decode("utf-8-sig")
    lines = text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
    csv_body = "\n".join(lines[header_idx:])

    # keep_default_na=False: avoid pandas reading the ticker "NA" (National Bank of Canada) as NaN
    df = pd.read_csv(io.StringIO(csv_body), thousands=",", keep_default_na=False,
                      na_values=["", " "])
    return df


def build_universe(size: int = UNIVERSE_SIZE) -> pd.DataFrame:
    raw = fetch_holdings()

    equity = raw[raw["Asset Class"] == "Equity"].copy()
    equity = equity[equity["Weight (%)"] > 0]
    equity = equity.sort_values("Weight (%)", ascending=False).head(size)

    out = pd.DataFrame({
        "ticker": equity["Ticker"].str.strip(),
        "name": equity["Name"].str.strip(),
        "sector": equity["Sector"].str.strip(),
        "weight": equity["Weight (%)"].astype(float),
    })
    out["yf_ticker"] = out["ticker"].apply(_to_yf_ticker)
    out = out.reset_index(drop=True)
    return out


def load_universe() -> pd.DataFrame:
    """Read back data/raw/universe.csv, guarding against pandas reading the
    ticker "NA" (National Bank of Canada) as a missing value."""
    return pd.read_csv(UNIVERSE_FILE, keep_default_na=False, na_values=[""])


def main():
    universe = build_universe()
    universe.to_csv(UNIVERSE_FILE, index=False)
    print(f"Wrote {len(universe)} tickers to {UNIVERSE_FILE}")
    print(universe["sector"].value_counts())


if __name__ == "__main__":
    main()
