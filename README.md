# TSX Top-100 Return Ranking Model

*This is a toy, proof of conecpt model. Not a serious portfolio project*

Predicts 12-month forward total return for the top 100 TSX stocks (by S&P/TSX
Capped Composite weight), ranks all 100 from highest to lowest expected
return, and renders a static dashboard with predicted-vs-actual tracking,
financial snapshots, and SHAP explanations per stock.

**Live dashboard:** `docs/index.html` (deploy via GitHub Pages, see below).

## Known limitation

Point forecasts of 12-month equity returns have inherently low R² in the
literature — markets are close to efficient at that horizon, and no model
here changes that. This isn't a bug to fix; it's why evaluation leans on
**rank quality** (Spearman IC, decile spread) rather than point accuracy as
the headline metric. See `docs/backtest_history.json` / the "Predicted vs.
Actual" tab for the honest picture.

Fundamental ratios (P/E, ROE, margins, etc.) are point-in-time joined from
each company's annual filings via yfinance, which only exposes ~5 years of
history for most TSX names. Rows before a ticker's first available filing
have NaN fundamentals — technical/macro features still populate, and
LightGBM's native missing-value handling deals with the rest natively (no
imputation). See `src/features.py` for the point-in-time join logic.

## Architecture

```
OFFLINE (Python, scheduled)              STATIC SITE (GitHub Pages)
ingest -> features -> model  ────────▶   reads rankings.json,
-> SHAP -> docs/*.json         writes    shap.json, backtest.json
```

GitHub Pages is static-only, so all computation happens offline in a
scheduled GitHub Actions job (`.github/workflows/refresh.yml`, monthly). The
site itself does no computation — it just renders the committed JSON.

## Repo layout

```
data/
  raw/          cached price/fundamental/macro pulls (gitignored)
  processed/    feature panel + labeled panel (parquet, gitignored)
src/
  config.py         shared paths/constants
  universe.py       top-100 TSX ticker list from iShares XIC holdings
  ingest.py         pulls price, fundamentals, macro
  features.py       builds the monthly feature panel (point-in-time)
  labels.py         forward 12mo return labels + purge/embargo fold logic
  model.py          LightGBM + Elastic Net baseline + naive momentum baseline
  backtest.py       walk-forward validation -> docs/backtest_history.json
  shap_explain.py   SHAP values -> docs/shap_values.json
  predict.py        current rankings -> docs/rankings.json
  price_history.py  recent price + predicted glide path -> docs/price_history.json
docs/               GitHub Pages source (static HTML/CSS/JS + Chart.js)
.github/workflows/refresh.yml   monthly scheduled pipeline run
```

## Running the pipeline locally

```bash
python -m venv .venv
.venv/Scripts/activate       # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cd src
python universe.py           # -> data/raw/universe.csv
python ingest.py             # -> data/raw/{prices,fundamentals,macro}.parquet
                              #    (pass an integer arg, e.g. `python ingest.py 10`,
                              #    to test on a small ticker slice first)
python features.py           # -> data/processed/feature_panel.parquet
python labels.py             # -> data/processed/labeled_panel.parquet
python backtest.py           # -> docs/backtest_history.json
python shap_explain.py       # -> docs/shap_values.json
python predict.py            # -> docs/rankings.json
python price_history.py      # -> docs/price_history.json (needs rankings.json)
```

## Viewing the dashboard locally

```bash
cd docs
python -m http.server 8123
# open http://localhost:8123
```

## Deploying to GitHub Pages

1. Push this repo to GitHub.
2. Repo Settings -> Pages -> Source: "Deploy from a branch" -> branch `main`,
   folder `/docs`.
3. `.github/workflows/refresh.yml` needs `contents: write` permission
   (already set) to commit refreshed JSON back to the branch — Pages
   redeploys automatically on that push.

## Data sources (v1, free/automatable only)

| Category | Source |
|---|---|
| Price/OHLCV, dividends | `yfinance` (`auto_adjust=True`, so `Close` is already a total-return series) |
| Fundamentals (P/E, P/B, ROE, margins, D/E, revenue growth) | `yfinance` annual financial statements |
| Macro: BoC overnight rate, 2yr/10yr yields | Bank of Canada Valet API |
| Macro: CAD/USD | `yfinance` (`CAD=X`) — BoC's own FX series only starts 2017 |
| Macro: WTI oil | `yfinance` (`CL=F`) |
| Universe / sector classification | iShares XIC (S&P/TSX Capped Composite) holdings CSV |

## Validation: purge + embargo

12-month forward labels overlap heavily between adjacent monthly rebalance
dates, so a naive time-based split leaks. `labels.purge_embargo_folds`
excludes any training date whose 12-month label window would overlap the
test period — a 12-month gap between the last training date and the test
date, applied on every walk-forward fold, not as a one-time cleanup.

Metrics, in order of how much they're trusted for this project:
1. **Rank correlation (Spearman IC)** between predicted and actual return.
2. **Decile spread** — top-10 predicted minus bottom-10 predicted actual
   return, per period.
3. **RMSE / R²** — reported for completeness only (see Known limitation).

The GBM is benchmarked against an Elastic Net baseline and a naive
"rank by trailing 12-month momentum" rule in every fold.
