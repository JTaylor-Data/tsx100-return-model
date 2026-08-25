const FEATURE_LABELS = {
  mom_1m: "1mo momentum", mom_3m: "3mo momentum", mom_6m: "6mo momentum", mom_12m: "12mo momentum",
  vol_63d: "63d volatility", vol_252d: "252d volatility", beta_252d: "Beta vs TSX (252d)",
  rel_strength_tsx: "Rel. strength vs TSX", rel_strength_sector: "Rel. strength vs sector",
  pe_ratio: "P/E", pb_ratio: "P/B", ev_ebitda: "EV/EBITDA", div_yield: "Dividend yield",
  payout_ratio: "Payout ratio", roe: "ROE", roa: "ROA", gross_margin: "Gross margin",
  operating_margin: "Operating margin", revenue_growth_yoy: "Revenue growth YoY",
  debt_equity: "Debt/Equity", boc_rate: "BoC overnight rate", yield_curve_2s10s: "2s10s yield curve",
  usdcad_chg_3m: "CAD/USD 3mo chg", oil_chg_3m: "WTI oil 3mo chg",
  value_z_universe: "Value z-score (universe)", momentum_z_universe: "Momentum z-score (universe)",
  sector: "Sector",
};

const PCT_FEATURES = new Set([
  "mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_63d", "vol_252d", "rel_strength_tsx",
  "rel_strength_sector", "div_yield", "payout_ratio", "roe", "roa", "gross_margin",
  "operating_margin", "revenue_growth_yoy", "boc_rate", "usdcad_chg_3m", "oil_chg_3m",
]);

function fmtLabel(f) { return FEATURE_LABELS[f] || f; }

function fmtFeatureValue(f, v) {
  if (v === null || v === undefined) return "—";
  if (f === "sector") return v;
  if (PCT_FEATURES.has(f)) return (v * 100).toFixed(1) + "%";
  return v.toFixed(2);
}

function fmtPct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
}

function retClass(v) {
  if (v === null || v === undefined) return "";
  return v >= 0 ? "ret-pos" : "ret-neg";
}

const state = { rankings: null, shap: null, backtest: null, priceHistory: null, sortKey: "rank", sortDir: 1, filterSector: "", search: "" };

async function loadData() {
  const [rankings, shap, backtest, priceHistory] = await Promise.all([
    fetch("rankings.json").then(r => r.json()),
    fetch("shap_values.json").then(r => r.json()),
    fetch("backtest_history.json").then(r => r.json()),
    fetch("price_history.json").then(r => r.json()),
  ]);
  state.rankings = rankings;
  state.shap = shap;
  state.backtest = backtest;
  state.priceHistory = priceHistory;
}

function renderHeader() {
  const asOf = state.rankings.as_of_date;
  document.getElementById("as-of-badge").textContent = `Updated ${asOf}`;
}

function renderSummaryCards() {
  const s = state.backtest.summary;
  const lgb = s.lightgbm, en = s.elastic_net, naive = s.naive_momentum;
  const cards = [
    { label: "Rank IC (LightGBM)", value: lgb.mean_rank_ic.toFixed(3), sub: `${lgb.n_folds} walk-forward folds`, cls: lgb.mean_rank_ic >= 0 ? "pos" : "neg" },
    { label: "Decile spread (LightGBM)", value: fmtPct(lgb.mean_decile_spread), sub: "top 10 minus bottom 10 predicted", cls: lgb.mean_decile_spread >= 0 ? "pos" : "neg" },
    { label: "vs. naive momentum", value: `${lgb.mean_rank_ic >= naive.mean_rank_ic ? "+" : ""}${(lgb.mean_rank_ic - naive.mean_rank_ic).toFixed(3)} IC`, sub: `naive IC ${naive.mean_rank_ic.toFixed(3)}`, cls: lgb.mean_rank_ic >= naive.mean_rank_ic ? "pos" : "neg" },
    { label: "vs. Elastic Net", value: `${lgb.mean_rank_ic >= en.mean_rank_ic ? "+" : ""}${(lgb.mean_rank_ic - en.mean_rank_ic).toFixed(3)} IC`, sub: `Elastic Net IC ${en.mean_rank_ic.toFixed(3)}`, cls: lgb.mean_rank_ic >= en.mean_rank_ic ? "pos" : "neg" },
    { label: "Universe", value: state.rankings.n_stocks, sub: `${state.rankings.n_features} features / ${state.rankings.n_training_rows.toLocaleString()} training rows` },
  ];
  document.getElementById("summary-cards").innerHTML = cards.map(c => `
    <div class="card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls || ""}">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>`).join("");
}

function populateSectorFilter() {
  const sectors = [...new Set(state.rankings.stocks.map(s => s.sector))].sort();
  const sel = document.getElementById("sector-filter");
  sel.innerHTML = `<option value="">All sectors</option>` + sectors.map(s => `<option value="${s}">${s}</option>`).join("");
}

function getFilteredSortedStocks() {
  let rows = state.rankings.stocks.slice();
  if (state.filterSector) rows = rows.filter(r => r.sector === state.filterSector);
  if (state.search) {
    const q = state.search.toLowerCase();
    rows = rows.filter(r => r.ticker.toLowerCase().includes(q) || (r.name || "").toLowerCase().includes(q));
  }
  const key = state.sortKey, dir = state.sortDir;
  rows.sort((a, b) => {
    let av = key === "predicted_return_12m" || key === "current_price" || key === "rank" ? a[key] : a[key];
    let bv = key === "predicted_return_12m" || key === "current_price" || key === "rank" ? b[key] : b[key];
    if (typeof av === "string") return av.localeCompare(bv) * dir;
    return ((av ?? -Infinity) - (bv ?? -Infinity)) * dir;
  });
  return rows;
}

function renderTable() {
  const rows = getFilteredSortedStocks();
  const tbody = document.getElementById("rankings-tbody");
  tbody.innerHTML = rows.map(r => `
    <tr data-ticker="${r.ticker}">
      <td class="rank-cell">${r.rank}</td>
      <td class="ticker-cell">${r.ticker}</td>
      <td>${r.name || ""}</td>
      <td><span class="sector-pill">${r.sector}</span></td>
      <td class="${retClass(r.predicted_return_12m)}">${fmtPct(r.predicted_return_12m)}</td>
      <td>${r.current_price != null ? "$" + r.current_price.toFixed(2) : "—"}</td>
    </tr>`).join("");
  tbody.querySelectorAll("tr").forEach(tr => tr.addEventListener("click", () => openDrawer(tr.dataset.ticker)));
}

function wireSortHandlers() {
  document.querySelectorAll("#rankings-table thead th[data-key]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (state.sortKey === key) state.sortDir *= -1; else { state.sortKey = key; state.sortDir = key === "rank" ? 1 : -1; }
      renderTable();
    });
  });
}

function openDrawer(ticker) {
  const stock = state.rankings.stocks.find(s => s.ticker === ticker);
  const shapEntry = state.shap.stocks.find(s => s.ticker === ticker);
  if (!stock) return;

  document.getElementById("drawer-ticker").textContent = stock.ticker;
  document.getElementById("drawer-name").textContent = stock.name || "";
  document.getElementById("drawer-sector").textContent = stock.sector;
  document.getElementById("drawer-return").textContent = fmtPct(stock.predicted_return_12m);
  document.getElementById("drawer-return").className = retClass(stock.predicted_return_12m);
  document.getElementById("drawer-price").textContent = stock.current_price != null ? "$" + stock.current_price.toFixed(2) : "—";
  document.getElementById("drawer-rank").textContent = `Rank ${stock.rank} of ${state.rankings.n_stocks}`;

  const snapKeys = ["pe_ratio", "pb_ratio", "div_yield", "roe", "roa", "debt_equity", "mom_12m", "vol_252d"];
  document.getElementById("snapshot-grid").innerHTML = snapKeys.map(k => `
    <div class="snapshot-item">
      <div class="k">${fmtLabel(k)}</div>
      <div class="v">${fmtFeatureValue(k, stock.snapshot[k])}</div>
    </div>`).join("");

  renderShapChart(shapEntry);
  renderPriceChart(stock.ticker);
  document.getElementById("overlay").classList.add("open");
}

let priceChart = null;
function renderPriceChart(ticker) {
  const ctx = document.getElementById("price-chart");
  if (priceChart) priceChart.destroy();
  const entry = state.priceHistory.stocks[ticker];
  if (!entry) return;

  const labels = [...entry.history_dates, ...entry.projected_dates];
  const historyData = [...entry.history_close, ...entry.projected_dates.map(() => null)];
  // repeat the last actual close as the first projected point so the dashed
  // line visually continues from where the solid line ends
  const projectedData = [
    ...entry.history_dates.slice(0, -1).map(() => null),
    entry.history_close[entry.history_close.length - 1],
    ...entry.projected_close,
  ];

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual price",
          data: historyData,
          borderColor: "#4f8cff",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          spanGaps: false,
        },
        {
          label: "Predicted path (12mo)",
          data: projectedData,
          borderColor: "#e8a33d",
          backgroundColor: "transparent",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          spanGaps: false,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#8b92a8" } },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: $${c.raw?.toFixed(2) ?? "—"}` } },
      },
      scales: {
        x: { ticks: { color: "#8b92a8", maxTicksLimit: 10 }, grid: { color: "#262c3d" } },
        y: { ticks: { color: "#8b92a8", callback: v => "$" + v }, grid: { color: "#262c3d" } },
      },
    },
  });
}

function closeDrawer() { document.getElementById("overlay").classList.remove("open"); }

let shapChart = null;
function renderShapChart(shapEntry) {
  const ctx = document.getElementById("shap-waterfall-chart");
  if (shapChart) shapChart.destroy();
  if (!shapEntry) { return; }
  const feats = shapEntry.top_features.slice().reverse();
  shapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: feats.map(f => `${fmtLabel(f.feature)} (${fmtFeatureValue(f.feature, f.value)})`),
      datasets: [{
        data: feats.map(f => f.shap),
        backgroundColor: feats.map(f => f.shap >= 0 ? "#2fbf71" : "#ef5a5a"),
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `SHAP: ${c.raw.toFixed(3)}` } } },
      scales: {
        x: { ticks: { color: "#8b92a8" }, grid: { color: "#262c3d" }, title: { display: true, text: "Contribution to predicted return", color: "#8b92a8" } },
        y: { ticks: { color: "#e6e9f0", font: { size: 11 } }, grid: { display: false } },
      },
    },
  });
}

function renderGlobalShapChart() {
  const ctx = document.getElementById("global-shap-chart");
  const top = state.shap.global_importance.slice(0, 15).reverse();
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map(f => fmtLabel(f.feature)),
      datasets: [{ data: top.map(f => f.mean_abs_shap), backgroundColor: "#4f8cff", borderRadius: 3 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b92a8" }, grid: { color: "#262c3d" }, title: { display: true, text: "Mean |SHAP| across universe", color: "#8b92a8" } },
        y: { ticks: { color: "#e6e9f0", font: { size: 11.5 } }, grid: { display: false } },
      },
    },
  });
}

let scatterChart = null;
function populatePeriodSelector() {
  const sel = document.getElementById("period-select");
  const periods = state.backtest.periods;
  sel.innerHTML = periods.map((p, i) => `<option value="${i}">${p.test_date}</option>`).join("");
  sel.value = String(periods.length - 1);
  sel.addEventListener("change", () => renderScatter(periods[Number(sel.value)]));
  renderScatter(periods[periods.length - 1]);
}

function renderScatter(period) {
  const ctx = document.getElementById("pred-vs-actual-chart");
  if (scatterChart) scatterChart.destroy();
  const points = period.ticker.map((t, i) => ({ x: period.predicted_return[i], y: period.actual_return[i], t }));
  const min = Math.min(...points.map(p => Math.min(p.x, p.y)), 0);
  const max = Math.max(...points.map(p => Math.max(p.x, p.y)), 0);
  scatterChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        { label: "Stocks", data: points, backgroundColor: "#4f8cff", pointRadius: 4 },
        { label: "Perfect prediction", data: [{ x: min, y: min }, { x: max, y: max }], type: "line", borderColor: "#8b92a8", borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#8b92a8" } },
        tooltip: { callbacks: { label: c => `${c.raw.t}: predicted ${fmtPct(c.raw.x)}, actual ${fmtPct(c.raw.y)}` } },
      },
      scales: {
        x: { title: { display: true, text: "Predicted 12mo return", color: "#8b92a8" }, ticks: { color: "#8b92a8", callback: v => (v * 100).toFixed(0) + "%" }, grid: { color: "#262c3d" } },
        y: { title: { display: true, text: "Actual 12mo return", color: "#8b92a8" }, ticks: { color: "#8b92a8", callback: v => (v * 100).toFixed(0) + "%" }, grid: { color: "#262c3d" } },
      },
    },
  });
}

function wireTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

async function init() {
  await loadData();
  renderHeader();
  renderSummaryCards();
  populateSectorFilter();
  renderTable();
  wireSortHandlers();
  wireTabs();
  renderGlobalShapChart();
  populatePeriodSelector();

  document.getElementById("sector-filter").addEventListener("change", e => { state.filterSector = e.target.value; renderTable(); });
  document.getElementById("search-input").addEventListener("input", e => { state.search = e.target.value; renderTable(); });
  document.getElementById("close-drawer").addEventListener("click", closeDrawer);
  document.getElementById("overlay").addEventListener("click", e => { if (e.target.id === "overlay") closeDrawer(); });
}

init();
