// Dashboard XGB Paper Trader.
// - Velas en vivo: el NAVEGADOR pide klines a Binance (la IP del usuario no
//   esta geo-bloqueada). Si falla, fallback a Coinbase.
// - Estado del bot (wallets, trades, senales): JSONs generados por el bot
//   en ./data/*.json (commiteados por GitHub Actions en cada tick).

const TFS = ["15m", "1h", "4h"];
const TF_SECONDS = { "1m": 60, "15m": 900, "1h": 3600, "4h": 14400 };

const COLORS = {
  up: "#3fb950", down: "#f85149",
  grid: "#232936", text: "#8a93a6",
  line: "#58a6ff", area: "rgba(88,166,255,.12)",
};

const fmtEur = v => (v == null || Number.isNaN(+v)) ? "-" : (+v).toFixed(2) + " EUR";
const fmtPct = v => (v == null || Number.isNaN(+v)) ? "-" : (+v * 100).toFixed(2) + " %";
const fmtP   = v => (v == null || v === "" || Number.isNaN(+v)) ? "-" : (+v).toFixed(4);
const fmtPrice = v => (v == null || v === "" || Number.isNaN(+v)) ? "-" : (+v).toLocaleString("en-US", {maximumFractionDigits: 2});
const fmtTime = v => {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toISOString().replace("T", " ").slice(0, 16);
};
const toUnix = v => {
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : Math.floor(d.getTime() / 1000);
};

async function fetchJSON(path, fallback) {
  try {
    const r = await fetch(path + "?t=" + Date.now());
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch { return fallback; }
}

// ---------------------------------------------------------------------------
// Klines: Binance desde el navegador, fallback Coinbase
// ---------------------------------------------------------------------------

async function fetchKlinesBinance(interval, limit) {
  const url = `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=${interval}&limit=${limit}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error("binance " + r.status);
  const raw = await r.json();
  return raw.map(k => ({
    time: Math.floor(k[0] / 1000),
    open: +k[1], high: +k[2], low: +k[3], close: +k[4],
  }));
}

async function fetchKlinesCoinbase(interval, limit) {
  const gran = TF_SECONDS[interval];
  const url = `https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=${gran}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error("coinbase " + r.status);
  const raw = await r.json();   // [[time, low, high, open, close, vol], ...] DESC
  return raw.reverse().slice(-limit).map(k => ({
    time: k[0], open: k[3], high: k[2], low: k[1], close: k[4],
  }));
}

let klineSource = "binance";
async function fetchKlines(interval, limit) {
  if (klineSource === "binance") {
    try { return await fetchKlinesBinance(interval, limit); }
    catch { klineSource = "coinbase"; }
  }
  try { return await fetchKlinesCoinbase(interval, limit); }
  catch (e) {
    // reintentar binance la proxima vez
    klineSource = "binance";
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Charts (lightweight-charts)
// ---------------------------------------------------------------------------

const baseChartOptions = (h) => ({
  height: h,
  layout: { background: { color: "transparent" }, textColor: COLORS.text },
  grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
  timeScale: { timeVisible: true, secondsVisible: false, borderColor: COLORS.grid },
  rightPriceScale: { borderColor: COLORS.grid },
  crosshair: { mode: 0 },
  autoSize: true,
});

function makeCandleChart(containerId, height) {
  const el = document.getElementById(containerId);
  const chart = LightweightCharts.createChart(el, baseChartOptions(height));
  const series = chart.addCandlestickSeries({
    upColor: COLORS.up, downColor: COLORS.down,
    borderUpColor: COLORS.up, borderDownColor: COLORS.down,
    wickUpColor: COLORS.up, wickDownColor: COLORS.down,
  });
  return { chart, series };
}

function makeLineChart(containerId, height) {
  const el = document.getElementById(containerId);
  const chart = LightweightCharts.createChart(el, baseChartOptions(height));
  const series = chart.addAreaSeries({
    lineColor: COLORS.line, topColor: COLORS.area,
    bottomColor: "transparent", lineWidth: 2,
  });
  return { chart, series };
}

// ---------------------------------------------------------------------------
// Markers de trades: triangulo verde (arrowUp) = entrada long,
// triangulo rojo invertido (arrowDown) = entrada short, circulo = salida.
// ---------------------------------------------------------------------------

function buildMarkers(tf, trades, openPositions) {
  const markers = [];
  for (const t of trades) {
    if (t.timeframe !== tf) continue;
    const entryTime = toUnix(t.entry_time);
    const exitTime = toUnix(t.exit_time);
    if (entryTime) {
      markers.push(t.side === "long"
        ? { time: entryTime, position: "belowBar", color: COLORS.up,
            shape: "arrowUp", text: "LONG @" + fmtPrice(t.entry_price) }
        : { time: entryTime, position: "aboveBar", color: COLORS.down,
            shape: "arrowDown", text: "SHORT @" + fmtPrice(t.entry_price) });
    }
    if (exitTime) {
      const win = (+t.pnl_eur || 0) >= 0;
      markers.push({
        time: exitTime, position: win ? "aboveBar" : "belowBar",
        color: win ? COLORS.up : COLORS.down, shape: "circle",
        text: (t.exit_reason || "EXIT") + " " + fmtEur(t.pnl_eur),
      });
    }
  }
  for (const p of openPositions) {
    if (p.timeframe !== tf) continue;
    const entryTime = toUnix(p.entry_time);
    if (entryTime) {
      markers.push(p.side === "long"
        ? { time: entryTime, position: "belowBar", color: COLORS.up,
            shape: "arrowUp", text: "LONG (abierta)" }
        : { time: entryTime, position: "aboveBar", color: COLORS.down,
            shape: "arrowDown", text: "SHORT (abierta)" });
    }
  }
  return markers.sort((a, b) => a.time - b.time);
}

// ---------------------------------------------------------------------------
// Render de tablas y cards
// ---------------------------------------------------------------------------

function renderSummary(summary) {
  const root = document.getElementById("summary-cards");
  root.innerHTML = "";
  const wallets = summary.wallets || {};
  for (const tf of TFS) {
    const w = wallets[tf] || {};
    const pnl = (w.equity_eur || 100) - (w.initial_capital_eur || 100);
    const pnlClass = pnl >= 0 ? "green" : "red";
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `
      <h3>Cartera ${tf}</h3>
      <div class="kpi-grid">
        <span class="kpi-label">Inicial</span><span class="kpi-value">${fmtEur(w.initial_capital_eur ?? 100)}</span>
        <span class="kpi-label">Equity</span><span class="kpi-value">${fmtEur(w.equity_eur ?? 100)}</span>
        <span class="kpi-label">PnL</span><span class="kpi-value ${pnlClass}">${fmtEur(pnl)}</span>
        <span class="kpi-label">Trades</span><span class="kpi-value">${w.n_trades || 0}</span>
        <span class="kpi-label">Win rate</span><span class="kpi-value">${fmtPct(w.win_rate || 0)}</span>
        <span class="kpi-label">Posicion</span><span class="kpi-value small">${w.open_position_id ? "ABIERTA" : "-"}</span>
      </div>`;
    root.appendChild(div);
  }
  document.getElementById("generated-at").textContent =
    "Ultimo tick del bot: " + fmtTime(summary.generated_at);
}

function renderOpenPositions(data) {
  const root = document.getElementById("open-positions");
  const pos = data.open_positions || [];
  if (pos.length === 0) {
    root.innerHTML = '<div class="empty">No hay posiciones abiertas.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>TF</th><th>Side</th><th>Entrada</th><th>Precio</th><th>TP</th><th>SL</th>" +
    "<th>Timeout</th><th>p_win</th><th>EV</th><th>Notional</th></tr></thead><tbody>";
  for (const p of pos) {
    html += `<tr>
      <td>${p.timeframe}</td><td class="${p.side}">${p.side === "long" ? "&#9650; long" : "&#9660; short"}</td>
      <td>${fmtTime(p.entry_time)}</td><td>${fmtPrice(p.entry_price)}</td>
      <td class="tp">${fmtPrice(p.tp_price)}</td><td class="sl">${fmtPrice(p.sl_price)}</td>
      <td>${fmtTime(p.timeout_time)}</td><td>${fmtP(p.p_win)}</td>
      <td>${fmtP(p.EV_pred)}</td><td>${fmtEur(p.notional_eur)}</td></tr>`;
  }
  root.innerHTML = html + "</tbody></table></div>";
}

function renderSignals(data) {
  const root = document.getElementById("signals-table");
  const signals = (data.signals || []).slice().reverse().slice(0, 100);
  if (signals.length === 0) {
    root.innerHTML = '<div class="empty">Sin senales registradas aun.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>Vela t (cierre = pregunta)</th><th>TF</th><th>BTC close</th>" +
    "<th>vol decile</th><th>candidatos</th><th>en banda</th>" +
    "<th>p_win max</th><th>EV max</th><th>Decision</th><th>Detalle</th>" +
    "<th>Side</th><th>p_win elegido</th><th>Entrada t+1</th></tr></thead><tbody>";
  for (const s of signals) {
    const yes = s.decision === "YES";
    html += `<tr class="${yes ? "row-yes" : ""}">
      <td>${fmtTime(s.candle_close_time)}</td>
      <td>${s.timeframe}</td>
      <td>${fmtPrice(s.btc_close)}</td>
      <td>${s.vol_decile ?? "-"}</td>
      <td>${s.n_candidates_initial ?? "-"}</td>
      <td>${s.n_in_band ?? "-"}</td>
      <td>${fmtP(s.p_win_max)}</td>
      <td>${fmtP(s.EV_max)}</td>
      <td class="${yes ? "tp" : "muted"}"><b>${s.decision || "-"}</b></td>
      <td class="muted small">${(s.reason_no_signal || "").replaceAll("_", " ").toLowerCase()}</td>
      <td class="${s.winner_side || ""}">${s.winner_side ? (s.winner_side === "long" ? "&#9650; long" : "&#9660; short") : "-"}</td>
      <td>${fmtP(s.winner_p_win_calibrated)}</td>
      <td>${s.entry_price ? fmtPrice(s.entry_price) : "-"}</td></tr>`;
  }
  root.innerHTML = html + "</tbody></table></div>";
}

function renderTrades(data) {
  const root = document.getElementById("trades-table");
  const trades = data.trades || [];
  if (trades.length === 0) {
    root.innerHTML = '<div class="empty">Sin trades cerrados aun.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>Entrada</th><th>Salida</th><th>TF</th><th>Side</th>" +
    "<th>Entry $</th><th>Exit $</th><th>TP $</th><th>SL $</th><th>Motivo</th>" +
    "<th>p_win</th><th>EV</th><th>PnL EUR</th><th>PnL %</th><th>Equity</th></tr></thead><tbody>";
  for (const t of trades) {
    const reasonCls = (t.exit_reason || "").toLowerCase();
    const pnlCls = (+t.pnl_eur >= 0) ? "tp" : "sl";
    html += `<tr>
      <td>${fmtTime(t.entry_time)}</td><td>${fmtTime(t.exit_time)}</td>
      <td>${t.timeframe}</td>
      <td class="${t.side}">${t.side === "long" ? "&#9650; long" : "&#9660; short"}</td>
      <td>${fmtPrice(t.entry_price)}</td><td>${fmtPrice(t.exit_price)}</td>
      <td>${fmtPrice(t.tp_price)}</td><td>${fmtPrice(t.sl_price)}</td>
      <td class="${reasonCls}">${t.exit_reason}</td>
      <td>${fmtP(t.p_win)}</td><td>${fmtP(t.EV_pred)}</td>
      <td class="${pnlCls}">${fmtEur(t.pnl_eur)}</td>
      <td class="${pnlCls}">${fmtPct(t.pnl_pct)}</td>
      <td>${fmtEur(t.wallet_equity_after)}</td></tr>`;
  }
  root.innerHTML = html + "</tbody></table></div>";
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

const charts = {};

async function refreshLiveChart() {
  try {
    const candles = await fetchKlines("1m", 240);
    charts.live.series.setData(candles);
    const last = candles[candles.length - 1];
    const prev = candles[candles.length - 2];
    const el = document.getElementById("live-price-value");
    el.textContent = fmtPrice(last.close);
    el.className = last.close >= prev.close ? "green" : "red";
    document.getElementById("live-price-source").textContent = "(" + klineSource + ")";
  } catch (e) { console.warn("live chart refresh failed", e); }
}

async function refreshTfCharts(trades, openPositions) {
  for (const tf of TFS) {
    try {
      const candles = await fetchKlines(tf, 150);
      charts[tf].series.setData(candles);
      charts[tf].series.setMarkers(buildMarkers(tf, trades, openPositions));
    } catch (e) { console.warn("tf chart failed", tf, e); }
  }
}

async function refreshBotData() {
  const [summary, openPos, trades, signals] = await Promise.all([
    fetchJSON("data/summary.json", { wallets: {}, generated_at: null }),
    fetchJSON("data/open_positions.json", { open_positions: [] }),
    fetchJSON("data/trades.json", { trades: [] }),
    fetchJSON("data/signals.json", { signals: [] }),
  ]);
  renderSummary(summary);
  renderOpenPositions(openPos);
  renderSignals(signals);
  renderTrades(trades);

  // Equity charts
  for (const tf of TFS) {
    const eq = await fetchJSON(`data/equity_${tf}.json`, { curve: [], initial_capital_eur: 100 });
    let points = (eq.curve || [])
      .map(p => ({ time: toUnix(p.ts), value: +p.equity }))
      .filter(p => p.time != null);
    if (points.length === 0) {
      points = [{ time: Math.floor(Date.now() / 1000), value: eq.initial_capital_eur || 100 }];
    }
    charts["eq-" + tf].series.setData(points);
  }
  return { trades: trades.trades || [], openPositions: openPos.open_positions || [] };
}

async function init() {
  charts.live = makeCandleChart("chart-live", 380);
  for (const tf of TFS) {
    charts[tf] = makeCandleChart("chart-tf-" + tf, 260);
    charts["eq-" + tf] = makeLineChart("chart-eq-" + tf, 180);
  }

  const botData = await refreshBotData();
  await refreshLiveChart();
  await refreshTfCharts(botData.trades, botData.openPositions);

  // Live BTC: cada 5s. TF charts: cada 60s. Datos del bot: cada 60s.
  setInterval(refreshLiveChart, 5000);
  setInterval(async () => {
    const d = await refreshBotData();
    await refreshTfCharts(d.trades, d.openPositions);
  }, 60000);
}

init();
