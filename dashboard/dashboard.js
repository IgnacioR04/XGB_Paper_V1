// Dashboard XGB Paper Trader v4
// - Velas en vivo con selector de timeframe
// - Bookmap heatmap de liquidez (order book acumulado)
// - Order book depth chart (Binance public)
// - Tablas paginadas + filtradas por timeframe

const TFS = ["15m", "1h", "4h"];
const TF_SECONDS = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400 };

const COLORS = {
  up: "#3fb950", down: "#f85149",
  grid: "#232936", text: "#8a93a6",
  line: "#58a6ff", area: "rgba(88,166,255,.12)",
  bid: "#3fb950", bidArea: "rgba(63,185,80,.25)",
  ask: "#f85149", askArea: "rgba(248,81,73,.25)",
};

// ---------------------------------------------------------------------------
// Bookmap: heatmap de liquidez sobre el chart principal
// Acumula snapshots del order book cada 5s y los pinta como celdas de calor
// detras de las velas usando la API de primitivas de lightweight-charts.
// ---------------------------------------------------------------------------

const HEATMAP = {
  snapshots: [],
  maxSnapshots: 2160,   // 3 h a 5 s/snap
  bucketUSD: 20,
  norm: 0,              // normalizacion por percentil (EMA suavizada)
  primitive: null,
};

// Paleta estilo bookmap: azul oscuro -> azul -> cyan -> verde -> amarillo
// -> naranja -> rojo. Cada stop: [frac, r, g, b, alpha]
const HEAT_STOPS = [
  [0.00,  15,  28,  85, 0.32],
  [0.18,  10, 105, 185, 0.48],
  [0.35,   0, 178, 172, 0.58],
  [0.52,  48, 188,  78, 0.68],
  [0.70, 238, 218,  48, 0.80],
  [0.85, 255, 142,  12, 0.90],
  [1.00, 255,  48,  48, 0.96],
];

function heatColor(frac, alphaScale = 1) {
  if (!(frac > 0.03)) return null;
  const f = Math.pow(Math.min(frac, 1), 0.55);
  let i = 1;
  while (i < HEAT_STOPS.length - 1 && HEAT_STOPS[i][0] < f) i++;
  const [f0, r0, g0, b0, a0] = HEAT_STOPS[i - 1];
  const [f1, r1, g1, b1, a1] = HEAT_STOPS[i];
  const t = (f - f0) / Math.max(f1 - f0, 1e-9);
  const r = Math.round(r0 + (r1 - r0) * t);
  const g = Math.round(g0 + (g1 - g0) * t);
  const b = Math.round(b0 + (b1 - b0) * t);
  const a = (a0 + (a1 - a0) * t) * alphaScale;
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

function captureSnapshot(book) {
  if (!book?.bids?.length || !book?.asks?.length) return;
  const bk = HEATMAP.bucketUSD;
  const levels = new Map();
  for (const arr of [book.bids, book.asks]) {
    for (const [p, q] of arr) {
      const key = Math.floor(+p / bk) * bk;
      levels.set(key, (levels.get(key) || 0) + +q);
    }
  }
  // Normalizacion robusta: percentil 95 de los buckets del snapshot,
  // suavizado con EMA para que la escala no salte entre refrescos.
  const vals = [...levels.values()].sort((a, b) => a - b);
  const p95 = vals[Math.floor(vals.length * 0.95)] || 1;
  HEATMAP.norm = HEATMAP.norm === 0 ? p95 : HEATMAP.norm * 0.97 + p95 * 0.03;
  HEATMAP.snapshots.push({ time: Math.floor(Date.now() / 1000), levels });
  while (HEATMAP.snapshots.length > HEATMAP.maxSnapshots) HEATMAP.snapshots.shift();
  if (HEATMAP.primitive?._requestUpdate) HEATMAP.primitive._requestUpdate();
}

class LiqHeatmap {
  constructor() {
    this._chart = null;
    this._series = null;
    this._requestUpdate = null;
    this._view = new LiqHeatmapView(this);
    this._groups = new Map();
    this._lastLen = 0;
    this._lastTf = "";
  }
  attached({ chart, series, requestUpdate }) {
    this._chart = chart; this._series = series; this._requestUpdate = requestUpdate;
  }
  detached() { this._chart = this._series = this._requestUpdate = null; }
  paneViews() { return [this._view]; }

  recompute() {
    if (HEATMAP.snapshots.length === this._lastLen && liveTfInterval === this._lastTf) return;
    this._lastLen = HEATMAP.snapshots.length;
    this._lastTf = liveTfInterval;
    const cSec = TF_SECONDS[liveTfInterval] || 900;
    const groups = new Map();
    for (const snap of HEATMAP.snapshots) {
      const ct = Math.floor(snap.time / cSec) * cSec;
      if (!groups.has(ct)) groups.set(ct, []);
      groups.get(ct).push(snap);
    }
    this._groups = new Map();
    for (const [ct, snaps] of groups) {
      const agg = new Map();
      for (const snap of snaps) {
        for (const [price, qty] of snap.levels) {
          agg.set(price, Math.max(agg.get(price) || 0, qty));
        }
      }
      this._groups.set(ct, agg);
    }
  }
}

class LiqHeatmapView {
  constructor(src) { this._r = new LiqHeatmapRenderer(src); }
  renderer() { return this._r; }
  zOrder() { return "bottom"; }
}

class LiqHeatmapRenderer {
  constructor(src) { this._src = src; }

  _drawBand(ctx, series, price, bk, color, x0, x1) {
    const yB = series.priceToCoordinate(price);
    const yT = series.priceToCoordinate(price + bk);
    if (yB === null || yT === null) return;
    const cH = Math.abs(yB - yT);
    if (cH < 0.3) return;
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x0), Math.round(Math.min(yT, yB)),
                 Math.ceil(x1 - x0), Math.max(Math.ceil(cH), 1));
  }

  draw(target) {
    this._src.recompute();
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      const { _chart: chart, _series: series, _groups: groups } = this._src;
      if (!chart || !series) return;
      const cSec = TF_SECONDS[liveTfInterval] || 900;
      const bk = HEATMAP.bucketUSD;
      const norm = HEATMAP.norm || 1;
      const latest = HEATMAP.snapshots[HEATMAP.snapshots.length - 1];
      if (!latest) return;

      // 1) Libro ACTUAL como bandas horizontales a ancho completo (como en
      //    bookmap/TradingView: los niveles de liquidez en reposo se ven como
      //    franjas que cruzan todo el chart).
      for (const [price, qty] of latest.levels) {
        const color = heatColor(qty / norm, 0.55);
        if (!color) continue;
        this._drawBand(ctx, series, price, bk, color, 0, mediaSize.width);
      }

      // 2) Historia acumulada por vela encima (donde tenemos snapshots reales
      //    el mapa muestra la evolucion del libro, celda a celda).
      const firstSnapTime = HEATMAP.snapshots[0].time;
      for (const [ct, agg] of groups) {
        const x = chart.timeScale().timeToCoordinate(ct);
        if (x === null || x < -60 || x > mediaSize.width + 60) continue;
        const xN = chart.timeScale().timeToCoordinate(ct + cSec);
        let bW;
        if (xN !== null) bW = Math.max(Math.abs(xN - x), 3);
        else {
          const xP = chart.timeScale().timeToCoordinate(ct - cSec);
          bW = xP !== null ? Math.max(Math.abs(x - xP), 3) : 8;
        }
        // limpiar la franja de fondo bajo esta vela para que la celda
        // historica no se mezcle con la banda del libro actual
        const xL = x - bW / 2, xR = x + bW / 2;
        for (const [price, qty] of agg) {
          const color = heatColor(qty / norm, 1.0);
          if (!color) continue;
          this._drawBand(ctx, series, price, bk, color, xL, xR);
        }
      }
    });
  }
}

const PAGE_SIZE = 25;
const LOCAL_TZ = Intl.DateTimeFormat().resolvedOptions().timeZone;

// ---------------------------------------------------------------------------
// Formateo de tiempo en hora LOCAL del usuario (no UTC)
// ---------------------------------------------------------------------------

const fmtEur = v => (v == null || Number.isNaN(+v)) ? "-" : (+v).toFixed(2) + " EUR";
const fmtPct = v => (v == null || Number.isNaN(+v)) ? "-" : (+v * 100).toFixed(2) + " %";
const fmtP   = v => (v == null || v === "" || Number.isNaN(+v)) ? "-" : (+v).toFixed(4);
const fmtPrice = v => (v == null || v === "" || Number.isNaN(+v)) ? "-" : (+v).toLocaleString("en-US", {maximumFractionDigits: 2});

const _timeFormatter = new Intl.DateTimeFormat("es-ES", {
  year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit",
  hour12: false,
});
const fmtTime = v => {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return _timeFormatter.format(d).replace(",", "");
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
  const gran = interval === "4h" ? 3600 : TF_SECONDS[interval];
  const url = `https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=${gran}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error("coinbase " + r.status);
  const raw = await r.json();
  let candles = raw.reverse().map(k => ({
    time: k[0], open: k[3], high: k[2], low: k[1], close: k[4],
  }));
  if (interval === "4h") {
    const buckets = new Map();
    for (const c of candles) {
      const b = Math.floor(c.time / 14400) * 14400;
      const cur = buckets.get(b);
      if (!cur) buckets.set(b, { time: b, open: c.open, high: c.high, low: c.low, close: c.close });
      else {
        cur.high = Math.max(cur.high, c.high);
        cur.low = Math.min(cur.low, c.low);
        cur.close = c.close;
      }
    }
    candles = [...buckets.values()].sort((a, b) => a.time - b.time);
  }
  return candles.slice(-limit);
}

let klineSource = "binance";
async function fetchKlines(interval, limit) {
  if (klineSource === "binance") {
    try { return await fetchKlinesBinance(interval, limit); }
    catch { klineSource = "coinbase"; }
  }
  try { return await fetchKlinesCoinbase(interval, limit); }
  catch (e) { klineSource = "binance"; throw e; }
}

// ---------------------------------------------------------------------------
// Order book depth chart (libro de ordenes en vivo)
// ---------------------------------------------------------------------------

async function fetchOrderBook() {
  const url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5000";
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch {
    // Coinbase product book (level 2)
    try {
      const r = await fetch("https://api.exchange.coinbase.com/products/BTC-USD/book?level=2");
      if (!r.ok) throw new Error(r.status);
      const d = await r.json();
      return {
        bids: (d.bids || []).map(b => [b[0], b[1]]),
        asks: (d.asks || []).map(a => [a[0], a[1]]),
      };
    } catch { return null; }
  }
}

function buildCumulativeDepth(book) {
  if (!book || !book.bids || !book.asks) return { bids: [], asks: [], mid: null };
  // Bids ya vienen DESC por precio; asks ASC.
  const bids = book.bids.map(b => [+b[0], +b[1]]).sort((a, b) => b[0] - a[0]);
  const asks = book.asks.map(a => [+a[0], +a[1]]).sort((a, b) => a[0] - b[0]);
  const mid = (bids[0][0] + asks[0][0]) / 2;

  // Limitar a +/- 1% del mid para que el chart sea legible
  const minP = mid * 0.99, maxP = mid * 1.01;

  const bidPts = [];
  let cum = 0;
  for (const [p, q] of bids) {
    if (p < minP) break;
    cum += q;
    bidPts.push({ price: p, cum });
  }
  bidPts.reverse(); // ASC en precio para el chart

  const askPts = [];
  cum = 0;
  for (const [p, q] of asks) {
    if (p > maxP) break;
    cum += q;
    askPts.push({ price: p, cum });
  }

  return { bids: bidPts, asks: askPts, mid, minP, maxP };
}

let depthChart = null;
let depthBidSeries = null;
let depthAskSeries = null;

function initDepthChart() {
  const el = document.getElementById("chart-depth");
  // No usamos lightweight-charts aqui porque el eje X seria precio, no tiempo.
  // Implementamos un canvas propio simple.
  el.innerHTML = '<canvas id="depth-canvas"></canvas>';
}

function renderDepth(depth) {
  const canvas = document.getElementById("depth-canvas");
  if (!canvas || !depth || !depth.mid) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight || 280;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const allPts = [...depth.bids, ...depth.asks];
  if (allPts.length === 0) return;
  const maxCum = Math.max(...allPts.map(p => p.cum));
  const minP = depth.minP, maxP = depth.maxP;

  const x = p => ((p - minP) / (maxP - minP)) * w;
  const y = c => h - (c / maxCum) * (h - 24) - 12;

  // Grid horizontal
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const yy = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(w, yy); ctx.stroke();
  }

  // Bids (verde, lado izquierdo)
  if (depth.bids.length > 1) {
    ctx.beginPath();
    ctx.moveTo(x(depth.bids[0].price), h);
    for (const p of depth.bids) ctx.lineTo(x(p.price), y(p.cum));
    ctx.lineTo(x(depth.bids[depth.bids.length - 1].price), h);
    ctx.closePath();
    ctx.fillStyle = COLORS.bidArea;
    ctx.fill();
    ctx.strokeStyle = COLORS.bid;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(depth.bids[0].price), y(depth.bids[0].cum));
    for (const p of depth.bids) ctx.lineTo(x(p.price), y(p.cum));
    ctx.stroke();
  }
  // Asks (rojo, lado derecho)
  if (depth.asks.length > 1) {
    ctx.beginPath();
    ctx.moveTo(x(depth.asks[0].price), h);
    for (const p of depth.asks) ctx.lineTo(x(p.price), y(p.cum));
    ctx.lineTo(x(depth.asks[depth.asks.length - 1].price), h);
    ctx.closePath();
    ctx.fillStyle = COLORS.askArea;
    ctx.fill();
    ctx.strokeStyle = COLORS.ask;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(depth.asks[0].price), y(depth.asks[0].cum));
    for (const p of depth.asks) ctx.lineTo(x(p.price), y(p.cum));
    ctx.stroke();
  }
  // Linea del mid
  ctx.strokeStyle = COLORS.text;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x(depth.mid), 4);
  ctx.lineTo(x(depth.mid), h - 4);
  ctx.stroke();
  ctx.setLineDash([]);

  // Etiquetas de precio
  ctx.fillStyle = COLORS.text;
  ctx.font = "11px ui-sans-serif, system-ui";
  ctx.textAlign = "left";
  ctx.fillText(fmtPrice(minP), 4, h - 4);
  ctx.textAlign = "center";
  ctx.fillText("mid " + fmtPrice(depth.mid), x(depth.mid), 14);
  ctx.textAlign = "right";
  ctx.fillText(fmtPrice(maxP), w - 4, h - 4);
}

async function refreshDepth() {
  const book = await fetchOrderBook();
  if (!book) return;
  captureSnapshot(book);
  const depth = buildCumulativeDepth(book);
  renderDepth(depth);

  // Stats: bid/ask totales, imbalance
  const totalBid = depth.bids.reduce((a, b) => a + b.cum, 0) / depth.bids.length || 0;
  const totalAsk = depth.asks.reduce((a, b) => a + b.cum, 0) / depth.asks.length || 0;
  const top = (arr, n) => arr.slice(-n).reduce((a, b) => a + b.cum, 0);
  const bidWall = depth.bids.length ? depth.bids[0].cum : 0;
  const askWall = depth.asks.length ? depth.asks[depth.asks.length - 1].cum : 0;
  const imbalance = bidWall + askWall > 0 ? (bidWall - askWall) / (bidWall + askWall) : 0;
  const imbCls = imbalance >= 0 ? "green" : "red";

  document.getElementById("depth-stats").innerHTML = `
    <div><span class="muted">Mid</span> <b>${fmtPrice(depth.mid)}</b></div>
    <div><span class="muted">Best bid</span> <b class="green">${fmtPrice(depth.bids.length ? depth.bids[depth.bids.length-1].price : null)}</b></div>
    <div><span class="muted">Best ask</span> <b class="red">${fmtPrice(depth.asks.length ? depth.asks[0].price : null)}</b></div>
    <div><span class="muted">Liquidez bid (+/-1%)</span> <b class="green">${bidWall.toFixed(2)} BTC</b></div>
    <div><span class="muted">Liquidez ask (+/-1%)</span> <b class="red">${askWall.toFixed(2)} BTC</b></div>
    <div><span class="muted">Imbalance</span> <b class="${imbCls}">${(imbalance*100).toFixed(1)}%</b></div>
  `;
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
  localization: {
    timeFormatter: ts => {
      const d = new Date(ts * 1000);
      return _timeFormatter.format(d).replace(",", "");
    },
  },
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
// Render summary cards y posiciones abiertas
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

// ---------------------------------------------------------------------------
// Estado de filtros + paginacion (signals/trades)
// ---------------------------------------------------------------------------

const tableState = {
  signals: { all: [], filter: "all", page: 0 },
  trades:  { all: [], filter: "all", page: 0 },
};

function filterRows(rows, filter) {
  if (filter === "all") return rows;
  return rows.filter(r => r.timeframe === filter);
}

function pageSlice(rows, page) {
  const start = page * PAGE_SIZE;
  return rows.slice(start, start + PAGE_SIZE);
}

function updatePager(target, total) {
  const st = tableState[target];
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (st.page >= totalPages) st.page = totalPages - 1;
  const root = document.querySelector(`.pager[data-target="${target}"]`);
  if (!root) return;
  root.querySelector(".pager-info").textContent =
    total === 0 ? "0 filas" :
    `${st.page * PAGE_SIZE + 1}-${Math.min(total, (st.page + 1) * PAGE_SIZE)} de ${total}`;
  root.querySelector(".pager-prev").disabled = st.page <= 0;
  root.querySelector(".pager-next").disabled = st.page >= totalPages - 1;
}

function renderSignalsTable() {
  const st = tableState.signals;
  const root = document.getElementById("signals-table");
  const filtered = filterRows(st.all, st.filter)
    .slice().sort((a, b) => String(b.tick_ts_utc).localeCompare(String(a.tick_ts_utc)));
  updatePager("signals", filtered.length);
  const slice = pageSlice(filtered, st.page);
  if (slice.length === 0) {
    root.innerHTML = '<div class="empty">Sin senales en esta vista.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>Vela t (cierre = pregunta)</th><th>TF</th><th>Regimen</th><th>BTC close</th>" +
    "<th>vol decile</th><th>candidatos</th><th>en banda</th>" +
    "<th>p_win max</th><th>EV max</th><th>Decision</th><th>Detalle</th>" +
    "<th>Side</th><th>p_win elegido</th><th>Entrada t+1</th></tr></thead><tbody>";
  for (const s of slice) {
    const yes = s.decision === "YES";
    const regTxt = s.regime
      ? `${s.regime}${s.leverage ? " " + s.leverage + "x" : ""}`
      : "-";
    html += `<tr class="${yes ? "row-yes" : ""}">
      <td>${fmtTime(s.candle_close_time)}</td>
      <td>${s.timeframe}</td>
      <td class="small">${regTxt}</td>
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

function renderTradesTable() {
  const st = tableState.trades;
  const root = document.getElementById("trades-table");
  const filtered = filterRows(st.all, st.filter)
    .slice().sort((a, b) => String(b.exit_time).localeCompare(String(a.exit_time)));
  updatePager("trades", filtered.length);
  const slice = pageSlice(filtered, st.page);
  if (slice.length === 0) {
    root.innerHTML = '<div class="empty">Sin trades cerrados en esta vista.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>Entrada</th><th>Salida</th><th>TF</th><th>Side</th>" +
    "<th>Entry $</th><th>Exit $</th><th>TP $</th><th>SL $</th><th>Motivo</th>" +
    "<th>p_win</th><th>EV</th><th>PnL EUR</th><th>PnL %</th><th>Equity</th></tr></thead><tbody>";
  for (const t of slice) {
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

function wireToolbars() {
  document.querySelectorAll(".tf-buttons[data-target]").forEach(group => {
    const target = group.dataset.target;
    group.querySelectorAll(".tf-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        group.querySelectorAll(".tf-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        tableState[target].filter = btn.dataset.tf;
        tableState[target].page = 0;
        if (target === "signals") renderSignalsTable();
        else if (target === "trades") renderTradesTable();
      });
    });
  });
  document.querySelectorAll(".pager[data-target]").forEach(pager => {
    const target = pager.dataset.target;
    pager.querySelector(".pager-prev").addEventListener("click", () => {
      tableState[target].page = Math.max(0, tableState[target].page - 1);
      if (target === "signals") renderSignalsTable();
      else renderTradesTable();
    });
    pager.querySelector(".pager-next").addEventListener("click", () => {
      tableState[target].page += 1;
      if (target === "signals") renderSignalsTable();
      else renderTradesTable();
    });
  });
}

// ---------------------------------------------------------------------------
// Init + refresh loops
// ---------------------------------------------------------------------------

const charts = {};
let liveTfInterval = "15m";
let liveLimitByTf = { "1m": 240, "5m": 240, "15m": 200, "1h": 200, "4h": 200, "1d": 200 };

async function refreshLiveChart() {
  try {
    const candles = await fetchKlines(liveTfInterval, liveLimitByTf[liveTfInterval]);
    charts.live.series.setData(candles);
    const last = candles[candles.length - 1];
    const prev = candles[candles.length - 2] || last;
    const el = document.getElementById("live-price-value");
    el.textContent = fmtPrice(last.close);
    el.className = last.close >= prev.close ? "green" : "red";
    document.getElementById("live-price-source").textContent = "(" + klineSource + " " + liveTfInterval + ")";
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

// Directorio de datos: "data" (bot principal) o "data_oof" (modelo OOF).
// Lo fija cada pagina via window.DATA_DIR antes de cargar este script.
const DATA_DIR = (typeof window !== "undefined" && window.DATA_DIR) ? window.DATA_DIR : "data";

async function refreshBotData() {
  const [summary, openPos, trades, signals] = await Promise.all([
    fetchJSON(DATA_DIR + "/summary.json", { wallets: {}, generated_at: null }),
    fetchJSON(DATA_DIR + "/open_positions.json", { open_positions: [] }),
    fetchJSON(DATA_DIR + "/trades.json", { trades: [] }),
    fetchJSON(DATA_DIR + "/signals.json", { signals: [] }),
  ]);
  renderSummary(summary);
  renderOpenPositions(openPos);

  tableState.signals.all = signals.signals || [];
  tableState.trades.all  = trades.trades   || [];
  renderSignalsTable();
  renderTradesTable();

  for (const tf of TFS) {
    const eq = await fetchJSON(`${DATA_DIR}/equity_${tf}.json`, { curve: [], initial_capital_eur: 100 });
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

function wireLiveTfButtons() {
  document.querySelectorAll("#live-tf-buttons .tf-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#live-tf-buttons .tf-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      liveTfInterval = btn.dataset.tf;
      refreshLiveChart();
    });
  });
}

async function init() {
  document.getElementById("local-tz-label").textContent = "Hora local (" + LOCAL_TZ + ")";

  charts.live = makeCandleChart("chart-live", 380);
  try {
    HEATMAP.primitive = new LiqHeatmap();
    charts.live.series.attachPrimitive(HEATMAP.primitive);
  } catch (e) { console.warn("Heatmap primitive not supported:", e); }
  for (const tf of TFS) {
    charts[tf] = makeCandleChart("chart-tf-" + tf, 260);
    charts["eq-" + tf] = makeLineChart("chart-eq-" + tf, 180);
  }
  initDepthChart();

  wireToolbars();
  wireLiveTfButtons();

  const botData = await refreshBotData();
  await refreshLiveChart();
  await refreshTfCharts(botData.trades, botData.openPositions);
  await refreshDepth();

  setInterval(refreshLiveChart, 5000);
  setInterval(refreshDepth, 5000);
  setInterval(async () => {
    const d = await refreshBotData();
    await refreshTfCharts(d.trades, d.openPositions);
  }, 60000);
  setInterval(() => {
    const n = HEATMAP.snapshots.length;
    const el = document.getElementById("heatmap-status");
    if (el && n > 0) {
      const mins = Math.round(n * 5 / 60);
      el.textContent = `Heatmap de liquidez: ${n} snapshots (${mins} min acumulados)`;
    }
  }, 10000);
  window.addEventListener("resize", () => refreshDepth());
}

init();
