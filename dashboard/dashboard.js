// Dashboard logic. Carga JSONs desde ./data/*.json y los pinta.

const TFS = ["15m", "1h", "4h"];
const fmtEur = v => (v == null || Number.isNaN(v)) ? "-" : Number(v).toFixed(2) + " EUR";
const fmtPct = v => (v == null || Number.isNaN(v)) ? "-" : (Number(v) * 100).toFixed(2) + " %";
const fmtP = v => (v == null || Number.isNaN(v)) ? "-" : Number(v).toFixed(4);
const fmtPrice = v => (v == null || Number.isNaN(v)) ? "-" : Number(v).toFixed(2);
const fmtTime = v => {
  if (!v) return "-";
  const d = new Date(v);
  return d.toISOString().replace("T", " ").slice(0, 19);
};

async function fetchJSON(path, fallback) {
  try {
    const r = await fetch(path + "?t=" + Date.now());
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch {
    return fallback;
  }
}

function renderSummary(summary) {
  const root = document.getElementById("summary-cards");
  root.innerHTML = "";
  const wallets = summary.wallets || {};
  for (const tf of TFS) {
    const w = wallets[tf] || {};
    const pnl = (w.equity_eur || 0) - (w.initial_capital_eur || 0);
    const pnlClass = pnl >= 0 ? "green" : "red";
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `
      <h3>Cartera ${tf}</h3>
      <div class="kpi-grid">
        <span class="kpi-label">Inicial</span>
        <span class="kpi-value">${fmtEur(w.initial_capital_eur)}</span>
        <span class="kpi-label">Equity</span>
        <span class="kpi-value">${fmtEur(w.equity_eur)}</span>
        <span class="kpi-label">PnL realizado</span>
        <span class="kpi-value ${pnlClass}">${fmtEur(pnl)}</span>
        <span class="kpi-label">Trades</span>
        <span class="kpi-value">${w.n_trades || 0}</span>
        <span class="kpi-label">Win rate</span>
        <span class="kpi-value">${fmtPct(w.win_rate || 0)}</span>
        <span class="kpi-label">Posicion abierta</span>
        <span class="kpi-value">${w.open_position_id || "-"}</span>
      </div>`;
    root.appendChild(div);
  }
  const gen = document.getElementById("generated-at");
  gen.textContent = "Ultima actualizacion: " + fmtTime(summary.generated_at);
}

async function renderEquityCharts() {
  for (const tf of TFS) {
    const data = await fetchJSON(`data/equity_${tf}.json`, {curve: [], initial_capital_eur: 100});
    const ctx = document.getElementById(`chart-${tf}`).getContext("2d");
    const points = (data.curve || []).map(p => ({x: p.ts, y: p.equity}));
    if (points.length === 0) {
      points.push({x: new Date().toISOString(), y: data.initial_capital_eur || 100});
    }
    new Chart(ctx, {
      type: "line",
      data: {
        datasets: [{
          label: "Equity EUR",
          data: points,
          borderColor: "#58a6ff",
          backgroundColor: "rgba(88,166,255,.12)",
          fill: true,
          tension: 0.15,
          pointRadius: 1.5,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { type: "time", time: { tooltipFormat: "yyyy-MM-dd HH:mm" },
               ticks: { color: "#8a93a6", maxRotation: 0 },
               grid: { color: "#232936" } },
          y: { ticks: { color: "#8a93a6" }, grid: { color: "#232936" } }
        }
      }
    });
  }
}

function renderOpenPositions(data) {
  const root = document.getElementById("open-positions");
  const pos = data.open_positions || [];
  if (pos.length === 0) {
    root.innerHTML = '<div class="empty">No hay posiciones abiertas.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>TF</th><th>Side</th><th>Symbol</th><th>Entry</th><th>TP</th><th>SL</th>" +
    "<th>Timeout</th><th>p_win</th><th>EV</th><th>Notional</th>" +
    "</tr></thead><tbody>";
  for (const p of pos) {
    html += `<tr>
      <td>${p.timeframe}</td>
      <td class="${p.side}">${p.side}</td>
      <td>${p.symbol}</td>
      <td>${fmtPrice(p.entry_price)}</td>
      <td>${fmtPrice(p.tp_price)}</td>
      <td>${fmtPrice(p.sl_price)}</td>
      <td>${fmtTime(p.timeout_time)}</td>
      <td>${fmtP(p.p_win)}</td>
      <td>${fmtP(p.EV_pred)}</td>
      <td>${fmtEur(p.notional_eur)}</td>
    </tr>`;
  }
  html += "</tbody></table></div>";
  root.innerHTML = html;
}

function renderTrades(data) {
  const root = document.getElementById("trades-table");
  const trades = data.trades || [];
  if (trades.length === 0) {
    root.innerHTML = '<div class="empty">Sin trades aun.</div>';
    return;
  }
  let html = '<div class="table-wrap"><table><thead><tr>' +
    "<th>Entry</th><th>Exit</th><th>TF</th><th>Side</th>" +
    "<th>Entry $</th><th>Exit $</th><th>TP $</th><th>SL $</th><th>Reason</th>" +
    "<th>p_win</th><th>EV</th><th>PnL EUR</th><th>PnL %</th><th>Equity</th>" +
    "</tr></thead><tbody>";
  for (const t of trades) {
    const reasonCls = (t.exit_reason || "").toLowerCase();
    const pnlCls = (t.pnl_eur >= 0) ? "tp" : "sl";
    html += `<tr>
      <td>${fmtTime(t.entry_time)}</td>
      <td>${fmtTime(t.exit_time)}</td>
      <td>${t.timeframe}</td>
      <td class="${t.side}">${t.side}</td>
      <td>${fmtPrice(t.entry_price)}</td>
      <td>${fmtPrice(t.exit_price)}</td>
      <td>${fmtPrice(t.tp_price)}</td>
      <td>${fmtPrice(t.sl_price)}</td>
      <td class="${reasonCls}">${t.exit_reason}</td>
      <td>${fmtP(t.p_win)}</td>
      <td>${fmtP(t.EV_pred)}</td>
      <td class="${pnlCls}">${fmtEur(t.pnl_eur)}</td>
      <td class="${pnlCls}">${fmtPct(t.pnl_pct)}</td>
      <td>${fmtEur(t.wallet_equity_after)}</td>
    </tr>`;
  }
  html += "</tbody></table></div>";
  root.innerHTML = html;
}

async function init() {
  const [summary, openPos, trades] = await Promise.all([
    fetchJSON("data/summary.json", {wallets: {}, generated_at: null}),
    fetchJSON("data/open_positions.json", {open_positions: []}),
    fetchJSON("data/trades.json", {trades: []}),
  ]);
  renderSummary(summary);
  renderOpenPositions(openPos);
  renderTrades(trades);
  await renderEquityCharts();
}

// Chart.js v4 con escala 'time' requiere el adapter; lo cargamos diferido si falla.
if (typeof window !== "undefined" && !window._chartTimeAdapterLoaded) {
  const s = document.createElement("script");
  s.src = "https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js";
  s.onload = init;
  s.onerror = init;  // si falla, sigue (los ticks se mostraran como string)
  window._chartTimeAdapterLoaded = true;
  document.head.appendChild(s);
} else {
  init();
}
