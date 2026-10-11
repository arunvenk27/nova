// Shared UI helpers: colours (read from CSS so light/dark mode just works),
// labels, chart creation with common styling, and small HTML builders.

export const TREATMENTS = ["control", "drought", "salinity", "heat", "famine"];
export const TREATMENT_LABEL = {
  control: "Control", drought: "Drought", salinity: "Salinity", heat: "Heat", famine: "Famine",
};
export const TREATMENT_INFO = {
  control: "Normal water and nutrients",
  drought: "Watering stopped for two periods",
  salinity: "Salt (EC) raised gradually",
  heat: "Room heated for 4-day spells",
  famine: "Plain water only + reduced watering",
};
const CROP_LABEL = { lettuce: "Lettuce", spinach: "Spinach", pak_choi: "Pak choi", kale: "Kale", basil: "Basil", tomato: "Tomato" };
export const cropLabel = (c) => CROP_LABEL[c] || c;
export const GROWTH_LABEL = { rosette_diam_cm: "rosette diameter", height_cm: "height" };

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const tcolor = (t) => css(`--t-${t}`);
export const isDark = () => css("color-scheme") === "dark";

export function hexAlpha(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "-" : Number(x).toFixed(d));
export const pct = (x, d = 0) => (x == null ? "-" : `${(100 * x).toFixed(d)}%`);
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ status (always icon + label)
const ICON = {
  low: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
  medium: '<svg viewBox="0 0 24 24"><path d="M12 6v8M12 18v.5"/></svg>',
  high: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
};
const LEVEL_LABEL = { low: "Low risk", medium: "Watch", high: "High risk", unknown: "No data" };
export function levelOf(p) {
  if (p == null) return "unknown";
  return p >= 0.6 ? "high" : p >= 0.3 ? "medium" : "low";
}
export const pill = (level) => `<span class="pill ${level}">${ICON[level] || ""}${LEVEL_LABEL[level]}</span>`;
export function prettyCategory(cat) {
  if (!cat) return "-";
  return cat.charAt(0) + cat.slice(1).toLowerCase().replace(/ - /g, " – ");
}

// ------------------------------------------------------------------ charts
const charts = new Set();
let resizeBound = false;

export function makeChart(el) {
  const chart = echarts.init(el, null, { renderer: "canvas" });
  charts.add(chart);
  if (!resizeBound) {
    resizeBound = true;
    window.addEventListener("resize", () => charts.forEach((c) => c.resize()));
  }
  return chart;
}

export function disposeCharts() {
  charts.forEach((c) => c.dispose());
  charts.clear();
}

// Common look: recessive solid hairline grid, ink-coloured text, themed tooltip.
export function base() {
  const ink = css("--ink"), ink2 = css("--ink-2"), ink3 = css("--ink-3"), grid = css("--grid"), axis = css("--axis");
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', color: ink2 },
    animationDuration: 900,
    animationEasing: "cubicOut",
    animationDurationUpdate: 600,
    tooltip: {
      backgroundColor: css("--surface"), borderColor: css("--border"), borderWidth: 1,
      textStyle: { color: ink, fontSize: 13 }, extraCssText: "box-shadow:0 6px 20px rgba(0,0,0,.15);border-radius:10px;",
    },
    legend: { textStyle: { color: ink2, fontSize: 12.5 }, itemWidth: 12, itemHeight: 12, icon: "roundRect", top: 0 },
    axisCommon: {
      axisLine: { lineStyle: { color: axis } }, axisTick: { show: false },
      axisLabel: { color: ink3, fontSize: 12 }, splitLine: { lineStyle: { color: grid, type: "solid" } },
      nameTextStyle: { color: ink3, fontSize: 12 },
    },
  };
}

export function axis(extra = {}) {
  return { ...base().axisCommon, ...extra };
}

export function setLoading(root, text = "Loading…") {
  root.innerHTML = `<div class="loading">${esc(text)}</div>`;
}

export function showError(root, err) {
  root.innerHTML = `<div class="error"><b>Could not load data.</b><br>${esc(err.message || err)}<br>
    <small>Is the server still running? Check the PowerShell window.</small></div>`;
}

// Count-up animation for KPI numbers
export function countUp(el, to, { decimals = 0, suffix = "", duration = 900 } = {}) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { el.textContent = to.toFixed(decimals) + suffix; return; }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const v = to * (1 - Math.pow(1 - t, 3));
    el.textContent = v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// Hash-route query helpers (#/page?a=1&b=2)
export function routeParams() {
  const q = location.hash.split("?")[1] || "";
  return Object.fromEntries(new URLSearchParams(q));
}
export function go(page, params = {}) {
  const q = new URLSearchParams(params).toString();
  location.hash = `#/${page}${q ? "?" + q : ""}`;
}
