// YOUR DATA - upload a simple CSV of your own plants and see NOVA's analysis.
// Kept separate from the six simulated crops: results live only in this page (nothing is saved).
import { api } from "../api.js";
import { axis, base, cropLabel, css, esc, fmt, hexAlpha, makeChart, pill, pct, setLoading, showError } from "../ui.js";

const COLUMNS = [
  ["plant_id", "Name of the plant", "P1", true],
  ["day", "Day number (0, 1, 2 …) or use a 'date' column (YYYY-MM-DD)", "0", true],
  ["soil_moisture_pct", "Soil moisture (%)", "34.5", true],
  ["air_temp_c", "Air temperature (°C)", "21.0", true],
  ["humidity_pct", "Air humidity (%)", "62", true],
  ["ec_ds_m", "Salt / nutrient level, EC (dS/m)", "1.4", true],
  ["height_cm", "Plant height (cm)", "6.2", true],
  ["leaf_count", "Number of leaves", "8", true],
  ["light_par", "Light (PAR)", "300", false],
  ["Re_kohm", "Leaf signal Re (kΩ), from the EIS clip", "22.1", false],
  ["Ri_kohm", "Leaf signal Ri (kΩ), from the EIS clip", "9.0", false],
];
const ENV = {
  soil: { key: "soil", label: "Soil moisture", unit: "%", lines: (l) => [[l.dry, `dry below ${l.dry}%`]] },
  temp: { key: "temp", label: "Temperature", unit: "°C", lines: (l) => [[l.heat, `too hot above ${l.heat} °C`]] },
  ec: { key: "ec", label: "Salt / nutrients (EC)", unit: "dS/m", lines: (l) => [[l.ec_high, `too salty above ${l.ec_high}`], [l.ec_low, `too few nutrients below ${l.ec_low}`]] },
  humidity: { key: "humidity", label: "Humidity", unit: "%", lines: () => [] },
};

// page state survives switching to other pages and back
const state = { crop: "lettuce", fileName: null, text: null, result: null, plant: null, env: "soil" };

export async function render(root) {
  setLoading(root);
  let species;
  try { species = await api.species(); } catch (e) { return showError(root, e); }
  if (!species.some((s) => s.species === state.crop)) state.crop = species[0].species;

  root.innerHTML = `
    <div class="page-head">
      <div><h1>Your Data</h1>
        <p>Upload a CSV of your own plants to get a stress forecast for each one. This page is
           <b>separate from the six simulated crops</b>: your file is analysed on the spot and nothing is saved.</p></div>
    </div>

    <div class="grid up-top">
      <div class="card">
        <h2>1 · Choose the crop and the file</h2>
        <p class="sub">The crop sets the safe limits and the model used.</p>
        <div class="controls" style="align-items:flex-end">
          <label class="field">Crop
            <select id="cropSel">${species.map((s) => `<option value="${s.species}" ${s.species === state.crop ? "selected" : ""}>${cropLabel(s.species)}</option>`).join("")}</select>
          </label>
          <label class="drop" id="drop" tabindex="0">
            <input type="file" id="file" accept=".csv,text/csv,text/plain" hidden>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5M4 16v4h16v-4"/></svg>
            <span id="dropText">${state.fileName ? `<b>${esc(state.fileName)}</b> · click to change` : "<b>Choose a CSV file</b> or drop it here"}</span>
          </label>
          <button class="btn primary" id="go" ${state.text ? "" : "disabled"}>Analyse</button>
        </div>
        <p class="sub" style="margin-top:14px">No file yet? Download a sample (5 plants, 30 days):
          <a href="/static/samples/sample_lettuce.csv" download>lettuce</a> ·
          <a href="/static/samples/sample_tomato.csv" download>tomato</a>.
          Plants: P1 healthy, P2 drought, P3 salt, P4 heatwave, P5 low nutrients.</p>
      </div>
      <div class="card">
        <h2>2 · File format</h2>
        <p class="sub">One row per plant per day. Column names are not case-sensitive.</p>
        <div class="table-wrap"><table class="data">
          <thead><tr><th>Column</th><th>What it is</th><th>Example</th></tr></thead>
          <tbody>${COLUMNS.map(([c, d, ex, req]) => `<tr><td><code>${c}</code>${req ? "" : ' <small style="color:var(--ink-3)">optional</small>'}</td><td>${d}</td><td>${ex}</td></tr>`).join("")}</tbody>
        </table></div>
      </div>
    </div>
    <div id="results"></div>`;

  const fileIn = root.querySelector("#file"), drop = root.querySelector("#drop"), goBtn = root.querySelector("#go");
  const loadFile = (f) => {
    if (!f) return;
    if (f.size > 2_000_000) { toastError(root, "The file is larger than 2 MB."); return; }
    const r = new FileReader();
    r.onload = () => {
      state.text = String(r.result); state.fileName = f.name;
      root.querySelector("#dropText").innerHTML = `<b>${esc(f.name)}</b> · click to change`;
      goBtn.disabled = false;
    };
    r.readAsText(f);
  };
  drop.addEventListener("click", (e) => { if (e.target !== fileIn) fileIn.click(); });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } });
  fileIn.addEventListener("change", () => loadFile(fileIn.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
  root.querySelector("#cropSel").addEventListener("change", (e) => { state.crop = e.target.value; });
  goBtn.addEventListener("click", () => analyse(root));

  if (state.result) showResults(root);
}

function toastError(root, msg) {
  root.querySelector("#results").innerHTML = `<div class="error mt"><b>Could not analyse the file.</b><br>${esc(msg)}</div>`;
}

async function analyse(root) {
  const out = root.querySelector("#results"), goBtn = root.querySelector("#go");
  goBtn.disabled = true;
  out.innerHTML = `<div class="loading">Analysing… the first file for a crop takes about 10 seconds while its model is trained.</div>`;
  try {
    state.result = await api.analyzeUpload(state.crop, state.text);
    state.plant = state.result.plants[0]?.plant_id;
    showResults(root);
  } catch (e) {
    state.result = null;
    toastError(root, e.message);
  } finally {
    goBtn.disabled = false;
  }
}

function showResults(root) {
  const R = state.result, S = R.summary, L = R.limits;
  if (!R.plants.some((p) => p.plant_id === state.plant)) state.plant = R.plants[0].plant_id;
  const out = root.querySelector("#results");
  out.innerHTML = `
    <h2 class="section-title">Results · ${cropLabel(R.crop)} <span style="font-weight:500;font-size:14px;color:var(--ink-3)">${esc(state.fileName || "")}</span></h2>
    ${R.warnings.length ? `<div class="note-box">${R.warnings.map((w) => `<div>⚠ ${esc(w)}</div>`).join("")}</div>` : ""}
    <div class="grid g-4">
      <div class="card kpi"><div class="label">Plants · days</div><div class="value">${S.plants}</div><div class="hint">${S.rows} readings over ${S.days} days</div></div>
      <div class="card kpi"><div class="label">${pill("high")}</div><div class="value">${S.high}</div><div class="hint">stress chance 60% or more</div></div>
      <div class="card kpi"><div class="label">${pill("medium")}</div><div class="value">${S.medium}</div><div class="hint">stress chance 30–60%</div></div>
      <div class="card kpi"><div class="label">${pill("low")}</div><div class="value">${S.low}</div><div class="hint">stress chance below 30%</div></div>
    </div>

    <div class="card mt">
      <div class="card-head"><div><h2>Time spent outside safe limits</h2>
        <p class="sub">Share of all readings beyond the ${cropLabel(R.crop).toLowerCase()} limits.</p></div></div>
      <div class="chart" style="height:200px" id="limitsChart" role="img" aria-label="Share of readings outside each safe limit"></div>
    </div>

    <h2 style="font-size:17px;margin:24px 0 12px">Plants <span style="font-weight:400;color:var(--ink-3);font-size:14px">highest risk first · click one for its charts</span></h2>
    <div class="plant-grid">
      ${R.plants.map((p) => `
        <div class="ptile ${p.level} ${p.plant_id === state.plant ? "sel" : ""}" tabindex="0" role="button" data-plant="${esc(p.plant_id)}"
             aria-label="${esc(p.plant_id)}, stress chance ${Math.round(100 * p.stress_prob)}%">
          <div class="top"><span class="id">${esc(p.plant_id)}</span>${pill(p.level)}</div>
          <div class="meta">Day ${p.last_day} · height ${fmt(p.height_cm, 1)} cm</div>
          <div class="cat">${p.reasons.length ? esc(p.reasons.join(", ")) : "All readings inside safe limits"}</div>
          ${p.eis_note ? `<div class="cat" style="min-height:0;margin-top:4px">🍃 ${esc(p.eis_note)}</div>` : ""}
          <div class="bar"><div style="width:0;background:var(--${p.level === "high" ? "critical" : p.level === "medium" ? "warning" : "good"})" data-w="${100 * p.stress_prob}"></div></div>
          <div style="font-size:12px;color:var(--ink-2);margin-top:4px">Stress chance (next 3 days) <b>${Math.round(100 * p.stress_prob)}%</b></div>
          <div style="font-size:12px;color:var(--ink-2)">Expected growth <b>${fmt(p.predicted_growth, 2)} cm/day</b></div>
        </div>`).join("")}
    </div>

    <div id="plantDetail"></div>

    <p class="sub mt" style="font-size:12.5px;color:var(--ink-3)">How this works: your readings are compared with the crop's safe limits and scored by a model
      trained on ${R.model.trained_on_plants} simulated ${cropLabel(R.crop).toLowerCase()} plants (tested on unseen simulated plants: stress score F1 ${fmt(R.model.stress_f1, 2)},
      growth error ${fmt(R.model.growth_mae_cm_per_day, 2)} cm/day). Treat results on real plants as a first estimate until the model is retrained on real data.</p>`;

  requestAnimationFrame(() => out.querySelectorAll(".bar > div").forEach((d) => { d.style.width = d.dataset.w + "%"; }));
  out.querySelectorAll(".ptile").forEach((t) => {
    const pick = () => { state.plant = t.dataset.plant; out.querySelectorAll(".ptile").forEach((x) => x.classList.toggle("sel", x === t)); plantDetail(out); };
    t.addEventListener("click", pick);
    t.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
  limitsChart(out, S);
  plantDetail(out);
  out.scrollIntoView({ behavior: "smooth", block: "start" });
}

function limitsChart(out, S) {
  const b = base();
  const rows = [["Too dry", S.share_dry], ["Too hot", S.share_hot], ["Too salty", S.share_salty], ["Too few nutrients", S.share_low_nutrients]];
  makeChart(out.querySelector("#limitsChart")).setOption({
    ...b,
    grid: { left: 8, right: 60, top: 6, bottom: 6, containLabel: true },
    tooltip: { ...b.tooltip, trigger: "axis", axisPointer: { type: "none" }, formatter: (p) => `${p[0].name}: <b>${pct(p[0].value, 0)}</b> of readings` },
    xAxis: axis({ type: "value", max: 1, show: false }),
    yAxis: axis({ type: "category", data: rows.map((r) => r[0]), inverse: true, axisLine: { show: false }, splitLine: { show: false }, axisLabel: { color: css("--ink-2"), fontSize: 13 } }),
    series: [{ type: "bar", data: rows.map((r) => r[1]), barWidth: 16, itemStyle: { color: css("--accent"), borderRadius: [0, 6, 6, 0] },
      label: { show: true, position: "right", formatter: (p) => pct(p.value, 0), color: css("--ink"), fontWeight: 600 } }],
  });
}

function plantDetail(out) {
  const R = state.result, T = R.timelines[state.plant], p = R.plants.find((x) => x.plant_id === state.plant);
  const box = out.querySelector("#plantDetail");
  box.innerHTML = `
    <div class="card mt">
      <div class="card-head"><div><h2>${esc(p.plant_id)} · chance of stress in the next 3 days</h2>
        <p class="sub">Above the orange line = watch, above the red line = high risk.</p></div>${pill(p.level)}</div>
      <div class="chart short" id="riskChart" role="img" aria-label="Stress chance by day"></div>
    </div>
    <div class="grid g-2 mt up-two">
      <div class="card">
        <div class="card-head"><h2>Conditions</h2>
          <div class="seg" id="envSeg">${Object.entries(ENV).map(([k, v]) => `<button data-v="${k}" class="${k === state.env ? "on" : ""}">${v.label.split(" (")[0]}</button>`).join("")}</div></div>
        <div class="chart short" id="envChart" role="img" aria-label="Selected condition by day with its safe limit"></div>
      </div>
      <div class="card">
        <h2>Growth</h2><p class="sub">Height (line) and leaf count (bars).</p>
        <div class="chart short" id="growChart" role="img" aria-label="Height and leaf count by day"></div>
      </div>
    </div>
    ${R.has_eis ? `<div class="card mt">
      <h2>Leaf signal (EIS)</h2>
      <p class="sub">Relative to the plant's first reading (1.0 = unchanged). Rising <b>Re</b> suggests water loss, falling <b>Re</b> suggests salt, falling <b>Ri</b> suggests leaky membranes.</p>
      <div class="chart short" id="eisChart" role="img" aria-label="Relative Re and Ri by day"></div>
    </div>` : `<p class="sub mt" style="color:var(--ink-3)">Add <code>Re_kohm</code> and <code>Ri_kohm</code> columns to also see the leaf signal.</p>`}`;

  box.querySelectorAll("#envSeg button").forEach((b) => b.addEventListener("click", () => {
    state.env = b.dataset.v;
    box.querySelectorAll("#envSeg button").forEach((x) => x.classList.toggle("on", x === b));
    envChart(box, T, R.limits);
  }));
  riskChart(box, T);
  envChart(box, T, R.limits);
  growChart(box, T);
  if (R.has_eis) eisChart(box, T);
}

function lineOpts(T, yName, extra = {}) {
  const b = base();
  return {
    ...b, legend: { show: false }, grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
    tooltip: { ...b.tooltip, trigger: "axis" },
    xAxis: axis({ type: "category", data: T.day, name: "day", nameLocation: "end", boundaryGap: false, splitLine: { show: false } }),
    yAxis: axis({ type: "value", name: yName, scale: true }), ...extra,
  };
}

const refLine = (v, text, color) => ({ yAxis: v, lineStyle: { color, type: "dashed", width: 1.5 }, label: { formatter: text, color, position: "insideEndTop", fontSize: 11.5 } });

function riskChart(box, T) {
  const brand = css("--brand");
  makeChart(box.querySelector("#riskChart")).setOption(lineOpts(T, "stress chance", {
    yAxis: axis({ type: "value", min: 0, max: 1, axisLabel: { color: css("--ink-3"), formatter: (v) => `${Math.round(100 * v)}%` } }),
    tooltip: { ...base().tooltip, trigger: "axis", valueFormatter: (v) => `${Math.round(100 * v)}%` },
    series: [{ type: "line", name: "Stress chance", data: T.stress_prob, smooth: true, symbol: "none", lineStyle: { width: 2.5, color: brand },
      areaStyle: { color: hexAlpha(brand, 0.12) },
      markLine: { silent: true, symbol: "none", data: [refLine(0.6, "high risk", css("--critical")), refLine(0.3, "watch", css("--warning"))] } }],
  }));
}

function envChart(box, T, L) {
  const v = ENV[state.env], col = css("--earth");
  makeChart(box.querySelector("#envChart")).setOption(lineOpts(T, v.unit, {
    tooltip: { ...base().tooltip, trigger: "axis", valueFormatter: (x) => `${x} ${v.unit}` },
    series: [{ type: "line", name: v.label, data: T[v.key], symbol: "none", lineStyle: { width: 2.2, color: col },
      markLine: { silent: true, symbol: "none", data: v.lines(L).map(([y, t]) => refLine(y, t, css("--critical"))) } }],
  }), true);
}

function growChart(box, T) {
  const b = base();
  makeChart(box.querySelector("#growChart")).setOption({
    ...lineOpts(T, "cm"),
    legend: { ...b.legend, show: true, data: ["Height", "Leaves"] },
    yAxis: [axis({ type: "value", name: "cm", scale: true }), axis({ type: "value", name: "leaves", splitLine: { show: false } })],
    xAxis: axis({ type: "category", data: T.day, splitLine: { show: false } }),
    series: [
      { type: "bar", name: "Leaves", yAxisIndex: 1, data: T.leaves, itemStyle: { color: hexAlpha(css("--sage"), 0.55), borderRadius: [3, 3, 0, 0] } },
      { type: "line", name: "Height", data: T.height, symbol: "none", smooth: true, lineStyle: { width: 2.5, color: css("--brand") }, itemStyle: { color: css("--brand") } },
    ],
  });
}

function eisChart(box, T) {
  const b = base();
  makeChart(box.querySelector("#eisChart")).setOption(lineOpts(T, "× day 1", {
    legend: { ...b.legend, show: true, data: ["Re", "Ri"] },
    tooltip: { ...b.tooltip, trigger: "axis", valueFormatter: (x) => (x == null ? "-" : x.toFixed(2)) },
    series: [
      { type: "line", name: "Re", data: T.Re_rel, symbol: "none", lineStyle: { width: 2.2, color: css("--eis-re") }, itemStyle: { color: css("--eis-re") },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 1, lineStyle: { color: css("--axis"), type: "solid" }, label: { show: false } }] } },
      { type: "line", name: "Ri", data: T.Ri_rel, symbol: "none", lineStyle: { width: 2.2, color: css("--eis-ri") }, itemStyle: { color: css("--eis-ri") } },
    ],
  }));
}
