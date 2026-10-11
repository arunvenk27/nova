// EIS LAB - the physics: how a leaf's impedance curve changes under stress, and
// (Objective 3) whether the EIS signal tracks the environment and growth.
import { api } from "../api.js";
import {
  TREATMENT_LABEL, axis, base, cropLabel, css, esc, fmt, go, makeChart, routeParams,
  setLoading, showError, tcolor,
} from "../ui.js";

const VAR_LABEL = {
  soil_moisture_pct: "Soil moisture", ec_ds_m: "Salt (EC)", air_temp_c: "Air temp",
  wilting_score: "Wilting", growth_past3: "Growth (last 3 d)", growth_next3: "Growth (next 3 d)",
};
const GROUP_LABEL = { "all plants": "All plants", ...TREATMENT_LABEL };
let feature = "Re_rel";

export async function render(root) {
  setLoading(root);
  let species;
  try { species = await api.species(); } catch (e) { return showError(root, e); }
  const params = routeParams();
  const crop = species.some((s) => s.species === params.crop) ? params.crop : "lettuce";
  let plants;
  try { plants = await api.plants(crop); } catch (e) { return showError(root, e); }
  const plantId = plants.some((p) => p.plant_id === params.plant) ? params.plant
    : (plants.find((p) => p.treatment === "drought") || plants[0]).plant_id;
  const plant = plants.find((p) => p.plant_id === plantId);

  root.innerHTML = `
    <div class="page-head">
      <div><h1>EIS Lab</h1>
        <p>A tiny current is passed through a leaf at 31 frequencies. The resulting curve reveals the water outside
           the cells (<b>Re</b>) and inside them (<b>Ri</b>). Move the slider to watch the curve change as the plant is stressed.</p></div>
      <div class="controls">
        <label class="field">Crop
          <select id="cropSel">${species.map((s) => `<option value="${s.species}" ${s.species === crop ? "selected" : ""}>${cropLabel(s.species)}</option>`).join("")}</select></label>
        <label class="field">Plant
          <select id="plantSel">${plants.map((p) => `<option value="${esc(p.plant_id)}" ${p.plant_id === plantId ? "selected" : ""}>${esc(p.plant_id)} · ${TREATMENT_LABEL[p.treatment]}</option>`).join("")}</select></label>
      </div>
    </div>

    <div class="grid g-3">
      <div class="card span-2">
        <div class="card-head">
          <div><h2>Impedance curve (Nyquist plot)</h2>
            <p class="sub">Dots = measurement, line = fitted model. A bigger arc means higher resistance.</p></div>
        </div>
        <div class="chart" id="nyq" role="img" aria-label="Nyquist plot comparing the baseline sweep with the selected day"></div>
        <div class="slider-row" style="margin-top:6px">
          <span style="font-size:13px;color:var(--ink-2);white-space:nowrap">Day</span>
          <input type="range" id="daySlider" min="0" max="0" step="1" aria-label="Measurement day">
          <output id="dayOut">-</output>
        </div>
      </div>
      <div class="card">
        <h2>Fitted values</h2>
        <p class="sub" id="valSub">Selected day compared with the first sweep.</p>
        <div class="vals" id="vals" style="grid-template-columns:1fr 1fr"></div>
        <ul class="tight" style="margin-top:14px">
          <li><b>Re</b> - resistance of the fluid <i>between</i> cells. Drying leaves → Re rises.</li>
          <li><b>Ri</b> - resistance <i>inside</i> cells, behind the membranes.</li>
          <li><b>fc</b> - frequency at the top of the arc; shifts when membranes change.</li>
        </ul>
        <p class="sub" style="font-size:12px;margin-top:10px">Re and Ri use the standard Fricke circuit. The patent-based method is not implemented yet.</p>
      </div>
    </div>

    <div class="card mt">
      <div class="card-head">
        <div><h2>Does the EIS signal track stress? <span style="font-weight:400;color:var(--ink-3)">(${cropLabel(crop)})</span></h2>
          <p class="sub">Correlation between the EIS signal and each variable, within each plant, after removing the effect of plant age.
             <b style="color:var(--div-pos)">Red</b> = rise together, <b style="color:var(--div-neg)">blue</b> = move opposite, ★ = reliable (95% interval excludes zero).</p></div>
        <div class="seg" id="featSeg">
          <button data-f="Re_rel" class="${feature === "Re_rel" ? "on" : ""}">Re (outside cells)</button>
          <button data-f="Ri_rel" class="${feature === "Ri_rel" ? "on" : ""}">Ri (inside cells)</button>
        </div>
      </div>
      <div class="chart" id="corr" role="img" aria-label="Correlation heatmap by treatment group and variable"></div>
    </div>`;

  root.querySelector("#cropSel").addEventListener("change", (e) => go("eis", { crop: e.target.value }));
  root.querySelector("#plantSel").addEventListener("change", (e) => go("eis", { crop, plant: e.target.value }));

  // ---------------- Nyquist + slider
  const nyq = makeChart(root.querySelector("#nyq"));
  const slider = root.querySelector("#daySlider");
  let first;
  try { first = await api.eis(crop, plantId); } catch (e) { return showError(root.querySelector("#nyq"), e); }
  const days = first.days;
  slider.max = days.length - 1;
  const startIdx = params.day != null ? Math.max(0, days.indexOf(+params.day)) : bestStressDay(days, plant, first);
  slider.value = startIdx;

  let pending = 0;
  const show = async (idx) => {
    const day = days[idx];
    root.querySelector("#dayOut").textContent = `day ${day}`;
    const ticket = ++pending;
    let e;
    try { e = await api.eis(crop, plantId, day); } catch (err) { return; }
    if (ticket !== pending) return;            // a newer slider position won
    drawNyquist(nyq, e, plant.treatment);
    drawValues(root, e);
  };
  let timer;
  slider.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => show(+slider.value), 60); });
  await show(startIdx);

  // ---------------- correlation heatmap
  const corrChart = makeChart(root.querySelector("#corr"));
  const drawCorr = async () => {
    try { drawCorrelations(corrChart, await api.correlations(crop, feature)); }
    catch (e) { showError(root.querySelector("#corr"), e); }
  };
  root.querySelectorAll("#featSeg button").forEach((b) => b.addEventListener("click", () => {
    feature = b.dataset.f;
    root.querySelectorAll("#featSeg button").forEach((x) => x.classList.toggle("on", x === b));
    drawCorr();
  }));
  drawCorr();
}

// start the slider somewhere interesting: for stressed plants, ~60% through the trial
function bestStressDay(days, plant) {
  return plant.treatment === "control" ? days.length - 1 : Math.round(days.length * 0.6);
}

function drawNyquist(chart, e, treatment) {
  const b = base();
  const baseColor = css("--ink-3"), selColor = tcolor(treatment);
  const B = e.baseline, S = e.selected;
  const all = [...B.points, ...S.points, ...B.fit_curve, ...S.fit_curve];
  const xmax = Math.max(...all.map((p) => p[0])), ymax = Math.max(...all.map((p) => p[1]));
  const k = (v) => v / 1000;
  const ser = (sw, color, name, z) => [
    { type: "line", name, data: sw.fit_curve.map((p) => [k(p[0]), k(p[1])]), showSymbol: false, smooth: false, z,
      lineStyle: { width: 2, color }, itemStyle: { color }, tooltip: { show: false } },
    { type: "scatter", name, data: sw.points.map((p) => [k(p[0]), k(p[1]), p[2]]), symbolSize: 8, z: z + 1,
      itemStyle: { color, borderColor: css("--surface"), borderWidth: 2 } },
  ];
  chart.setOption({
    ...b,
    legend: { ...b.legend, data: [`Day ${B.day} (first sweep)`, `Day ${S.day} (selected)`] },
    grid: { left: 12, right: 18, top: 34, bottom: 30, containLabel: true },
    tooltip: { ...b.tooltip, trigger: "item",
      formatter: (p) => `${p.marker}<b>${p.seriesName}</b><br>${fmtHz(p.value[2])}<br>Z′ ${p.value[0].toFixed(1)} kΩ · −Z″ ${p.value[1].toFixed(1)} kΩ` },
    xAxis: axis({ type: "value", name: "Z′  real part (kΩ)", nameLocation: "middle", nameGap: 28, min: 0, max: Math.ceil(k(xmax) * 1.05) }),
    yAxis: axis({ type: "value", name: "−Z″ (kΩ)", min: 0, max: Math.ceil(k(ymax) * 1.25) }),
    series: [...ser(B, baseColor, `Day ${B.day} (first sweep)`, 2), ...ser(S, selColor, `Day ${S.day} (selected)`, 4)],
  }, true);
}

const fmtHz = (f) => (f >= 1e6 ? `${(f / 1e6).toFixed(1)} MHz` : f >= 1e3 ? `${(f / 1e3).toFixed(1)} kHz` : `${f.toFixed(0)} Hz`);

function drawValues(root, e) {
  const B = e.baseline, S = e.selected;
  const ch = (a, b) => { const d = 100 * (a / b - 1); return { d, txt: `${d >= 0 ? "+" : ""}${d.toFixed(0)}%` }; };
  const tile = (k, v, d, unit) => `<div class="val"><div class="k">${k}</div><div class="v">${v}<small style="font-size:12px;font-weight:500"> ${unit}</small></div>
      <div class="d ${d == null ? "" : d.d > 2 ? "up" : d.d < -2 ? "down" : ""}">${d == null ? "&nbsp;" : d.txt + " vs day " + B.day}</div></div>`;
  root.querySelector("#valSub").textContent = `Day ${S.day} compared with the first sweep (day ${B.day}).`;
  root.querySelector("#vals").innerHTML =
    tile("Re (outside cells)", (S.Re / 1000).toFixed(1), ch(S.Re, B.Re), "kΩ") +
    tile("Ri (inside cells)", (S.Ri / 1000).toFixed(1), ch(S.Ri, B.Ri), "kΩ") +
    tile("fc (arc peak)", (S.fc_hz / 1000).toFixed(1), ch(S.fc_hz, B.fc_hz), "kHz") +
    tile("Fit error", (100 * S.rel_rmse).toFixed(1), null, "%");
}

function drawCorrelations(chart, c) {
  const b = base();
  const xs = c.variables, ys = [...c.groups].reverse();
  const data = c.cells.map((cell) => ({
    value: [xs.indexOf(cell.variable), ys.indexOf(cell.group), cell.rho],
    cell,
    label: { color: Math.abs(cell.rho) > 0.45 ? "#fff" : css("--ink") },
  })).filter((d) => d.value[0] >= 0 && d.value[1] >= 0);
  chart.setOption({
    ...b,
    grid: { left: 8, right: 70, top: 6, bottom: 4, containLabel: true },
    tooltip: { ...b.tooltip, formatter: (p) => {
      const x = p.data.cell;
      return `<b>${GROUP_LABEL[x.group]} · ${VAR_LABEL[x.variable]}</b><br>ρ = <b>${x.rho.toFixed(2)}</b>
              (95%: ${x.lo.toFixed(2)} to ${x.hi.toFixed(2)})<br>${x.significant ? "★ reliable" : "not reliable (interval includes 0)"}`;
    } },
    xAxis: axis({ type: "category", data: xs.map((v) => VAR_LABEL[v]), splitLine: { show: false }, axisLine: { show: false },
      axisLabel: { color: css("--ink-2"), interval: 0, fontSize: 12 } }),
    yAxis: axis({ type: "category", data: ys.map((g) => GROUP_LABEL[g]), splitLine: { show: false }, axisLine: { show: false },
      axisLabel: { color: css("--ink-2") } }),
    visualMap: { min: -1, max: 1, calculable: false, right: 0, top: "middle", itemHeight: 150, itemWidth: 12,
      text: ["+1", "−1"], textStyle: { color: css("--ink-3"), fontSize: 11 },
      inRange: { color: [css("--div-neg"), css("--div-mid"), css("--div-pos")] } },
    series: [{
      type: "heatmap", data,
      label: { show: true, fontSize: 12, formatter: (p) => `${p.data.cell.rho.toFixed(2)}${p.data.cell.significant ? " ★" : ""}` },
      itemStyle: { borderColor: css("--surface"), borderWidth: 3, borderRadius: 6 },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,.25)" } },
    }],
  }, true);
}
