// STRESS MONITOR - every plant's latest modelled stress risk, as a farmer would see it.
import { api } from "../api.js";
import {
  TREATMENTS, TREATMENT_LABEL, axis, base, cropLabel, css, esc, go, makeChart, pill,
  prettyCategory, routeParams, setLoading, showError, tcolor,
} from "../ui.js";

const LEVELS = ["high", "medium", "low"];

export async function render(root) {
  setLoading(root);
  let rows;
  try { rows = await api.status(); } catch (e) { return showError(root, e); }
  const crops = [...new Set(rows.map((r) => r.species))];
  const params = routeParams();
  const crop = crops.includes(params.crop) ? params.crop : "all";
  const shown = rows.filter((r) => crop === "all" || r.species === crop)
    .sort((a, b) => (b.modelled_stress_prob ?? -1) - (a.modelled_stress_prob ?? -1));
  const n = (lvl) => shown.filter((r) => r.level === lvl).length;

  root.innerHTML = `
    <div class="page-head">
      <div><h1>Stress Monitor</h1>
        <p>The chance that each plant will be stressed in the next 3 days, from its latest readings.
           Click a plant to see its history.</p></div>
      <div class="seg" id="cropSeg">
        <button data-c="all" class="${crop === "all" ? "on" : ""}">All crops</button>
        ${crops.map((c) => `<button data-c="${c}" class="${c === crop ? "on" : ""}">${cropLabel(c)}</button>`).join("")}
      </div>
    </div>

    <div class="grid g-3">
      <div class="card kpi"><div class="label">${pill("high")}</div><div class="value">${n("high")}</div><div class="hint">chance of stress 60% or more</div></div>
      <div class="card kpi"><div class="label">${pill("medium")}</div><div class="value">${n("medium")}</div><div class="hint">chance of stress 30-60%</div></div>
      <div class="card kpi"><div class="label">${pill("low")}</div><div class="value">${n("low")}</div><div class="hint">chance of stress below 30%</div></div>
    </div>

    <div class="card mt">
      <h2>Average stress risk by crop and treatment</h2>
      <p class="sub">Each square is the average chance of stress for that group (scale on the right). Hover for details.</p>
      <div class="chart short" id="heat" role="img" aria-label="Heatmap of average stress probability by crop and treatment"></div>
    </div>

    <h2 style="font-size:17px;margin:24px 0 12px">Plants (${shown.length}) <span style="font-weight:400;color:var(--ink-3);font-size:14px">highest risk first</span></h2>
    <div class="plant-grid">
      ${shown.map((r) => {
        const p = r.modelled_stress_prob;
        return `<div class="ptile ${r.level}" tabindex="0" role="button" data-crop="${r.species}" data-plant="${esc(r.plant_id)}"
                 aria-label="${esc(r.plant_id)}, ${TREATMENT_LABEL[r.treatment]}, stress chance ${p == null ? "unknown" : Math.round(100 * p) + "%"}">
          <div class="top"><span class="id">${esc(r.plant_id)}</span>${pill(r.level)}</div>
          <div class="meta"><i style="background:${tcolor(r.treatment)}"></i>${cropLabel(r.species)} · ${TREATMENT_LABEL[r.treatment]}</div>
          <div class="cat">${esc(prettyCategory(r.category))}</div>
          <div class="bar"><div style="width:0;background:var(--${r.level === "high" ? "critical" : r.level === "medium" ? "warning" : "good"})" data-w="${p == null ? 0 : 100 * p}"></div></div>
          <div style="font-size:12px;color:var(--ink-2);margin-top:4px">Stress chance <b>${p == null ? "-" : Math.round(100 * p) + "%"}</b></div>
        </div>`;
      }).join("")}
    </div>`;

  requestAnimationFrame(() => root.querySelectorAll(".bar > div").forEach((d) => { d.style.width = d.dataset.w + "%"; }));
  root.querySelectorAll("#cropSeg button").forEach((b) =>
    b.addEventListener("click", () => go("monitor", b.dataset.c === "all" ? {} : { crop: b.dataset.c })));
  root.querySelectorAll(".ptile").forEach((t) => {
    const open = () => go("explorer", { crop: t.dataset.crop, plant: t.dataset.plant });
    t.addEventListener("click", open);
    t.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  });

  heatmap(root, rows, crops);
}

function heatmap(root, rows, crops) {
  const b = base();
  const data = [];
  crops.forEach((c, yi) => TREATMENTS.forEach((t, xi) => {
    const ps = rows.filter((r) => r.species === c && r.treatment === t && r.modelled_stress_prob != null)
      .map((r) => r.modelled_stress_prob);
    if (ps.length) { const v = ps.reduce((a, x) => a + x, 0) / ps.length; data.push({ value: [xi, yi, v], label: { color: labelInk(v) } }); }
  }));
  const chart = makeChart(root.querySelector("#heat"));
  chart.setOption({
    ...b,
    grid: { left: 8, right: 90, top: 4, bottom: 4, containLabel: true },
    tooltip: { ...b.tooltip, formatter: (p) => `<b>${cropLabel(crops[p.value[1]])} · ${TREATMENT_LABEL[TREATMENTS[p.value[0]]]}</b><br>Average stress chance: <b>${Math.round(100 * p.value[2])}%</b>` },
    xAxis: axis({ type: "category", data: TREATMENTS.map((t) => TREATMENT_LABEL[t]), splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: css("--ink-2") } }),
    yAxis: axis({ type: "category", data: crops.map(cropLabel), splitLine: { show: false }, axisLine: { show: false }, axisLabel: { color: css("--ink-2") } }),
    visualMap: {
      min: 0, max: 1, calculable: false, orient: "vertical", right: 0, top: "middle", itemHeight: 140, itemWidth: 12,
      text: ["100%", "0%"], textStyle: { color: css("--ink-3"), fontSize: 11 },
      inRange: { color: [css("--seq-0"), css("--seq-1"), css("--seq-2"), css("--seq-3")] },
    },
    series: [{
      type: "heatmap", data,
      label: { show: true, formatter: (p) => `${Math.round(100 * p.value[2])}%`, fontSize: 12, textBorderWidth: 0 },
      itemStyle: { borderColor: css("--surface"), borderWidth: 3, borderRadius: 6 },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,.25)" } },
    }],
  });
}

function labelInk(v) {
  const dark = css("color-scheme") === "dark";
  // light theme: ramp goes light->dark, so low values need dark ink; dark theme: the reverse
  return dark ? (v > 0.5 ? "#0b0b0b" : "#f0efec") : (v > 0.45 ? "#ffffff" : "#0b0b0b");
}
