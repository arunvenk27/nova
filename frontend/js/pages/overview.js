// OVERVIEW - the 30-second story: headline numbers, harvest under stress,
// how well stress is predicted, and what we learned.
import { api } from "../api.js";
import {
  TREATMENTS, TREATMENT_LABEL, TREATMENT_INFO, axis, base, countUp, cropLabel, css, esc, fmt,
  makeChart, setLoading, showError, tcolor,
} from "../ui.js";

export async function render(root) {
  setLoading(root);
  let data;
  try { data = await api.overview(); } catch (e) { return showError(root, e); }
  const k = data.kpis;

  root.innerHTML = `
    <section class="hero">
      <div>
        <span class="eyebrow"><i></i>NOVA · Natural Observation &amp; Vegetative Analysis</span>
        <h1>See plant stress <em>before</em> it shows.</h1>
        <p>NOVA reads a leaf's electrical signal (EIS) alongside soil, air and growth data to forecast growth
           and flag stress three days ahead - tested here on six greenhouse crops.</p>
        <div class="actions">
          <a class="btn primary" href="#/explorer">Explore a plant &rarr;</a>
          <a class="btn ghost" href="#/about">How it works</a>
        </div>
      </div>
      <div class="hero-art" aria-hidden="true">
        <div class="ring r1"></div><div class="ring r2"></div>
        <svg class="leaf" viewBox="0 0 32 32"><path d="M16 29V15M16 17C16 9 10 5 4 5c0 8 5 12 12 12zm0-2c0-7 5-11 12-11 0 7-5 11-12 11z"/></svg>
        <div class="tag t1">Stress forecast<b>${fmt(k.f1_min)}–${fmt(k.f1_max)} F1</b></div>
        <div class="tag t2">Famine harvest loss<b>up to ${Math.round(k.worst_famine_loss_pct)}%</b></div>
      </div>
    </section>

    <div class="grid g-5">
      <div class="card kpi"><div class="label">Crops studied</div><div class="value" id="k1">0</div>
        <div class="hint">lettuce, spinach, pak choi, kale, basil, tomato</div></div>
      <div class="card kpi"><div class="label">Virtual plants</div><div class="value" id="k2">0</div>
        <div class="hint">${k.treatments} treatments × 3 planting batches per crop</div></div>
      <div class="card kpi"><div class="label">Measurements</div><div class="value" id="k3">0</div>
        <div class="hint">daily records, plus ${k.eis_sweeps.toLocaleString()} EIS sweeps</div></div>
      <div class="card kpi accent"><div class="label">Worst harvest loss (famine)</div><div class="value" id="k4">0</div>
        <div class="hint">compared with well-fed plants</div></div>
      <div class="card kpi"><div class="label">Stress predicted 3 days ahead</div>
        <div class="value"><span id="k5">0</span><span style="font-size:18px;font-weight:600;color:var(--ink-3)"> – ${fmt(k.f1_max)}</span></div>
        <div class="hint">F1 score on unseen plants (1.0 = perfect)</div></div>
    </div>

    <div class="grid g-3 mt">
      <div class="card span-2">
        <div class="card-head">
          <div><h2>How much harvest each stress costs</h2>
            <p class="sub">Fresh weight at harvest as a % of well-fed (control) plants. The line at 100% is control.</p></div>
          <div class="legend" id="hvLegend"></div>
        </div>
        <div class="chart" id="harvestChart" role="img" aria-label="Grouped bar chart of harvest as percent of control by crop and stress"></div>
        <details class="datatable"><summary>Show as table</summary><div class="table-wrap" id="harvestTable"></div></details>
      </div>
      <div class="card">
        <h2>How reliably stress is predicted</h2>
        <p class="sub">F1 score per crop, on plants the model never saw.</p>
        <div class="chart" id="f1Chart" role="img" aria-label="Bar chart of stress prediction F1 score by crop"></div>
      </div>
    </div>

    <h2 class="section-title">What we learned</h2>
    <div class="grid g-4">
      ${data.findings.map((f, i) => `
        <div class="card finding"><div class="num">${i + 1}</div>
          <div><h3>${esc(f.title)}</h3><p>${esc(f.text)}</p></div></div>`).join("")}
    </div>`;

  countUp(root.querySelector("#k1"), k.crops);
  countUp(root.querySelector("#k2"), k.plants);
  countUp(root.querySelector("#k3"), k.daily_records);
  countUp(root.querySelector("#k4"), k.worst_famine_loss_pct, { suffix: "%" });
  countUp(root.querySelector("#k5"), k.f1_min, { decimals: 2 });

  harvestChart(root, data.harvest);
  f1Chart(root, data.detection);
}

function harvestChart(root, rows) {
  const crops = [...new Set(rows.map((r) => r.species))];
  const stresses = TREATMENTS.filter((t) => t !== "control");
  const val = (c, t) => rows.find((r) => r.species === c && r.treatment === t);
  const b = base();

  root.querySelector("#hvLegend").innerHTML = stresses
    .map((t) => `<span title="${esc(TREATMENT_INFO[t])}"><i style="background:${tcolor(t)}"></i>${TREATMENT_LABEL[t]}</span>`).join("");

  const chart = makeChart(root.querySelector("#harvestChart"));
  chart.setOption({
    ...b,
    legend: { show: false },
    grid: { left: 8, right: 16, top: 26, bottom: 4, containLabel: true },
    tooltip: {
      ...b.tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,127,127,.08)" } },
      formatter: (ps) => {
        const crop = crops[ps[0].dataIndex];
        const ctrl = val(crop, "control").mean_fresh_weight_g;
        return `<b>${cropLabel(crop)}</b> · control ${ctrl.toFixed(0)} g<br>` + ps.map((p) => {
          const r = val(crop, stresses[p.seriesIndex]);
          return `${p.marker}${TREATMENT_LABEL[stresses[p.seriesIndex]]}: <b>${p.value.toFixed(0)}%</b> (${r.mean_fresh_weight_g.toFixed(0)} g)`;
        }).join("<br>");
      },
    },
    xAxis: axis({ type: "category", data: crops.map(cropLabel), axisLabel: { color: css("--ink-2"), fontSize: 12.5 } }),
    yAxis: axis({ type: "value", min: 0, max: 100, interval: 25, axisLabel: { formatter: "{value}%", color: css("--ink-3") } }),
    series: stresses.map((t, i) => ({
      name: TREATMENT_LABEL[t], type: "bar", barGap: "12%", barCategoryGap: "28%",
      data: crops.map((c) => val(c, t).pct_of_control),
      itemStyle: { color: tcolor(t), borderRadius: [4, 4, 0, 0] },
      emphasis: { focus: "series" },
      animationDelay: (idx) => idx * 60 + i * 120,
      markLine: i === 0 ? {
        silent: true, symbol: "none",
        lineStyle: { color: tcolor("control"), width: 2, type: "solid" },
        label: { formatter: "Control = 100%", color: css("--ink-2"), position: "insideEndTop", fontSize: 11.5 },
        data: [{ yAxis: 100 }],
      } : undefined,
    })),
  });

  root.querySelector("#harvestTable").innerHTML = `<table class="data"><thead><tr><th>Crop</th>
    ${TREATMENTS.map((t) => `<th>${TREATMENT_LABEL[t]} (g)</th>`).join("")}</tr></thead><tbody>
    ${crops.map((c) => `<tr><td>${cropLabel(c)}</td>${TREATMENTS.map((t) => {
      const r = val(c, t); return `<td>${r.mean_fresh_weight_g.toFixed(0)}${t === "control" ? "" : ` <small>(${r.pct_of_control.toFixed(0)}%)</small>`}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
}

function f1Chart(root, det) {
  const rows = [...det].sort((a, b) => a.f1 - b.f1);
  const b = base();
  const color = css("--brand");
  const chart = makeChart(root.querySelector("#f1Chart"));
  chart.setOption({
    ...b,
    grid: { left: 8, right: 44, top: 6, bottom: 4, containLabel: true },
    tooltip: {
      ...b.tooltip, trigger: "item",
      formatter: (p) => {
        const r = rows[p.dataIndex];
        return `<b>${cropLabel(r.species)}</b><br>F1: <b>${fmt(r.f1)}</b><br>Precision: ${fmt(r.precision)} · Recall: ${fmt(r.recall)}<br>AUC: ${fmt(r.auc)}`;
      },
    },
    xAxis: axis({ type: "value", min: 0, max: 1, interval: 0.25 }),
    yAxis: axis({ type: "category", data: rows.map((r) => cropLabel(r.species)), axisLabel: { color: css("--ink-2"), fontSize: 12.5 } }),
    series: [{
      type: "bar", data: rows.map((r) => r.f1), barWidth: "52%",
      itemStyle: { color, borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: "right", color: css("--ink-2"), formatter: (p) => p.value.toFixed(2), fontSize: 12 },
      animationDelay: (i) => i * 90,
    }],
  });
}
