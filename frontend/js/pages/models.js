// MODEL RESULTS - the honest test (Objective 4): does adding EIS make predictions better?
import { api } from "../api.js";
import { axis, base, cropLabel, css, fmt, makeChart, setLoading, showError } from "../ui.js";

export async function render(root) {
  setLoading(root);
  let rows;
  try { rows = await api.models(); } catch (e) { return showError(root, e); }

  const avg = (k) => rows.reduce((a, r) => a + r[k], 0) / rows.length;
  const gain = 100 * (1 - avg("mae_without_eis") / avg("baseline_mae"));
  const eisDiff = 100 * (avg("mae_with_eis") / avg("mae_without_eis") - 1);
  const f1Diff = avg("f1_with_eis") - avg("f1_without_eis");

  root.innerHTML = `
    <div class="page-head">
      <div><h1>Model Results</h1>
        <p>Each model was trained on some plants and tested on <b>different</b> plants it had never seen.
           We compare a model using sensors + growth only with the same model plus EIS.</p></div>
    </div>

    <div class="grid g-3">
      <div class="card kpi accent"><div class="label">Better than a simple guess</div>
        <div class="value">${gain.toFixed(0)}%</div><div class="hint">less growth-forecast error than "same as the last 3 days"</div></div>
      <div class="card kpi"><div class="label">Effect of adding EIS on growth error</div>
        <div class="value">${eisDiff >= 0 ? "+" : ""}${eisDiff.toFixed(1)}%</div><div class="hint">average across crops (≈ no change)</div></div>
      <div class="card kpi"><div class="label">Effect of adding EIS on stress F1</div>
        <div class="value">${f1Diff >= 0 ? "+" : ""}${f1Diff.toFixed(2)}</div><div class="hint">average across crops (≈ no change)</div></div>
    </div>

    <div class="grid g-2 mt">
      <div class="card">
        <h2>Growth forecast error <span style="font-weight:400;color:var(--ink-3)">(lower is better)</span></h2>
        <p class="sub">Average error predicting the next 3 days of growth, in cm per day.</p>
        <div class="legend" id="lg1"></div>
        <div class="chart" id="mae" role="img" aria-label="Grouped bar chart of growth forecast error per crop"></div>
      </div>
      <div class="card">
        <h2>Stress detection <span style="font-weight:400;color:var(--ink-3)">(higher is better)</span></h2>
        <p class="sub">F1 score for "will this plant be stressed within 3 days?"</p>
        <div class="legend" id="lg2"></div>
        <div class="chart" id="f1" role="img" aria-label="Grouped bar chart of stress detection F1 per crop"></div>
      </div>
    </div>

    <div class="card mt">
      <h2>Why EIS didn't help here - and what it means</h2>
      <ul class="tight">
        <li>The soil-moisture, temperature and EC sensors already "see" every stress in this experiment, so EIS mostly repeats the same information.</li>
        <li>EIS <b>does</b> react to stress (see the EIS Lab) - it just isn't <i>extra</i> information when the sensors already show the cause.</li>
        <li>Where EIS could add value: stresses no sensor measures (disease, root damage), or catching the plant's response before growth slows. A real trial should test exactly that.</li>
        <li>The 90% prediction intervals contained the true value ${fmt(100 * avg("interval_coverage"), 0)}% of the time on average - the uncertainty estimates are honest.</li>
      </ul>
      <details class="datatable"><summary>Show all numbers</summary><div class="table-wrap">
        <table class="data"><thead><tr><th>Crop</th><th>Baseline error</th><th>Without EIS</th><th>With EIS</th><th>Interval coverage</th>
          <th>F1 without EIS</th><th>F1 with EIS</th><th>AUC without</th><th>AUC with</th></tr></thead><tbody>
          ${rows.map((r) => `<tr><td>${cropLabel(r.species)}</td><td>${fmt(r.baseline_mae, 3)}</td><td>${fmt(r.mae_without_eis, 3)}</td>
            <td>${fmt(r.mae_with_eis, 3)}</td><td>${fmt(100 * r.interval_coverage, 0)}%</td><td>${fmt(r.f1_without_eis)}</td>
            <td>${fmt(r.f1_with_eis)}</td><td>${fmt(r.auc_without_eis)}</td><td>${fmt(r.auc_with_eis)}</td></tr>`).join("")}
        </tbody></table></div></details>
    </div>`;

  const crops = rows.map((r) => cropLabel(r.species));
  const C = { base: css("--m-baseline"), wo: css("--m-without"), w: css("--m-with") };
  const legend = (items) => items.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("");
  root.querySelector("#lg1").innerHTML = legend([["Simple guess (last 3 days)", C.base], ["Sensors + growth", C.wo], ["Sensors + growth + EIS", C.w]]);
  root.querySelector("#lg2").innerHTML = legend([["Sensors + growth", C.wo], ["Sensors + growth + EIS", C.w]]);

  const bar = (name, data, color, i) => ({
    name, type: "bar", data, barGap: "10%", barCategoryGap: "30%",
    itemStyle: { color, borderRadius: [4, 4, 0, 0] }, emphasis: { focus: "series" },
    animationDelay: (idx) => idx * 70 + i * 140,
  });
  const opt = (series, yAxisExtra, digits) => {
    const b = base();
    return {
      ...b,
      legend: { show: false },
      grid: { left: 8, right: 12, top: 12, bottom: 4, containLabel: true },
      tooltip: { ...b.tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,127,127,.08)" } },
        valueFormatter: (v) => (v == null ? "-" : v.toFixed(digits)) },
      xAxis: axis({ type: "category", data: crops, axisLabel: { color: css("--ink-2") } }),
      yAxis: axis({ type: "value", ...yAxisExtra }),
      series,
    };
  };
  makeChart(root.querySelector("#mae")).setOption(opt([
    bar("Simple guess (last 3 days)", rows.map((r) => r.baseline_mae), C.base, 0),
    bar("Sensors + growth", rows.map((r) => r.mae_without_eis), C.wo, 1),
    bar("Sensors + growth + EIS", rows.map((r) => r.mae_with_eis), C.w, 2),
  ], {}, 3));
  makeChart(root.querySelector("#f1")).setOption(opt([
    bar("Sensors + growth", rows.map((r) => r.f1_without_eis), C.wo, 0),
    bar("Sensors + growth + EIS", rows.map((r) => r.f1_with_eis), C.w, 1),
  ], { min: 0, max: 1 }, 2));
}
