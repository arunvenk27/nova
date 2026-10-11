// PLANT EXPLORER - follow one plant through its life: what the environment did,
// how the leaf's electrical signal (EIS) responded, and how growth changed.
import { api } from "../api.js";
import {
  TREATMENTS, TREATMENT_LABEL, GROWTH_LABEL, axis, base, cropLabel, css, esc, fmt, go, hexAlpha,
  levelOf, makeChart, pill, prettyCategory, routeParams, setLoading, showError, tcolor,
} from "../ui.js";

const ENV_VARS = {
  soil: { key: "soil_moisture", label: "Soil moisture", unit: "%", threshold: () => ({ v: 20, text: "dry below 20%" }) },
  temp: { key: "air_temp", label: "Air temperature", unit: "°C", threshold: (sp) => ({ v: sp.temp_high_flag_c, text: `heat above ${sp.temp_high_flag_c} °C` }) },
  ec: { key: "ec", label: "Nutrient / salt (EC)", unit: "dS/m", threshold: (sp) => ({ v: sp.ec_high_flag, text: `salty above ${sp.ec_high_flag}` }) },
  spad: { key: "spad", label: "Leaf greenness (SPAD)", unit: "", threshold: (sp) => ({ v: +(0.8 * sp.healthy_spad).toFixed(1), text: "pale below 80% of healthy" }) },
};
let envVar = "soil";

export async function render(root) {
  setLoading(root);
  let species;
  try { species = await api.species(); } catch (e) { return showError(root, e); }
  const params = routeParams();
  const crop = species.some((s) => s.species === params.crop) ? params.crop : "lettuce";
  const sp = species.find((s) => s.species === crop);
  let plants;
  try { plants = await api.plants(crop); } catch (e) { return showError(root, e); }
  const plantId = plants.some((p) => p.plant_id === params.plant) ? params.plant
    : (plants.find((p) => p.treatment === "drought") || plants[0]).plant_id;
  const p = plants.find((x) => x.plant_id === plantId);

  root.innerHTML = `
    <div class="page-head">
      <div><h1>Plant Explorer</h1>
        <p>Pick a crop and a plant to follow it day by day. Red bands mark days when the plant was under stress.</p></div>
      <div class="controls">
        <label class="field">Crop
          <select id="cropSel">${species.map((s) => `<option value="${s.species}" ${s.species === crop ? "selected" : ""}>${cropLabel(s.species)}</option>`).join("")}</select>
        </label>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h2>Choose a plant</h2>
        <div class="legend">${TREATMENTS.map((t) => `<span><i style="background:${tcolor(t)};border-radius:50%"></i>${TREATMENT_LABEL[t]}</span>`).join("")}</div>
      </div>
      <div class="chips mt" style="margin-top:10px">
        ${plants.map((x) => `<button class="chip ${x.plant_id === plantId ? "on" : ""}" data-plant="${esc(x.plant_id)}"
            title="${TREATMENT_LABEL[x.treatment]}, batch ${x.cohort}"><i style="background:${tcolor(x.treatment)}"></i>${esc(x.plant_id)}</button>`).join("")}
      </div>
    </div>

    <div class="grid mt" style="grid-template-columns: 300px minmax(0,1fr)" id="exGrid">
      <div class="card">
        <h2>${esc(p.plant_id)}</h2>
        <p class="sub">${cropLabel(crop)} · <i>${esc(sp.latin_name)}</i></p>
        <div style="margin:4px 0 14px">${pill(levelOf(p.modelled_stress_prob))}</div>
        <dl class="facts">
          <dt>Treatment</dt><dd><span style="color:${tcolor(p.treatment)}">●</span> ${TREATMENT_LABEL[p.treatment]}</dd>
          <dt>What happened</dt><dd style="font-weight:400;color:var(--ink-2)">${esc(p.treatment_detail)}</dd>
          <dt>Planting batch</dt><dd>${p.cohort}</dd>
          <dt>Harvest weight</dt><dd>${fmt(p.harvest_fresh_weight_g, 0)} g</dd>
          <dt>Stress chance</dt><dd>${p.modelled_stress_prob == null ? "-" : Math.round(100 * p.modelled_stress_prob) + "%"} <small style="color:var(--ink-3)">(last day)</small></dd>
          <dt>Diagnosis</dt><dd style="font-weight:400">${esc(prettyCategory(p.category))}</dd>
        </dl>
        <p class="sub" style="margin-top:16px;font-size:12.5px">EIS lines show the leaf's resistance relative to the plant's own first week
          (1.0 = unchanged). Rising <b>Re</b> usually means the leaf is losing water.</p>
        <a href="#/eis?crop=${crop}&plant=${encodeURIComponent(p.plant_id)}" style="font-size:13.5px">See this plant's EIS curves &rarr;</a>
      </div>
      <div class="card">
        <div class="card-head">
          <div><h2>Life of ${esc(p.plant_id)}</h2><p class="sub">Drag the handles at the bottom to zoom into a period.</p></div>
          <div class="seg" id="envSeg">${Object.entries(ENV_VARS).map(([k, v]) =>
            `<button data-v="${k}" class="${k === envVar ? "on" : ""}">${v.label.split(" (")[0]}</button>`).join("")}</div>
        </div>
        <div class="chart tall" id="tlChart" role="img" aria-label="Three stacked time charts: environment, EIS signal and growth rate"></div>
      </div>
    </div>`;

  if (matchMedia("(max-width: 1100px)").matches) root.querySelector("#exGrid").style.gridTemplateColumns = "1fr";

  root.querySelector("#cropSel").addEventListener("change", (e) => go("explorer", { crop: e.target.value }));
  root.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => go("explorer", { crop, plant: b.dataset.plant })));

  let tl;
  try { tl = await api.timeline(crop, plantId); } catch (e) { return showError(root.querySelector("#tlChart"), e); }
  const chart = makeChart(root.querySelector("#tlChart"));
  const draw = () => chart.setOption(timelineOption(tl, sp), true);
  draw();
  root.querySelectorAll("#envSeg button").forEach((b) => b.addEventListener("click", () => {
    envVar = b.dataset.v;
    root.querySelectorAll("#envSeg button").forEach((x) => x.classList.toggle("on", x === b));
    draw();
  }));
}

function stressBands(tl) {
  const bands = [];
  let start = null;
  tl.stress.forEach((s, i) => {
    if (s === 1 && start === null) start = i;
    if ((s !== 1 || i === tl.stress.length - 1) && start !== null) {
      const end = s === 1 ? i : i - 1;
      bands.push([{ xAxis: String(tl.plant_age[start]) }, { xAxis: String(tl.plant_age[end]) }]);
      start = null;
    }
  });
  return bands;
}

function timelineOption(tl, sp) {
  const b = base();
  const v = ENV_VARS[envVar];
  const thr = v.threshold(sp);
  const x = tl.plant_age.map(String);
  const band = { silent: true, itemStyle: { color: hexAlpha(css("--critical"), isDarkish() ? 0.18 : 0.10) }, data: stressBands(tl) };
  const tColor = tcolor(tl.plant.treatment);
  const envColor = css("--seq-2");
  const reColor = css("--eis-re"), riColor = css("--eis-ri");
  const gUnit = GROWTH_LABEL[tl.plant.growth_col] || tl.plant.growth_col;
  const titles = [`${v.label}${v.unit ? " (" + v.unit + ")" : ""}`, "Leaf electrical signal (EIS, relative to first week)", `Growth rate (${gUnit}, cm per day)`];
  const tops = [34, 222, 410], h = 126;
  const xa = (i) => axis({ type: "category", data: x, gridIndex: i, boundaryGap: false,
    axisLabel: { show: i === 2, color: css("--ink-3") }, name: i === 2 ? "plant age (days)" : "", nameLocation: "middle", nameGap: 26 });

  return {
    ...b,
    legend: { ...b.legend, data: ["Re (extracellular)", "Ri (intracellular)"], top: 186, right: 10 },
    title: titles.map((t, i) => ({ text: t, left: 4, top: tops[i] - 28, textStyle: { fontSize: 13, fontWeight: 600, color: css("--ink-2") } })),
    grid: tops.map((t) => ({ left: 48, right: 18, top: t, height: h })),
    axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: css("--ink-3") } },
    tooltip: {
      ...b.tooltip, trigger: "axis",
      formatter: (ps) => {
        const i = ps[0].dataIndex;
        const st = tl.stress[i] === 1 ? ' · <b style="color:var(--critical-ink)">stressed</b>' : "";
        const lines = [
          `${v.label}: <b>${fmt(tl[v.key][i], 1)}${v.unit ? " " + v.unit : ""}</b>`,
          `Re: <b>${fmt(tl.re_rel[i])}</b> · Ri: <b>${fmt(tl.ri_rel[i])}</b>`,
          `Growth: <b>${fmt(tl.growth_rate[i])}</b> cm/day`,
        ];
        return `<b>Age ${tl.plant_age[i]} d</b> (day ${tl.days[i]})${st}<br>${lines.join("<br>")}`;
      },
    },
    xAxis: [xa(0), xa(1), xa(2)],
    yAxis: [
      axis({ gridIndex: 0, scale: true, splitNumber: 3 }),
      axis({ gridIndex: 1, scale: true, splitNumber: 3 }),
      axis({ gridIndex: 2, splitNumber: 3 }),
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2] },
      { type: "slider", xAxisIndex: [0, 1, 2], bottom: 4, height: 16, borderColor: css("--border"),
        fillerColor: hexAlpha(css("--brand"), 0.12), handleStyle: { color: css("--brand") },
        textStyle: { color: css("--ink-3") }, dataBackground: { lineStyle: { color: css("--axis") }, areaStyle: { color: css("--grid") } } },
    ],
    series: [
      { type: "line", name: v.label, xAxisIndex: 0, yAxisIndex: 0, data: tl[v.key], showSymbol: false, connectNulls: true,
        lineStyle: { width: 2, color: envColor }, itemStyle: { color: envColor },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: hexAlpha(envColor, 0.25) }, { offset: 1, color: hexAlpha(envColor, 0) }]) },
        markArea: band,
        markLine: thr.v == null ? undefined : { silent: true, symbol: "none", data: [{ yAxis: thr.v }],
          lineStyle: { color: css("--critical"), width: 1.5, type: "solid", opacity: .7 },
          label: { formatter: thr.text, color: css("--ink-2"), fontSize: 11, position: "insideEndTop" } } },
      { type: "line", name: "Re (extracellular)", xAxisIndex: 1, yAxisIndex: 1, data: tl.re_rel, showSymbol: false,
        connectNulls: true, lineStyle: { width: 2, color: reColor }, itemStyle: { color: reColor }, markArea: band,
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 1 }], lineStyle: { color: css("--axis"), width: 1, type: "solid" }, label: { show: false } } },
      { type: "line", name: "Ri (intracellular)", xAxisIndex: 1, yAxisIndex: 1, data: tl.ri_rel, showSymbol: false,
        connectNulls: true, lineStyle: { width: 2, color: riColor }, itemStyle: { color: riColor } },
      { type: "bar", name: "Growth rate", xAxisIndex: 2, yAxisIndex: 2, data: tl.growth_rate, barWidth: "62%",
        itemStyle: { color: tColor, borderRadius: [3, 3, 0, 0] }, markArea: band,
        animationDelay: (i) => i * 12 },
    ],
  };
}

const isDarkish = () => css("color-scheme") === "dark";
