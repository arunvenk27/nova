"""
Analytics layer: turns database rows into the numbers the dashboard shows.
Uses the project's own analysis code (pipeline/plantsense_v3.py) for the EIS fitting,
so the website and the research pipeline always agree.
"""
import sys
from pathlib import Path

import numpy as np

from . import db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import plantsense_v3 as ps  # noqa: E402

TREATMENTS = ["control", "drought", "salinity", "heat", "famine"]
HORIZON = 3
GROWTH_TASK, STRESS_TASK = "growth (per day)", "stress within 3 d"

def _findings(famine_lo, famine_hi, f1_lo, f1_hi) -> list[dict]:
    """Key findings; the numbers are filled in from the database so the text never goes stale."""
    return [
        {"title": "Famine is the most damaging stress",
         "text": f"Cutting nutrients and water reduced harvest to {famine_lo:.0f}-{famine_hi:.0f} % of control in every "
                 "crop - far more than drought, salinity or heat alone."},
        {"title": "Stress can be predicted 3 days ahead",
         "text": f"On plants the models had never seen, stress was predicted with F1 {f1_lo:.2f}-{f1_hi:.2f} for all six crops."},
        {"title": "The leaf's electrical signal reacts to stress",
         "text": "Extracellular resistance (Re) rises as soil dries and falls as salt rises, in all six crops - "
                 "the pattern the pipeline was built to detect."},
        {"title": "...but cheap sensors already capture it",
         "text": "Adding EIS did not make forecasts more accurate here. A real trial must test EIS where "
                 "soil and air sensors are blind."},
    ]


def _num(x):
    """JSON-safe number (NaN/inf -> None)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 4)


# ------------------------------------------------------------------ overview
def overview() -> dict:
    harvest = db.harvest()
    ctrl = {r["species"]: r["mean_fresh_weight_g"] for r in harvest if r["treatment"] == "control"}
    pct = [dict(species=r["species"], treatment=r["treatment"], mean_fresh_weight_g=r["mean_fresh_weight_g"],
                pct_of_control=_num(100 * r["mean_fresh_weight_g"] / ctrl[r["species"]]))
           for r in harvest]
    famine = [p["pct_of_control"] for p in pct if p["treatment"] == "famine"]
    det = detection_by_crop()
    f1 = [d["f1"] for d in det if d["f1"] is not None]
    c = db.counts()
    kpis = dict(**c,
                worst_famine_loss_pct=_num(100 - min(famine)),
                f1_min=_num(min(f1)), f1_max=_num(max(f1)))
    return dict(kpis=kpis, harvest=pct, detection=det, data_source="SIMULATED",
                findings=_findings(min(famine), max(famine), min(f1), max(f1)))


def detection_by_crop() -> list[dict]:
    out = []
    for sp in db.species_list():
        rows = db.model_results(sp["species"])
        r = next((x for x in rows if x["task"] == STRESS_TASK and x["model"] == "RandomForest"
                  and x["inputs"] == "A"), None)
        out.append(dict(species=sp["species"], f1=_num(r and r["F1"]), auc=_num(r and r["ROC_AUC"]),
                        precision=_num(r and r["Precision"]), recall=_num(r and r["Recall"])))
    return out


def models_summary() -> list[dict]:
    """Per crop: growth-forecast error (baseline vs without EIS vs with EIS) and stress-detection F1."""
    out = []
    for sp in db.species_list():
        rows = db.model_results(sp["species"])
        get = lambda task, model, inp, col: next(
            (_num(x[col]) for x in rows if x["task"] == task and x["model"] == model and x["inputs"] == inp), None)
        out.append(dict(
            species=sp["species"], growth_unit=sp["recommended_growth_col"],
            baseline_mae=get(GROWTH_TASK, "Persistence baseline", "-", "MAE"),
            mae_without_eis=get(GROWTH_TASK, "RandomForest", "A", "MAE"),
            mae_with_eis=get(GROWTH_TASK, "RandomForest", "B", "MAE"),
            interval_coverage=get(GROWTH_TASK, "RandomForest", "A", "PI90_coverage"),
            f1_without_eis=get(STRESS_TASK, "RandomForest", "A", "F1"),
            f1_with_eis=get(STRESS_TASK, "RandomForest", "B", "F1"),
            auc_without_eis=get(STRESS_TASK, "RandomForest", "A", "ROC_AUC"),
            auc_with_eis=get(STRESS_TASK, "RandomForest", "B", "ROC_AUC")))
    return out


# ------------------------------------------------------------------ plant timeline
def timeline(species: str, plant_id: str) -> dict | None:
    p = db.plant(species, plant_id)
    if not p:
        return None
    rows = db.daily(species, plant_id)
    gcol = p["recommended_growth_col"]
    by_day = {r["day"]: r for r in rows}
    d0, d1 = min(by_day), max(by_day)
    days = list(range(d0, d1 + 1))
    size = [by_day.get(d, {}).get(gcol) for d in days]
    rate = []
    for i, d in enumerate(days):           # growth rate over the previous 3 days (gap -> None)
        a, b = size[i], size[i - HORIZON] if i >= HORIZON else None
        rate.append(_num((a - b) / HORIZON) if a is not None and b is not None else None)
    eis = {r["day"]: r for r in db.eis_features(species, plant_id)}
    col = lambda k: [_num(by_day[d][k]) if d in by_day else None for d in days]
    return dict(
        plant=dict(plant_id=p["plant_id"], species=species, treatment=p["treatment"], cohort=p["cohort"],
                   transplant_day=p["transplant_day"], treatment_detail=p["treatment_detail"],
                   harvest_fresh_weight_g=p["harvest_fresh_weight_g"], growth_col=gcol),
        days=days, plant_age=[d - d0 for d in days],
        soil_moisture=col("soil_moisture_pct"), air_temp=col("air_temp_c"), ec=col("ec_ds_m"),
        spad=col("spad"), wilting=col("wilting_score"), size=[_num(s) for s in size], growth_rate=rate,
        stress=[int(by_day[d]["stress_truth"]) if d in by_day and by_day[d]["stress_truth"] is not None else None
                for d in days],
        re_rel=[_num(eis[d]["Re_rel"]) if d in eis else None for d in days],
        ri_rel=[_num(eis[d]["Ri_rel"]) if d in eis else None for d in days],
        data_source="SIMULATED")


# ------------------------------------------------------------------ EIS spectra + Cole fit
def _fit_sweep(species, plant_id, day):
    rows = db.spectrum(species, plant_id, day)
    if len(rows) < 5:
        return None
    f = np.array([r["freq_hz"] for r in rows])
    zr = np.array([r["z_real"] for r in rows])
    zi = np.array([r["z_imag"] for r in rows])
    fit = ps.fit_cole(f, zr, zi)
    fdense = np.logspace(np.log10(f.min()), np.log10(f.max()), 120)
    m = ps.fit_curve(fit, fdense)
    return dict(day=day, sweep_temp_c=_num(rows[0]["sweep_temp_c"]),
                points=[[_num(a), _num(-b), _num(c)] for a, b, c in zip(zr, zi, f)],   # Z', -Z'', freq
                fit_curve=[[_num(a), _num(-b)] for a, b in zip(m.real, m.imag)],
                Re=_num(fit["Re"]), Ri=_num(fit["Ri"]), fc_hz=_num(fit["fc_hz"]), alpha=_num(fit["alpha"]),
                rel_rmse=_num(fit["rel_rmse"]))


def eis(species: str, plant_id: str, day: int | None) -> dict | None:
    days = db.sweep_days(species, plant_id)
    if not days:
        return None
    day = day if day in days else days[-1] if day is None else min(days, key=lambda d: abs(d - day))
    base = _fit_sweep(species, plant_id, days[0])
    sel = _fit_sweep(species, plant_id, day)
    feats = {r["day"]: r for r in db.eis_features(species, plant_id)}
    if sel and day in feats:
        sel["Re_rel"], sel["Ri_rel"] = _num(feats[day]["Re_rel"]), _num(feats[day]["Ri_rel"])
    return dict(days=days, baseline=base, selected=sel, data_source="SIMULATED")


# ------------------------------------------------------------------ Objective 3 correlations
def correlations(species: str, feature: str = "Re_rel") -> dict:
    rows = [r for r in db.correlations(species) if r["eis_feature"] == feature]
    groups = ["all plants"] + TREATMENTS
    variables = ["soil_moisture_pct", "ec_ds_m", "air_temp_c", "wilting_score", "growth_past3", "growth_next3"]
    cells = []
    for r in rows:
        if r["group"] in groups and r["variable"] in variables:
            cells.append(dict(group=r["group"], variable=r["variable"], rho=_num(r["age_adj_rho"]),
                              lo=_num(r["age_adj_ci95_lo"]), hi=_num(r["age_adj_ci95_hi"]),
                              significant=bool(r["age_adj_ci_excludes_0"])))
    present = {c["group"] for c in cells}
    return dict(feature=feature, groups=[g for g in groups if g in present], variables=variables, cells=cells)


# ------------------------------------------------------------------ stress monitor
def status(species: str | None = None) -> list[dict]:
    rows = db.status_all()
    if species:
        rows = [r for r in rows if r["species"] == species]
    for r in rows:
        p = r["modelled_stress_prob"]
        r["level"] = "unknown" if p is None else "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"
        for k in ("modelled_stress_prob", "observed_growth", "predicted_next3"):
            r[k] = _num(r[k])
    return rows
