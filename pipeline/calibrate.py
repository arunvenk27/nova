#!/usr/bin/env python3
"""
Turn the Phase-1 calibration measurements into numbers PlantSense v3 needs (Phase 3, step 16).

    python calibrate.py --data data --out calib.json
    python plantsense_v3.py ... --calib calib.json

  calib_sensor_repeatability.csv -> 1-sigma of every sensor (SD of repeat readings)
  calib_eis_repeatability.csv    -> 1-sigma of every EIS feature (re-clipped sweeps)
  calib_eis_temperature.csv      -> TEMP_COEF from 1/R0 vs temperature
Derived sigmas: a 3-day rate uses two readings, so sigma_rate = sqrt(2) * sigma / 3.
"""
import argparse
import json

import numpy as np
import pandas as pd

import plantsense_v3 as ps
from plantsense_v3 import HORIZON, fit_cole


def fit_sweeps(df, key):
    rows = []
    for k, g in df.groupby(key):
        g = g.sort_values("freq_hz")
        r = fit_cole(g.freq_hz.to_numpy(), g.z_real.to_numpy(), g.z_imag.to_numpy())
        r[key] = k
        r["sweep_temp_c"] = g.sweep_temp_c.iloc[0]
        r["phase_min_deg"] = np.degrees(np.arctan2(g.z_imag, g.z_real)).min()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--growth-col", default="rosette_diam_cm")
    ap.add_argument("--electrode-cpe", action="store_true", help="2-electrode data: fit electrode polarisation")
    a = ap.parse_args()
    ps.ELECTRODE_CPE = a.electrode_cpe

    rep = pd.read_csv(f"{a.data}/calib_sensor_repeatability.csv")
    stats = rep.groupby("sensor").agg(reference=("reference_value", "first"), mean=("reading", "mean"),
                                      sd=("reading", "std"), n=("reading", "size"))
    stats["bias"] = stats["mean"] - stats["reference"]
    print("SENSOR REPEATABILITY (vs reference)")
    print(stats.round(4).to_string())
    sig = {}
    for s, r in stats.iterrows():
        sig[s] = ("rel", r.sd / r.reference) if s == "light_par" else ("abs", r.sd)
    gsd = stats.loc[a.growth_col, "sd"]
    sig["growth_past3"] = ("abs", np.sqrt(2) * gsd / HORIZON)
    if "leaf_count" in stats.index:
        sig["leaf_rate_past3"] = ("abs", np.sqrt(2) * stats.loc["leaf_count", "sd"] / HORIZON)
    sig["age_days"] = ("abs", 0.0)
    for derived, src in [("sm_deficit", "soil_moisture_pct"), ("sm_excess", "soil_moisture_pct"),
                         ("heat_excess", "air_temp_c"), ("ec_excess", "ec_ds_m"), ("ec_deficit", "ec_ds_m")]:
        if src in sig:
            sig[derived] = sig[src]

    # EIS repeatability (features relative to the median of the repeats)
    er = fit_sweeps(pd.read_csv(f"{a.data}/calib_eis_repeatability.csv"), "sweep")
    sig["Re_rel"] = ("abs", float((er.Re / er.Re.median()).std()))
    sig["Ri_rel"] = ("abs", float((er.Ri / er.Ri.median()).std()))
    sig["log10_fc"] = ("abs", float(np.log10(er.fc_hz).std()))
    sig["alpha"] = ("abs", float(er.alpha.std()))
    sig["phase_min_deg"] = ("abs", float(er.phase_min_deg.std()))
    print(f"\nEIS REPEATABILITY ({len(er)} re-clipped sweeps): median fit rel-RMSE {er.rel_rmse.median():.4f}")
    for k in ["Re_rel", "Ri_rel", "log10_fc", "alpha", "phase_min_deg"]:
        print(f"  {k:14s} sigma = {sig[k][1]:.4f}")

    # temperature coefficient: R_T = R25/(1 + a(T-25))  ->  1/R_T = (1/R25) + (a/R25)(T-25)
    tc = pd.read_csv(f"{a.data}/calib_eis_temperature.csv")
    tc["sweep_id"] = tc.set_temp_c.astype(str) + "_" + tc.repeat.astype(str)
    ft = fit_sweeps(tc, "sweep_id")
    out = {}
    for name in ("Re", "Ri"):
        slope, icpt = np.polyfit(ft.sweep_temp_c - 25, 1 / ft[name], 1)
        out[name] = slope / icpt
    temp_coef = float(np.mean(list(out.values())))
    print(f"\nEIS TEMPERATURE COEFFICIENT: from Re {out['Re']:.4f}/degC, from Ri {out['Ri']:.4f}/degC "
          f"-> TEMP_COEF = {temp_coef:.4f}/degC")

    calib = dict(SENSOR_SIGMA={k: [v[0], round(float(v[1]), 6)] for k, v in sig.items()},
                 TEMP_COEF=round(temp_coef, 5),
                 source="SIMULATED calibration data from generate_lettuce_trial.py")
    with open(a.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
