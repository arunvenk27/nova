#!/usr/bin/env python3
"""
Generate SIMULATED PlantSense trials for six greenhouse crops, following the same Phase-1
protocol as generate_lettuce_trial.py, with a fifth "famine" treatment.

    python generate_species_trials.py                    # all species -> data_species/<species>/
    python generate_species_trials.py --species tomato   # one species

Every number is a [BIOLOGICAL/ENGINEERING ASSUMPTION] unless marked [CITED]. NOT measured data.

TREATMENTS (5 per species x 3 staggered cohorts = 15 plants per species)
  control  : normal nutrient solution, re-watered to 36 % VWC whenever VWC < 28 %
  drought  : irrigation withheld at two age windows, re-watered next day
  salinity : nutrient EC ramped +0.3 dS/m/day from age 10 to the species' salinity-test level
  heat     : chamber B raised to the species' heat-test temperature for 4 days every 13 days
  famine   : from age 10, plain water instead of nutrient solution (EC ~0.3 dS/m) AND
             deficit irrigation (re-watered to 24 % only when VWC < 16 %)

NEW MEASUREMENT: SPAD (chlorophyll meter, 3 readings on the youngest fully expanded leaf,
averaged) every day at 09:00 - the observable sign of nutrient starvation.

LABELS (fixed before the trial, independent of the soil-moisture sensor)
  drought  : pot weighing shows VWC < 20 % during withholding, + 1 day after rewatering
  salinity : calibrated EC meter > species ec_high
  heat     : heat days + 1 day
  famine   : SPAD < 80 % of the species' healthy SPAD, OR pot VWC < 20 %

SALT TOLERANCE [CITED: FAO Irrigation & Drainage Paper 61, Annex 1, Table A1.1 (Maas-Hoffman)]
  lettuce 1.3 dS/m, 13 %/dS/m | spinach 2.0, 7.6 | tomato 2.5, 9.9 | cabbage 1.8, 9.7
  kale and pak choi use the cabbage values (kale is only rated MS* in FAO; pak choi not listed);
  basil is not listed -> assumed 1.5, 10. FAO thresholds are soil-extract ECe; the nutrient-
  solution growth threshold is assumed to be ECe threshold + 0.7 dS/m (matches lettuce: growth
  unaffected up to ~2.0 dS/m in Purdue HO-309-W).
"""
import argparse
import os

import numpy as np
import pandas as pd

FREQS = np.logspace(3, 6, 31)
TEMP_COEF_TRUE = 0.021
EIS_NOISE = 0.004
BAD_SWEEP_P, MISSING_SWEEP_P, MISSING_ENV_P = 0.02, 0.03, 0.02
POT_DRY_G = 350.0
TRUE_SIGMA = dict(air_temp_c=0.3, soil_moisture_pct=1.5, rel_humidity_pct=2.0, light_par_rel=0.04,
                  ec_ds_m=0.04, ph=0.05, rosette_diam_cm=0.35, height_cm=0.3, leaf_count=0.6,
                  pot_weight_g=2.0, spad=1.0)

# ------------------------------------------------------------------ species profiles
# growth: logistic leaf area LA (cm2): LA0, LAmax, r (/day); cardinal temps (base, opt, max) for
# expansion; phyllochron (degC.day per leaf above base); size = how rosette diameter / height follow LA.
SPECIES = {
    "lettuce": dict(
        latin="Lactuca sativa (butterhead)", days=40, room_temp=20, growth_col="rosette_diam_cm",
        LA0=18, LAmax=950, r=0.17, cardinal=(4, 20, 33), phyllo=22, leaves0=4,
        diam=(3.0, 0.33), height=(4, 0.012), fw_per_cm2=0.24, pot_cm3=1000,
        ec_normal=1.4, salt_thr=1.3, salt_slope=0.13, ec_high=2.5, temp_high=27,
        eis=(22e3, 9e3, 1.8e-10, 0.78), spad=30),
    "spinach": dict(
        latin="Spinacia oleracea", days=40, room_temp=18, growth_col="rosette_diam_cm",
        LA0=12, LAmax=600, r=0.16, cardinal=(2, 18, 30), phyllo=25, leaves0=4,
        diam=(2.8, 0.33), height=(3, 0.015), fw_per_cm2=0.30, pot_cm3=1000,
        ec_normal=1.8, salt_thr=2.0, salt_slope=0.076, ec_high=3.2, temp_high=26,
        eis=(15e3, 7e3, 2.2e-10, 0.76), spad=55),
    "pak_choi": dict(
        latin="Brassica rapa subsp. chinensis", days=40, room_temp=20, growth_col="rosette_diam_cm",
        LA0=15, LAmax=900, r=0.18, cardinal=(4, 20, 32), phyllo=30, leaves0=4,
        diam=(2.9, 0.33), height=(5, 0.018), fw_per_cm2=0.28, pot_cm3=1200,
        ec_normal=1.8, salt_thr=1.8, salt_slope=0.097, ec_high=3.0, temp_high=27,
        eis=(18e3, 8e3, 2.0e-10, 0.77), spad=40),
    "kale": dict(
        latin="Brassica oleracea var. sabellica", days=55, room_temp=19, growth_col="rosette_diam_cm",
        LA0=15, LAmax=1500, r=0.14, cardinal=(4, 19, 31), phyllo=40, leaves0=4,
        diam=(3.1, 0.33), height=(6, 0.02), fw_per_cm2=0.20, pot_cm3=2500,
        ec_normal=2.0, salt_thr=1.8, salt_slope=0.097, ec_high=3.0, temp_high=27,
        eis=(30e3, 11e3, 1.6e-10, 0.79), spad=60),
    "basil": dict(
        latin="Ocimum basilicum", days=50, room_temp=25, growth_col="height_cm",
        LA0=10, LAmax=700, r=0.15, cardinal=(10, 27, 38), phyllo=30, leaves0=4,
        diam=(2.2, 0.33), height=(5, 0.035), fw_per_cm2=0.12, pot_cm3=1500,
        ec_normal=1.6, salt_thr=1.5, salt_slope=0.10, ec_high=2.7, temp_high=35,
        eis=(20e3, 8.5e3, 1.9e-10, 0.77), spad=40),
    "tomato": dict(
        latin="Solanum lycopersicum (vegetative stage)", days=50, room_temp=24, growth_col="height_cm",
        LA0=25, LAmax=3000, r=0.12, cardinal=(10, 25, 38), phyllo=45, leaves0=4,
        diam=(2.0, 0.33), height=(8, 0.025), fw_per_cm2=0.15, pot_cm3=2500,
        ec_normal=2.5, salt_thr=2.5, salt_slope=0.099, ec_high=3.7, temp_high=32,
        eis=(25e3, 10e3, 1.7e-10, 0.78), spad=50),
}
TREATMENTS = ["control", "drought", "salinity", "heat", "famine"]


def in_any(t, windows, lag_start=0, lag_end=0):
    return any(a + lag_start <= t <= b + lag_end for a, b in windows)


def ar1(rng, n, sd, rho=0.7):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    return x


def f_temp(T, tb, to, tm):
    if T <= tb or T >= tm:
        return 0.0
    return ((T - tb) / (to - tb)) * ((tm - T) / (tm - to)) ** ((tm - to) / (to - tb))


def cole(f, Re, Ri, Cm, alpha):
    R0, Rinf = Re, Re * Ri / (Re + Ri)
    return Rinf + (R0 - Rinf) / (1 + (1j * 2 * np.pi * f * Cm * (Re + Ri)) ** alpha)


def leaf_impedance(rng, f, Re25, Ri25, Cm, alpha, T, bad=False):
    k = 1 + TEMP_COEF_TRUE * (T - 25.0)
    Z = cole(f, Re25 / k, Ri25 / k, Cm, alpha) * (1 + rng.normal(0, 0.005))
    if bad:
        Rc, Cc = rng.uniform(2e4, 8e4), rng.uniform(5e-9, 2e-8)
        Z = Z + Rc / (1 + 1j * 2 * np.pi * f * Rc * Cc) + 1 / (1j * 2 * np.pi * f * 1e-8)
    return Z + EIS_NOISE * np.abs(Z) * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(2)


def simulate_species(name, seed, cohorts=3, stagger=7, reps=1):
    sp = SPECIES[name]
    rng = np.random.default_rng(seed)
    DAYS = sp["days"]
    ncal = DAYS + stagger * (cohorts - 1)
    d = np.arange(ncal)
    dry = [(10, int(0.5 * DAYS)), (int(0.68 * DAYS), int(0.85 * DAYS))]
    heat_cal = [(s, s + 3) for s in range(18, ncal - 3, 13)]
    reclip = int(0.52 * DAYS)
    heat_T = sp["temp_high"] + 3
    salt_target = sp["ec_high"] + 1.5
    salt_thr_sol = sp["salt_thr"] + 0.7
    Tb, To, Tm = sp["cardinal"]
    room = sp["room_temp"]
    TA = room + 1.0 * np.sin(2 * np.pi * d / 9) + ar1(rng, ncal, 0.6)
    TB = TA + rng.normal(0, 0.2, ncal) + np.array([heat_T - room if in_any(t, heat_cal) else 0.0 for t in d])
    RH_A = np.clip(65 + ar1(rng, ncal, 4), 50, 85)
    RH_B = np.clip(RH_A - 1.5 * (TB - TA), 35, 85)
    PAR = np.clip(300 * (1 + ar1(rng, ncal, 0.07)), 200, 380)
    pot_scale = 1000.0 / sp["pot_cm3"]

    detail = {"control": "normal nutrient solution, re-watered to 36 % VWC whenever VWC < 28 %",
              "drought": f"irrigation withheld at plant age {dry[0][0]}-{dry[0][1]} and {dry[1][0]}-{dry[1][1]}",
              "salinity": f"EC +0.3 dS/m per day from age 10, held at {salt_target:.1f} dS/m",
              "heat": f"chamber B at {heat_T} C for 4 days every 13 days (calendar days {heat_cal})",
              "famine": "from age 10: plain water (EC ~0.3) and deficit irrigation to 24 % when VWC < 16 %"}
    plants, env_rows, eis_rows = [], [], []
    for c in range(cohorts):
        t0 = c * stagger
        for g in TREATMENTS:
            for i in range(1, reps + 1):
                pid = f"{name[:2].upper()}-{g[0].upper()}{c + 1}{chr(96 + i) if reps > 1 else ''}"
                LAmax = sp["LAmax"] * (1 + rng.normal(0, 0.08))
                LA = sp["LA0"] * (1 + rng.normal(0, 0.10))
                r = sp["r"] * (1 + rng.normal(0, 0.05))
                leaves = float(sp["leaves0"])
                vwc, W, D, N = 36 + rng.normal(0, 1), 1.0, 0.0, 1.0   # N = nutrient status
                Re_b, Ri_b, Cm_b, a_b = sp["eis"]
                base = {1: (Re_b * (1 + rng.normal(0, 0.15)), Ri_b * (1 + rng.normal(0, 0.15)))}
                base[2] = (base[1][0] * (1 + rng.normal(0, 0.15)), base[1][1] * (1 + rng.normal(0, 0.15)))
                Cm0 = Cm_b * (1 + rng.normal(0, 0.10))
                T_off = rng.normal(0, 0.3)
                label_prev = 0
                plants.append(dict(
                    species=name, latin_name=sp["latin"], plant_id=pid, treatment=g, cohort=c + 1,
                    transplant_day=t0, days_grown=DAYS, treatment_detail=detail[g],
                    chamber="B" if g == "heat" else "A", electrode_config="4-electrode clip",
                    electrode_session_1=f"leaf 4, mid-lamina, plant age 0-{reclip - 1}",
                    electrode_session_2=f"younger fully expanded leaf, plant age {reclip}-{DAYS - 1}",
                    recommended_growth_col=sp["growth_col"], pot_volume_cm3=sp["pot_cm3"]))
                for a in range(DAYS):
                    t = t0 + a
                    T = (TB[t] if g == "heat" else TA[t]) + T_off
                    RH = RH_B[t] if g == "heat" else RH_A[t]
                    ec = sp["ec_normal"] + rng.normal(0, 0.03)
                    if g == "salinity" and a >= 10:
                        ec = min(sp["ec_normal"] + 0.3 * (a - 9), salt_target) + rng.normal(0, 0.03)
                    if g == "famine" and a >= 10:
                        ec = max(0.3, sp["ec_normal"] - 0.25 * (a - 9)) + rng.normal(0, 0.03)  # flushed out
                    withheld = g == "drought" and in_any(a, dry)
                    starving = g == "famine" and a >= 10
                    if a > 0:
                        demand = (1.5 + 0.004 * LA) * pot_scale * (1 + 0.06 * (T - 20)) * (0.3 + 0.7 * W)
                        vwc = max(vwc - demand, 8.0)
                        if starving:
                            if vwc < 16:
                                vwc = 24 + rng.normal(0, 0.8)
                        elif not withheld and vwc < 28:
                            vwc = 36 + rng.normal(0, 0.8)
                    target = float(np.clip((vwc - 12) / 14, 0, 1)) - 0.02 * max(T - (sp["temp_high"] + 1), 0)
                    W = 0.6 * W + 0.4 * float(np.clip(target, 0, 1))
                    D = float(np.clip(0.85 * D + 0.08 * max(T - (sp["temp_high"] + 1), 0), 0, 1))
                    N += 0.06 * (float(np.clip(ec / sp["ec_normal"], 0, 1)) - N)      # nutrient status
                    fW = 0.1 + 0.9 * W ** 1.5
                    fEC = float(np.clip(1 - sp["salt_slope"] * (ec - salt_thr_sol), 0.3, 1))
                    fN = 0.15 + 0.85 * N
                    fL = min((PAR[t] / (PAR[t] + 150)) / (300 / 450), 1.1)
                    if a > 0:
                        LA += r * LA * (1 - LA / LAmax) * f_temp(T, Tb, To, Tm) * fW * fEC * fN * fL * (1 - 0.3 * D)
                        leaves += max(T - Tb, 0) / sp["phyllo"] * (0.5 + 0.5 * fW) * np.sqrt(fEC * fN)
                    diam = sp["diam"][0] * LA ** sp["diam"][1] * (1 - 0.10 * (1 - W))
                    height = (sp["height"][0] + sp["height"][1] * LA) * (1 - 0.05 * (1 - W))
                    pot = POT_DRY_G + sp["pot_cm3"] * vwc / 100 + sp["fw_per_cm2"] * LA
                    spad = sp["spad"] * (0.45 + 0.55 * N) * (1 - 0.2 * D)
                    if g == "drought":
                        label = int((withheld and vwc < 20) or (not withheld and label_prev and in_any(a, dry, lag_end=1)))
                    elif g == "salinity":
                        label = int(ec > sp["ec_high"])
                    elif g == "heat":
                        label = int(in_any(t, heat_cal, lag_end=1))
                    elif g == "famine":
                        label = int(spad < 0.8 * sp["spad"] or vwc < 20)
                    else:
                        label = 0
                    label_prev = label
                    wilt = int(np.clip(np.round(3 * (1 - W) ** 1.3 + 1.5 * D + rng.normal(0, 0.35)), 0, 3))
                    if rng.random() > MISSING_ENV_P or a in (0, DAYS - 1):
                        S = TRUE_SIGMA
                        env_rows.append(dict(
                            species=name, plant_id=pid, day=t, treatment=g, cohort=c + 1, data_source="SIMULATED",
                            air_temp_c=round(T + rng.normal(0, S["air_temp_c"]), 2),
                            soil_moisture_pct=round(vwc + rng.normal(0, S["soil_moisture_pct"]), 2),
                            rel_humidity_pct=round(RH + rng.normal(0, S["rel_humidity_pct"]), 1),
                            light_par=round(PAR[t] * (1 + rng.normal(0, S["light_par_rel"])), 1),
                            ec_ds_m=round(ec + rng.normal(0, S["ec_ds_m"]), 3),
                            ph=round(6.0 + 0.03 * (ec - sp["ec_normal"]) + rng.normal(0, S["ph"]), 2),
                            rosette_diam_cm=round(diam + rng.normal(0, S["rosette_diam_cm"]), 2),
                            height_cm=round(height + rng.normal(0, S["height_cm"]), 2),
                            leaf_count=int(round(leaves + rng.normal(0, S["leaf_count"]))),
                            pot_weight_g=round(pot + rng.normal(0, S["pot_weight_g"]), 1),
                            spad=round(spad + rng.normal(0, S["spad"]), 1),
                            wilting_score=wilt, stress_truth=label))
                    session = 1 if a < reclip else 2
                    leaf_age = a if session == 1 else a - reclip
                    Re0, Ri0 = base[session]
                    # [ASSUMPTION] EIS responses: drought raises Re/Ri, salinity lowers Re, heat damage
                    # lowers Re/Ri/Cm/alpha, nutrient starvation raises Re (fewer apoplast ions)
                    mult_e = (1 + 0.9 * (1 - W)) * (1 - 0.06 * (ec - sp["ec_normal"])) * (1 - 0.25 * D) * (1 + 0.5 * (1 - N))
                    mult_i = (1 + 0.35 * (1 - W)) * (1 - 0.30 * D) * (1 + 0.15 * (1 - N))
                    Re25 = Re0 * (1 + 0.003 * leaf_age) * mult_e
                    Ri25 = Ri0 * (1 + 0.003 * leaf_age) * mult_i
                    Cm = Cm0 * (1 - 0.35 * D) * (1 - 0.05 * (1 - W)) * (1 - 0.1 * (1 - N))
                    alpha = a_b - 0.08 * D + rng.normal(0, 0.01)
                    Tleaf = T - 0.8 + rng.normal(0, 0.3)
                    sweeps = [(session, Re25, Ri25)]
                    if a == reclip:
                        ReO, RiO = base[1]
                        lf = (1 + 0.003 * a) / (1 + 0.003 * leaf_age)
                        sweeps.insert(0, (1, Re25 * ReO / Re0 * lf, Ri25 * RiO / Ri0 * lf))
                    elif rng.random() < MISSING_SWEEP_P:
                        continue
                    for sess, re_, ri_ in sweeps:
                        Z = leaf_impedance(rng, FREQS, re_, ri_, Cm, alpha, Tleaf,
                                           bad=(a != reclip) and rng.random() < BAD_SWEEP_P)
                        for f, z in zip(FREQS, Z):
                            eis_rows.append(dict(species=name, plant_id=pid, day=t, freq_hz=round(f, 3),
                                                 z_real=z.real, z_imag=z.imag, sweep_temp_c=round(Tleaf, 2),
                                                 electrode_session=sess))
                plants[-1]["harvest_fresh_weight_g"] = round(sp["fw_per_cm2"] * LA * (0.9 + 0.1 * W) + rng.normal(0, 3), 1)
    return pd.DataFrame(plants), pd.DataFrame(env_rows), pd.DataFrame(eis_rows)


def calibration_data(name, seed):
    sp = SPECIES[name]
    rng = np.random.default_rng(seed + 100)
    refs = [("air_temp_c", 25.0, "degC", "probe in stirred water bath vs reference thermometer"),
            ("soil_moisture_pct", 30.0, "% VWC", "probe in substrate packed to known gravimetric water content"),
            ("rel_humidity_pct", 75.3, "%", "sensor sealed over saturated NaCl solution at 25 C"),
            ("light_par", 300.0, "umol m-2 s-1", "sensor at fixed position under one LED panel"),
            ("ec_ds_m", 1.413, "dS/m", "1413 uS/cm calibration solution"),
            ("ph", 7.00, "pH", "pH 7.00 buffer"),
            ("rosette_diam_cm", 18.6, "cm", "one plant photographed and measured 30x by 3 team members"),
            ("height_cm", 12.4, "cm", "one plant measured 30x by 3 team members"),
            ("leaf_count", 22, "leaves", "one plant counted 30x by 3 team members"),
            ("pot_weight_g", 1000.0, "g", "1 kg reference mass weighed 30x"),
            ("spad", 40.0, "SPAD", "SPAD meter on its calibration check film, 30x")]
    rows = []
    for s, ref, unit, how in refs:
        sd = TRUE_SIGMA["light_par_rel"] * ref if s == "light_par" else TRUE_SIGMA[s]
        for k in range(30):
            x = ref + rng.normal(0, sd)
            rows.append(dict(species=name, sensor=s, repeat=k + 1, reference_value=ref,
                             reading=int(round(x)) if s == "leaf_count" else round(x, 3), unit=unit, procedure=how))
    Re25, Ri25, Cm, a = sp["eis"]
    trows, rrows = [], []
    for T in (15, 20, 25, 30, 35):
        for k in range(3):
            Tl = T + rng.normal(0, 0.2)
            Z = leaf_impedance(rng, FREQS, Re25, Ri25, Cm, a, Tl)
            trows += [dict(species=name, set_temp_c=T, repeat=k + 1, sweep_temp_c=round(Tl, 2), freq_hz=round(f, 3),
                           z_real=z.real, z_imag=z.imag) for f, z in zip(FREQS, Z)]
    for k in range(20):
        Z = leaf_impedance(rng, FREQS, Re25, Ri25, Cm, a + rng.normal(0, 0.005), 25 + rng.normal(0, 0.2))
        rrows += [dict(species=name, sweep=k + 1, freq_hz=round(f, 3), z_real=z.real, z_imag=z.imag, sweep_temp_c=25.0)
                  for f, z in zip(FREQS, Z)]
    return pd.DataFrame(rows), pd.DataFrame(trows), pd.DataFrame(rrows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=list(SPECIES))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data_species")
    ap.add_argument("--cohorts", type=int, default=3)
    ap.add_argument("--stagger", type=int, default=7)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    profiles = []
    for k, name in enumerate(a.species):
        out = f"{a.out}/{name}"
        os.makedirs(out, exist_ok=True)
        plants, env, eis = simulate_species(name, a.seed + 17 * k, a.cohorts, a.stagger, a.reps)
        rep, tcal, erep = calibration_data(name, a.seed + 17 * k)
        plants.to_csv(f"{out}/plants.csv", index=False)
        env.to_csv(f"{out}/env.csv", index=False)
        eis.to_csv(f"{out}/eis.csv", index=False)
        rep.to_csv(f"{out}/calib_sensor_repeatability.csv", index=False)
        tcal.to_csv(f"{out}/calib_eis_temperature.csv", index=False)
        erep.to_csv(f"{out}/calib_eis_repeatability.csv", index=False)
        fw = plants.groupby("treatment").harvest_fresh_weight_g.mean().round(0).to_dict()
        print(f"{name:9s} {len(plants)} plants, {len(env)} env rows, {eis.groupby(['plant_id', 'day', 'electrode_session']).ngroups}"
              f" sweeps | mean harvest FW (g): " + ", ".join(f"{t} {fw[t]:.0f}" for t in TREATMENTS))
        sp = SPECIES[name]
        profiles.append(dict(species=name, latin_name=sp["latin"], days_grown=sp["days"], room_temp_c=sp["room_temp"],
                             recommended_growth_col=sp["growth_col"], cardinal_temps_c="/".join(map(str, sp["cardinal"])),
                             ec_normal_ds_m=sp["ec_normal"], fao_salt_threshold_ece=sp["salt_thr"],
                             fao_salt_slope_pct=100 * sp["salt_slope"], ec_high_flag=sp["ec_high"],
                             temp_high_flag_c=sp["temp_high"], healthy_spad=sp["spad"],
                             eis_Re_kohm=sp["eis"][0] / 1e3, eis_Ri_kohm=sp["eis"][1] / 1e3,
                             salt_source="FAO I&D 61 Table A1.1" if name in ("lettuce", "spinach", "tomato") else
                             ("FAO cabbage values (proxy)" if name in ("kale", "pak_choi") else "assumed (not in FAO table)")))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(profiles).to_csv(f"{a.out}/species_profiles.csv", index=False)
    print(f"\nSIMULATED data written to ./{a.out}/<species>/ and species_profiles.csv")


if __name__ == "__main__":
    main()
