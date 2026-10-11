#!/usr/bin/env python3
"""
Generate a SIMULATED lettuce pilot trial that follows the PlantSense Phase-1 protocol.

    python generate_lettuce_trial.py                         # 3 staggered cohorts (default)
    python generate_lettuce_trial.py --cohorts 1 --reps 3    # single transplant date
    python generate_lettuce_trial.py --electrodes 2          # 2-electrode clip (polarisation)
    python generate_lettuce_trial.py --seed 3                # another replicate trial

Every number below is a [BIOLOGICAL/ENGINEERING ASSUMPTION] chosen to be plausible for
butterhead lettuce in a controlled growth room. It is NOT measured data. Use it to
rehearse the protocol and test the pipeline, never as evidence about real plants.

PHASE-1 PROTOCOL IMPLEMENTED
  Step 1  growth: rosette diameter (fixed top-view photo), leaf count, height, every day
          at 09:00; pots weighed daily; harvest fresh weight at plant age 39 days.
  Step 2  sensors: air temp, soil moisture (VWC), RH, PAR + optional EC and pH.
  Step 3  EIS: 4-electrode clip (default) on tagged leaf 4, mid-lamina. Leaf 4 is shaded by
          plant age 21, so the clip is moved to leaf 10 and electrode_session becomes 2.
          Same 31 frequencies every day (1 kHz - 1 MHz), sweep at 10:00, leaf temperature
          logged (sweep_temp_c). The -Z'' peak (~20-40 kHz) is inside the range.
          On the re-clip day both leaves are swept back-to-back so the baselines can be linked.
  Step 4  labels fixed before the trial (independent of the soil-moisture sensor):
            drought  : pot weighing shows VWC < 20 %, during withholding, + 1 day after rewatering
            salinity : calibrated EC meter reads > 2.5 dS/m
            heat     : chamber-B heat days + 1 day
          plus a blind 0-3 wilting score.
  Step 5  4 treatments x cohorts x reps. Cohorts are transplanted `stagger` days apart so that
          on any calendar day several growth stages are present (needed for a plant+time
          split). Treatments start at plant age 10, so ages 0-9 are an unstressed baseline.
  Calibration: repeat readings of every sensor against a reference, a 20-sweep EIS
          repeatability test and a 5-temperature EIS calibration on one spare plant.
"""
import argparse
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ design
DAYS = 40                                             # days each plant is grown
FREQS = np.logspace(3, 6, 31)                         # 1 kHz - 1 MHz, 31 points
TREATMENTS = {
    "control":  "re-watered to 36 % VWC whenever VWC < 28 %",
    "drought":  "irrigation withheld at plant age 10-20 and 27-34, re-watered to 36 % next day",
    "salinity": "nutrient EC raised by 0.3 dS/m per day from plant age 10, held at 4.0 dS/m",
    "heat":     "chamber B at 30 C on calendar days 18-21, 31-34 and 45-48",
}
DRY_AGE = [(10, 20), (27, 34)]
HEAT_CAL = [(18, 21), (31, 34), (45, 48)]
RECLIP_AGE = 21

# ------------------------------------------------------------------ assumptions (true values)
TRUE_SIGMA = dict(air_temp_c=0.3, soil_moisture_pct=1.5, rel_humidity_pct=2.0, light_par_rel=0.04,
                  ec_ds_m=0.04, ph=0.05, rosette_diam_cm=0.35, height_cm=0.3, leaf_count=0.6, pot_weight_g=2.0)
TEMP_COEF_TRUE = 0.021        # tissue resistance falls ~2.1 % per degC
EIS_NOISE = 0.004             # 0.4 % complex noise
BAD_SWEEP_P = 0.02            # electrode slipped -> contact artefact
MISSING_SWEEP_P = 0.03
MISSING_ENV_P = 0.02          # whole-day sensor outage
POT_DRY_G, POT_VOL_CM3 = 350.0, 1000.0


def in_any(t, windows, lag_start=0, lag_end=0):
    return any(a + lag_start <= t <= b + lag_end for a, b in windows)


def ar1(rng, n, sd, rho=0.7):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    return x


def f_temp(T, tb=4.0, to=20.0, tm=33.0):
    """Beta temperature response for lettuce leaf expansion (cardinal temps assumed)."""
    if T <= tb or T >= tm:
        return 0.0
    return ((T - tb) / (to - tb)) * ((tm - T) / (tm - to)) ** ((tm - to) / (to - tb))


def cole(f, Re, Ri, Cm, alpha):
    """Fricke circuit with a CPE-broadened membrane: R0 = Re, Rinf = Re*Ri/(Re+Ri)."""
    R0, Rinf = Re, Re * Ri / (Re + Ri)
    tau = Cm * (Re + Ri)
    return Rinf + (R0 - Rinf) / (1 + (1j * 2 * np.pi * f * tau) ** alpha)


def leaf_impedance(rng, f, Re25, Ri25, Cm, alpha, T, contact_sd=0.005, bad=False, electrodes=4):
    k = 1 + TEMP_COEF_TRUE * (T - 25.0)                 # R_T = R_25 / (1 + a (T - 25))
    Z = cole(f, Re25 / k, Ri25 / k, Cm, alpha) * (1 + rng.normal(0, contact_sd))
    if electrodes == 2:                                  # electrode polarisation in series (CPE)
        Q, n = 1.1e-7 * (1 + rng.normal(0, 0.1)), 0.80 + rng.normal(0, 0.01)
        Z = Z + 1 / (Q * (1j * 2 * np.pi * f) ** n)
    if bad:                                              # slipped clip: series contact + polarisation
        Rc, Cc = rng.uniform(2e4, 8e4), rng.uniform(5e-9, 2e-8)
        Z = Z + Rc / (1 + 1j * 2 * np.pi * f * Rc * Cc) + 1 / (1j * 2 * np.pi * f * 1e-8)
    noise = EIS_NOISE * np.abs(Z) * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(2)
    return Z + noise


# ------------------------------------------------------------------ trial
def simulate_trial(seed, cohorts=3, stagger=7, reps=1, electrodes=4):
    rng = np.random.default_rng(seed)
    ncal = DAYS + stagger * (cohorts - 1)
    d = np.arange(ncal)
    # growth room A (control, drought, salinity) and B (heat); 16 h photoperiod
    TA = 20 + 1.0 * np.sin(2 * np.pi * d / 9) + ar1(rng, ncal, 0.6)
    TB = TA + rng.normal(0, 0.2, ncal) + np.array([10.0 if in_any(t, HEAT_CAL) else 0.0 for t in d])
    RH_A = np.clip(65 + ar1(rng, ncal, 4), 50, 85)
    RH_B = np.clip(RH_A - 1.5 * (TB - TA), 35, 85)
    PAR = np.clip(280 * (1 + ar1(rng, ncal, 0.07)), 200, 360)    # daily mean during photoperiod

    plants, env_rows, eis_rows = [], [], []
    for c in range(cohorts):
        t0 = c * stagger
        for g in TREATMENTS:
            for i in range(1, reps + 1):
                pid = f"{g[0].upper()}{c + 1}{chr(96 + i) if reps > 1 else ''}"
                LAmax = 950 * (1 + rng.normal(0, 0.08))         # cm2 total leaf area at maturity
                LA = 18 * (1 + rng.normal(0, 0.10))             # cm2 at transplant (3-4 true leaves)
                leaves = 4.0
                r = 0.17 * (1 + rng.normal(0, 0.05))            # /day relative expansion rate
                vwc, W, D = 36 + rng.normal(0, 1), 1.0, 0.0     # soil moisture, water status, heat damage
                base = {1: (22e3 * (1 + rng.normal(0, 0.15)), 9e3 * (1 + rng.normal(0, 0.15)))}
                base[2] = (base[1][0] * (1 + rng.normal(0, 0.15)), base[1][1] * (1 + rng.normal(0, 0.15)))
                Cm0 = 1.8e-10 * (1 + rng.normal(0, 0.10))
                T_off = rng.normal(0, 0.3)                      # bench position
                label_prev = 0
                plants.append(dict(
                    plant_id=pid, treatment=g, cohort=c + 1, transplant_day=t0, treatment_detail=TREATMENTS[g],
                    chamber="B" if g == "heat" else "A", electrode_config=f"{electrodes}-electrode clip",
                    electrode_session_1=f"leaf 4, mid-lamina, plant age 0-{RECLIP_AGE - 1}",
                    electrode_session_2=f"leaf 10, mid-lamina, plant age {RECLIP_AGE}-{DAYS - 1}",
                    sweep_time="10:00", growth_photo_time="09:00", pot_dry_mass_g=POT_DRY_G))
                for a in range(DAYS):
                    t = t0 + a                                   # calendar day
                    T = (TB[t] if g == "heat" else TA[t]) + T_off
                    RH = RH_B[t] if g == "heat" else RH_A[t]
                    ec = 1.4 + rng.normal(0, 0.03)
                    if g == "salinity" and a >= 10:
                        ec = min(1.4 + 0.3 * (a - 9), 4.0) + rng.normal(0, 0.03)
                    withheld = g == "drought" and in_any(a, DRY_AGE)
                    if a > 0:
                        demand = (1.5 + 0.004 * LA) * (1 + 0.06 * (T - 20)) * (0.3 + 0.7 * W)
                        vwc = max(vwc - demand, 8.0)
                        if not withheld and vwc < 28:
                            vwc = 36 + rng.normal(0, 0.8)
                    target = float(np.clip((vwc - 12) / 14, 0, 1)) - 0.02 * max(T - 28, 0)
                    W = 0.6 * W + 0.4 * float(np.clip(target, 0, 1))
                    D = float(np.clip(0.85 * D + 0.08 * max(T - 28, 0), 0, 1))
                    fW = 0.1 + 0.9 * W ** 1.5
                    fEC = float(np.clip(1 - 0.13 * (ec - 2.0), 0.3, 1))   # growth falls above ~2 dS/m
                    fL = min((PAR[t] / (PAR[t] + 150)) / (280 / 430), 1.1)
                    if a > 0:
                        LA += r * LA * (1 - LA / LAmax) * f_temp(T) * fW * fEC * fL * (1 - 0.3 * D)
                        leaves += max(T - 4, 0) / 22 * (0.5 + 0.5 * fW) * np.sqrt(fEC)
                    diam = 3.0 * LA ** 0.33 * (1 - 0.10 * (1 - W))     # wilting shrinks the rosette
                    height = 4 + 0.012 * LA
                    pot = POT_DRY_G + POT_VOL_CM3 * vwc / 100 + 0.24 * LA
                    # labels (protocol, independent of the soil-moisture sensor)
                    if g == "drought":
                        label = int((withheld and vwc < 20) or (not withheld and label_prev and
                                                                  in_any(a, DRY_AGE, lag_end=1)))
                    elif g == "salinity":
                        label = int(ec > 2.5)
                    elif g == "heat":
                        label = int(in_any(t, HEAT_CAL, lag_end=1))
                    else:
                        label = 0
                    label_prev = label
                    wilt = int(np.clip(np.round(3 * (1 - W) ** 1.3 + 1.5 * D + rng.normal(0, 0.35)), 0, 3))
                    if rng.random() > MISSING_ENV_P or a in (0, DAYS - 1):
                        env_rows.append(dict(
                            plant_id=pid, day=t, treatment=g, cohort=c + 1, data_source="SIMULATED",
                            air_temp_c=round(T + rng.normal(0, TRUE_SIGMA["air_temp_c"]), 2),
                            soil_moisture_pct=round(vwc + rng.normal(0, TRUE_SIGMA["soil_moisture_pct"]), 2),
                            rel_humidity_pct=round(RH + rng.normal(0, TRUE_SIGMA["rel_humidity_pct"]), 1),
                            light_par=round(PAR[t] * (1 + rng.normal(0, TRUE_SIGMA["light_par_rel"])), 1),
                            ec_ds_m=round(ec + rng.normal(0, TRUE_SIGMA["ec_ds_m"]), 3),
                            ph=round(6.0 + 0.03 * (ec - 1.4) + rng.normal(0, TRUE_SIGMA["ph"]), 2),
                            rosette_diam_cm=round(diam + rng.normal(0, TRUE_SIGMA["rosette_diam_cm"]), 2),
                            height_cm=round(height + rng.normal(0, TRUE_SIGMA["height_cm"]), 2),
                            leaf_count=int(round(leaves + rng.normal(0, TRUE_SIGMA["leaf_count"]))),
                            pot_weight_g=round(pot + rng.normal(0, TRUE_SIGMA["pot_weight_g"]), 1),
                            wilting_score=wilt, stress_truth=label))
                    # EIS on the tagged leaf (25 C-referenced tissue parameters)
                    session = 1 if a < RECLIP_AGE else 2
                    leaf_age = a if session == 1 else a - RECLIP_AGE
                    Re0, Ri0 = base[session]
                    Re25 = Re0 * (1 + 0.003 * leaf_age) * (1 + 0.9 * (1 - W)) * (1 - 0.06 * (ec - 1.4)) * (1 - 0.25 * D)
                    Ri25 = Ri0 * (1 + 0.003 * leaf_age) * (1 + 0.35 * (1 - W)) * (1 - 0.30 * D)
                    Cm = Cm0 * (1 - 0.35 * D) * (1 - 0.05 * (1 - W))
                    alpha = 0.78 - 0.08 * D + rng.normal(0, 0.01)
                    Tleaf = T - 0.8 + rng.normal(0, 0.3)
                    # on the re-clip day BOTH leaves are swept back-to-back (overlap links the baselines)
                    sweeps = [(session, Re25, Ri25)]
                    if a == RECLIP_AGE:
                        ReO, RiO = base[1]
                        lf = (1 + 0.003 * a) / (1 + 0.003 * leaf_age)
                        sweeps.insert(0, (1, Re25 * ReO / Re0 * lf, Ri25 * RiO / Ri0 * lf))
                    elif rng.random() < MISSING_SWEEP_P:
                        continue
                    for sess, re_, ri_ in sweeps:
                        Z = leaf_impedance(rng, FREQS, re_, ri_, Cm, alpha, Tleaf,
                                           bad=(a != RECLIP_AGE) and rng.random() < BAD_SWEEP_P, electrodes=electrodes)
                        for f, z in zip(FREQS, Z):
                            eis_rows.append(dict(plant_id=pid, day=t, freq_hz=round(f, 3), z_real=z.real, z_imag=z.imag,
                                                 sweep_temp_c=round(Tleaf, 2), electrode_session=sess))
                plants[-1]["harvest_fresh_weight_g"] = round(0.24 * LA * (0.9 + 0.1 * W) + rng.normal(0, 5), 1)
    return pd.DataFrame(plants), pd.DataFrame(env_rows), pd.DataFrame(eis_rows)


# ------------------------------------------------------------------ calibration datasets
def calibration_data(seed, electrodes=4):
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
            ("pot_weight_g", 1000.0, "g", "1 kg reference mass weighed 30x")]
    rows = []
    for name, ref, unit, how in refs:
        s = TRUE_SIGMA["light_par_rel"] * ref if name == "light_par" else TRUE_SIGMA[name]
        for k in range(30):
            x = ref + rng.normal(0, s)
            rows.append(dict(sensor=name, repeat=k + 1, reference_value=ref,
                             reading=int(round(x)) if name == "leaf_count" else round(x, 3),
                             unit=unit, procedure=how))
    rep = pd.DataFrame(rows)
    Re25, Ri25, Cm, a = 24e3, 9.5e3, 1.8e-10, 0.78
    trows = []
    for T in (15, 20, 25, 30, 35):
        for k in range(3):
            Tl = T + rng.normal(0, 0.2)
            Z = leaf_impedance(rng, FREQS, Re25, Ri25, Cm, a, Tl, electrodes=electrodes)
            trows += [dict(set_temp_c=T, repeat=k + 1, sweep_temp_c=round(Tl, 2), freq_hz=round(f, 3),
                           z_real=z.real, z_imag=z.imag) for f, z in zip(FREQS, Z)]
    rrows = []
    for k in range(20):
        Z = leaf_impedance(rng, FREQS, Re25, Ri25, Cm, a + rng.normal(0, 0.005), 25 + rng.normal(0, 0.2),
                           electrodes=electrodes)
        rrows += [dict(sweep=k + 1, freq_hz=round(f, 3), z_real=z.real, z_imag=z.imag, sweep_temp_c=25.0)
                  for f, z in zip(FREQS, Z)]
    return rep, pd.DataFrame(trows), pd.DataFrame(rrows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data")
    ap.add_argument("--cohorts", type=int, default=3, help="transplant cohorts (staggered)")
    ap.add_argument("--stagger", type=int, default=7, help="days between cohorts")
    ap.add_argument("--reps", type=int, default=1, help="plants per treatment per cohort")
    ap.add_argument("--electrodes", type=int, choices=[2, 4], default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plants, env, eis = simulate_trial(a.seed, a.cohorts, a.stagger, a.reps, a.electrodes)
    rep, tcal, erep = calibration_data(a.seed, a.electrodes)
    plants.to_csv(f"{a.out}/plants.csv", index=False)
    env.to_csv(f"{a.out}/env.csv", index=False)
    eis.to_csv(f"{a.out}/eis.csv", index=False)
    rep.to_csv(f"{a.out}/calib_sensor_repeatability.csv", index=False)
    tcal.to_csv(f"{a.out}/calib_eis_temperature.csv", index=False)
    erep.to_csv(f"{a.out}/calib_eis_repeatability.csv", index=False)
    print(f"SIMULATED lettuce trial (seed {a.seed}, {a.cohorts} cohort(s) x {a.reps} rep(s), "
          f"{a.electrodes}-electrode) written to ./{a.out}/")
    print(f"  plants.csv {len(plants)} plants | env.csv {len(env)} rows | eis.csv {len(eis)} rows "
          f"({eis.groupby(['plant_id', 'day']).ngroups} sweeps)")
    print(plants[["plant_id", "treatment", "cohort", "harvest_fresh_weight_g"]].to_string(index=False))


if __name__ == "__main__":
    main()
