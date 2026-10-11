#!/usr/bin/env python3
"""
PlantSense v3 - explainable plant growth / stress prediction (research prototype).

RUN
    pip install numpy pandas scipy scikit-learn matplotlib
    python plantsense_v3.py                                   # SIMULATED demo data
    python plantsense_v3.py --seeds 5                         # repeat on 5 simulated datasets
    python plantsense_v3.py --csv-env env.csv --csv-eis eis.csv --crop lettuce \
                            --growth-col rosette_diam_cm      # your MEASURED data

CHANGES IN v3
  - Objective 3: correlation study (within-plant Spearman, plant-bootstrap 95 % CI, lag 0 and
    EIS-today vs growth over the next 3 days) -> eis_correlations.csv + figure 08.
  - Wilting score: --label-source stress_truth | wilting | either, and Cohen's kappa agreement.
  - Tiered, noise-aware stress categories ("POSSIBLE X STRESS" vs "X CONDITIONS - no plant
    response yet", "UNEXPLAINED GROWTH DROP"); a growth drop must exceed 2 sigma of the
    growth-rate measurement noise.
  - Electrode sessions: electrode_session column; baselines linked on the overlap day where
    both leaves are swept (fallback: adjacent days, with a warning).
  - --electrode-cpe: fits a series electrode-polarisation CPE (for 2-electrode set-ups).
  - Time-aware conformal intervals in plant+time mode.
  - Growth models predict the change from the last 3 days (persistence + learned correction).
  - Threshold-relative stress inputs (sm_deficit, sm_excess, heat_excess, ec_excess, ec_deficit)
    from the crop profile, so "too high OR too low" is learnable; low-EC NUTRIENT-DEFICIENCY flag.
  - Crop profiles: lettuce, spinach, pak_choi, kale, basil, tomato (--crop).
  - fc is temperature-referred to 25 C (fc_25 = fc_T / (1 + a (T - 25))), not only |Z|.
  - --primary auto (default: lower held-out growth MAE) | Linear | RandomForest drives status,
    sensitivity and figures. Final deployed models are trained on all labelled rows.
  - Expected-growth model includes current plant size.
  - Lettuce thresholds cited: temp_high 27 C (lettuce commonly grown at 20-25 C, Utah State
    Univ. Extension), ec_high 2.5 dS/m (growth higher at 1.3-2.0 than 2.9-3.9 dS/m, Purdue HO-309-W).

CHANGES FROM v1 (numbers match the review)
  1  Re and Ri are derived from the Cole fit with the standard Fricke circuit
     (Re = R0, Ri = R0*Rinf/(R0-Rinf)). [STANDARD] - this is NOT the patent method.
  2  Evaluation holds out whole PLANTS (GroupKFold over plant_id) as well as later days.
  3  Missing days are handled: each plant is put on a full daily calendar before
     growth rates are computed, so a gap gives NaN instead of a wrong rate.
  4  Growth column is selectable (--growth-col); leaf_count is used if present.
  5  EC and pH are optional.
  6  EIS is temperature-normalised to 25 C (sweep_temp_c column, else daily air temp).
  7  Spectra with a poor Cole fit, or whose -Z'' peak is outside the measured range,
     are rejected.
  8  Conformal 90% prediction intervals (model error) are reported next to the
     input-noise Monte Carlo, with their measured coverage on held-out plants.
  10 Threshold profiles (--crop generic | lettuce).
  11 Guards for real data: missing days/plants, day 0, missing EIS days.
  13 The Cole fit forces R0 > Rinf; impedance sign convention is checked.
  --split plant | plant+time  (plant for a single transplant cohort; current size is a feature)
  --calib calib.json  loads measured sensor sigmas and TEMP_COEF (from calibrate.py).
  Lettuce ec_high is 2.5 dS/m (nutrient-solution EC, Purdue HO-309-W).

EVIDENCE CATEGORIES (keep these separate in your report)
  [STANDARD]   Z = V/I, |Z|, phase, single-dispersion Cole model, Fricke Re/Ri.
  [PROJECT]    growth model, growth deficit, stress rules, sensitivity, uncertainty,
               Monte Carlo, conformal intervals, Model A vs Model B.
  [ASSUMPTION] every threshold, sensor sigma, temperature coefficient, the simulator.
  [PATENT]     NOT used anywhere.

REAL-DATA CSV SCHEMA
  env.csv  (one row per plant per day)
      required: plant_id, day, air_temp_c, soil_moisture_pct, rel_humidity_pct,
                light_par, <growth column, default height_cm>
      optional: ec_ds_m, ph, leaf_count, stress_truth (0/1),
                data_source ("SIMULATED" forces simulated labels on every output)
  eis.csv  (one row per plant, day and frequency)
      required: plant_id, day, freq_hz and EITHER z_real, z_imag OR v_re, v_im, i_re, i_im
      optional: sweep_temp_c (tissue/air temperature during the sweep)

Results on simulated data only test the PIPELINE. The simulator encodes the
assumption that EIS responds to plant water status.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
HORIZON = 3                                   # forecast horizon in days
FREQS_HZ = np.logspace(2, 6, 25)              # 100 Hz - 1 MHz (simulator only)
ENV_REQUIRED = ["air_temp_c", "soil_moisture_pct", "rel_humidity_pct", "light_par"]
ENV_OPTIONAL = ["ec_ds_m", "ph"]
EIS_COLS = ["Re_rel", "Ri_rel", "log10_fc", "alpha", "phase_min_deg"]
GTASK, STASK = "growth (per day)", "stress within 3 d"

TEMP_COEF = 0.02      # [ASSUMPTION] ~2 %/degC tissue conductivity change - measure it for your crop
T_REF = 25.0
MAX_FIT_RMSE = 0.10   # [ASSUMPTION] reject spectra whose Cole fit is worse than this
CONF_LEVEL = 0.90

# [ASSUMPTION] sensor / feature 1-sigma uncertainties: ("abs", value) or ("rel", fraction)
SENSOR_SIGMA = {
    "air_temp_c": ("abs", 0.5), "soil_moisture_pct": ("abs", 2.0),
    "rel_humidity_pct": ("abs", 3.0), "light_par": ("rel", 0.05),
    "ec_ds_m": ("abs", 0.15), "ph": ("abs", 0.1),
    "growth_past3": ("abs", 0.07), "age_days": ("abs", 0.0),
    "leaf_count": ("abs", 0.5), "leaf_rate_past3": ("abs", 0.25),
    "height_cm": ("abs", 0.15), "rosette_diam_cm": ("abs", 0.35),
    "sm_deficit": ("abs", 2.0), "sm_excess": ("abs", 2.0), "heat_excess": ("abs", 0.5),
    "ec_excess": ("abs", 0.15), "ec_deficit": ("abs", 0.15),
    "Re_rel": ("abs", 0.05), "Ri_rel": ("abs", 0.08), "log10_fc": ("abs", 0.03),
    "alpha": ("abs", 0.03), "phase_min_deg": ("abs", 1.0),
}
CLIP = {"sm_deficit": (0, None), "sm_excess": (0, None), "heat_excess": (0, None), "ec_excess": (0, None),
        "ec_deficit": (0, None), "soil_moisture_pct": (0, 100), "rel_humidity_pct": (0, 100), "light_par": (0, None),
        "ec_ds_m": (0, None), "ph": (0, 14), "alpha": (0.3, 1.0), "leaf_count": (0, None)}

# [ASSUMPTION] threshold profiles. The lettuce values are starting points to CHECK against
# an agronomy source and your own substrate; growth_low=None means "20th percentile of
# training-period growth" because lettuce growth units depend on --growth-col.
THRESH_PROFILES = {
    "generic": dict(sm_low=20, temp_high=34, sm_high=40, ec_high=2.5, ec_low=0.0, deficit=0.3, re_rise=1.35, growth_low=0.6),
    "lettuce": dict(sm_low=20, temp_high=27, sm_high=40, ec_high=2.5, ec_low=0.7, deficit=0.3, re_rise=1.35, growth_low=None),
    # other greenhouse crops: ec_high = FAO I&D 61 Table A1.1 salt threshold + 1.2 dS/m (solution EC);
    # ec_low = half the normal nutrient-solution EC; temp_high values are [ASSUMPTIONS] to check
    "spinach":  dict(sm_low=20, temp_high=26, sm_high=40, ec_high=3.2, ec_low=0.9, deficit=0.3, re_rise=1.35, growth_low=None),
    "pak_choi": dict(sm_low=20, temp_high=27, sm_high=40, ec_high=3.0, ec_low=0.9, deficit=0.3, re_rise=1.35, growth_low=None),
    "kale":     dict(sm_low=20, temp_high=27, sm_high=40, ec_high=3.0, ec_low=1.0, deficit=0.3, re_rise=1.35, growth_low=None),
    "basil":    dict(sm_low=20, temp_high=35, sm_high=40, ec_high=2.7, ec_low=0.8, deficit=0.3, re_rise=1.35, growth_low=None),
    "tomato":   dict(sm_low=20, temp_high=32, sm_high=40, ec_high=3.7, ec_low=1.25, deficit=0.3, re_rise=1.35, growth_low=None),
}
# lettuce sources: temp - Utah State Univ. Extension (lettuce commonly grown at 20-25 C);
#                  EC   - Purdue Extension HO-309-W (1.3 dS/m recommended; growth lower at 2.9-3.9 dS/m)
ELECTRODE_CPE = False     # True: fit a series electrode-polarisation CPE (2-electrode set-ups)
EIS_DEV = 0.35            # [ASSUMPTION] |Re/baseline - 1| above this counts as an EIS response
THRESH = dict(THRESH_PROFILES["generic"])


# ============================================================ 1. SIMULATED DATA
def simulate(seed=SEED, n_plants=12, n_days=100):
    """SIMULATED DATA. Latent water status W drives growth and EIS - an ASSUMPTION."""
    rng = np.random.default_rng(seed)
    d = np.arange(n_days)

    def ar1(sd, rho=0.8):
        x = np.zeros(n_days)
        for i in range(1, n_days):
            x[i] = rho * x[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
        return x
    heat = ((d >= 12) & (d <= 16)) | ((d >= 38) & (d <= 42)) | ((d >= 70) & (d <= 75))
    T = 27 + 3 * np.sin(2 * np.pi * d / 35) + ar1(1.2) + np.where(heat, 8, 0)
    rain = np.where(rng.random(n_days) < 0.08, rng.uniform(5, 20, n_days), 0)
    light = np.clip(420 + 80 * np.sin(2 * np.pi * d / 35 + 1) + ar1(60) - np.where(rain > 0, 120, 0), 150, 650)
    RH = np.clip(70 - 1.5 * (T - 27) + rng.normal(0, 4, n_days) + np.where(rain > 0, 10, 0), 25, 95)
    scenarios = ["control", "control", "drought", "drought", "waterlogging", "salinity"]
    env_rows, eis_rows = [], []
    for p in range(n_plants):
        scen, pid = scenarios[p % 6], f"P{p + 1}"
        shift = int(rng.integers(0, 6))
        dry_windows = [(20 + shift, 34 + shift), (58 + shift, 72 + shift)]
        wet_windows = [(18 + shift, 30 + shift), (60 + shift, 72 + shift)]
        gmax = 1.4 * (1 + rng.normal(0, 0.08))
        base = 50e3 * (1 + rng.normal(0, 0.2))
        sm, W, H, leaves = 30 + rng.normal(0, 1.5), 1.0, 10 + rng.normal(0, 0.5), 4.0
        for t in d:
            irr = 15 if sm < 24 else 0
            if scen == "drought" and any(a <= t <= b for a, b in dry_windows):
                irr = 0
            if scen == "waterlogging" and any(a <= t <= b for a, b in wet_windows):
                irr = 25
            if t > 0:
                sm = np.clip(sm + 0.5 * (irr + rain[t]) - (0.6 + 0.05 * max(T[t] - 20, 0)), 5, 50)
            ec = 1.2 + (min(0.08 * (t - 20), 3.0) if scen == "salinity" and t >= 20 else 0)
            W = 0.5 * W + 0.5 * np.clip(np.clip((sm - 12) / 16, 0, 1) - 0.025 * max(T[t] - 32, 0), 0, 1)
            A = np.clip((sm - 38) / 10, 0, 1)
            g = (gmax * np.exp(-((T[t] - 26) / 9) ** 2) * np.clip(light[t] / 450, 0.4, 1.0)
                 * (0.15 + 0.85 * W) * (1 - 0.8 * A) * np.clip(1 - 0.2 * max(ec - 2, 0), 0.2, 1)
                 * (1 - 0.0005 * t) + rng.normal(0, 0.03))
            H += max(g, 0)
            leaves += max(g, 0) / 2.5
            stress = int(W < 0.55 or T[t] > 34 or sm > 40 or ec > 2.5)
            env_rows.append(dict(
                plant_id=pid, day=t, scenario=scen,
                air_temp_c=T[t] + rng.normal(0, 0.5), soil_moisture_pct=sm + rng.normal(0, 2.0),
                rel_humidity_pct=RH[t] + rng.normal(0, 3.0), light_par=light[t] * (1 + rng.normal(0, 0.05)),
                ec_ds_m=ec + rng.normal(0, 0.15), ph=6.4 + rng.normal(0, 0.1),
                height_cm=H + rng.normal(0, 0.15), leaf_count=int(np.floor(leaves)),
                stress_truth=stress, water_status_latent=W))
            R0 = base * (1 + 0.8 * (1 - W)) * (1 - 0.08 * (ec - 1.2)) * (1 + rng.normal(0, 0.03))
            Rinf = 0.24 * base * (1 + 0.3 * (1 - W)) * (1 + rng.normal(0, 0.03))
            fc = 5e3 * (1 + 0.2 * (1 - W))
            alpha = 0.8 + rng.normal(0, 0.02)
            contact = 1 + rng.normal(0, 0.04)
            Z = contact * cole_model([np.log10(R0), np.log10(Rinf), np.log10(1 / (2 * np.pi * fc)), alpha], FREQS_HZ)
            Tsw = T[t] + rng.normal(0, 0.3)
            Z = Z / (1 + TEMP_COEF * (Tsw - T_REF))          # [ASSUMPTION] temperature effect
            I0 = 1e-5
            cn = lambda: rng.normal(0, 0.01, len(FREQS_HZ)) + 1j * rng.normal(0, 0.01, len(FREQS_HZ))
            V = Z * I0 * (1 + cn())
            I = I0 * (1 + cn())
            for k, f in enumerate(FREQS_HZ):
                eis_rows.append(dict(plant_id=pid, day=t, freq_hz=f, v_re=V[k].real, v_im=V[k].imag,
                                     i_re=I[k].real, i_im=I[k].imag, sweep_temp_c=Tsw))
    return pd.DataFrame(env_rows), pd.DataFrame(eis_rows)


def load_real(env_csv, eis_csv=None, growth_col="height_cm"):
    env = pd.read_csv(env_csv)
    need = ["plant_id", "day", growth_col] + ENV_REQUIRED
    missing = [c for c in need if c not in env.columns]
    if missing:
        raise ValueError(f"env CSV is missing columns: {missing}")
    env["day"] = env.day.astype(int)
    if env.duplicated(["plant_id", "day"]).any():
        raise ValueError("env CSV has duplicate plant_id/day rows - average or remove them first")
    eis = None
    if eis_csv:
        eis = pd.read_csv(eis_csv)
        base = [c for c in ["plant_id", "day", "freq_hz"] if c not in eis.columns]
        has_z = {"z_real", "z_imag"} <= set(eis.columns)
        has_vi = {"v_re", "v_im", "i_re", "i_im"} <= set(eis.columns)
        if base or not (has_z or has_vi):
            raise ValueError(f"eis CSV needs plant_id, day, freq_hz and z_real/z_imag or v_re/v_im/i_re/i_im "
                             f"(missing: {base or 'impedance columns'})")
        eis["day"] = eis.day.astype(int)
    return env, eis


# ============================================================ 2. EIS (standard maths)
def impedance_from_vi(df):
    """[STANDARD] Z = V/I ; |Z| = sqrt(Z'^2 + Z''^2) ; phi = atan2(Z'', Z')."""
    out = df.copy()
    if "z_real" not in out:
        Z = (out.v_re.to_numpy() + 1j * out.v_im.to_numpy()) / (out.i_re.to_numpy() + 1j * out.i_im.to_numpy())
        out["z_real"], out["z_imag"] = np.real(Z), np.imag(Z)
    out["z_mag"] = np.sqrt(out.z_real ** 2 + out.z_imag ** 2)
    out["phase_deg"] = np.degrees(np.arctan2(out.z_imag, out.z_real))
    return out


def standardise_eis(eis, env):
    """Z from V/I, sign-convention check, and temperature normalisation to T_REF."""
    out = impedance_from_vi(eis)
    if (out.z_imag > 0).mean() > 0.5:
        print("NOTE: most Z'' values are positive - your analyser reports +Z'' for capacitive "
              "behaviour, so the sign was flipped.")
        out["z_imag"] = -out.z_imag
    if "sweep_temp_c" not in out and env is not None:
        out = out.merge(env[["plant_id", "day", "air_temp_c"]].rename(columns={"air_temp_c": "sweep_temp_c"}),
                        on=["plant_id", "day"], how="left")
        print("NOTE: no sweep_temp_c in the EIS file - daily air temperature used for the "
              "temperature correction (rough; log temperature at each sweep).")
    if "sweep_temp_c" in out:
        k = 1 + TEMP_COEF * (out.sweep_temp_c.fillna(T_REF) - T_REF)     # R25 = R_T (1 + a (T - 25))
        out["z_real"], out["z_imag"] = out.z_real * k, out.z_imag * k
    out["z_mag"] = np.sqrt(out.z_real ** 2 + out.z_imag ** 2)
    out["phase_deg"] = np.degrees(np.arctan2(out.z_imag, out.z_real))
    return out


def cole_model(p, f):
    """[STANDARD] Z = Rinf + (R0 - Rinf)/(1 + (j w tau)^alpha); p = [log10 R0, log10 Rinf, log10 tau, alpha]."""
    return 10 ** p[1] + (10 ** p[0] - 10 ** p[1]) / (1 + (1j * 2 * np.pi * f * 10 ** p[2]) ** p[3])


# Internally the fit uses q = [log10 R0, log10(R0/Rinf), log10 tau, alpha] so that R0 > Rinf always.
_QLO, _QHI = np.array([0, 0.001, -9, 0.3]), np.array([9, 6, -1, 1.0])


def _q2p(q):
    return [q[0], q[0] - q[1], q[2], q[3]]


def _start(f, z):
    mag = np.abs(z)
    fc0 = f[np.argmax(-z.imag)]
    q0 = np.array([np.log10(mag[0]), np.log10(mag[0] / mag[-1]), np.log10(1 / (2 * np.pi * fc0)), 0.8])
    return np.clip(q0, _QLO + 1e-6, _QHI - 1e-6)


def _solve(resid_p, q0):
    res = least_squares(lambda q: resid_p(_q2p(q)), q0, bounds=(_QLO, _QHI))
    return _q2p(res.x), res


def _result(p, rel_rmse=np.nan):
    R0, Rinf = 10 ** p[0], 10 ** p[1]
    return dict(R0=R0, Rinf=Rinf, Re=R0, Ri=R0 * Rinf / (R0 - Rinf),      # [STANDARD] Fricke circuit
                fc_hz=1 / (2 * np.pi * 10 ** p[2]), alpha=p[3], rel_rmse=rel_rmse)


def electrode_cpe(f, logQ, n):
    """Series electrode-polarisation constant-phase element Z = 1/(Q (j w)^n)."""
    return 1 / (10 ** logQ * (1j * 2 * np.pi * f) ** n)


def fit_curve(fit, f):
    """Model spectrum for a fit_cole result (adds the electrode CPE if it was fitted)."""
    m = cole_model([np.log10(fit["R0"]), np.log10(fit["Rinf"]), np.log10(1 / (2 * np.pi * fit["fc_hz"])), fit["alpha"]], f)
    if np.isfinite(fit.get("cpe_logQ", np.nan)):
        m = m + electrode_cpe(f, fit["cpe_logQ"], fit["cpe_n"])
    return m


def fit_cole(f, zr, zi, cpe=None):
    """Least squares on relative residuals of Z' and Z''. Goodness of fit = rel_rmse.
    cpe=True adds a series electrode CPE (2 extra parameters) so electrode polarisation is
    not mistaken for tissue Re."""
    cpe = ELECTRODE_CPE if cpe is None else cpe
    z = zr + 1j * zi
    mag = np.abs(z)
    if not cpe:
        def resid(p):
            m = cole_model(p, f)
            return np.concatenate([(z.real - m.real) / mag, (z.imag - m.imag) / mag])
        p, res = _solve(resid, _start(f, z))
        return _result(p, float(np.sqrt(np.mean(res.fun ** 2))))
    lo, hi = np.concatenate([_QLO, [-12, 0.3]]), np.concatenate([_QHI, [-3, 1.0]])
    k0 = len(f) // 5                                    # tissue peak search above the polarisation tail
    fc0 = f[k0 + np.argmax(-zi[k0:])]
    q0 = _start(f, z)
    q0[0] = np.log10(np.abs(z[k0]))
    q0[2] = np.log10(1 / (2 * np.pi * fc0))
    logQ0 = -np.log10(0.3 * mag[0] * (2 * np.pi * f[0]) ** 0.8)
    q0 = np.clip(np.concatenate([q0, [logQ0, 0.8]]), lo + 1e-6, hi - 1e-6)

    def resid_q(q):
        m = cole_model(_q2p(q[:4]), f) + electrode_cpe(f, q[4], q[5])
        return np.concatenate([(z.real - m.real) / mag, (z.imag - m.imag) / mag])
    res = least_squares(resid_q, q0, bounds=(lo, hi))
    out = _result(_q2p(res.x[:4]), float(np.sqrt(np.mean(res.fun ** 2))))
    out.update(cpe_logQ=float(res.x[4]), cpe_n=float(res.x[5]))
    return out


def one_sweep(raw, plant, day):
    """Rows of one sweep; if two electrode sessions share the day, the later session."""
    g = raw[(raw.plant_id == plant) & (raw.day == day)]
    if "electrode_session" in g and g.electrode_session.nunique() > 1:
        g = g[g.electrode_session == g.electrode_session.max()]
    return g.sort_values("freq_hz")


def extract_eis_features(eis):
    """One Cole fit per plant per day. fit_ok is False if the fit is poor or the -Z'' peak
    (the dispersion) is not inside the measured frequency range."""
    rows = []
    eis = eis.copy()
    if "electrode_session" not in eis:
        eis["electrode_session"] = 1
    for (pid, day, sess), g in eis.groupby(["plant_id", "day", "electrode_session"]):
        g = g.sort_values("freq_hz")
        if len(g) < 5:
            continue
        try:
            f = g.freq_hz.to_numpy()
            r = fit_cole(f, g.z_real.to_numpy(), g.z_imag.to_numpy())
        except Exception:
            continue
        r.update(plant_id=pid, day=day, electrode_session=int(sess), phase_min_deg=g.phase_deg.min(),
                 peak_inside=bool(f.min() < r["fc_hz"] < f.max()))
        if "sweep_temp_c" in g and g.sweep_temp_c.notna().any():
            # tau = Cm (Re + Ri) and R falls with temperature, so fc rises: refer fc to 25 C as well
            r["fc_hz"] = r["fc_hz"] / (1 + TEMP_COEF * (g.sweep_temp_c.iloc[0] - T_REF))
        rows.append(r)
    feats = pd.DataFrame(rows)
    feats["fit_ok"] = (feats.rel_rmse <= MAX_FIT_RMSE) & feats.peak_inside
    if "cpe_logQ" in feats:
        # the electrode CPE can absorb a slipped-clip artefact, so the fit error stays low;
        # but the electrode term should be stable within one clip placement -> robust outlier test
        def robust_z(x):
            mad = 1.4826 * np.median(np.abs(x - np.median(x)))
            return np.abs(x - np.median(x)) / max(mad, 1e-9)
        grp = feats.groupby(["plant_id", "electrode_session"])
        z = np.maximum(grp.cpe_logQ.transform(robust_z), grp.cpe_n.transform(robust_z))
        feats["electrode_outlier"] = z > 5
        feats["fit_ok"] &= ~feats.electrode_outlier
        print(f"Electrode CPE: {feats.electrode_outlier.sum()} sweep(s) rejected as electrode outliers "
              "(sudden change in the polarisation term = clip moved/slipped)")
    return feats


def eis_fit_mc(f, zr, zi, n=300, rel_noise=0.01, seed=SEED):
    """[PROJECT] parametric bootstrap of EIS fit uncertainty (assumed rel_noise x |Z|)."""
    rng = np.random.default_rng(seed)
    fit = fit_cole(f, zr, zi)
    p = [np.log10(fit["R0"]), np.log10(fit["Rinf"]), np.log10(1 / (2 * np.pi * fit["fc_hz"])), fit["alpha"]]
    z0 = cole_model(p, f)
    names = ["Re", "Ri", "fc_hz", "alpha"]
    out = []
    for _ in range(n):
        zn = z0 + np.abs(z0) * rel_noise * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
        r = fit_cole(f, zn.real, zn.imag)
        out.append([r[c] for c in names])
    out = np.array(out)
    return pd.DataFrame(dict(parameter=names, mean=out.mean(0), sd=out.std(0, ddof=1),
                             rel_sd=out.std(0, ddof=1) / np.abs(out.mean(0))))


def frequency_count_study(f, zr, zi, counts=(5, 10, 25), n=300, rel_noise=0.01):
    rows = []
    for k in counts:
        idx = np.unique(np.round(np.linspace(0, len(f) - 1, k)).astype(int))
        m = eis_fit_mc(f[idx], zr[idx], zi[idx], n, rel_noise)
        m.insert(0, "n_freq", len(idx))
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def bin_spectrum(f, z, k):
    edges = np.linspace(np.log10(f.min()), np.log10(f.max()), k + 1)
    idx = np.clip(np.digitize(np.log10(f), edges[1:-1]), 0, k - 1)
    keep = [i for i in range(k) if (idx == i).any()]
    fb = np.array([10 ** np.mean(np.log10(f[idx == i])) for i in keep])
    zb = np.array([z[idx == i].mean() for i in keep])
    nb = np.array([(idx == i).sum() for i in keep])
    return fb, zb, nb, idx, keep


def fit_cole_binned(f, zr, zi, k, consistent=True):
    z = zr + 1j * zi
    fb, zb, nb, idx, keep = bin_spectrum(f, z, k)
    mag = np.abs(zb)

    def model_b(p):
        if consistent:
            m = cole_model(p, f)
            return np.array([m[idx == i].mean() for i in keep])
        return cole_model(p, fb)

    def resid(p):
        m = model_b(p)
        w = np.sqrt(nb) / mag
        return np.concatenate([(zb.real - m.real) * w, (zb.imag - m.imag) * w])
    p, _ = _solve(resid, _start(fb, zb))
    return _result(p)


def compression_study(f, zr, zi, ks=(4, 6, 8), n=200, rel_noise=0.01, seed=SEED):
    rng = np.random.default_rng(seed)
    ref = fit_cole(f, zr, zi)
    keys = ["Re", "Ri", "fc_hz", "alpha"]
    truth = np.array([ref[c] for c in keys])
    p = [np.log10(ref["R0"]), np.log10(ref["Rinf"]), np.log10(1 / (2 * np.pi * ref["fc_hz"])), ref["alpha"]]
    z0 = cole_model(p, f)
    spectra = [z0 + np.abs(z0) * rel_noise * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
               for _ in range(n)]
    methods = {"all frequencies (reference)": (len(f), lambda zn: fit_cole(f, zn.real, zn.imag))}
    for k in ks:
        idx = np.unique(np.round(np.linspace(0, len(f) - 1, k)).astype(int))
        methods[f"{k} pts: subsample"] = (k, lambda zn, i=idx: fit_cole(f[i], zn.real[i], zn.imag[i]))
        methods[f"{k} pts: bin-average, naive"] = (k, lambda zn, k=k: fit_cole_binned(f, zn.real, zn.imag, k, False))
        methods[f"{k} pts: bin-average, model-consistent"] = (k, lambda zn, k=k: fit_cole_binned(f, zn.real, zn.imag, k, True))
    rows = []
    for name, (npts, fn) in methods.items():
        est = np.array([[fn(zn)[c] for c in keys] for zn in spectra])
        bias, sd = est.mean(0) / truth - 1, est.std(0, ddof=1) / truth
        rows.append(dict(method=name, n_points=npts,
                         **{f"{c}_bias%": 100 * b for c, b in zip(keys, bias)},
                         **{f"{c}_sd%": 100 * v for c, v in zip(keys, sd)},
                         total_rmse_pct=100 * float(np.sqrt(np.mean(bias ** 2 + sd ** 2)))))
    return pd.DataFrame(rows)


# ============================================================ 3. DATASET
def complete_days(env):
    """Put every plant on a full daily calendar so that shifting by rows = shifting by days."""
    parts, n_missing = [], 0
    for pid, g in env.groupby("plant_id"):
        g = g.sort_values("day").set_index("day")
        full = pd.Index(np.arange(int(g.index.min()), int(g.index.max()) + 1), name="day")
        n_missing += len(full) - len(g)
        g = g.reindex(full)
        g["plant_id"] = pid
        parts.append(g.reset_index())
    if n_missing:
        print(f"NOTE: {n_missing} missing plant-days filled with NaN (growth over those gaps is not computed).")
    return pd.concat(parts, ignore_index=True)


def prepare_dataset(env, eis_feats=None, growth_col="height_cm", baseline_days=7):
    """Growth rate per day over HORIZON days. Features use data up to day t; targets look
    HORIZON days ahead. EIS features are relative to each plant's first `baseline_days`
    days (ASSUMPTION: those days are unstressed)."""
    df = complete_days(env)
    g = df.groupby("plant_id")[growth_col]
    df["growth_past3"] = (df[growth_col] - g.shift(HORIZON)) / HORIZON
    df["growth_next3"] = (g.shift(-HORIZON) - df[growth_col]) / HORIZON
    if "leaf_count" in df and df.leaf_count.notna().any():
        lc = df.groupby("plant_id")["leaf_count"]
        df["leaf_rate_past3"] = (df.leaf_count - lc.shift(HORIZON)) / HORIZON
    df["age_days"] = df.day - df.groupby("plant_id")["day"].transform("min")
    if "stress_truth" in df and df.stress_truth.notna().any():
        s = df.groupby("plant_id")["stress_truth"]
        fut = pd.concat([s.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
        df["stress_next3"] = fut.max(axis=1, skipna=False)
    if eis_feats is not None and eis_feats.fit_ok.any():
        e = link_session_baselines(eis_feats[eis_feats.fit_ok].copy(), baseline_days)
        e["log10_fc"] = np.log10(e.fc_hz)
        keep = ["plant_id", "day", "Re", "Ri", "Re_rel", "Ri_rel", "log10_fc", "alpha", "phase_min_deg", "rel_rmse"]
        df = df.merge(e[keep], on=["plant_id", "day"], how="left")
    return df


def link_session_baselines(e, baseline_days=7):
    """Re_rel, Ri_rel relative to each plant's unstressed baseline. Session 1 baseline = median
    of the first `baseline_days` days. When the clip moves (new electrode_session), the new
    baseline is set so Re_rel is continuous on the OVERLAP day where both leaves were swept
    (ASSUMPTION: same physiology at the same moment). Without an overlap day the adjacent days
    are used and a warning is printed (a stress at the switch would then be carried over)."""
    out, warned = [], 0
    for pid, g in e.groupby("plant_id"):
        g = g.sort_values(["day", "electrode_session"]).copy()
        sessions = sorted(g.electrode_session.unique())
        first = g[g.electrode_session == sessions[0]]
        early = first[first.day < first.day.min() + baseline_days]
        base = {sessions[0]: (early.Re.median(), early.Ri.median())}
        for prev, cur in zip(sessions[:-1], sessions[1:]):
            gp, gc = g[g.electrode_session == prev], g[g.electrode_session == cur]
            both = sorted(set(gp.day) & set(gc.day))
            if both:
                dp = dc = both[-1]
            else:
                warned += 1
                dp, dc = gp.day.max(), gc.day.min()
            rp, rc = gp[gp.day == dp].iloc[0], gc[gc.day == dc].iloc[0]
            re_rel_p, ri_rel_p = rp.Re / base[prev][0], rp.Ri / base[prev][1]
            base[cur] = (rc.Re / re_rel_p, rc.Ri / ri_rel_p)
        g["Re_base"] = g.electrode_session.map(lambda k: base[k][0])
        g["Ri_base"] = g.electrode_session.map(lambda k: base[k][1])
        g = g.sort_values("electrode_session").drop_duplicates("day", keep="last")   # keep newest leaf
        out.append(g)
    if warned:
        print(f"WARNING: {warned} electrode re-clip(s) without an overlap day - baselines linked on "
              "adjacent days; sweep both leaves on the re-clip day.")
    e = pd.concat(out)
    e["Re_rel"], e["Ri_rel"] = e.Re / e.Re_base, e.Ri / e.Ri_base
    return e


def env_signal_flags(df):
    t, no = THRESH, pd.Series(False, index=df.index)
    return dict(low_sm=df.soil_moisture_pct < t["sm_low"], high_t=df.air_temp_c > t["temp_high"],
                high_sm=df.soil_moisture_pct > t["sm_high"],
                high_ec=(df.ec_ds_m > t["ec_high"]) if "ec_ds_m" in df else no,
                low_ec=(df.ec_ds_m < t.get("ec_low", 0.0)) if "ec_ds_m" in df else no)


def add_status_columns(df, ref_model, env_cols):
    """Growth deficit = 1 - actual/expected [PROJECT-DEFINED]. Rule-based category uses
    several signals; EIS alone never decides."""
    X = df[env_cols]                              # = the expected-growth model's inputs
    ok = X.notna().all(axis=1)
    exp = pd.Series(np.nan, index=df.index)
    exp[ok] = ref_model.predict(X[ok].to_numpy())
    floor = max(0.05 * np.nanmedian(df.growth_past3), 1e-6)
    df["expected_growth"] = exp.clip(lower=floor)
    df["growth_deficit"] = 1 - df.growth_past3 / df.expected_growth
    f, t = env_signal_flags(df), THRESH
    # a growth drop must exceed both the deficit fraction AND 2 sigma of the rate's measurement noise
    noise = 2 * SENSOR_SIGMA.get("growth_past3", ("abs", 0.0))[1]
    shortfall = df.expected_growth - df.growth_past3
    drop = (shortfall > t["deficit"] * df.expected_growth) & (shortfall > noise)
    eis_resp = ((df["Re_rel"] - 1).abs() > EIS_DEV) if "Re_rel" in df else pd.Series(False, index=df.index)
    responds = drop | eis_resp
    insuff = df[["air_temp_c", "soil_moisture_pct", "growth_deficit"]].isna().any(axis=1)
    any_env = f["low_sm"] | f["high_t"] | f["high_sm"] | f["high_ec"] | f["low_ec"]
    kinds = [("low_sm", "WATER"), ("high_t", "HEAT"), ("high_sm", "EXCESS-MOISTURE"), ("high_ec", "SALINITY"),
             ("low_ec", "NUTRIENT-DEFICIENCY")]
    labels = []
    for i in df.index:                            # every matching stress is listed, not just the first
        if insuff[i]:
            labels.append("INSUFFICIENT DATA")
            continue
        flagged = [n for k, n in kinds if f[k][i]]
        if flagged and responds[i]:
            labels.append("POSSIBLE " + " + ".join(flagged) + " STRESS")
        elif flagged:
            labels.append(" + ".join(flagged) + " CONDITIONS - no plant response yet")
        elif eis_resp[i]:
            labels.append("ANOMALOUS PHYSIOLOGICAL RESPONSE")
        elif drop[i]:
            labels.append("UNEXPLAINED GROWTH DROP")
        else:
            labels.append("NORMAL")
    df["growth_drop"], df["eis_response"] = drop, eis_resp
    df["stress_category"] = labels
    return df


# ============================================================ 4. MODELS (A vs B)
class PersistenceResidual:
    """Predict growth_next3 = growth_past3 + f(X): the model learns the CHANGE from the last
    3 days, so 'no information' falls back to the persistence baseline instead of below it."""
    def __init__(self, inner, base_idx):
        self.inner, self.base_idx = inner, base_idx

    def fit(self, X, y):
        self.inner.fit(X, y - X[:, self.base_idx])
        return self

    def predict(self, X):
        return X[:, self.base_idx] + self.inner.predict(X)


def make_reg(kind, base_idx=None):
    inner = (make_pipeline(StandardScaler(), LinearRegression()) if kind == "Linear"
             else RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1))
    return inner if base_idx is None else PersistenceResidual(inner, base_idx)


def make_clf(kind):
    if kind == "Linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1)


def reg_metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=np.sqrt(mean_squared_error(y, p)), R2=r2_score(y, p))


def clf_metrics(y, pr):
    yh = (pr >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, pr)
    except ValueError:
        auc = np.nan
    return dict(Accuracy=accuracy_score(y, yh), Precision=precision_score(y, yh, zero_division=0),
                Recall=recall_score(y, yh, zero_division=0), F1=f1_score(y, yh, zero_division=0), ROC_AUC=auc)


def conformal_q(kind, X, y, groups, level=CONF_LEVEL, days=None, base_idx=None):
    """[PROJECT] split-conformal half-width from plant-wise out-of-fold residuals on the
    training rows. With `days` (plant+time mode) the calibration also mimics the time shift:
    inner models train on early days of some plants and are scored on LATER days of other
    plants. Its real coverage is still measured on the test set."""
    k = min(3, len(np.unique(groups)))
    if k < 2:
        return np.nan
    res = []
    t_cal = np.quantile(days, 0.6) if days is not None else None
    for tr, te in GroupKFold(k).split(X, y, groups):
        if t_cal is not None:
            tr = tr[days[tr] + HORIZON < t_cal]
            te = te[days[te] >= t_cal]
            if len(tr) < 10 or len(te) == 0:
                continue
        m = make_reg(kind, base_idx).fit(X[tr], y[tr])
        res.append(np.abs(y[te] - m.predict(X[te])))
    if not res:
        return np.nan
    r = np.concatenate(res)
    return float(np.quantile(r, min(1.0, np.ceil((len(r) + 1) * level) / len(r))))


def run_experiment(df, split_day, sets, n_folds=6, split_mode="plant"):
    """Model A: environment + growth.  Model B: A + EIS.  Same rows for both.
    split_mode="plant"      : each fold trains on some plants (all days) and tests on OTHER
                              plants (all days). Right for a single transplant cohort.
    split_mode="plant+time" : train on some plants' days before split_day, test on OTHER plants'
                              days from split_day. Use only with staggered cohorts, otherwise
                              train and test are different growth stages."""
    timed = split_mode == "plant+time"
    longest = max(sets.values(), key=len)
    has_cls = "stress_next3" in df
    lab = df.dropna(subset=longest + ["growth_next3"] + (["stress_next3"] if has_cls else []))
    plants = np.array(sorted(lab.plant_id.unique()))
    k = min(n_folds, len(plants))
    if k < 2:
        raise ValueError("plant-wise evaluation needs at least 2 plants with complete data")
    tests, fold_rows = [], []
    for fold, (tr_i, te_i) in enumerate(GroupKFold(k).split(plants, groups=plants)):
        train = lab[lab.plant_id.isin(plants[tr_i]) & ((lab.day + HORIZON < split_day) | (not timed))]
        test = lab[lab.plant_id.isin(plants[te_i]) & ((lab.day >= split_day) | (not timed))].copy()
        if len(train) < 10 or test.empty:
            continue
        test["fold"] = fold
        test["pred_Persistence"] = test.growth_past3
        fold_rows.append(dict(fold=fold, task=GTASK, model="Persistence baseline", inputs="-",
                              **reg_metrics(test.growth_next3, test.pred_Persistence)))
        ytr, gtr = train.growth_next3.to_numpy(), train.plant_id.to_numpy()
        for key, feats in sets.items():
            Xtr, Xte = train[feats].to_numpy(float), test[feats].to_numpy(float)
            bi = feats.index("growth_past3")
            for kind in ("Linear", "RandomForest"):
                p = make_reg(kind, bi).fit(Xtr, ytr).predict(Xte)
                q = conformal_q(kind, Xtr, ytr, gtr, days=train.day.to_numpy() if timed else None, base_idx=bi)
                test[f"pred_{kind}_{key}"], test[f"lo_{kind}_{key}"], test[f"hi_{kind}_{key}"] = p, p - q, p + q
                fold_rows.append(dict(fold=fold, task=GTASK, model=kind, inputs=key, **reg_metrics(test.growth_next3, p)))
                if has_cls and train.stress_next3.nunique() > 1:
                    pr = make_clf(kind).fit(Xtr, train.stress_next3.astype(int)).predict_proba(Xte)[:, 1]
                    test[f"prob_{kind}_{key}"] = pr
                    fold_rows.append(dict(fold=fold, task=STASK, model=kind, inputs=key,
                                          **clf_metrics(test.stress_next3.astype(int), pr)))
        tests.append(test)
    if not tests:
        raise ValueError("no fold had both training and test rows - check split_day and data length")
    test = pd.concat(tests)
    rows = [dict(task=GTASK, model="Persistence baseline", inputs="-",
                 **reg_metrics(test.growth_next3, test.pred_Persistence))]
    for key in sets:
        for kind in ("Linear", "RandomForest"):
            c = f"pred_{kind}_{key}"
            lo, hi = test[f"lo_{kind}_{key}"], test[f"hi_{kind}_{key}"]
            rows.append(dict(task=GTASK, model=kind, inputs=key, **reg_metrics(test.growth_next3, test[c]),
                             PI90_coverage=float(((test.growth_next3 >= lo) & (test.growth_next3 <= hi)).mean()),
                             PI90_width=float((hi - lo).mean())))
            pc = f"prob_{kind}_{key}"
            if pc in test and test[pc].notna().any():
                t = test.dropna(subset=[pc])
                rows.append(dict(task=STASK, model=kind, inputs=key, **clf_metrics(t.stress_next3.astype(int), t[pc])))
    # final (deployed) models: evaluation is done, so they learn from ALL labelled rows of all
    # plants; this stops the status table extrapolating beyond the training period
    train_all = lab
    models, confq = {}, {}
    y, gr = train_all.growth_next3.to_numpy(), train_all.plant_id.to_numpy()
    for key, feats in sets.items():
        X = train_all[feats].to_numpy(float)
        bi = feats.index("growth_past3")
        for kind in ("Linear", "RandomForest"):
            models[("reg", kind, key)] = make_reg(kind, bi).fit(X, y)
            confq[(kind, key)] = conformal_q(kind, X, y, gr, days=train_all.day.to_numpy() if timed else None,
                                             base_idx=bi)
            if has_cls and train_all.stress_next3.nunique() > 1:
                models[("clf", kind, key)] = make_clf(kind).fit(X, train_all.stress_next3.astype(int))
    return dict(sets=sets, train=train_all, test=test, models=models, confq=confq,
                metrics=pd.DataFrame(rows), fold_metrics=pd.DataFrame(fold_rows), split_day=split_day, n_folds=k,
                split_mode=split_mode)


def apply_label_source(env, source):
    """Choose the stress label. wilting: blind score >= 2; either: stress_truth OR wilting >= 2.
    Prints Cohen's kappa between the two when both exist."""
    agree = None
    if "wilting_score" in env and "stress_truth" in env:
        ok = env[["wilting_score", "stress_truth"]].dropna()
        if len(ok) and ok.stress_truth.nunique() > 1:
            agree = cohen_kappa_score(ok.stress_truth.astype(int), (ok.wilting_score >= 2).astype(int))
            print(f"Label agreement: Cohen's kappa(stress_truth, wilting>=2) = {agree:.2f} "
                  "(<0.4 poor, 0.4-0.6 moderate, >0.6 good)")
            if "treatment" in env:
                for grp, g in env.dropna(subset=["wilting_score", "stress_truth"]).groupby("treatment"):
                    if g.stress_truth.nunique() > 1:
                        k = cohen_kappa_score(g.stress_truth.astype(int), (g.wilting_score >= 2).astype(int))
                        print(f"   {grp:10s} kappa = {k:.2f}")
    if source == "wilting":
        if "wilting_score" not in env:
            raise ValueError("--label-source wilting needs a wilting_score column")
        env["stress_truth"] = (env.wilting_score >= 2).astype(float).where(env.wilting_score.notna())
    elif source == "either" and "wilting_score" in env and "stress_truth" in env:
        env["stress_truth"] = ((env.stress_truth == 1) | (env.wilting_score >= 2)).astype(float)
    return env, agree


def correlation_study(df, n_boot=2000, seed=SEED):
    """Run _correlations for all plants and, if a 'treatment' column exists, for each group
    (a response that only appears in stressed plants is diluted when controls are pooled in)."""
    out = [_correlations(df, n_boot, seed).assign(group="all plants")]
    if "treatment" in df and df.treatment.notna().any():
        for grp, g in df.groupby("treatment"):
            out.append(_correlations(g, n_boot, seed).assign(group=grp))
    out = pd.concat(out, ignore_index=True)
    return out[["group"] + [c for c in out.columns if c != "group"]] if len(out) else out


def _correlations(df, n_boot=2000, seed=SEED):
    """[PROJECT] Objective 3. For each EIS feature x environment/growth variable pair:
    pooled Spearman rho, and the mean WITHIN-PLANT Spearman rho with a 95 % CI from
    resampling plants (days within a plant are not independent, so plain p-values are
    over-confident). age_adj_rho is the partial Spearman controlling for plant age, so a
    shared time trend (leaf ageing, EC ramp) is not mistaken for a physiological link. With 3
    plants per group the bootstrap CI is rough. lag 0 = same day; 'growth_next3' = EIS today vs growth over the next 3 days."""
    eis_f = [c for c in ["Re_rel", "Ri_rel", "log10_fc", "alpha"] if c in df]
    others = [c for c in ["soil_moisture_pct", "air_temp_c", "ec_ds_m", "growth_past3", "growth_next3",
                          "growth_deficit", "wilting_score"] if c in df]
    rng = np.random.default_rng(seed)
    rows = []
    for a in eis_f:
        for b in others:
            d = df[["plant_id", "age_days", a, b]].dropna()
            if len(d) < 10:
                continue
            per, per_adj = [], []
            for _, g in d.groupby("plant_id"):
                if len(g) >= 6 and g[a].nunique() > 1 and g[b].nunique() > 1:
                    per.append(spearmanr(g[a], g[b])[0])
                    # partial Spearman controlling for plant age (both drift with time)
                    r = g[[a, b, "age_days"]].rank().to_numpy()
                    A = np.c_[np.ones(len(r)), r[:, 2]]
                    ra = r[:, 0] - A @ np.linalg.lstsq(A, r[:, 0], rcond=None)[0]
                    rb = r[:, 1] - A @ np.linalg.lstsq(A, r[:, 1], rcond=None)[0]
                    per_adj.append(np.corrcoef(ra, rb)[0, 1] if ra.std() > 0 and rb.std() > 0 else np.nan)
            keep = np.isfinite(per) & np.isfinite(per_adj)
            per, per_adj = np.array(per)[keep], np.array(per_adj)[keep]
            if len(per) < 2:
                continue
            idx = [rng.integers(0, len(per), len(per)) for _ in range(n_boot)]
            boot, boot_adj = [per[i].mean() for i in idx], [per_adj[i].mean() for i in idx]
            rows.append(dict(eis_feature=a, variable=b, n_rows=len(d), n_plants=len(per),
                             pooled_rho=spearmanr(d[a], d[b])[0], within_plant_rho=per.mean(),
                             ci95_lo=np.percentile(boot, 2.5), ci95_hi=np.percentile(boot, 97.5),
                             age_adj_rho=per_adj.mean(), age_adj_ci95_lo=np.percentile(boot_adj, 2.5),
                             age_adj_ci95_hi=np.percentile(boot_adj, 97.5)))
    out = pd.DataFrame(rows)
    if len(out):
        out["ci_excludes_0"] = (out.ci95_lo > 0) | (out.ci95_hi < 0)
        out["age_adj_ci_excludes_0"] = (out.age_adj_ci95_lo > 0) | (out.age_adj_ci95_hi < 0)
    return out


def build_context(env, eis_raw=None, simulated=True, split_day=None, growth_col="height_cm", n_folds=6,
                  split_mode="plant+time", primary="Linear", label_source="stress_truth"):
    env = env.copy()
    env["day"] = env.day.astype(int)
    env, kappa = apply_label_source(env, label_source)
    env_cols = ENV_REQUIRED + [c for c in ENV_OPTIONAL if c in env and env[c].notna().any()]
    eis_std = standardise_eis(eis_raw, env) if eis_raw is not None else None
    eis_feats = extract_eis_features(eis_std) if eis_std is not None else None
    df = prepare_dataset(env, eis_feats, growth_col)
    # distance outside the crop's healthy band (0 inside it): lets even a linear model represent
    # "too high OR too low" (e.g. salinity vs nutrient starvation) using species thresholds
    t = THRESH
    df["sm_deficit"] = (t["sm_low"] - df.soil_moisture_pct).clip(lower=0)
    df["sm_excess"] = (df.soil_moisture_pct - t["sm_high"]).clip(lower=0)
    df["heat_excess"] = (df.air_temp_c - t["temp_high"]).clip(lower=0)
    stress_cols = ["sm_deficit", "sm_excess", "heat_excess"]
    if "ec_ds_m" in env_cols:
        df["ec_excess"] = (df.ec_ds_m - t["ec_high"]).clip(lower=0)
        stress_cols.append("ec_excess")
        if t.get("ec_low", 0) > 0:
            df["ec_deficit"] = (t["ec_low"] - df.ec_ds_m).clip(lower=0)
            stress_cols.append("ec_deficit")
    growth_cols = [growth_col, "growth_past3", "age_days"]        # current size: growth depends on stage
    if "leaf_rate_past3" in df:
        growth_cols += ["leaf_count", "leaf_rate_past3"]
    sets = {"A": env_cols + stress_cols + growth_cols}
    if eis_feats is not None and all(c in df for c in EIS_COLS) and df[EIS_COLS].notna().all(axis=1).any():
        sets["B"] = sets["A"] + EIS_COLS
    d0, d1 = int(df.day.min()), int(df.day.max())
    split_day = split_day if split_day is not None else d0 + int(0.65 * (d1 - d0)) + 1
    if split_mode == "plant":
        split_day = d1 + HORIZON + 1          # no time cut: every day counts as "training period"
    if THRESH["growth_low"] is None:
        THRESH["growth_low"] = float(df[df.day < split_day].growth_past3.quantile(0.2))
        print(f"NOTE: growth_low set to the 20th percentile of training growth = {THRESH['growth_low']:.3f} per day")
    f = env_signal_flags(df)
    quiet = ~(f["low_sm"] | f["high_t"] | f["high_sm"] | f["high_ec"] | f["low_ec"])
    df["size_sq"] = df[growth_col] ** 2          # growth slows near maturity (logistic hump)
    ref_x = env_cols + ["age_days", growth_col, "size_sq"]
    ref_rows = df[quiet & (df.day < split_day)].dropna(subset=ref_x + ["growth_past3"])
    if len(ref_rows) < 10:
        print("NOTE: too few no-stress-signal rows for the expected-growth model; using all training rows.")
        ref_rows = df[df.day < split_day].dropna(subset=ref_x + ["growth_past3"])
    ref = LinearRegression().fit(ref_rows[ref_x].to_numpy(), ref_rows.growth_past3.to_numpy())
    df = add_status_columns(df, ref, ref_x)
    ctx = run_experiment(df, split_day, sets, n_folds, split_mode)
    if primary == "auto":
        m = ctx["metrics"]
        g = m[(m.task == GTASK) & (m.inputs == "A")].set_index("model").MAE
        primary = g[["Linear", "RandomForest"]].idxmin()
        print(f"--primary auto: {primary} chosen (held-out growth MAE: Linear {g['Linear']:.3f}, "
              f"RandomForest {g['RandomForest']:.3f})")
    ctx.update(df=df, eis_raw=eis_std, eis_feats=eis_feats, simulated=simulated, growth_col=growth_col,
               env_cols=env_cols, label="SIMULATED DATA" if simulated else "MEASURED DATA", primary=primary,
               kappa=kappa, label_source=label_source,
               correlations=correlation_study(df) if "Re_rel" in df else pd.DataFrame())
    return ctx


# ============================================================ 5. SENSITIVITY / UNCERTAINTY / MONTE CARLO
def local_sensitivity(predict, x0, step):
    """Central difference S_i = dG/dX_i (secant for random forests)."""
    S = np.zeros(len(x0))
    for i in range(len(x0)):
        if step[i] == 0:
            continue
        up, dn = x0.copy(), x0.copy()
        up[i] += step[i]
        dn[i] -= step[i]
        S[i] = (predict(up[None, :])[0] - predict(dn[None, :])[0]) / (2 * step[i])
    return S


def sigma_vector(feats, x0):
    out = []
    for f, x in zip(feats, x0):
        kind, v = SENSOR_SIGMA.get(f, ("abs", 0.0))
        out.append(v * abs(x) if kind == "rel" else v)
    return np.array(out)


def monte_carlo(reg, clf, feats, x0, sig, n=5000, threshold=None, seed=SEED):
    """Propagates INPUT measurement uncertainty only. Probabilities are MODELLED."""
    threshold = THRESH["growth_low"] if threshold is None else threshold
    rng = np.random.default_rng(seed)
    X = x0 + rng.normal(0, 1, (n, len(x0))) * sig
    for j, f in enumerate(feats):
        lo, hi = CLIP.get(f, (None, None))
        X[:, j] = np.clip(X[:, j], lo, hi)
    pred = reg.predict(X)
    prob = clf.predict_proba(X)[:, 1] if clf is not None else None
    return dict(pred=pred, prob=prob, mean=pred.mean(), sd=pred.std(ddof=1),
                lo95=np.percentile(pred, 2.5), hi95=np.percentile(pred, 97.5),
                p_below=float((pred < threshold).mean()),
                p_stress=float(prob.mean()) if prob is not None else np.nan)


def pick_point(ctx, plant, day):
    """Return a (plant, day) that actually has all features; nearest valid day if not."""
    feats = ctx["sets"]["B" if "B" in ctx["sets"] else "A"]
    ok = ctx["df"].dropna(subset=feats)
    if plant is None or plant not in set(ok.plant_id):
        new = "P3" if "P3" in set(ok.plant_id) else ok.plant_id.iloc[0]
        if plant is not None:
            print(f"NOTE: plant {plant} has no complete rows - using {new}")
        plant = new
    days = ok[ok.plant_id == plant].day.to_numpy()
    target = day if day is not None else (ctx["split_day"] + 5 if ctx["split_mode"] != "plant" else int(np.median(days)))
    new = int(days[np.argmin(np.abs(days - target))])
    if day is not None and new != day:
        print(f"NOTE: {plant} has no complete data on day {day} - using nearest day {new}")
    return plant, new


def analyse_point(ctx, plant, day, n_mc=5000):
    df, out = ctx["df"], {}
    row = df[(df.plant_id == plant) & (df.day == day)]
    for key, feats in ctx["sets"].items():
        x0 = row[feats].iloc[0].to_numpy(float)
        reg = ctx["models"][("reg", ctx["primary"], key)]
        clf = ctx["models"].get(("clf", ctx["primary"], key))
        sd = ctx["train"][feats].std().to_numpy()
        pred = lambda X, m=reg: m.predict(X)
        G0 = pred(x0[None, :])[0]
        sig = sigma_vector(feats, x0)
        S = local_sensitivity(pred, x0, 0.5 * sd)
        S_loc = local_sensitivity(pred, x0, np.where(sig > 0, sig, 0.5 * sd))
        sens = pd.DataFrame(dict(feature=feats, S_per_unit=S,
                                 S_normalised=(x0 * S / G0) if abs(G0) > 1e-6 else np.nan,
                                 effect_per_1SD=S * sd))
        sens = sens.reindex(sens.effect_per_1SD.abs().sort_values(ascending=False).index)
        var = (S_loc * sig) ** 2
        budget = pd.DataFrame(dict(feature=feats, sigma=sig, variance_contrib=var,
                                   fraction=var / var.sum() if var.sum() > 0 else 0.0))
        budget = budget.sort_values("fraction", ascending=False)
        q = ctx["confq"][(ctx["primary"], key)]
        out[key] = dict(x0=x0, G0=G0, sens=sens, budget=budget, sigma_G_first_order=float(np.sqrt(var.sum())),
                        pi90=(G0 - q, G0 + q), mc=monte_carlo(reg, clf, feats, x0, sig, n_mc))
    return out


def plant_status(ctx, n_mc=1000):
    key = "B" if "B" in ctx["sets"] else "A"
    feats, rows = ctx["sets"][key], []
    reg = ctx["models"][("reg", ctx["primary"], key)]
    clf = ctx["models"].get(("clf", ctx["primary"], key))
    q = ctx["confq"][(ctx["primary"], key)]
    for pid, g in ctx["df"].groupby("plant_id"):
        g = g.dropna(subset=feats)
        if g.empty:
            continue
        r = g.iloc[-1]
        x0 = r[feats].to_numpy(float)
        mc = monte_carlo(reg, clf, feats, x0, sigma_vector(feats, x0), n_mc)
        G0 = reg.predict(x0[None, :])[0]
        rows.append(dict(plant_id=pid, day=int(r.day), observed_growth=r.growth_past3,
                         growth_drop=bool(r.growth_drop), eis_response=bool(r.eis_response),
                         expected_growth=r.expected_growth, growth_deficit=r.growth_deficit,
                         predicted_next3=G0, pi90_lo=G0 - q, pi90_hi=G0 + q, input_noise_sd=mc["sd"],
                         modelled_stress_prob=mc["p_stress"], category=r.stress_category))
    return pd.DataFrame(rows)


# ============================================================ 6. FIGURES
def _t(ctx, ax, text):
    ax.set_title(f"{ctx['label']} - {text}", fontsize=9)


def fig_timeseries(ctx):
    df, gc = ctx["df"], ctx["growth_col"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for pid, g in df.groupby("plant_id"):
        ax[0, 0].plot(g.day, g[gc], label=pid)
        ax[0, 1].plot(g.day, g.growth_past3, label=pid)
        ax[0, 2].plot(g.day, g.soil_moisture_pct, label=pid)
    ax[1, 0].plot(df.groupby("day").air_temp_c.mean(), color="tab:red")
    for a, (txt, yl) in zip(ax.flat[:4], [(f"1. {gc} vs time", gc), ("2. growth rate (3-day mean) vs time", f"{gc} per day"),
                                          ("3. soil moisture vs time", "%"), ("4. air temperature vs time", "deg C")]):
        _t(ctx, a, txt); a.set_ylabel(yl); a.set_xlabel("day")
    for p, (x, xl) in zip(ax.flat[4:6], [("soil_moisture_pct", "soil moisture (%)"), ("air_temp_c", "air temp (deg C)")]):
        p.scatter(df[x], df.growth_past3, s=6, alpha=0.4)
        p.set_xlabel(xl); p.set_ylabel(f"growth ({gc} per day)")
    _t(ctx, ax[1, 1], "5. growth vs soil moisture (association only)")
    _t(ctx, ax[1, 2], "6. growth vs temperature (association only)")
    ax[0, 0].legend(fontsize=7)
    for a in ax.flat[:4]:
        if ctx["split_mode"] != "plant":
            a.axvline(ctx["split_day"], color="k", ls=":", lw=0.8)
    fig.tight_layout()
    return fig


def fig_pred_actual(ctx):
    test, keys = ctx["test"], list(ctx["sets"])
    fig, ax = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 5), squeeze=False)
    for a, k in zip(ax[0], keys):
        pk = ctx["primary"]
        p = test[f"pred_{pk}_{k}"]
        a.errorbar(test.growth_next3, p, yerr=(p - test[f"lo_{pk}_{k}"]), fmt="o", ms=2,
                   alpha=0.3, elinewidth=0.4)
        lim = [min(0, test.growth_next3.min()), max(test.growth_next3.max(), p.max()) * 1.05]
        a.plot(lim, lim, "k--", lw=0.8)
        m = ctx["metrics"].query("task == @GTASK and model == @pk and inputs == @k").iloc[0]
        a.set_xlabel("actual 3-day growth (per day)"); a.set_ylabel("predicted (bars = 90% conformal PI)")
        _t(ctx, a, f"7. Model {k} ({pk}), held-out plants: MAE={m.MAE:.3f}, R2={m.R2:.2f}, PI cover={m.PI90_coverage:.0%}")
    fig.tight_layout()
    return fig


def fig_sensitivity_uncertainty(ctx, res, plant, day):
    key = "B" if "B" in res else "A"
    r = res[key]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    s = r["sens"].iloc[::-1]
    ax[0].barh(s.feature, s.effect_per_1SD, color=["tab:blue" if v >= 0 else "tab:orange" for v in s.effect_per_1SD])
    ax[0].set_xlabel("change in predicted growth per +1 SD of input")
    _t(ctx, ax[0], f"8. sensitivity ranking, Model {key}, {plant} day {day}")
    b = r["budget"].iloc[::-1]
    ax[1].barh(b.feature, 100 * b.fraction, color="tab:green")
    ax[1].set_xlabel("share of first-order variance (%)")
    _t(ctx, ax[1], f"uncertainty budget (sigma_G = {r['sigma_G_first_order']:.3f})")
    for k, c in zip(res, ["tab:gray", "tab:purple"]):
        ax[2].hist(res[k]["mc"]["pred"], bins=50, alpha=0.55, color=c, label=f"Model {k}: sd={res[k]['mc']['sd']:.3f}")
    ax[2].set_xlabel("predicted 3-day growth (per day)"); ax[2].legend()
    _t(ctx, ax[2], "9. Monte Carlo output (input noise only)")
    fig.tight_layout()
    return fig


def fig_stress_prob(ctx, plant):
    g = ctx["test"][ctx["test"].plant_id == plant].sort_values("day")
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, c in zip(ctx["sets"], ["tab:gray", "tab:purple"]):
        pc = f"prob_{ctx['primary']}_{k}"
        if pc in g:
            ax.plot(g.day, g[pc], color=c, label=f"Model {k} ({ctx['primary']}) modelled probability")
    if "stress_next3" in g:
        ax.fill_between(g.day, 0, g.stress_next3, color="red", alpha=0.15, step="mid", label="stress within 3 d (label)")
    ax.set_ylim(0, 1.02); ax.set_xlabel("day"); ax.legend(fontsize=8)
    _t(ctx, ax, f"10. modelled stress probability, {plant} (held-out plant; not experimentally validated)")
    fig.tight_layout()
    return fig


def fig_plant_comparison(ctx, status):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, (c, t) in zip(ax, [("observed_growth", "growth rate (per day)"), ("growth_deficit", "growth deficit (project-defined)"),
                              ("modelled_stress_prob", "modelled stress probability"), ("input_noise_sd", "prediction sd (input noise)")]):
        a.bar(status.plant_id, status[c]); _t(ctx, a, t); a.tick_params(labelsize=8)
    fig.suptitle("11. plant comparison - no composite 'best plant' score", fontsize=9)
    fig.tight_layout()
    return fig


def fig_eis(ctx, plant, healthy_day, stressed_day):
    raw = ctx["eis_raw"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for day, c in [(healthy_day, "tab:green"), (stressed_day, "tab:red")]:
        g = one_sweep(raw, plant, day)
        if len(g) < 5:
            continue
        f, zr, zi = g.freq_hz.to_numpy(), g.z_real.to_numpy(), g.z_imag.to_numpy()
        fit = fit_cole(f, zr, zi)
        m = fit_curve(fit, f)
        lbl = f"day {day}: Re={fit['Re']:.3g}, Ri={fit['Ri']:.3g} ohm (fit RMSE {fit['rel_rmse']:.3f})"
        ax[0].loglog(f, g.z_mag, "o", ms=3, color=c, label=lbl); ax[0].loglog(f, np.abs(m), "-", color=c)
        ax[1].semilogx(f, g.phase_deg, "o", ms=3, color=c); ax[1].semilogx(f, np.degrees(np.angle(m)), "-", color=c)
        ax[2].plot(zr, -zi, "o", ms=3, color=c); ax[2].plot(m.real, -m.imag, "-", color=c)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("|Z| at 25 C (ohm)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("phase (deg)")
    ax[2].set_xlabel("Z' (ohm)"); ax[2].set_ylabel("-Z'' (ohm)"); ax[2].set_aspect("equal", adjustable="datalim")
    _t(ctx, ax[0], f"12. Bode magnitude, {plant}"); _t(ctx, ax[1], "12. Bode phase (dots=data, line=Cole fit)")
    _t(ctx, ax[2], "13. Nyquist")
    fig.tight_layout()
    return fig


def fig_impedance_vs_stress(ctx):
    df = ctx["df"].dropna(subset=["Re_rel"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    c = df.stress_truth if "stress_truth" in df else None
    ax[0].scatter(df.soil_moisture_pct, df.Re_rel, c=c, cmap="coolwarm", s=8)
    ax[0].set_xlabel("soil moisture (%)"); ax[0].set_ylabel("Re / plant baseline")
    ax[1].scatter(df.growth_deficit, df.Re_rel, c=c, cmap="coolwarm", s=8)
    ax[1].set_xlabel("growth deficit"); ax[1].set_ylabel("Re / plant baseline")
    ax[2].scatter(df.soil_moisture_pct, df.Ri_rel, c=c, cmap="coolwarm", s=8)
    ax[2].set_xlabel("soil moisture (%)"); ax[2].set_ylabel("Ri / plant baseline")
    _t(ctx, ax[0], "14. Re vs soil moisture (colour = stress label)"); _t(ctx, ax[1], "14. Re vs growth deficit")
    _t(ctx, ax[2], "14. Ri vs soil moisture")
    fig.tight_layout()
    return fig


def fig_timeline(ctx, plant):
    g = ctx["df"][ctx["df"].plant_id == plant]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(g.day, g.soil_moisture_pct, label="soil moisture (%)")
    ax[0].plot(g.day, g.air_temp_c, color="tab:red", label="air temp (deg C)"); ax[0].legend(fontsize=8)
    if "Re_rel" in g:
        ax[1].plot(g.day, g.Re_rel, color="tab:purple", label="Re / baseline")
        ax[1].plot(g.day, g.Ri_rel, color="tab:brown", label="Ri / baseline"); ax[1].legend(fontsize=8)
    ax[2].plot(g.day, g.growth_past3, color="tab:green"); ax[2].set_ylabel("growth per day"); ax[2].set_xlabel("day")
    if "stress_truth" in g:
        for a in ax:
            a.fill_between(g.day, 0, 1, where=g.stress_truth == 1, color="red", alpha=0.1, transform=a.get_xaxis_transform())
    _t(ctx, ax[0], f"15. environment + physiology + growth timeline, {plant} (red = stress label)")
    fig.tight_layout()
    return fig


def fig_correlations(ctx):
    c = ctx["correlations"]
    c = c[c.group == "all plants"]
    piv = c.pivot(index="eis_feature", columns="variable", values="age_adj_rho")
    sig = c.pivot(index="eis_feature", columns="variable", values="age_adj_ci_excludes_0").reindex_like(piv)
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 3, 0.8 * piv.shape[0] + 2))
    im = ax.imshow(piv.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}{'*' if sig.iat[i, j] else ''}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean within-plant partial Spearman rho (age-adjusted)")
    _t(ctx, ax, "16. Objective 3 - EIS vs environment & growth, all plants (* = 95% CI excludes 0)")
    fig.tight_layout()
    return fig


def all_figures(ctx, res, status, plant, day):
    figs = {"01_timeseries_and_scatter": fig_timeseries(ctx), "02_pred_vs_actual": fig_pred_actual(ctx),
            "03_sensitivity_uncertainty": fig_sensitivity_uncertainty(ctx, res, plant, day),
            "04_stress_probability": fig_stress_prob(ctx, plant), "05_plant_comparison": fig_plant_comparison(ctx, status),
            "07_timeline": fig_timeline(ctx, plant)}
    if ctx["eis_raw"] is not None:
        edays = sorted(set(ctx["eis_raw"][ctx["eis_raw"].plant_id == plant].day))
        if edays:
            figs["06_eis_bode_nyquist"] = fig_eis(ctx, plant, edays[0], day if day in edays else edays[-1])
        if "Re_rel" in ctx["df"]:
            figs["06b_impedance_vs_stress"] = fig_impedance_vs_stress(ctx)
        if len(ctx["correlations"]):
            figs["08_eis_correlations"] = fig_correlations(ctx)
    return figs


# ============================================================ 7. MAIN
def seed_summary(n_seeds, split_day=None, n_folds=6, split_mode="plant+time"):
    rows = []
    for s in range(n_seeds):
        env, eis = simulate(seed=1000 + s)
        m = build_context(env, eis, True, split_day, "height_cm", n_folds, split_mode)["metrics"]
        m["seed"] = s
        rows.append(m)
    allm = pd.concat(rows)
    cols = [c for c in ["MAE", "RMSE", "R2", "PI90_coverage", "F1", "ROC_AUC"] if c in allm]
    summ = allm.groupby(["task", "model", "inputs"])[cols].agg(["mean", "std"]).round(3)
    wins = allm[(allm.task == GTASK) & (allm.inputs != "-")].pivot_table(index=["seed", "model"], columns="inputs", values="MAE")
    if "B" not in wins:
        return summ, None
    wins["B_better"] = wins["B"] < wins["A"]
    return summ, wins.groupby("model").B_better.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-env"); ap.add_argument("--csv-eis")
    ap.add_argument("--outdir", default="plantsense_out"); ap.add_argument("--seeds", type=int, default=0)
    ap.add_argument("--plant", default=None); ap.add_argument("--day", type=int, default=None)
    ap.add_argument("--split-day", type=int, default=None)
    ap.add_argument("--growth-col", default="height_cm", help="e.g. rosette_diam_cm or leaf_area_cm2 for lettuce")
    ap.add_argument("--crop", choices=list(THRESH_PROFILES), default=None,
                    help="threshold profile (default: generic for simulation, lettuce for real data)")
    ap.add_argument("--growth-threshold", type=float, default=None, help="override 'low growth' threshold (per day)")
    ap.add_argument("--folds", type=int, default=6, help="number of plant-wise folds")
    ap.add_argument("--split", choices=["plant", "plant+time"], default="plant+time",
                    help="plant: hold out whole plants (single cohort); plant+time: also later days (staggered cohorts)")
    ap.add_argument("--eis-mc", action="store_true", help="Monte Carlo of EIS fit uncertainty")
    ap.add_argument("--eis-compress", action="store_true", help="compare <9-point spectrum compression methods")
    ap.add_argument("--calib", default=None, help="calib.json from calibrate.py (sensor sigmas, TEMP_COEF)")
    ap.add_argument("--primary", choices=["auto", "Linear", "RandomForest"], default="auto",
                    help="model for status, sensitivity and figures; auto = lower held-out growth MAE")
    ap.add_argument("--label-source", choices=["stress_truth", "wilting", "either"], default="stress_truth")
    ap.add_argument("--electrode-cpe", action="store_true", help="fit electrode polarisation (2-electrode set-ups)")
    a = ap.parse_args()
    global ELECTRODE_CPE
    ELECTRODE_CPE = a.electrode_cpe
    if a.calib:
        global TEMP_COEF
        with open(a.calib) as fh:
            cal = json.load(fh)
        SENSOR_SIGMA.update({k: tuple(v) for k, v in cal.get("SENSOR_SIGMA", {}).items()})
        TEMP_COEF = cal.get("TEMP_COEF", TEMP_COEF)
        print(f"Calibration loaded from {a.calib}: TEMP_COEF = {TEMP_COEF}, {len(cal.get('SENSOR_SIGMA', {}))} sigmas")
    os.makedirs(a.outdir, exist_ok=True)
    simulated = a.csv_env is None
    THRESH.update(THRESH_PROFILES[a.crop or ("generic" if simulated else "lettuce")])
    if a.growth_threshold is not None:
        THRESH["growth_low"] = a.growth_threshold
    if simulated:
        env, eis = simulate()
        print("*** SIMULATED DATA - results test the pipeline, NOT real plants ***")
    else:
        env, eis = load_real(a.csv_env, a.csv_eis, a.growth_col)
        if "data_source" in env and env.data_source.astype(str).str.upper().str.startswith("SIM").any():
            print("*** CSV rows are tagged data_source=SIMULATED - all outputs are labelled SIMULATED ***")
            simulated = True
    ctx = build_context(env, eis, simulated, a.split_day, a.growth_col, a.folds, a.split, a.primary, a.label_source)
    pd.set_option("display.width", 220)
    if ctx["split_mode"] == "plant":
        print(f"\nEvaluation: {ctx['n_folds']} plant-wise folds; whole plants held out, all days "
              f"(pooled test n={len(ctx['test'])})")
    else:
        print(f"\nEvaluation: {ctx['n_folds']} plant-wise folds; train days < {ctx['split_day']}, test days >= "
              f"{ctx['split_day']} on plants NOT used for training (pooled test n={len(ctx['test'])})")
    print("Feature sets:", {k: len(v) for k, v in ctx["sets"].items()}, "| env inputs used:", ctx["env_cols"])
    print("\nMODEL A (environment+growth) vs MODEL B (+EIS), pooled over held-out plants:")
    print(ctx["metrics"].round(3).to_string(index=False))
    fm = ctx["fold_metrics"]
    print("\nFold-to-fold spread (std across folds) - differences smaller than this are not meaningful:")
    print(fm.groupby(["task", "model", "inputs"])[[c for c in ["MAE", "R2", "F1"] if c in fm]].std().round(3).to_string())
    print(f"\nPrimary model for status/sensitivity/figures: {ctx['primary']} (Linear recommended with few plants;"
          " RandomForest cannot extrapolate)")
    ctx["metrics"].to_csv(f"{a.outdir}/model_comparison.csv", index=False)
    fm.to_csv(f"{a.outdir}/fold_metrics.csv", index=False)
    if ctx["eis_feats"] is not None:
        ef = ctx["eis_feats"]
        print(f"\nCole fit quality: median rel. RMSE {ef.rel_rmse.median():.3f}; rejected {(~ef.fit_ok).mean():.1%} "
              f"of spectra (RMSE > {MAX_FIT_RMSE}, fitted peak outside the frequency range, or electrode outlier)")
        print("Re = R0 and Ri = R0*Rinf/(R0-Rinf) [STANDARD Fricke circuit, not the patent method]")
    cor = ctx["correlations"]
    if len(cor):
        cor.to_csv(f"{a.outdir}/eis_correlations.csv", index=False)
        print("\nOBJECTIVE 3 - EIS vs environment/growth (mean within-plant Spearman rho, plant-bootstrap 95% CI):")
        cols = ["group", "eis_feature", "variable", "n_plants", "within_plant_rho", "age_adj_rho",
                "age_adj_ci95_lo", "age_adj_ci95_hi"]
        print(cor[cor.group == "all plants"][cols + ["age_adj_ci_excludes_0"]].round(3).to_string(index=False))
        sub = cor[(cor.group != "all plants") & cor.age_adj_ci_excludes_0 & cor.eis_feature.isin(["Re_rel", "Ri_rel"])]
        if len(sub):
            print("\nBy treatment group (Re_rel/Ri_rel pairs whose CI excludes 0; full table in eis_correlations.csv):")
            print(sub[cols].round(3).to_string(index=False))
    plant, day = pick_point(ctx, a.plant, a.day)
    res = analyse_point(ctx, plant, day)
    for k, r in res.items():
        mc = r["mc"]
        print(f"\n--- Model {k}, {plant} day {day}: predicted 3-day growth {r['G0']:.3f} per day ---")
        print(f"90% conformal interval (model error) [{r['pi90'][0]:.3f}, {r['pi90'][1]:.3f}]")
        print(f"input-noise only: first-order sigma_G = {r['sigma_G_first_order']:.3f}; Monte Carlo sd {mc['sd']:.3f}, "
              f"95% [{mc['lo95']:.3f}, {mc['hi95']:.3f}]")
        print(f"modelled P(growth < {THRESH['growth_low']:.3f}) = {mc['p_below']:.2f}; modelled stress probability = "
              f"{mc['p_stress']:.2f} (MODELLED, not experimentally validated)")
        print("top sensitivities (effect per +1 SD of input):")
        print(r["sens"].head(5).round(4).to_string(index=False))
        print("top input-uncertainty contributors:")
        print(r["budget"].head(4).round(4).to_string(index=False))
    if (a.eis_mc or a.eis_compress) and ctx["eis_raw"] is not None:
        g = one_sweep(ctx["eis_raw"], plant, day)
        f_, zr_, zi_ = g.freq_hz.to_numpy(), g.z_real.to_numpy(), g.z_imag.to_numpy()
        if a.eis_mc:
            print(f"\nEIS FIT-UNCERTAINTY MONTE CARLO ({plant} day {day}, assumed 1% noise):")
            print(frequency_count_study(f_, zr_, zi_).round(4).to_string(index=False))
        if a.eis_compress:
            print(f"\nSPECTRUM COMPRESSION to fewer than 9 points ({plant} day {day}; bias/sd in % of true value):")
            print(compression_study(f_, zr_, zi_).round(2).to_string(index=False))
    status = plant_status(ctx)
    print("\nPLANT STATUS (latest complete day):")
    print(status.round(3).to_string(index=False))
    status.to_csv(f"{a.outdir}/plant_status.csv", index=False)
    for name, fig in all_figures(ctx, res, status, plant, day).items():
        fig.savefig(f"{a.outdir}/{name}.png", dpi=130)
        plt.close(fig)
    print(f"\nFigures and tables saved in ./{a.outdir}/")
    if a.seeds > 0 and a.csv_env is None:
        summ, wins = seed_summary(a.seeds, a.split_day, a.folds, a.split)
        print(f"\nREPLICATION over {a.seeds} simulated datasets (mean, std):")
        print(summ.to_string())
        if wins is not None:
            print("\nFraction of simulated datasets where Model B had lower MAE than Model A:")
            print(wins.round(2).to_string())
        print("(reflects the simulator's own assumptions about EIS, not real-plant evidence)")


if __name__ == "__main__":
    main()
