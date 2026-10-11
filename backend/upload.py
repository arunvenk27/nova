"""
"Try your own data": analyse a simple CSV uploaded from the browser.

Kept SEPARATE from the six simulated crops: nothing is written to the database.
The uploaded rows are checked, turned into the same simple features the model
was trained on, and scored by a model trained on the simulated trial of the
chosen crop.

Simple CSV format (one row per plant per day):
    plant_id, day, soil_moisture_pct, air_temp_c, humidity_pct, ec_ds_m, height_cm, leaf_count
    optional: light_par, Re_kohm, Ri_kohm
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

from . import db

REQUIRED = ["plant_id", "day", "soil_moisture_pct", "air_temp_c", "humidity_pct",
            "ec_ds_m", "height_cm", "leaf_count"]
OPTIONAL = ["light_par", "Re_kohm", "Ri_kohm"]
ALIASES = {  # accept a few friendly spellings
    "plant": "plant_id", "id": "plant_id", "soil_moisture": "soil_moisture_pct", "moisture": "soil_moisture_pct",
    "temperature": "air_temp_c", "temp_c": "air_temp_c", "air_temp": "air_temp_c", "temperature_c": "air_temp_c",
    "humidity": "humidity_pct", "rel_humidity_pct": "humidity_pct", "ec": "ec_ds_m", "ec_dsm": "ec_ds_m",
    "height": "height_cm", "leaves": "leaf_count", "light": "light_par", "re": "Re_kohm", "ri": "Ri_kohm",
}
RANGES = {  # physically possible values; rows outside are rejected
    "soil_moisture_pct": (0, 100), "air_temp_c": (-10, 60), "humidity_pct": (0, 100), "ec_ds_m": (0, 20),
    "height_cm": (0, 500), "leaf_count": (0, 500), "light_par": (0, 3000), "Re_kohm": (0.01, 10000),
    "Ri_kohm": (0.01, 10000),
}
DRY_LIMIT = 20.0          # soil moisture % below which plants are water-stressed (same as the dashboard)
HORIZON = 3               # predict 3 days ahead
MAX_BYTES = 2_000_000
FEATURES = ["soil_moisture_pct", "air_temp_c", "humidity_pct", "ec_ds_m", "light_par", "age",
            "dry_deficit", "heat_excess", "ec_excess", "ec_deficit",
            "soil_3d", "temp_3d", "ec_3d", "growth_3d", "height_cm", "leaf_count"]


class UploadError(ValueError):
    """A problem with the uploaded file that the user can fix."""


# ------------------------------------------------------------------ crop limits
def limits(crop: str) -> dict:
    sp = next(s for s in db.species_list() if s["species"] == crop)
    return dict(dry=DRY_LIMIT, heat=float(sp["temp_high_flag_c"]), ec_high=float(sp["ec_high_flag"]),
                ec_low=round(0.5 * float(sp["ec_normal_ds_m"]), 2))


# ------------------------------------------------------------------ features (same for training and upload)
def add_features(df: pd.DataFrame, lim: dict) -> pd.DataFrame:
    df = df.sort_values(["plant_id", "day"]).copy()
    g = df.groupby("plant_id", sort=False)
    df["age"] = df["day"] - g["day"].transform("min")
    df["dry_deficit"] = (lim["dry"] - df["soil_moisture_pct"]).clip(lower=0)
    df["heat_excess"] = (df["air_temp_c"] - lim["heat"]).clip(lower=0)
    df["ec_excess"] = (df["ec_ds_m"] - lim["ec_high"]).clip(lower=0)
    df["ec_deficit"] = (lim["ec_low"] - df["ec_ds_m"]).clip(lower=0)
    for col, out in [("soil_moisture_pct", "soil_3d"), ("air_temp_c", "temp_3d"), ("ec_ds_m", "ec_3d")]:
        df[out] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
    prev = g["height_cm"].shift(3)
    days = (df["day"] - g["day"].shift(3)).replace(0, np.nan)
    df["growth_3d"] = ((df["height_cm"] - prev) / days).fillna(0.0)
    return df


# ------------------------------------------------------------------ model trained on the simulated crop
@lru_cache(maxsize=8)
def model_for(crop: str):
    """Train (once per crop, then cached) on that crop's simulated plants."""
    lim = limits(crop)
    d = pd.DataFrame(db.query(
        """SELECT plant_id, day, soil_moisture_pct, air_temp_c, rel_humidity_pct AS humidity_pct, ec_ds_m,
                  light_par, height_cm, leaf_count, stress_truth FROM env_daily WHERE species=?""", (crop,)))
    d = add_features(d, lim)
    g = d.groupby("plant_id", sort=False)
    # targets: stressed at any point in the next 3 days; mean height growth over the next 3 days
    fut = [g["stress_truth"].shift(-k) for k in range(1, HORIZON + 1)]
    d["y_stress"] = pd.concat(fut, axis=1).max(axis=1)
    d["y_growth"] = (g["height_cm"].shift(-HORIZON) - d["height_cm"]) / HORIZON
    d = d.dropna(subset=["y_stress", "y_growth"])
    X, ys, yg, groups = d[FEATURES].to_numpy(), d["y_stress"].astype(int).to_numpy(), d["y_growth"].to_numpy(), d["plant_id"]

    # honest check on unseen simulated plants (5 plant-wise folds)
    f1s, maes = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, ys, groups):
        c = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X[tr], ys[tr])
        r = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X[tr], yg[tr])
        f1s.append(f1_score(ys[te], c.predict(X[te]), zero_division=0))
        maes.append(mean_absolute_error(yg[te], r.predict(X[te])))

    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, ys)
    reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, yg)
    quality = dict(stress_f1=round(float(np.mean(f1s)), 2), growth_mae_cm_per_day=round(float(np.mean(maes)), 3),
                   trained_on_plants=int(d["plant_id"].nunique()), trained_on_rows=int(len(d)))
    return clf, reg, quality, lim


# ------------------------------------------------------------------ parse + validate the upload
def read_csv(text: str) -> tuple[pd.DataFrame, list[str]]:
    if not text or not text.strip():
        raise UploadError("The file is empty.")
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise UploadError("The file is larger than 2 MB. Split it into smaller files.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")   # detects comma / semicolon / tab
    except Exception as e:
        raise UploadError(f"Could not read the file as CSV ({e}).")
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for want in REQUIRED + OPTIONAL + ["date"]:
        if want not in df.columns and want.lower() in lower:
            rename[lower[want.lower()]] = want
    for alias, want in ALIASES.items():
        if want not in df.columns and want not in rename.values() and alias in lower:
            rename[lower[alias]] = want
    df = df.rename(columns=rename)

    warnings: list[str] = []
    if "day" not in df.columns and "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
        if dates.isna().all():
            raise UploadError("The 'date' column could not be read. Use YYYY-MM-DD, or a 'day' column with 0, 1, 2, ...")
        df["day"] = (dates - dates.min()).dt.days
        warnings.append("Dates were converted to day numbers (first date = day 0).")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise UploadError("Missing column(s): " + ", ".join(missing) +
                          ". Download the sample CSV to see the expected format.")

    df["plant_id"] = df["plant_id"].astype(str).str.strip()
    n0 = len(df)
    bad_rows: list[str] = []
    for col in [c for c in REQUIRED + OPTIONAL if c in df.columns and c != "plant_id"]:
        num = pd.to_numeric(df[col], errors="coerce")
        lo, hi = RANGES.get(col, (-np.inf, np.inf))
        if col in OPTIONAL:          # optional columns may be blank
            bad = num.notna() & ((num < lo) | (num > hi)) | (df[col].notna() & num.isna() & (df[col].astype(str).str.strip() != ""))
        else:
            bad = num.isna() | (num < lo) | (num > hi)
        for i in df.index[bad][:3]:
            bad_rows.append(f"row {i + 2}: {col} = '{df.at[i, col]}'")
        df[col] = num.where(~bad)
    if "light_par" not in df.columns or df["light_par"].isna().all():
        df["light_par"] = 300.0
        warnings.append("No light readings: a typical greenhouse value (300 PAR) was used.")
    else:
        df["light_par"] = df["light_par"].fillna(df["light_par"].median())

    df = df.dropna(subset=[c for c in REQUIRED if c != "plant_id"])
    df = df[df["plant_id"] != ""]
    df = df.drop_duplicates(subset=["plant_id", "day"], keep="last")
    if bad_rows:
        warnings.append(f"{n0 - len(df)} row(s) skipped because of missing or impossible values, e.g. " +
                        "; ".join(bad_rows[:5]) + ".")
    if df.empty:
        raise UploadError("No usable rows left after checking the values.")
    if df["plant_id"].nunique() > 200:
        raise UploadError("More than 200 plants in one file. Split it into smaller files.")
    df["day"] = df["day"].astype(int)
    return df, warnings


# ------------------------------------------------------------------ analyse
def reasons_for(row, lim) -> list[str]:
    r = []
    if row["soil_moisture_pct"] < lim["dry"]:
        r.append(f"soil dry ({row['soil_moisture_pct']:.0f}% < {lim['dry']:.0f}%)")
    if row["air_temp_c"] > lim["heat"]:
        r.append(f"too hot ({row['air_temp_c']:.1f} °C > {lim['heat']:.0f} °C)")
    if row["ec_ds_m"] > lim["ec_high"]:
        r.append(f"too salty (EC {row['ec_ds_m']:.1f} > {lim['ec_high']:.1f})")
    if row["ec_ds_m"] < lim["ec_low"]:
        r.append(f"too few nutrients (EC {row['ec_ds_m']:.2f} < {lim['ec_low']:.2f})")
    return r


def level(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


def _num(x, d=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def analyse(crop: str, text: str) -> dict:
    if not db.species_exists(crop):
        raise UploadError(f"Unknown crop '{crop}'.")
    df, warnings = read_csv(text)
    clf, reg, quality, lim = model_for(crop)
    df = add_features(df, lim)
    X = df[FEATURES].to_numpy()
    df["stress_prob"] = clf.predict_proba(X)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else 0.0
    df["pred_growth"] = reg.predict(X)
    has_eis = all(c in df.columns for c in ("Re_kohm", "Ri_kohm")) and df[["Re_kohm", "Ri_kohm"]].notna().any().all()
    if has_eis:
        g = df.groupby("plant_id", sort=False)
        for c in ("Re_kohm", "Ri_kohm"):
            first = g[c].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
            df[c.replace("_kohm", "_rel")] = df[c] / first
    elif any(c in df.columns for c in ("Re_kohm", "Ri_kohm")):
        warnings.append("Leaf signal needs both Re_kohm and Ri_kohm; it was ignored.")

    plants, timelines = [], {}
    for pid, p in df.groupby("plant_id", sort=False):
        last = p.iloc[-1]
        pr = float(last["stress_prob"])
        days_out = int(sum(bool(reasons_for(r, lim)) for _, r in p.iterrows()))
        eis_note = None
        if has_eis and pd.notna(last.get("Re_rel")):
            re, ri = float(last["Re_rel"]), float(last["Ri_rel"])
            if re > 1.15:
                eis_note = f"Re up {100 * (re - 1):.0f}% from day 1: leaf may be losing water"
            elif re < 0.85:
                eis_note = f"Re down {100 * (1 - re):.0f}% from day 1: salt may be building up in the leaf"
            elif ri < 0.85:
                eis_note = f"Ri down {100 * (1 - ri):.0f}% from day 1: cell membranes may be leaky (heat?)"
            else:
                eis_note = "Leaf signal close to day 1"
        plants.append(dict(
            plant_id=pid, days=int(len(p)), first_day=int(p["day"].min()), last_day=int(last["day"]),
            stress_prob=round(pr, 3), level=level(pr), reasons=reasons_for(last, lim),
            days_outside_limits=days_out, height_cm=_num(last["height_cm"], 1),
            recent_growth=_num(last["growth_3d"]), predicted_growth=_num(last["pred_growth"]), eis_note=eis_note))
        timelines[pid] = dict(
            day=p["day"].tolist(),
            soil=p["soil_moisture_pct"].round(1).tolist(), temp=p["air_temp_c"].round(1).tolist(),
            humidity=p["humidity_pct"].round(1).tolist(), ec=p["ec_ds_m"].round(2).tolist(),
            height=p["height_cm"].round(2).tolist(), leaves=p["leaf_count"].round(0).tolist(),
            stress_prob=p["stress_prob"].round(3).tolist(),
            Re_rel=[_num(v) for v in p["Re_rel"]] if has_eis else None,
            Ri_rel=[_num(v) for v in p["Ri_rel"]] if has_eis else None)
    plants.sort(key=lambda r: -r["stress_prob"])
    counts = {k: sum(p["level"] == k for p in plants) for k in ("high", "medium", "low")}
    share = lambda m: round(float(m.mean()), 3)
    return dict(
        crop=crop, limits=lim, model=quality, warnings=warnings, has_eis=bool(has_eis),
        summary=dict(plants=len(plants), rows=int(len(df)), days=int(df["day"].nunique()), **counts,
                     share_dry=share(df["soil_moisture_pct"] < lim["dry"]),
                     share_hot=share(df["air_temp_c"] > lim["heat"]),
                     share_salty=share(df["ec_ds_m"] > lim["ec_high"]),
                     share_low_nutrients=share(df["ec_ds_m"] < lim["ec_low"])),
        plants=plants, timelines=timelines)
