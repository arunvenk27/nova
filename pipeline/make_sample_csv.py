"""
Make small, simple CSV files to test the "Upload data" page.

Each file has 5 plants of one crop followed for 30 days:
    P1 healthy · P2 drought from day 12 · P3 salt rising from day 8 · P4 heatwave days 14-19 · P5 low nutrients
Columns are plain greenhouse readings (all SIMULATED):
    plant_id, day, soil_moisture_pct, air_temp_c, humidity_pct, ec_ds_m, light_par, height_cm, leaf_count, Re_kohm, Ri_kohm

Run:  python pipeline/make_sample_csv.py      -> frontend/samples/sample_<crop>.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "frontend" / "samples"
CROPS = {   # start height (cm), healthy growth (cm/day), room temp, normal EC, heat flag, EC flag, Re, Ri (kOhm)
    "lettuce": dict(h0=4.0, g=0.30, t=20, ec=1.4, heat=27, ec_hi=2.5, re=22, ri=9),
    "tomato": dict(h0=9.0, g=1.20, t=24, ec=2.5, heat=32, ec_hi=3.7, re=25, ri=10),
}
DAYS = 30


def plant(rng, c, pid, kind):
    rows, h, leaves, soil = [], c["h0"], 5.0, 34.0
    for d in range(DAYS):
        temp = c["t"] + 1.5 * np.sin(d / 3) + rng.normal(0, 0.5)
        ec = c["ec"] + rng.normal(0, 0.05)
        soil = 34 + rng.normal(0, 1.5)
        stress = 0.0
        if kind == "drought" and d >= 12:
            soil = max(8.0, 32 - 2.2 * (d - 12)) + rng.normal(0, 1)
            stress = max(0.0, (20 - soil) / 12)
        if kind == "salt" and d >= 8:
            ec = c["ec"] + 0.15 * (d - 8) + rng.normal(0, 0.05)
            stress = max(0.0, (ec - c["ec_hi"]) / 2)
        if kind == "heat" and 14 <= d <= 19:
            temp = c["heat"] + 4 + rng.normal(0, 0.8)
            stress = 0.8
        if kind == "nutrients":
            ec = 0.3 + rng.normal(0, 0.03)
            stress = min(1.0, d / 20)
        stress = min(stress, 1.0)
        h += c["g"] * (1 - 0.8 * stress) * (1 + rng.normal(0, 0.15))
        leaves += 0.35 * (1 - 0.7 * stress) + rng.normal(0, 0.05)
        # leaf signal: drought raises Re, salt lowers Re, heat lowers Ri (plus slow drift with age)
        re = c["re"] * (1 + 0.004 * d) * (1 + (0.6 * stress if kind == "drought" else 0)
                                          - (0.35 * stress if kind == "salt" else 0)) * (1 + rng.normal(0, 0.02))
        ri = c["ri"] * (1 + 0.003 * d) * (1 - (0.35 * stress if kind == "heat" else 0)) * (1 + rng.normal(0, 0.02))
        rows.append(dict(plant_id=pid, day=d, soil_moisture_pct=round(soil, 1), air_temp_c=round(temp, 1),
                         humidity_pct=round(62 + rng.normal(0, 4) - (8 if kind == "heat" and 14 <= d <= 19 else 0), 1),
                         ec_ds_m=round(max(ec, 0.1), 2), light_par=round(300 + 40 * np.sin(d / 4) + rng.normal(0, 10)),
                         height_cm=round(h, 1), leaf_count=int(round(leaves)), Re_kohm=round(re, 2), Ri_kohm=round(ri, 2)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for crop, c in CROPS.items():
        rng = np.random.default_rng(7)
        rows = []
        for pid, kind in [("P1", "healthy"), ("P2", "drought"), ("P3", "salt"), ("P4", "heat"), ("P5", "nutrients")]:
            rows += plant(rng, c, pid, kind)
        f = OUT / f"sample_{crop}.csv"
        pd.DataFrame(rows).to_csv(f, index=False)
        print("wrote", f, len(rows), "rows")


if __name__ == "__main__":
    main()
