#!/usr/bin/env python3
"""
Combine all SIMULATED PlantSense data into one SQLite database.

    python build_database.py                 # -> plantsense.db

Tables (every row has data_source = 'SIMULATED')
  species          one row per crop: thresholds, salt tolerance, EIS baseline, sources
  plants           one row per plant: species, treatment, cohort, electrode set-up, harvest weight
  env_daily        one row per plant per day: sensors, growth, SPAD, wilting, stress label
  eis_spectra      one row per plant, day, electrode session and frequency (impedance)
  calib_sensor     repeat readings of every sensor against a reference
  calib_eis_temp   EIS sweeps at 15-35 C (temperature coefficient)
  calib_eis_repeat repeated EIS sweeps (EIS feature noise)
  model_results    held-out model metrics per species (from results_species/)
  eis_correlations Objective-3 correlations per species and treatment group
  plant_status     latest status / stress category per plant

Open with DB Browser for SQLite, or:  sqlite3 plantsense.db "SELECT * FROM species;"
"""
import os
import sqlite3

import pandas as pd

DATA, RESULTS, DB = "data_species", "results_species", "plantsense.db"


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    species = pd.read_csv(f"{DATA}/species_profiles.csv")
    species["data_source"] = "SIMULATED"
    species.to_sql("species", con, index=False)
    tables = {"plants": "plants.csv", "env_daily": "env.csv", "eis_spectra": "eis.csv",
              "calib_sensor": "calib_sensor_repeatability.csv", "calib_eis_temp": "calib_eis_temperature.csv",
              "calib_eis_repeat": "calib_eis_repeatability.csv"}
    res = {"model_results": "model_comparison.csv", "eis_correlations": "eis_correlations.csv",
           "plant_status": "plant_status.csv"}
    for name in species.species:
        for table, f in tables.items():
            df = pd.read_csv(f"{DATA}/{name}/{f}")
            if "species" not in df:
                df.insert(0, "species", name)
            df["data_source"] = "SIMULATED"
            df.to_sql(table, con, index=False, if_exists="append")
        for table, f in res.items():
            path = f"{RESULTS}/{name}/{f}"
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.insert(0, "species", name)
                df["data_source"] = "SIMULATED"
                df.to_sql(table, con, index=False, if_exists="append")
    cur = con.cursor()
    for sql in ["CREATE INDEX ix_plants ON plants(species, plant_id)",
                "CREATE INDEX ix_env ON env_daily(species, plant_id, day)",
                "CREATE INDEX ix_eis ON eis_spectra(species, plant_id, day, electrode_session)",
                """CREATE VIEW harvest_by_treatment AS
                   SELECT species, treatment, COUNT(*) AS plants, ROUND(AVG(harvest_fresh_weight_g), 1) AS mean_fresh_weight_g
                   FROM plants GROUP BY species, treatment""",
                """CREATE VIEW daily_with_plant AS
                   SELECT e.*, p.latin_name, p.cohort AS plant_cohort, p.recommended_growth_col
                   FROM env_daily e JOIN plants p USING (species, plant_id)"""]:
        cur.execute(sql)
    con.commit()
    for t in ["species", "plants", "env_daily", "eis_spectra", "calib_sensor", "calib_eis_temp", "calib_eis_repeat",
              "model_results", "eis_correlations", "plant_status"]:
        print(f"{t:17s} {cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>7} rows")
    con.close()
    print(f"\nwrote {DB} ({os.path.getsize(DB) / 1e6:.1f} MB) - all rows SIMULATED")


if __name__ == "__main__":
    main()
