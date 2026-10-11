"""
Pre-compute EIS features (Re, Ri, fc, alpha, Re_rel, Ri_rel) for every sweep in the database
and store them in the table `eis_features`, so the website never has to fit thousands of
spectra while someone is waiting.

Run once (the server also runs it automatically if the table is missing):
    python -m backend.precompute

Uses the SAME functions as the analysis pipeline (pipeline/plantsense_v3.py):
temperature correction -> Cole fit -> quality check -> electrode-session baseline linking.
"""
import io
import json
import sqlite3
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import plantsense_v3 as ps  # noqa: E402

DB = ROOT / "data" / "plantsense.db"
CALIB = ROOT / "data" / "calib"


def _features_for(species: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    env = pd.read_sql("SELECT * FROM env_daily WHERE species = ?", con, params=(species,))
    eis = pd.read_sql("SELECT * FROM eis_spectra WHERE species = ?", con, params=(species,))
    con.close()
    calib_file = CALIB / f"{species}.json"
    if calib_file.exists():
        ps.TEMP_COEF = json.loads(calib_file.read_text()).get("TEMP_COEF", ps.TEMP_COEF)
    with redirect_stdout(io.StringIO()):                       # pipeline prints notes; keep quiet
        std = ps.standardise_eis(eis.drop(columns=["species", "data_source"]), env)
        feats = ps.extract_eis_features(std)
        linked = ps.link_session_baselines(feats[feats.fit_ok].copy())
    keep = ["plant_id", "day", "electrode_session", "Re", "Ri", "fc_hz", "alpha", "rel_rmse",
            "phase_min_deg", "Re_rel", "Ri_rel"]
    out = linked[keep].copy()
    out.insert(0, "species", species)
    out["data_source"] = "SIMULATED"
    rejected = feats[~feats.fit_ok][["plant_id", "day", "electrode_session"]].copy()
    rejected.insert(0, "species", species)
    return out, rejected


def precompute(verbose: bool = True) -> None:
    con = sqlite3.connect(DB)
    species = [r[0] for r in con.execute("SELECT species FROM species")]
    con.close()
    with Pool(min(len(species), 6)) as pool:
        results = pool.map(_features_for, species)
    feats = pd.concat([r[0] for r in results], ignore_index=True)
    rejected = pd.concat([r[1] for r in results], ignore_index=True)
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS eis_features")
    con.execute("DROP TABLE IF EXISTS eis_rejected")
    feats.to_sql("eis_features", con, index=False)
    rejected.to_sql("eis_rejected", con, index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_eisf ON eis_features(species, plant_id, day)")
    con.commit()
    con.close()
    if verbose:
        print(f"eis_features: {len(feats)} sweeps fitted, {len(rejected)} rejected (bad fit / electrode fault)")


if __name__ == "__main__":
    precompute()
