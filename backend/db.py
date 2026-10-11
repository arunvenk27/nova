"""
Data-access layer: the ONLY place that talks to the SQLite database.
Every function returns plain Python lists/dicts, ready to be sent as JSON.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "plantsense.db"


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query and return rows as dictionaries."""
    con = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)   # read-only; as_uri() is Windows-safe
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def table_exists(name: str) -> bool:
    return bool(query("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)))


# ------------------------------------------------------------------ species & plants
def species_list() -> list[dict]:
    return query("SELECT * FROM species ORDER BY species")


def species_exists(species: str) -> bool:
    return bool(query("SELECT 1 FROM species WHERE species=?", (species,)))


def plants(species: str) -> list[dict]:
    return query("""
        SELECT p.plant_id, p.treatment, p.cohort, p.harvest_fresh_weight_g, p.treatment_detail,
               s.day AS status_day, s.modelled_stress_prob, s.category, s.predicted_next3,
               s.observed_growth, s.expected_growth
        FROM plants p LEFT JOIN plant_status s USING (species, plant_id)
        WHERE p.species = ? ORDER BY p.treatment, p.cohort""", (species,))


def plant(species: str, plant_id: str) -> dict | None:
    rows = query("SELECT * FROM plants WHERE species=? AND plant_id=?", (species, plant_id))
    return rows[0] if rows else None


def daily(species: str, plant_id: str) -> list[dict]:
    return query("SELECT * FROM env_daily WHERE species=? AND plant_id=? ORDER BY day", (species, plant_id))


def eis_features(species: str, plant_id: str) -> list[dict]:
    return query("""SELECT day, electrode_session, Re, Ri, fc_hz, alpha, Re_rel, Ri_rel, rel_rmse
                    FROM eis_features WHERE species=? AND plant_id=? ORDER BY day""", (species, plant_id))


def spectrum(species: str, plant_id: str, day: int) -> list[dict]:
    """One impedance sweep. If the electrode was moved that day (two sessions), the newer leaf."""
    return query("""
        SELECT freq_hz, z_real, z_imag, sweep_temp_c, electrode_session FROM eis_spectra
        WHERE species=? AND plant_id=? AND day=? AND electrode_session = (
            SELECT MAX(electrode_session) FROM eis_spectra WHERE species=? AND plant_id=? AND day=?)
        ORDER BY freq_hz""", (species, plant_id, day, species, plant_id, day))


def sweep_days(species: str, plant_id: str) -> list[int]:
    return [r["day"] for r in query(
        "SELECT day FROM eis_features WHERE species=? AND plant_id=? ORDER BY day", (species, plant_id))]


# ------------------------------------------------------------------ results
def harvest() -> list[dict]:
    return query("SELECT * FROM harvest_by_treatment ORDER BY species, treatment")


def model_results(species: str | None = None) -> list[dict]:
    if species:
        return query("SELECT * FROM model_results WHERE species=?", (species,))
    return query("SELECT * FROM model_results")


def correlations(species: str) -> list[dict]:
    return query("""SELECT "group", eis_feature, variable, n_plants, within_plant_rho, age_adj_rho,
                           age_adj_ci95_lo, age_adj_ci95_hi, age_adj_ci_excludes_0
                    FROM eis_correlations WHERE species=?""", (species,))


def status_all() -> list[dict]:
    return query("""
        SELECT s.species, s.plant_id, p.treatment, p.cohort, s.day, s.modelled_stress_prob, s.category,
               s.observed_growth, s.predicted_next3
        FROM plant_status s JOIN plants p USING (species, plant_id)
        ORDER BY s.species, p.treatment, p.cohort""")


def counts() -> dict:
    one = lambda sql: query(sql)[0]["n"]
    return dict(crops=one("SELECT COUNT(*) n FROM species"),
                plants=one("SELECT COUNT(*) n FROM plants"),
                daily_records=one("SELECT COUNT(*) n FROM env_daily"),
                eis_sweeps=one("SELECT COUNT(*) n FROM (SELECT DISTINCT species, plant_id, day, electrode_session FROM eis_spectra)"),
                treatments=one("SELECT COUNT(DISTINCT treatment) n FROM plants"))
