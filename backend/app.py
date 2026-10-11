"""
NOVA web backend (FastAPI) - Natural Observation and Vegetative Analysis.

    Frontend (HTML/CSS/JS in /frontend)  --HTTP/JSON-->  this REST API  -->  analytics.py  -->  db.py  -->  SQLite

Run:  python -m uvicorn backend.app:app --port 8000      (or start.bat / start.sh)
Interactive API documentation: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import analytics, db, upload

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

@asynccontextmanager
async def lifespan(_app):
    """The EIS features table is pre-computed; build it once if this database lacks it."""
    if not db.table_exists("eis_features"):
        from .precompute import precompute
        print("eis_features table missing - computing it once (about 10-30 s)...")
        precompute()
    yield


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,              # served below from local files, so /docs works offline
    redoc_url=None,
    title="NOVA API",
    version="1.0",
    description="REST API behind the NOVA dashboard (Natural Observation and Vegetative Analysis). **All data is SIMULATED** "
                "(six greenhouse crops x five treatments, 90 plants).",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _check_species(species: str):
    if not db.species_exists(species):
        raise HTTPException(404, f"Unknown crop '{species}'")


# ------------------------------------------------------------------ API
@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "data_source": "SIMULATED"}


@app.get("/api/overview", tags=["dashboard"])
def overview():
    """Headline numbers, harvest by treatment, stress-detection accuracy and key findings."""
    return analytics.overview()


@app.get("/api/species", tags=["crops"])
def species():
    """Crop profiles: growth measure, stress thresholds, salt tolerance and its source."""
    return db.species_list()


@app.get("/api/species/{species}/plants", tags=["crops"])
def plants(species: str):
    """All plants of one crop with treatment, harvest weight and latest stress status."""
    _check_species(species)
    return db.plants(species)


@app.get("/api/species/{species}/plants/{plant_id}/timeline", tags=["plants"])
def timeline(species: str, plant_id: str):
    """Day-by-day environment, EIS signal (relative to the plant's baseline), growth and stress label."""
    _check_species(species)
    t = analytics.timeline(species, plant_id)
    if t is None:
        raise HTTPException(404, f"Unknown plant '{plant_id}'")
    return t


@app.get("/api/species/{species}/plants/{plant_id}/eis", tags=["plants"])
def eis(species: str, plant_id: str, day: int | None = Query(None, description="Calendar day; nearest sweep is used")):
    """Impedance spectrum (Nyquist points) and Cole-model fit for the baseline day and a chosen day."""
    _check_species(species)
    e = analytics.eis(species, plant_id, day)
    if e is None:
        raise HTTPException(404, f"No EIS data for '{plant_id}'")
    return e


@app.get("/api/species/{species}/correlations", tags=["results"])
def correlations(species: str, feature: str = Query("Re_rel", pattern="^(Re_rel|Ri_rel)$")):
    """Objective 3: age-adjusted within-plant correlation of an EIS feature with each variable, per treatment."""
    _check_species(species)
    return analytics.correlations(species, feature)


@app.get("/api/models", tags=["results"])
def models():
    """Per crop: growth-forecast error (baseline / without EIS / with EIS) and stress-detection F1."""
    return analytics.models_summary()


@app.get("/api/status", tags=["results"])
def status(species: str | None = None):
    """Latest modelled stress probability and category for every plant (optionally one crop)."""
    if species:
        _check_species(species)
    return analytics.status(species)


class UploadIn(BaseModel):
    crop: str = Field(..., description="Crop the data belongs to, e.g. lettuce")
    csv: str = Field(..., description="Contents of the CSV file (see /static/samples/ for the format)")


@app.post("/api/upload/analyze", tags=["your data"])
def analyze_upload(body: UploadIn):
    """Analyse a simple CSV of your own plants (kept separate from the simulated crops; nothing is stored).

    Columns: plant_id, day, soil_moisture_pct, air_temp_c, humidity_pct, ec_ds_m, height_cm, leaf_count;
    optional: light_par, Re_kohm, Ri_kohm."""
    try:
        return upload.analyse(body.crop, body.csv)
    except upload.UploadError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ frontend (static files)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/docs", include_in_schema=False)
def docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url, title="NOVA API - documentation",
        swagger_js_url="/static/vendor/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/vendor/swagger/swagger-ui.css",
        swagger_favicon_url="/static/vendor/swagger/favicon-32x32.png")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")
