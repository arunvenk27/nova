# NOVA – Natural Observation & Vegetative Analysis

An interactive dashboard for the NOVA (PlantSense) plant growth and stress prediction system.
A **Python backend** (FastAPI) does the science and serves the data, and an **HTML/CSS/JavaScript frontend** shows it.
The two communicate only through a REST API.

> **All data is SIMULATED.** The data covers six greenhouse crops, 90 virtual plants and five treatments. It shows what the system can do; it is not evidence about real plants.

---

## Run it (Windows)

1. Install **Python 3.10 or newer** from python.org. On the first installer screen, tick **"Add python.exe to PATH"**.
2. Unzip this folder anywhere, for example on the Desktop.
3. **Double-click `start.bat`.**
   - The first run installs the packages automatically (needs internet, takes a few minutes).
   - Your browser then opens at **http://localhost:8000**.
4. Keep the black window open while presenting. Close it to stop the server.

To use a different port, open PowerShell in the folder and run `.\start.bat 8080`.

**Manual start** (any system), from inside this folder:

```
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --port 8000
```

On macOS, Linux or WSL you can run `./start.sh` instead.

The site works **without internet** once the packages are installed: the chart library and the API documentation viewer are bundled in the folder.

---

## What's on the website

| Page | What it shows |
|---|---|
| **Overview** | Headline numbers, harvest lost to each stress, how well stress is predicted, key findings |
| **Plant Explorer** | One plant's life, day by day: environment, leaf EIS signal and growth, with stress periods shaded |
| **Stress Monitor** | Every plant's chance of stress in the next 3 days as a risk level, plus a crop × treatment heatmap |
| **EIS Lab** | The leaf's impedance curve (Nyquist plot) with a day slider, fitted Re/Ri values, and the Objective 3 correlation heatmap |
| **Model Results** | Growth and stress prediction with and without EIS, tested on unseen plants (Objective 4) |
| **Your Data** | Upload a simple CSV of your own plants and get a stress forecast per plant. Kept separate from the six simulated crops; nothing is saved |
| **How it works** | Architecture, method and honest limitations |

The site also has **dark mode** (sidebar button), works on **phones**, and shows the API documentation at **http://localhost:8000/docs**.

---

## How it's built (frontend vs backend)

```
 Browser (frontend)          REST API (backend)          Analytics                 Data
 HTML + CSS + JS   ──HTTP──▶  FastAPI           ──────▶   EIS fitting, results ──▶ SQLite
 ECharts graphs   ◀──JSON──   backend/app.py              backend/analytics.py     data/plantsense.db
                                                         (uses pipeline/ code)     via backend/db.py
```

```
nova_web/
├── start.bat / start.sh   one-click start
├── requirements.txt       Python packages
├── frontend/              WHAT YOU SEE
│   ├── index.html         page shell + sidebar menu
│   ├── css/style.css      NOVA colours + layout (all colours at the top)
│   ├── js/app.js          menu, page switching, dark mode
│   ├── js/api.js          every call to the backend
│   ├── js/ui.js           shared chart style, colours, helpers
│   ├── js/pages/*.js      one file per page
│   └── vendor/            ECharts + Swagger UI (bundled, offline)
├── backend/               PYTHON SERVER
│   ├── app.py             REST API endpoints (FastAPI)
│   ├── analytics.py       turns data into dashboard numbers
│   ├── db.py              the only code that reads the database
│   ├── upload.py          analyses an uploaded CSV (Your Data page)
│   └── precompute.py      pre-fits all EIS sweeps into the database
├── pipeline/              the research code (plantsense_v3.py, generators, calibration)
└── data/
    ├── plantsense.db      SQLite database (135k rows, all SIMULATED)
    └── calib/             calibration per crop
```

### API endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/overview` | KPIs, harvest by treatment, stress-detection accuracy, findings |
| `GET /api/species` | Crop profiles and thresholds |
| `GET /api/species/{crop}/plants` | Plants of one crop with latest status |
| `GET /api/species/{crop}/plants/{id}/timeline` | Daily environment, EIS, growth, stress label |
| `GET /api/species/{crop}/plants/{id}/eis?day=N` | Impedance spectrum + Cole fit (baseline and chosen day) |
| `GET /api/species/{crop}/correlations?feature=Re_rel` | Objective 3 correlations |
| `GET /api/models` | With vs without EIS, per crop |
| `GET /api/status` | Stress risk for every plant |
| `POST /api/upload/analyze` | Analyse an uploaded CSV (`{"crop": "lettuce", "csv": "<file text>"}`) |

### Your Data: CSV format

One row per plant per day. Sample files: `frontend/samples/sample_lettuce.csv` and `sample_tomato.csv`
(5 plants, 30 days: P1 healthy, P2 drought, P3 salt, P4 heatwave, P5 low nutrients). Rebuild them with
`python pipeline/make_sample_csv.py`.

| Column | Meaning | Required |
|---|---|---|
| `plant_id` | Plant name | yes |
| `day` (or `date` as YYYY-MM-DD) | Day number | yes |
| `soil_moisture_pct` | Soil moisture (%) | yes |
| `air_temp_c` | Air temperature (°C) | yes |
| `humidity_pct` | Air humidity (%) | yes |
| `ec_ds_m` | Salt / nutrient level (dS/m) | yes |
| `height_cm` | Plant height (cm) | yes |
| `leaf_count` | Number of leaves | yes |
| `light_par` | Light | no |
| `Re_kohm`, `Ri_kohm` | Leaf signal from the EIS clip (kΩ) | no |

The forecast uses a model trained on the simulated plants of the chosen crop (trained on first use, about 10 s),
plus the crop's safe limits. Rows with impossible values are skipped and reported.

---

## Customising

- **Colours:** edit the variables at the top of `frontend/css/style.css`. Brand colours are `--brand`, `--accent`, `--earth`, `--sage` and `--page`. Chart colours are `--t-*`; these were checked for colour-blind safety, so re-check them if you change them.
- **Text:** each page's wording is in `frontend/js/pages/<page>.js`. The key findings are in `backend/analytics.py`, and their numbers are filled in from the data automatically.
- **New data:** rebuild the database with `pipeline/build_database.py`. The server re-fits the EIS features automatically on its next start.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python was not found` | Install Python 3.10+ and tick "Add python.exe to PATH". Then in Settings → Apps → Advanced app settings → App execution aliases, turn off the two "python" App Installer entries. |
| `Application Control policy has blocked this file` | Windows Smart App Control is blocking NumPy. Turn it off: Windows Security → App & browser control → Smart App Control settings → Off. Or run the project in WSL. |
| `address already in use` / port busy | Run `.\start.bat 8080` and open http://localhost:8080 |
| Browser shows "can't reach this page" | The server needs a few seconds to start; refresh. Check that the black window is still open. |
| Charts empty / "Could not load data" | The server window was closed or crashed. Read its last lines, then restart `start.bat`. |
| `pip install` fails | Check the internet connection, then run `python -m pip install --upgrade pip` and try again. |

---

## Credits and limits

- **Re/Ri** use the standard Fricke circuit. The patent-based method is not implemented yet (it needs the verified equations).
- **Cited thresholds:** FAO Irrigation & Drainage Paper 61 (salt tolerance), Purdue Extension HO-309-W (lettuce EC) and Utah State University Extension (lettuce temperature). Everything else in the simulation is an assumption; see `pipeline/`.
- **Bundled libraries:** Apache ECharts 5.5.1 (Apache-2.0) and Swagger UI (Apache-2.0). Their licences are in `frontend/vendor/`.
