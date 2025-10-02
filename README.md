# Will It Rain On My Parade — README
Keep this repo at the project root (the commands assume you run them from the repository root).

---

## What this project does (short)

A minimal prototype that uses NASA POWER historical daily data to:

* fetch daily weather variables for a point (`fetch_power_point.py`),
* clean & prepare the data (`preprocess.py`),
* compute empirical exceedance probabilities and bootstrap CI (`probability.py`),
* compute trends and decadal summaries (`trend.py`),
* provide a simple REST API (`backend/main.py`) with endpoints:

  * `/probability` (probability + bootstrap),
  * `/trend` (value & exceedance trend + decadal summary),
  * `/history` (per-year series used to build charts),
* a tiny static frontend (`frontend/index.html`) that calls the API and plots results with Plotly.

Files of interest:

```
backend/main.py
data_pipeline/fetch_power_point.py
data_pipeline/preprocess.py
data_pipeline/probability.py
data_pipeline/trend.py
data_pipeline/cache/                # sample CSVs live here after fetching
frontend/index.html
```

---

# Quick start (Windows CMD, or PowerShell recommended)

> These are the exact steps I used and verified with the code in this repo. If you encounter issues, see Troubleshooting below.

1. Open PowerShell and allow script execution for this session (only if needed):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
```

2. Create & activate a venv and upgrade pip:

```powershell
cd C:\path\to\repo          # change to your repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

3. Install Python packages (pip install):

```powershell
pip install -r requirements.txt
# if there is no requirements.txt in repo, install these minimal libs:
# pip install pandas numpy scipy fastapi uvicorn plotly python-dotenv requests
```

# Initially we have to fetch date, later we directly run backend
# Data fetch → preprocess → run API (sequence)

1. **Fetch NASA POWER historical daily point data** (example location: Bengaluru 12.97, 77.59):

```powershell
python data_pipeline\fetch_power_point.py --lat 12.97 --lon 77.59 --start 19900101 --end 20231231
# output: data_pipeline/cache/sample_point_daily.csv
```

2. **Preprocess cleaned daily CSV**:

```powershell
python data_pipeline\preprocess.py `
  --in data_pipeline\cache\sample_point_daily.csv `
  --out_clean data_pipeline\cache\sample_point_daily_clean.csv `
  --out_doy data_pipeline\cache\sample_point_doy_agg.csv
```

3. **Run the backend API server (FastAPI / Uvicorn)**:

```powershell
# activate venv if not already
.\.venv\Scripts\Activate.ps1

# start uvicorn (use -reload during development)
python -m uvicorn backend.main:app --reload --port 8000
```

4. **Test endpoints via PowerShell / browser / curl**:

* Probability (single day):

```powershell
curl "http://127.0.0.1:8000/probability?var=t2m_max&doy=196&threshold=32&n_boot=500"
```

* Trend (value + exceedance trend; includes decadal summary):

```powershell
curl "http://127.0.0.1:8000/trend?var=t2m_max&doy=196&threshold=32"
```

* History (per-year values used by the probability/trend calculation):

```powershell
curl "http://127.0.0.1:8000/history?var=t2m_max&doy=196"
```

(If using PowerShell, `curl` returns a structured object; `Invoke-WebRequest` / `Invoke-RestMethod` also work.)

---

# Run the demo frontend (local)

1. Serve the `frontend` directory:

```powershell
cd frontend
python -m http.server 8080
```

2. Open the browser:

```
http://127.0.0.1:8080
```

3. Use the page to select variable/day/threshold and click **Get Probability & Trend**.
   The page calls the API endpoints, shows JSON and draws charts.

> If you get `Error contacting API: Failed to fetch` in the browser console, you likely need to enable CORS in `backend/main.py`. See Troubleshooting.

---

# Typical example (what you should expect)

A `/probability` JSON response includes:

```json
{
  "ok": true,
  "variable": "t2m_max",
  "doy": 196,
  "threshold": 30,
  "years_used": 34,
  "exceed_count": 6,
  "probability": 0.17647058823529413,
  "bootstrap": { "n_boot": 500, "ci_95_lower": 0.05882, "ci_95_upper": 0.2941 }
}
```

`/trend` returns `value_trend` (slope, p-value) and `exceedance_trend` (change in exceedance probability per year) plus `decadal_summary`.

---

# Troubleshooting (common issues & fixes)

* **`Failed to fetch` or CORS blocked in browser**
  Add CORS middleware to `backend/main.py`:

  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

  Restart the backend.

---

# Project checklist

1. Clone repo
2. Create & activate venv, install requirements
3. `python data_pipeline\fetch_power_point.py` for your coordinates / date range
4. `python data_pipeline\preprocess.py --in ...` to produce cleaned CSV + doy agg
5. Start backend: `python -m uvicorn backend.main:app --reload --port 8000`
6. Serve frontend: `cd frontend` → `python -m http.server 8080`
7. Open UI and test endpoints; or use curl to test directly

---
