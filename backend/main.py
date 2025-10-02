# backend/main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.stats import linregress
from functools import lru_cache
import time
import requests
import os
import tempfile

# Allow the frontend origin(s) used during development:
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    # add any other local dev URLs you might use
]

app = FastAPI(title="Will-It-Rain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # for dev: you may use ["*"], but prefer specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# NASA POWER data fetching functions
# -------------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "will_it_rain_cache")

def fetch_power(lat, lon, start, end, params, community="AG", fmt="JSON", retries=3, backoff=2):
    q = {
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "parameters": ",".join(params),
        "community": community,
        "format": fmt,
    }

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(POWER_BASE, params=q, timeout=60)
            if r.status_code != 200:
                print(f"ERROR: Request failed (status {r.status_code}) for {r.url}")
                print("Response body:", r.text)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = getattr(e.response, "status_code", None)
            if status and 400 <= status < 500:
                print(f"HTTP Error {status} (no retry): {e}")
                print("Response body:", e.response.text if e.response is not None else "(no body)")
                raise
            print(f"HTTP error on attempt {attempt}/{retries}: {e}. Retrying in {backoff} seconds...")
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"Request exception on attempt {attempt}/{retries}: {e}. Retrying in {backoff} seconds...")
        time.sleep(backoff)
        backoff *= 2

    raise last_exc if last_exc is not None else RuntimeError("Unknown error fetching POWER data")

def json_to_dataframe(j, params):
    param_block = j.get("properties", {}).get("parameter", {})
    if not param_block:
        raise ValueError("No parameter block found in JSON response.")

    first = next(iter(param_block.values()))
    dates = sorted(first.keys())

    rows = []
    for d in dates:
        row = {"date": datetime.strptime(d, "%Y%m%d").date()}
        for p in params:
            val = param_block.get(p, {}).get(d, None)
            if val is None:
                row[p] = float("nan")
            else:
                try:
                    row[p] = float(val)
                except Exception:
                    row[p] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

def chunk_date_ranges(start_str, end_str, days=365):
    start = datetime.strptime(start_str, "%Y%m%d").date()
    end = datetime.strptime(end_str, "%Y%m%d").date()
    cur_start = start
    while cur_start <= end:
        cur_end = min(end, cur_start + timedelta(days=days - 1))
        yield cur_start.strftime("%Y%m%d"), cur_end.strftime("%Y%m%d")
        cur_start = cur_end + timedelta(days=1)

def get_power_data(lat: float, lon: float, start_year: int = 1990, end_year: int = 2023) -> pd.DataFrame:
    cache_filename = f"power_{lat:.4f}_{lon:.4f}_{start_year}_{end_year}.csv"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        print(f"Loading from cache: {cache_filepath}")
        df = pd.read_csv(cache_filepath, parse_dates=["date"])
        return df

    print(f"Cache miss for {cache_filepath}. Fetching from NASA POWER API...")
    params = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS10M", "RH2M"]
    start_str = f"{start_year}0101"
    end_str = f"{end_year}1231"
    
    dfs = []
    for s_chunk, e_chunk in chunk_date_ranges(start_str, end_str, days=365*5):
        print(f"Fetching {s_chunk} -> {e_chunk} ...")
        try:
            j = fetch_power(lat, lon, s_chunk, e_chunk, params, community="AG")
            df_chunk = json_to_dataframe(j, params)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"Failed to fetch chunk: {s_chunk} -> {e_chunk}. Error: {e}")
            continue

    if not dfs:
        raise HTTPException(status_code=503, detail="Could not fetch any data from NASA POWER API.")

    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    
    df.reset_index(inplace=True)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_filepath, index=False)
    print(f"Saved to cache: {cache_filepath}")

    return df

# -------------------------
# Heat Index Calculation
# -------------------------
def calculate_heat_index(t, rh):
    # Using the Steadman formula for heat index in Celsius
    # t in Celsius, rh in %
    if t is None or rh is None or pd.isna(t) or pd.isna(rh):
        return None
    t_f = t * 9/5 + 32 # Convert to Fahrenheit
    hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    if hi_f < 80:
        return t # Return original temp if HI is low
    hi_c = (hi_f - 32) * 5/9 # Convert back to Celsius
    return hi_c

# -------------------------
# /percentiles endpoint
# -------------------------
@app.get("/percentiles")
def percentiles_api(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    var: str = Query(..., description="variable name, e.g., t2m_max"),
    doy: int = Query(..., description="Day of year to calculate percentiles for")
):
    df = get_power_data(lat, lon)
    var_upper = var.upper()
    if var == "heat_index":
        df["heat_index"] = df.apply(lambda row: calculate_heat_index(row["T2M"], row["RH2M"]), axis=1)
        var_upper = "heat_index"
    
    if var_upper not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset for this location.")

    day_data = df[df["day_of_year"] == doy][var_upper].dropna()

    if len(day_data) < 10:
        raise HTTPException(status_code=400, detail="Not enough data to calculate percentiles for this day.")

    percentiles = {
        "p10": day_data.quantile(0.10),
        "p25": day_data.quantile(0.25),
        "p50": day_data.quantile(0.50),
        "p75": day_data.quantile(0.75),
        "p90": day_data.quantile(0.90),
    }
    return sanitize_for_json(percentiles)

# -------------------------
# /discomfort_index endpoint
# -------------------------
@app.get("/discomfort_index")
def discomfort_index_api(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    doy: int = Query(..., description="Day of year to calculate discomfort index for")
):
    df = get_power_data(lat, lon)
    df["heat_index"] = df.apply(lambda row: calculate_heat_index(row["T2M"], row["RH2M"]), axis=1)
    
    yearly = extract_yearly_values(df, var="heat_index", doy=doy)
    years = [int(y) for y in yearly.index.tolist()]
    values = [v for v in yearly.values]

    return {
        "ok": True,
        "variable": "heat_index",
        "doy": doy,
        "years": years,
        "values": values,
        "years_used": len(years),
        "generated_on": datetime.now(timezone.utc).isoformat()
    }

# -------------------------
# Extraction helpers
# -------------------------
def extract_yearly_values(df: pd.DataFrame, var: str,
                          doy: Optional[int]=None,
                          doy_start: Optional[int]=None,
                          doy_end: Optional[int]=None,
                          agg: str="mean"):
    var_upper = var.upper()
    if var == "heat_index":
        df["heat_index"] = df.apply(lambda row: calculate_heat_index(row["T2M"], row["RH2M"]), axis=1)
        var_upper = "heat_index"

    if doy is not None:
        sel = df[df["day_of_year"] == int(doy)]
        yearly = sel.groupby("year")[var_upper].first().dropna()
    else:
        sel = df[(df["day_of_year"] >= int(doy_start)) & (df["day_of_year"] <= int(doy_end))]
        if agg == "mean":
            yearly = sel.groupby("year")[var_upper].mean().dropna()
        elif agg == "sum":
            yearly = sel.groupby("year")[var_upper].sum().dropna()
        elif agg == "max":
            yearly = sel.groupby("year")[var_upper].max().dropna()
        else:
            raise ValueError("Unsupported agg")
    return yearly.sort_index()

# -------------------------
# Probability helpers (same as you used)
# -------------------------
def empirical_probability(values, threshold):
    n = len(values)
    if n == 0:
        return None
    exceed = int(np.sum(values > threshold))
    return float(exceed) / n, exceed, int(n)

def bootstrap_ci(values, threshold, n_boot=1000, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return None
    probs = []
    vals = np.array(values)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        probs.append(np.sum(sample > threshold) / n)
    lower = float(np.percentile(probs, (100 - ci) / 2))
    upper = float(np.percentile(probs, 100 - (100 - ci) / 2))
    return lower, upper, float(np.mean(probs)), float(np.std(probs))

# -------------------------
# Trend helpers
# -------------------------
def compute_linear_trend(years, values):
    if len(years) < 3:
        return None
    res = linregress(years, values)
    return {
        "slope_per_year": float(res.slope),
        "intercept": float(res.intercept),
        "r_value": float(res.rvalue),
        "p_value": float(res.pvalue),
        "stderr": float(res.stderr)
    }

def compute_exceedance_trend(years, values, threshold):
    bin_vals = (np.array(values) > threshold).astype(float)
    return compute_linear_trend(years, bin_vals), int(bin_vals.sum()), int(len(bin_vals))

def decadal_summary(years, values, threshold=None, decade_span=10):
    df = pd.DataFrame({"year": list(years), "value": list(values)})
    df = df.sort_values("year")
    df["decade_start"] = (df["year"] // decade_span) * decade_span
    out = []
    for start, g in df.groupby("decade_start"):
        k = "mean_value" if threshold is None else "prob_exceed"
        years_used = int(g["year"].nunique())
        if threshold is None:
            val = float(g["value"].mean())
        else:
            val = float((g["value"] > threshold).sum() / years_used) if years_used>0 else None
        out.append({
            "decade_start": int(start),
            "decade_end": int(start + decade_span - 1),
            "years_used": years_used,
            k: val
        })
    return out

def _sanitize_value(v):
    # Convert numpy scalar types, floats with nan/inf, and nested structures to JSON-safe Python values
    if v is None:
        return None
    # numpy scalar
    if isinstance(v, (np.floating, np.integer)):
        try:
            py = v.item()
        except Exception:
            py = float(v)
        if isinstance(py, float):
            if math.isnan(py) or math.isinf(py):
                return None
        return py
    # builtin float/int
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (int, str, bool)):
        return v
    # list / tuple / ndarray
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_sanitize_value(x) for x in v]
    # dict -> sanitize recursively
    if isinstance(v, dict):
        return {str(k): _sanitize_value(val) for k, val in v.items()}
    # pandas types that might sneak in
    try:
        # detect pandas Timestamp / numpy datetime, convert to ISO string
        import pandas as pd
        if isinstance(v, (pd.Timestamp, pd.DatetimeTZDtype)):
            return str(v)
    except Exception:
        pass
    # fallback: try turning into native python type
    try:
        return v if isinstance(v, (str, bool)) else (float(v) if isinstance(v, (np.number,)) else v)
    except Exception:
        # last resort
        return None

def sanitize_for_json(obj):
    """
    Recursively sanitize a nested structure (dict/list) replacing nan/inf and numpy scalars
    with JSON-serializable equivalents (None or Python primitives).
    """
    return _sanitize_value(obj)



# -------------------------
# /probability endpoint (same behaviour)
# -------------------------
@app.get("/probability")
def probability_api(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    var: str = Query(..., description="variable name, e.g., t2m_max"),
    threshold: float = Query(...),
    doy: Optional[int] = Query(None, ge=1, le=366),
    doy_start: Optional[int] = Query(None, ge=1, le=366),
    doy_end: Optional[int] = Query(None, ge=1, le=366),
    agg: str = Query("mean", regex="^(mean|sum|max)$"),
    n_boot: int = Query(1000, ge=0),
    min_years: int = Query(5, ge=1)
):
    df = get_power_data(lat, lon)
    var_upper = var.upper()
    if var == "heat_index":
        df["heat_index"] = df.apply(lambda row: calculate_heat_index(row["T2M"], row["RH2M"]), axis=1)
        var_upper = "heat_index"

    if var_upper not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset for this location.")

    if doy is None and (doy_start is None or doy_end is None):
        raise HTTPException(status_code=400, detail="Either 'doy' or both 'doy_start' and 'doy_end' required.")

    yearly = extract_yearly_values(df, var=var, doy=doy, doy_start=doy_start, doy_end=doy_end, agg=agg)

    if len(yearly) < min_years:
        raise HTTPException(status_code=400, detail={"reason":"insufficient_years","years_available":len(yearly),"min_years_required":min_years})

    p_tuple = empirical_probability(yearly.values, threshold)
    p, exceed, n = p_tuple
    ci = bootstrap_ci(yearly.values, threshold, n_boot=n_boot) if n_boot>0 else (None,None,None,None)

    resp = {
        "ok": True,
        "variable": var,
        "mode": "single_day" if doy is not None else "range",
        "doy": int(doy) if doy is not None else None,
        "doy_start": int(doy_start) if doy_start is not None else None,
        "doy_end": int(doy_end) if doy_end is not None else None,
        "aggregation_over_range": agg,
        "threshold": threshold,
        "years_used": n,
        "exceed_count": exceed,
        "probability": p,
        "bootstrap": {
            "n_boot": n_boot,
            "ci_95_lower": ci[0] if ci is not None else None,
            "ci_95_upper": ci[1] if ci is not None else None,
            "bootstrap_mean": ci[2] if ci is not None else None,
            "bootstrap_std": ci[3] if ci is not None else None
        },
        "computed_on": datetime.now(timezone.utc).isoformat()
    }
    return sanitize_for_json(resp)

# -------------------------
# /trend endpoint
# -------------------------
@app.get("/trend")
def trend_api(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    var: str = Query(..., description="variable name, e.g., t2m_max"),
    doy: Optional[int] = Query(None, ge=1, le=366),
    doy_start: Optional[int] = Query(None, ge=1, le=366),
    doy_end: Optional[int] = Query(None, ge=1, le=366),
    agg: str = Query("mean", regex="^(mean|sum|max)$"),
    threshold: Optional[float] = Query(None, description="optional threshold to compute exceedance trend"),
    decade_span: int = Query(10, ge=1),
    min_years: int = Query(10, ge=1)
):
    df = get_power_data(lat, lon)
    var_upper = var.upper()
    if var == "heat_index":
        df["heat_index"] = df.apply(lambda row: calculate_heat_index(row["T2M"], row["RH2M"]), axis=1)
        var_upper = "heat_index"

    if var_upper not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset for this location.")

    if doy is None and (doy_start is None or doy_end is None):
        raise HTTPException(status_code=400, detail="Either 'doy' or both 'doy_start' and 'doy_end' required.")

    yearly = extract_yearly_values(df, var=var, doy=doy, doy_start=doy_start, doy_end=doy_end, agg=agg)
    years = list(yearly.index.astype(int))
    values = list(yearly.values.astype(float))

    if len(years) < min_years:
        raise HTTPException(status_code=400, detail={"reason":"insufficient_years","years_available":len(years),"min_years_required":min_years})

    value_trend = compute_linear_trend(years, values)
    exceedance_trend = None
    if threshold is not None:
        exceedance_trend, exceed_count, n = compute_exceedance_trend(years, values, threshold)

    decadal = decadal_summary(years, values, threshold=threshold, decade_span=decade_span)

    resp = {
        "ok": True,
        "variable": var,
        "mode": "single_day" if doy is not None else "range",
        "doy": int(doy) if doy is not None else None,
        "doy_start": int(doy_start) if doy_start is not None else None,
        "doy_end": int(doy_end) if doy_end is not None else None,
        "aggregation_over_range": agg,
        "years_used": len(years),
        "value_trend": value_trend,
        "exceedance_trend": exceedance_trend,
        "decadal_summary": decadal,
        "computed_on": datetime.now(timezone.utc).isoformat()
    }
    return sanitize_for_json(resp)


# -------------------------
# History endpoint
# -------------------------

@app.get("/history")
def history_api(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    var: str = Query(..., description="variable name e.g. t2m_max"),
    doy: Optional[int] = Query(None, ge=1, le=366),
    doy_start: Optional[int] = Query(None, ge=1, le=366),
    doy_end: Optional[int] = Query(None, ge=1, le=366),
    agg: str = Query("mean", regex="^(mean|sum|max)$")
):
    df = get_power_data(lat, lon)
    var_upper = var.upper()
    if var == "heat_index":
        df["heat_index"] = df.apply(lambda row: calculate_heat_index(row["T2M"], row["RH2M"]), axis=1)
        var_upper = "heat_index"

    if var_upper not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset for this location.")
    if doy is None and (doy_start is None or doy_end is None):
        raise HTTPException(status_code=400, detail="Either 'doy' OR both 'doy_start' and 'doy_end' required")

    yearly = extract_yearly_values(df, var=var, doy=doy, doy_start=doy_start, doy_end=doy_end, agg=agg)
    years = [int(y) for y in yearly.index.tolist()]
    values = []
    for v in yearly.values:
        # convert NaN -> null in JSON
        if isinstance(v, (np.floating,)) and (np.isnan(v) or np.isinf(v)):
            values.append(None)
        else:
            # cast to python float/int
            try:
                values.append(float(v))
            except Exception:
                values.append(None)

    return {
        "ok": True,
        "variable": var,
        "mode": "single_day" if doy is not None else "range",
        "doy": int(doy) if doy is not None else None,
        "doy_start": int(doy_start) if doy_start is not None else None,
        "doy_end": int(doy_end) if doy_end is not None else None,
        "aggregation": agg,
        "years": years,
        "values": values,
        "years_used": len(years),
        "generated_on": datetime.now(timezone.utc).isoformat()
    }
