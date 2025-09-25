# backend/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone

app = FastAPI(title="Will-It-Rain: Probability API")

CLEAN_CSV = "data_pipeline/cache/sample_point_daily_clean.csv"

def load_clean_csv(path=CLEAN_CSV):
    try:
        return pd.read_csv(path, comment="#", parse_dates=["date"])
    except Exception as e:
        raise RuntimeError(f"Cannot load cleaned CSV: {e}")

def extract_yearly_values(df, var, doy=None, doy_start=None, doy_end=None, agg="mean"):
    if doy is not None:
        sel = df[df["day_of_year"] == int(doy)]
        yearly = sel.groupby("year")[var].first().dropna()
    else:
        sel = df[(df["day_of_year"] >= int(doy_start)) & (df["day_of_year"] <= int(doy_end))]
        if agg == "mean":
            yearly = sel.groupby("year")[var].mean().dropna()
        elif agg == "sum":
            yearly = sel.groupby("year")[var].sum().dropna()
        elif agg == "max":
            yearly = sel.groupby("year")[var].max().dropna()
        else:
            raise ValueError("Unsupported agg")
    return yearly.sort_index()

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

class ProbabilityResponse(BaseModel):
    ok: bool
    variable: str
    mode: str
    doy: Optional[int]
    doy_start: Optional[int]
    doy_end: Optional[int]
    aggregation_over_range: Optional[str]
    threshold: float
    years_used: int
    exceed_count: int
    probability: float
    bootstrap: dict
    computed_on: str

@app.get("/probability", response_model=ProbabilityResponse)
def probability_api(
    var: str = Query(..., description="variable name, e.g., t2m_max"),
    threshold: float = Query(...),
    doy: Optional[int] = Query(None, ge=1, le=366),
    doy_start: Optional[int] = Query(None, ge=1, le=366),
    doy_end: Optional[int] = Query(None, ge=1, le=366),
    agg: str = Query("mean", regex="^(mean|sum|max)$"),
    n_boot: int = Query(1000, ge=0),
    min_years: int = Query(5, ge=1)
):
    df = load_clean_csv()
    if var not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{var}' not found")

    if doy is None and (doy_start is None or doy_end is None):
        raise HTTPException(status_code=400, detail="Either 'doy' or both 'doy_start' and 'doy_end' required")

    yearly = extract_yearly_values(df, var=var, doy=doy, doy_start=doy_start, doy_end=doy_end, agg=agg)

    if len(yearly) < min_years:
        raise HTTPException(status_code=400, detail={"reason":"insufficient_years","years_available":len(yearly),"min_years_required":min_years})

    prob_tuple = empirical_probability(yearly.values, threshold)
    p, exceed, n = prob_tuple
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
            "ci_95_lower": ci[0],
            "ci_95_upper": ci[1],
            "bootstrap_mean": ci[2],
            "bootstrap_std": ci[3]
        },
        "computed_on": datetime.now(timezone.utc).isoformat()
    }
    return resp

