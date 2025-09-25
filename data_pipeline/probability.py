#!/usr/bin/env python3
"""
Compute empirical exceedance probability and bootstrap CI from cleaned daily CSV.

Usage examples:
1) Single day (day_of_year):
   python data_pipeline/probability.py \
     --clean_csv data_pipeline/cache/sample_point_daily_clean.csv \
     --var t2m_max --doy 196 --threshold 32 --n_boot 1000

2) Day range (e.g., season) with aggregation across days per year:
   python data_pipeline/probability.py \
     --clean_csv data_pipeline/cache/sample_point_daily_clean.csv \
     --var prectot --doy_start 152 --doy_end 243 --agg sum --threshold 10 --n_boot 1000

Notes:
- For 'doy' mode, the script picks the single calendar day across years.
- For 'range' mode, for each year it aggregates days in the given doy range using 'agg' (mean, sum, max).
"""

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime

def load_clean_csv(path):
    return pd.read_csv(path, comment="#", parse_dates=["date"])

def extract_yearly_values(df, var, doy=None, doy_start=None, doy_end=None, agg="mean"):
    if doy is not None:
        sel = df[df["day_of_year"] == int(doy)]
        # For a single day, we want one value per year (the day's value)
        yearly = sel.groupby("year")[var].first().dropna()
    else:
        # range mode
        if doy_start is None or doy_end is None:
            raise ValueError("doy_start and doy_end required for range mode")
        sel = df[(df["day_of_year"] >= int(doy_start)) & (df["day_of_year"] <= int(doy_end))]
        if agg == "mean":
            yearly = sel.groupby("year")[var].mean().dropna()
        elif agg == "sum":
            yearly = sel.groupby("year")[var].sum().dropna()
        elif agg == "max":
            yearly = sel.groupby("year")[var].max().dropna()
        else:
            raise ValueError("Unsupported agg: choose mean,sum,max")
    return yearly.sort_index()

def empirical_probability(values, threshold):
    # values: 1D numeric array or Pandas Series
    n = len(values)
    if n == 0:
        return None
    exceed = np.sum(values > threshold)
    return float(exceed) / n, int(exceed), int(n)

def bootstrap_ci(values, threshold, n_boot=1000, ci=95, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return None
    probs = []
    vals = np.array(values)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        p = np.sum(sample > threshold) / n
        probs.append(p)
    lower = np.percentile(probs, (100 - ci) / 2)
    upper = np.percentile(probs, 100 - (100 - ci) / 2)
    return float(lower), float(upper), float(np.mean(probs)), float(np.std(probs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_csv", required=True, help="cleaned daily CSV (with day_of_year and year columns)")
    parser.add_argument("--var", required=True, help="variable column name (e.g., t2m_max, prectot)")
    parser.add_argument("--doy", type=int, help="day_of_year (1..366) for single-day mode")
    parser.add_argument("--doy_start", type=int, help="start day_of_year for range mode")
    parser.add_argument("--doy_end", type=int, help="end day_of_year for range mode")
    parser.add_argument("--agg", choices=["mean","sum","max"], default="mean", help="aggregation over day range (for range mode)")
    parser.add_argument("--threshold", type=float, required=True, help="threshold to test exceedance (same units as variable)")
    parser.add_argument("--n_boot", type=int, default=1000, help="bootstrap iterations")
    parser.add_argument("--min_years", type=int, default=10, help="minimum years required to return result")
    parser.add_argument("--out_csv", default=None, help="optional: path to save per-year values used (csv)")

    args = parser.parse_args()

    df = load_clean_csv(args.clean_csv)
    # ensure var exists
    if args.var not in df.columns:
        raise ValueError(f"Variable '{args.var}' not found in CSV columns: {df.columns.tolist()}")

    if args.doy is None and (args.doy_start is None or args.doy_end is None):
        raise ValueError("Either --doy or both --doy_start and --doy_end must be provided")

    yearly_series = extract_yearly_values(df, var=args.var, doy=args.doy, doy_start=args.doy_start, doy_end=args.doy_end, agg=args.agg)

    if args.out_csv:
        yearly_series.to_csv(args.out_csv, header=[args.var])

    # require minimum years
    if len(yearly_series) < args.min_years:
        result = {
            "ok": False,
            "reason": "insufficient_years",
            "years_available": int(len(yearly_series)),
            "min_years_required": int(args.min_years),
        }
        print(json.dumps(result, indent=2))
        return

    p, exceed, n = empirical_probability(yearly_series.values, args.threshold)
    ci = bootstrap_ci(yearly_series.values, args.threshold, n_boot=args.n_boot)

    out = {
        "ok": True,
        "variable": args.var,
        "mode": "single_day" if args.doy is not None else "range",
        "doy": int(args.doy) if args.doy is not None else None,
        "doy_start": int(args.doy_start) if args.doy_start is not None else None,
        "doy_end": int(args.doy_end) if args.doy_end is not None else None,
        "aggregation_over_range": args.agg,
        "threshold": args.threshold,
        "years_used": n,
        "years_list": list(map(int, yearly_series.index.tolist())),
        "exceed_count": exceed,
        "probability": p,
        "bootstrap": {
            "n_boot": args.n_boot,
            "ci_95_lower": ci[0] if ci is not None else None,
            "ci_95_upper": ci[1] if ci is not None else None,
            "bootstrap_mean": ci[2] if ci is not None else None,
            "bootstrap_std": ci[3] if ci is not None else None
        },
        "computed_on": datetime.utcnow().isoformat() + "Z"
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
