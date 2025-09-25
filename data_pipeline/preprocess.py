#!/usr/bin/env python3
"""
Preprocess a raw POWER CSV into cleaned CSV and day-of-year aggregations.

Usage:
    python data_pipeline/preprocess.py \
        --in data_pipeline/cache/sample_point_daily.csv \
        --out_clean data_pipeline/cache/sample_point_daily_clean.csv \
        --out_doy data_pipeline/cache/sample_point_doy_agg.csv
"""

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
import io

def read_csv_with_metadata(path):
    """
    Reads a CSV that may have a top block of commented metadata lines starting with '#'.
    Returns (metadata_dict, dataframe)
    """
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # separate header comment lines and CSV content
    header_lines = []
    csv_lines = []
    header_done = False
    for ln in lines:
        if not header_done and ln.lstrip().startswith("#"):
            header_lines.append(ln.strip()[1:].strip())
        else:
            header_done = True
            csv_lines.append(ln)

    # parse metadata lines like "key: value"
    for ln in header_lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            meta[k.strip()] = v.strip()

    if not csv_lines:
        raise ValueError("No CSV content found in file (maybe file only had metadata?).")

    csv_text = "".join(csv_lines)
    df = pd.read_csv(io.StringIO(csv_text))
    return meta, df

def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    possible_date_cols = [c for c in df.columns if "date" in c]
    if "date" not in df.columns and possible_date_cols:
        df.rename(columns={possible_date_cols[0]: "date"}, inplace=True)
    return df

def parse_dates(df):
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a date column")
    # try common POWER format YYYYMMDD
    try:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad > 0:
        print(f"Warning: {n_bad} rows had unparseable dates and will be dropped.")
        df = df[~df["date"].isna()].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    return df

def to_numeric_and_handle_missing(df):
    df = df.copy()
    for c in df.columns:
        if c in ["date", "year", "month", "day", "day_of_year"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def summarize_and_save(df, meta, out_clean, out_doy):
    os.makedirs(os.path.dirname(out_clean), exist_ok=True)
    os.makedirs(os.path.dirname(out_doy), exist_ok=True)

    # Save cleaned CSV with metadata header
    meta_lines = [f"# {k}: {v}" for k,v in meta.items()]
    meta_lines += [f"# cleaned_on: {datetime.utcnow().isoformat()}Z",
                   "# note: missing values are represented as empty fields (NaN)."]
    with open(out_clean, "w", encoding="utf-8") as f:
        for line in meta_lines:
            f.write(line + "\n")
        df.to_csv(f, index=False)

    # Prepare day_of_year aggregation
    vars_to_agg = [c for c in df.columns if c not in ["date","year","month","day","day_of_year"]]
    if not vars_to_agg:
        raise ValueError("No variables found to aggregate. Check input CSV columns.")

    agg_funcs = ["count", "mean", "std",
                 lambda x: x.quantile(0.10),
                 lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.50),
                 lambda x: x.quantile(0.75),
                 lambda x: x.quantile(0.90)]
    agg_names = ["count", "mean", "std", "p10", "p25", "median", "p75", "p90"]

    doy_agg = df.groupby("day_of_year")[vars_to_agg].agg(agg_funcs)

    # Flatten MultiIndex columns: map each (var, func) -> var_<agg_name>
    rename_map = {}
    for i, col in enumerate(doy_agg.columns):
        var = col[0]
        # pick agg name by order
        agg_name = agg_names[i % len(agg_names)]
        rename_map[col] = f"{var}_{agg_name}"
    doy_agg.rename(columns=rename_map, inplace=True)

    doy_agg.reset_index(inplace=True)

    meta2 = meta.copy()
    meta2["aggregation"] = "day_of_year statistics"
    meta2_lines = [f"# {k}: {v}" for k,v in meta2.items()]
    meta2_lines += [f"# cleaned_on: {datetime.utcnow().isoformat()}Z"]
    with open(out_doy, "w", encoding="utf-8") as f:
        for line in meta2_lines:
            f.write(line + "\n")
        doy_agg.to_csv(f, index=False)

    # Print summary
    print("CLEANED DATA SAVED:", out_clean)
    print("DOY AGGREGATION SAVED:", out_doy)
    print("Rows (clean):", len(df))
    print("Years:", df['year'].min(), "-", df['year'].max(), f"({df['year'].nunique()} distinct years)")
    print("Columns:", df.columns.tolist())
    missing_report = {v: float(df[v].isna().mean()) for v in vars_to_agg}
    print("Missing fraction per variable (0..1):")
    for k,v in missing_report.items():
        print(f"  {k}: {v:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="raw CSV path (with optional # metadata)")
    parser.add_argument("--out_clean", required=False, default="data_pipeline/cache/sample_point_daily_clean.csv")
    parser.add_argument("--out_doy", required=False, default="data_pipeline/cache/sample_point_doy_agg.csv")
    args = parser.parse_args()

    meta, df = read_csv_with_metadata(args.infile)
    df = normalize_columns(df)
    df = parse_dates(df)
    df = to_numeric_and_handle_missing(df)
    summarize_and_save(df, meta, args.out_clean, args.out_doy)

if __name__ == "__main__":
    main()

