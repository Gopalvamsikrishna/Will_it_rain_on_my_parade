#!/usr/bin/env python3
"""
Fetch daily point data from NASA POWER and save as CSV.

Usage:
  python data_pipeline/fetch_power_point.py --lat 12.97 --lon 77.59 --start 19900101 --end 20231231
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


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
                # Print body for debugging then raise for status to let caller handle it
                print(f"ERROR: Request failed (status {r.status_code}) for {r.url}")
                print("Response body:", r.text)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            last_exc = e
            # If 4xx (client) on first try, don't retry; show body and break
            status = getattr(e.response, "status_code", None)
            if status and 400 <= status < 500:
                # show response and break immediately
                print(f"HTTP Error {status} (no retry): {e}")
                print("Response body:", e.response.text if e.response is not None else "(no body)")
                raise
            # otherwise retry
            print(f"HTTP error on attempt {attempt}/{retries}: {e}. Retrying in {backoff} seconds...")
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"Request exception on attempt {attempt}/{retries}: {e}. Retrying in {backoff} seconds...")
        time.sleep(backoff)
        backoff *= 2

    # If we finish attempts without success, raise the last exception
    raise last_exc if last_exc is not None else RuntimeError("Unknown error fetching POWER data")


def json_to_dataframe(j, params):
    # POWER returns 'properties' -> 'parameter' -> param -> {date: value}
    param_block = j.get("properties", {}).get("parameter", {})
    if not param_block:
        raise ValueError("No parameter block found in JSON response.")

    # get sorted common dates from first parameter (if any)
    first = next(iter(param_block.values()))
    dates = sorted(first.keys())

    rows = []
    for d in dates:
        row = {"date": datetime.strptime(d, "%Y%m%d").date()}
        for p in params:
            val = param_block.get(p, {}).get(d, None)
            # POWER sometimes returns string "null" or None — handle robustly
            if val is None:
                row[p] = float("nan")
            else:
                try:
                    row[p] = float(val)
                except Exception:
                    row[p] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    # set date column as index for convenience
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def save_csv(df, out_path, metadata):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # write metadata as top commented header
    meta_lines = [f"# {k}: {v}" for k, v in metadata.items()]
    with open(out_path, "w", encoding="utf-8") as f:
        for line in meta_lines:
            f.write(line + "\n")
        df.to_csv(f, index=True)


def chunk_date_ranges(start_str, end_str, days=365):
    """Yield (start, end) strings in YYYYMMDD covering the full range in chunks."""
    start = datetime.strptime(start_str, "%Y%m%d").date()
    end = datetime.strptime(end_str, "%Y%m%d").date()
    cur_start = start
    while cur_start <= end:
        cur_end = min(end, cur_start + timedelta(days=days - 1))
        yield cur_start.strftime("%Y%m%d"), cur_end.strftime("%Y%m%d")
        cur_start = cur_end + timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--start", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--end", type=str, required=True, help="YYYYMMDD")
    parser.add_argument("--out", type=str, default="data_pipeline/cache/sample_point_daily.csv")
    parser.add_argument("--community", type=str, default="SB", help="POWER community (e.g. SB, AG, RE)")
    parser.add_argument("--chunk-days", type=int, default=365, help="Chunk size in days for API calls (default 365)")
    args = parser.parse_args()

    # Corrected parameter names (WS10M instead of WS_10M)
    params = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS10M"]

    # If the full range is large, break into chunks and combine results
    dfs = []
    for s_chunk, e_chunk in chunk_date_ranges(args.start, args.end, days=args.chunk_days):
        print(f"Fetching {s_chunk} -> {e_chunk} ...")
        try:
            j = fetch_power(args.lat, args.lon, s_chunk, e_chunk, params, community=args.community)
        except Exception as e:
            print("Failed to fetch chunk:", s_chunk, e_chunk, "Error:", e)
            raise
        df_chunk = json_to_dataframe(j, params)
        dfs.append(df_chunk)

    if not dfs:
        raise RuntimeError("No data fetched for the requested range.")

    # concatenate along dates, sort by index, and deduplicate
    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    metadata = {
        "source": "NASA POWER API",
        "api_endpoint": POWER_BASE,
        "latitude": args.lat,
        "longitude": args.lon,
        "start": args.start,
        "end": args.end,
        "parameters": ",".join(params),
        "retrieved": datetime.utcnow().isoformat() + "Z",
        "notes": "Units as returned by POWER. See NASA POWER docs for parameter definitions.",
    }
    save_csv(df, args.out, metadata)
    print(f"Saved {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

