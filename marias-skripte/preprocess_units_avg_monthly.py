#!/usr/bin/env python3
# ============================================================
# preprocess_units_and_monthly.py
# Converts WFDE5 or ISIMIP daily climate data to monthly files
# Applies unit conversions, combines Rainf + Snowf to total precip
# ============================================================

import os
import time
import xarray as xr
import numpy as np
import pandas as pd

# === USER PATHS ===============================================================
INPUT_ROOT = "/p/projects/ou/labs/ai/mariafe/data"
OUTPUT_ROOT = os.path.join(INPUT_ROOT, "monthly")

# Variables expected in the daily structure
VAR_GROUPS = {
    "fluxes_sum": ["Rainf", "Snowf"],  # summed → mm/day
    "temps_mean": ["Tair", "tmin", "tmax"],  # averaged → °C
    "radiation_sum": ["SWdown", "LWdown"],  # W/m², mean over time
    "met_mean": ["wind", "Qair", "PSurf"],  # other meteorological
}

# add near the top
VAR_NAME_MAP = {
    "tmin":  ("Tair", "tmin"),
    "tmax":  ( "Tair","tmax"),
    "Rainf": ("Rainf","Rainf"),
    "Snowf": ("Snowf","Snowf"),
    "wind":  ("Wind","wind"),
    "Tair":  ("Tair","Tair"),
    "SWdown":("SWdown","SWdown"),
    "LWdown":("LWdown","LWdown"),
    "Qair":  ("Qair","Qair"),
    "PSurf": ("PSurf","PSurf"),
}

# === HELPER FUNCTIONS =========================================================
def convert_units(ds, var):
    """Convert variable units to standard ones."""
    da = ds[var]
    u = da.attrs.get("units", "").lower()

    # Temperature K → °C
    if "k" == u or u == "kelvin":
        da = da - 273.15
        da.attrs["units"] = "°C"

    # Flux kg m-2 s-1 → mm/day
    elif "kg m-2 s-1" in u:
        da = da * 86400.0
        da.attrs["units"] = "mm/day"

    # Pressure Pa → kPa
    elif u == "pa":
        da = da / 1000.0
        da.attrs["units"] = "kPa"

    # Radiation (W/m²): keep, average later
    elif "w m-2" in u:
        da.attrs["units"] = "W/m²"

    da.attrs["processed_unit_conversion"] = "yes"
    return da


def aggregate_monthly(da, method="mean"):
    """Aggregate daily → monthly."""
    if method == "sum":
        return da.resample(time="1MS").sum(skipna=True)
    else:
        return da.resample(time="1MS").mean(skipna=True)

def process_variable(var, in_path, out_path):
    """
    Convert units and aggregate daily -> monthly for one variable folder.
    Uses VAR_NAME_MAP to read the actual variable name inside files and
    standardize the output variable name.
    """
    files = sorted([f for f in os.listdir(in_path) if f.endswith(".nc")])
    if not files:
        print(f"️  No files found for {var} in {in_path}")
        return

    os.makedirs(out_path, exist_ok=True)

    # Map folder name -> (actual var in file, standardized output var name)
    actual_var, out_var = VAR_NAME_MAP.get(var, (var, var))

    start_time = time.time()
    n_files = len(files)
    print(f"\n=== Processing {var} -> read '{actual_var}', write as '{out_var}' ({n_files} files) ===")

    monthly_datasets = []
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(in_path, fname)
        ds = xr.open_dataset(fpath)

        if actual_var not in ds.data_vars:
            print(f"  Skipping {fname}: variable '{actual_var}' not found (has {list(ds.data_vars)})")
            ds.close()
            continue

        da = convert_units(ds, actual_var)

        # Choose aggregation rule
        method = "sum" if out_var in ["Rainf", "Snowf", "Precip"] else "mean"
        da_m = aggregate_monthly(da, method)

        # Standardize output variable name
        da_m = da_m.rename(out_var)

        monthly_datasets.append(da_m)
        ds.close()

        if i % 25 == 0 or i == n_files:
            print(f"  Processed {i}/{n_files} files...")

    if monthly_datasets:
        combined = xr.concat(monthly_datasets, dim="time")
        # remove any duplicate monthly timestamps
        combined = combined.sel(time=~combined.get_index("time").duplicated())
        out_file = os.path.join(out_path, f"{out_var}_monthly.nc")
        combined.to_netcdf(out_file)
        print(f" Saved monthly {out_var} to {out_file}")
        print(f"   -> time range: {str(combined.time.values[0])} to {str(combined.time.values[-1])}")
    else:
        print(f"  No monthly data produced for {var}")

    print(f"Finished {var} in {(time.time() - start_time)/60:.2f} min")


# === MAIN WORKFLOW ============================================================
def main():
    t0 = time.time()
    print("\n=== START Monthly Preprocessing ===")

    # Handle Rainf + Snowf combined as total precipitation
    rain_path = os.path.join(INPUT_ROOT, "Rainf")
    snow_path = os.path.join(INPUT_ROOT, "Snowf")
    if os.path.exists(rain_path) and os.path.exists(snow_path):
        out_path = os.path.join(OUTPUT_ROOT, "Precip")
        os.makedirs(out_path, exist_ok=True)
        print("\n=== Combining Rainf + Snowf to total Precip ===")
        all_rain = sorted([os.path.join(rain_path, f) for f in os.listdir(rain_path) if f.endswith(".nc")])
        all_snow = sorted([os.path.join(snow_path, f) for f in os.listdir(snow_path) if f.endswith(".nc")])
        monthly_datasets = []
        for fr, fs in zip(all_rain, all_snow):
            dr = xr.open_dataset(fr)
            ds = xr.open_dataset(fs)
            rain = convert_units(dr, "Rainf")
            snow = convert_units(ds, "Snowf")
            total = rain + snow
            total.attrs["units"] = "mm/day"
            total.attrs["description"] = "Total precipitation (Rainf+Snowf)"
            total_m = aggregate_monthly(total, method="sum")
            monthly_datasets.append(total_m)
            dr.close(); ds.close()
        combined = xr.concat(monthly_datasets, dim="time")
        combined = combined.sel(time=~combined.get_index("time").duplicated())
        combined.to_netcdf(os.path.join(out_path, "Precip_monthly.nc"))
        print(f" Saved total monthly Precip to {out_path}")

    # Process all other variables individually
    for group, vars_ in VAR_GROUPS.items():
        for var in vars_:
            in_path = os.path.join(INPUT_ROOT, var)
            out_path = os.path.join(OUTPUT_ROOT, var)
            if os.path.exists(in_path):
                process_variable(var, in_path, out_path)
            else:
                print(f" {var} folder not found, skipping.")

    print(f"\n=== ALL DONE ===\nTotal time: {(time.time()-t0)/60:.2f} minutes")


# === ENTRY POINT ==============================================================
if __name__ == "__main__":
    main()

