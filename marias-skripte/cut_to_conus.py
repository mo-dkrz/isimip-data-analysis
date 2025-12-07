#!/usr/bin/env python
"""
Crop ISIMIP3b SSP (daily, bias-adjusted, global) data to CONUS extent.

Input root:
    /p/projects/isimip/isimip/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/

Output root:
    /p/projects/ou/labs/ai/mariafe/data/

For each scenario (ssp126, ssp370, ssp585) and model
(GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR, MRI-ESM2-0, UKESM1-0-LL),
the script creates:

    <out_root>/<scenario>/<model>_conus/

and writes all *_conus_*.nc files there.

You can re-run the script: existing output files are skipped.
"""

import os
import glob
import time
import xarray as xr
import numpy as np

# -------------------------------
# USER SETTINGS
# -------------------------------

IN_ROOT = "/p/projects/isimip/isimip/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily"
OUT_ROOT = "/p/projects/ou/labs/ai/mariafe/data"

SCENARIOS = ["ssp126", "ssp370", "ssp585"]
MODELS = [
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "UKESM1-0-LL",
]

# CONUS bounds
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.0, 49.5


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def infer_lat_lon_names(ds):
    """Return (lat_name, lon_name) for a dataset."""
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]

    lat_name = next((n for n in lat_candidates if n in ds.coords), None)
    lon_name = next((n for n in lon_candidates if n in ds.coords), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Could not find lat/lon coordinates in dataset.")

    return lat_name, lon_name


def crop_to_conus(ds):
    """Return dataset cropped to CONUS, handling lat order."""
    lat_name, lon_name = infer_lat_lon_names(ds)

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # lon: assume increasing; if 0..360, convert bounds to that (not expected here, but safe)
    lon_min, lon_max = LON_MIN, LON_MAX
    if lons.min() >= 0 and lons.max() > 180:
        lon_min = (LON_MIN + 360) % 360
        lon_max = (LON_MAX + 360) % 360

    # lat can be increasing or decreasing
    if lats[0] < lats[-1]:  # increasing
        lat_slice = slice(LAT_MIN, LAT_MAX)
    else:                   # decreasing (e.g. 90 -> -90)
        lat_slice = slice(LAT_MAX, LAT_MIN)

    ds_conus = ds.sel(
        {lon_name: slice(lon_min, lon_max),
         lat_name: lat_slice}
    )

    # drop coordinate encodings (sometimes cause trouble when writing)
    for c in ds_conus.coords:
        ds_conus[c].encoding = {}

    return ds_conus


def make_out_name(in_path):
    """Generate output filename from input filename."""
    base = os.path.basename(in_path)
    if "_global_" in base:
        return base.replace("_global_", "_conus_")
    elif "_daily_" in base:
        return base.replace("_daily_", "_conus_daily_")
    else:
        # fallback
        if base.endswith(".nc"):
            return base[:-3] + "_conus.nc"
        return base + "_conus.nc"


# -------------------------------
# MAIN PROCESSING
# -------------------------------

def main():
    t_start_total = time.perf_counter()
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    print("=== Start cropping ISIMIP3b SSP data to CONUS ===")
    print(f"Input root : {IN_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Scenarios  : {SCENARIOS}")
    print(f"Models     : {MODELS}")
    print(f"CONUS bounds: lon [{LON_MIN}, {LON_MAX}], lat [{LAT_MIN}, {LAT_MAX}]")
    print("===============================================")

    for scen in SCENARIOS:
        in_scen_dir = os.path.join(IN_ROOT, scen)
        if not os.path.isdir(in_scen_dir):
            print(f"\n[WARNING] Scenario directory not found, skipping: {in_scen_dir}")
            continue

        print(f"\n=== Scenario: {scen} ===")

        for model in MODELS:
            in_model_dir = os.path.join(in_scen_dir, model)
            if not os.path.isdir(in_model_dir):
                print(f"[WARNING] Model directory not found, skipping: {in_model_dir}")
                continue

            out_model_dir = os.path.join(OUT_ROOT, scen, f"{model}_conus")
            os.makedirs(out_model_dir, exist_ok=True)

            print(f"\n--- Model: {model} ---")
            print(f"Input dir : {in_model_dir}")
            print(f"Output dir: {out_model_dir}")

            files = sorted(glob.glob(os.path.join(in_model_dir, "*.nc")))
            if not files:
                print("[INFO] No .nc files found, skipping model.")
                continue

            for fpath in files:
                out_name = make_out_name(fpath)
                out_path = os.path.join(out_model_dir, out_name)

                # Skip if already processed
                if os.path.exists(out_path):
                    print(f"[SKIP] Output exists: {out_path}")
                    n_skipped += 1
                    continue

                print(f"\n[PROCESS] {fpath}")
                t0 = time.perf_counter()

                try:
                    # Open lazily (don't load into memory all at once)
                    ds = xr.open_dataset(fpath)
                    ds_conus = crop_to_conus(ds)

                    # Optional: print basic spatial info once per file
                    lat_name, lon_name = infer_lat_lon_names(ds_conus)
                    lats = ds_conus[lat_name].values
                    lons = ds_conus[lon_name].values
                    print(f"  -> lon: {lons.min()} to {lons.max()} (n={lons.size})")
                    print(f"  -> lat: {lats.min()} to {lats.max()} (n={lats.size})")

                    # Write output
                    ds_conus.to_netcdf(out_path, format="NETCDF4")

                    # Close datasets
                    ds.close()
                    ds_conus.close()

                    dt = time.perf_counter() - t0
                    print(f"[DONE] Saved: {out_path}")
                    print(f"       Processing time: {dt:.1f} s")
                    n_processed += 1

                except Exception as e:
                    dt = time.perf_counter() - t0
                    print(f"[ERROR] Failed processing file: {fpath}")
                    print(f"        Error: {e}")
                    print(f"        Time until error: {dt:.1f} s")
                    n_failed += 1
                    # continue with next file

    total_dt = time.perf_counter() - t_start_total
    print("\n===============================================")
    print("Finished cropping ISIMIP3b SSP data to CONUS.")
    print(f"Processed files : {n_processed}")
    print(f"Skipped (exists): {n_skipped}")
    print(f"Failed          : {n_failed}")
    print(f"Total runtime   : {total_dt/60:.1f} min")
    print("===============================================")


if __name__ == "__main__":
    main()

