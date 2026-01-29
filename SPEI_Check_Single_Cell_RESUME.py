#!/usr/bin/env python3
"""
SPEI_Check_Single_Cell.py

Pipeline (with intermediate outputs + plots):
1) Load daily inputs (pr, tasmin, tasmax) from comma-separated glob patterns
2) Select nearest grid cell at lat/lon -> 01_* (raw daily)
3) Convert pr to mm/day -> 02_*
4) Aggregate pr to monthly totals (mm/month) -> 03_*
5) Compute daily PET (Hargreaves) -> 04_*
6) Aggregate PET to monthly totals (mm/month) -> 05_*
7) Align monthly P and PET -> 07_* (and wb mm/month)

Adds: --resume to skip steps when outputs already exist.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def parse_comma_separated_paths(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]

def exists_all(*paths: Path) -> bool:
    return all(Path(p).exists() for p in paths)

def save_csv_1d(da: xr.DataArray, path_csv: Path):
    df = da.to_series().to_frame(name=da.name if da.name else "value")
    df.to_csv(path_csv)

def plot_timeseries_1d(da: xr.DataArray, path_png: Path, title: str, ylabel: str):
    # keep it simple and robust for long series
    plt.figure()
    da.to_pandas().plot(linewidth=0.6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("time")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def nearest_point_select(da: xr.DataArray, target_lat: float, target_lon: float) -> xr.DataArray:
    # Handle lon conventions (0..360 vs -180..180)
    lon = da["lon"]
    if lon.max() > 180 and target_lon < 0:
        target_lon = target_lon % 360
    elif lon.max() <= 180 and target_lon > 180:
        # unlikely, but safe
        target_lon = ((target_lon + 180) % 360) - 180

    return da.sel(lat=target_lat, lon=target_lon, method="nearest")

def hargreaves_pet_mm_day(tasmin_c: xr.DataArray, tasmax_c: xr.DataArray, lat_deg: float) -> xr.DataArray:
    """
    Daily Hargreaves PET (mm/day).

    Needs: tasmin, tasmax in °C.
    Uses Ra (extraterrestrial radiation) from FAO-56 style daily formula.
    """
    # convert time to day-of-year
    time = tasmin_c["time"]
    doy = xr.DataArray(time.dt.dayofyear, dims=["time"], coords={"time": time})

    lat_rad = np.deg2rad(lat_deg)

    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    # Solar declination
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    # Sunset hour angle
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1.0, 1.0))

    # Extraterrestrial radiation Ra (MJ m-2 day-1)
    Gsc = 0.0820  # MJ m-2 min-1
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )

    tmin = tasmin_c
    tmax = tasmax_c
    tmean = (tmin + tmax) / 2.0
    td = (tmax - tmin)

    # Hargreaves (FAO-56): ET0 = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * Ra
    pet = 0.0023 * (tmean + 17.8) * np.sqrt(xr.where(td > 0, td, 0)) * Ra
    pet = pet.clip(min=0)
    pet.name = "pet_mm_day"
    pet.attrs["units"] = "mm/day"
    return pet


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True, help="Comma-separated glob patterns to pr NetCDF files")
    parser.add_argument("--tasmin", required=True, help="Comma-separated glob patterns to tasmin NetCDF files")
    parser.add_argument("--tasmax", required=True, help="Comma-separated glob patterns to tasmax NetCDF files")
    parser.add_argument("--scale", type=int, default=3, help="SPEI accumulation scale (months). (Not computed here; kept for your workflow.)")
    parser.add_argument("--calibration", default="1979-2014", help="Calibration period (kept for your workflow)")
    parser.add_argument("--lat", type=float, required=True, help="Target latitude")
    parser.add_argument("--lon", type=float, required=True, help="Target longitude")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps whose outputs already exist; load intermediates instead of recomputing.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Filenames (match your log)
    f01_nc = outdir / "01_pr_raw_daily.nc"
    f01_csv = outdir / "01_pr_raw_daily.csv"
    f01_png = outdir / "01_pr_raw_daily.png"

    f02_nc = outdir / "02_pr_mm_day.nc"
    f02_csv = outdir / "02_pr_mm_day.csv"
    f02_png = outdir / "02_pr_mm_day.png"

    f03_nc = outdir / "03_pr_mm_month.nc"
    f03_csv = outdir / "03_pr_mm_month.csv"
    f03_png = outdir / "03_pr_mm_month.png"

    f04_nc = outdir / "04_pet_mm_day.nc"
    f04_csv = outdir / "04_pet_mm_day.csv"
    f04_png = outdir / "04_pet_mm_day.png"

    f05_nc = outdir / "05_pet_mm_month.nc"
    f05_csv = outdir / "05_pet_mm_month.csv"
    f05_png = outdir / "05_pet_mm_month.png"

    f07_prpet_nc = outdir / "07_pr_pet_aligned_mm_month.nc"
    f07_wb_nc = outdir / "07_wb_mm_month.nc"

    # -------------------------
    # STEP 1: load daily inputs (only if we need them)
    # -------------------------
    need_daily = True
    if args.resume:
        # if Steps 2-6 are all present, we don't need to open the big daily files
        if exists_all(f01_nc, f02_nc, f03_nc, f04_nc, f05_nc):
            need_daily = False

    if need_daily:
        print("\n=== STEP 1: Load daily inputs ===")
        pr_paths = parse_comma_separated_paths(args.pr)
        tasmin_paths = parse_comma_separated_paths(args.tasmin)
        tasmax_paths = parse_comma_separated_paths(args.tasmax)

        print(f"[INFO] Opening {len(pr_paths)} patterns for {args.pr}")
        ds_pr = xr.open_mfdataset(pr_paths, combine="by_coords")
        print(f"[INFO] Opening {len(tasmin_paths)} patterns for {args.tasmin}")
        ds_tmin = xr.open_mfdataset(tasmin_paths, combine="by_coords")
        print(f"[INFO] Opening {len(tasmax_paths)} patterns for {args.tasmax}")
        ds_tmax = xr.open_mfdataset(tasmax_paths, combine="by_coords")

        # Find variable names robustly
        pr_var = "pr" if "pr" in ds_pr.data_vars else list(ds_pr.data_vars)[0]
        tmin_var = "tasmin" if "tasmin" in ds_tmin.data_vars else list(ds_tmin.data_vars)[0]
        tmax_var = "tasmax" if "tasmax" in ds_tmax.data_vars else list(ds_tmax.data_vars)[0]

        pr = ds_pr[pr_var]
        tasmin = ds_tmin[tmin_var]
        tasmax = ds_tmax[tmax_var]
    else:
        pr = tasmin = tasmax = None

    # -------------------------
    # STEP 2: select nearest grid cell
    # -------------------------
    print("\n=== STEP 2: Select nearest grid cell ===")
    if args.resume and exists_all(f01_nc, f01_csv, f01_png):
        print("[RESUME] Step 2 outputs exist, skipping.")
        pr_cell = xr.open_dataset(f01_nc)[list(xr.open_dataset(f01_nc).data_vars)[0]]
    else:
        pr_cell = nearest_point_select(pr, args.lat, args.lon)
        pr_cell.name = "pr_raw"
        xr.Dataset({"pr_raw": pr_cell}).to_netcdf(f01_nc)
        save_csv_1d(pr_cell, f01_csv)
        plot_timeseries_1d(pr_cell, f01_png, "01 pr raw daily (native units)", "pr (native)")
        print("[SAVE]", f01_csv)
        print("[SAVE]", f01_nc)
        print("[PLOT]", f01_png)

    # -------------------------
    # STEP 3: convert precip to mm/day
    # -------------------------
    print("\n=== STEP 3: Convert precip to mm/day ===")
    if args.resume and exists_all(f02_nc, f02_csv, f02_png):
        print("[RESUME] Step 3 outputs exist, skipping.")
        pr_mm_day = xr.open_dataset(f02_nc)[list(xr.open_dataset(f02_nc).data_vars)[0]]
    else:
        # ISIMIP pr is typically kg m-2 s-1, equivalent to mm/s. Convert to mm/day:
        pr_mm_day = pr_cell * 86400.0
        pr_mm_day = pr_mm_day.astype("float32")
        pr_mm_day.name = "pr_mm_day"
        pr_mm_day.attrs["units"] = "mm/day"

        xr.Dataset({"pr_mm_day": pr_mm_day}).to_netcdf(f02_nc)
        save_csv_1d(pr_mm_day, f02_csv)
        plot_timeseries_1d(pr_mm_day, f02_png, "02 pr converted to mm/day", "mm/day")
        print("[SAVE]", f02_csv)
        print("[SAVE]", f02_nc)
        print("[PLOT]", f02_png)

    # -------------------------
    # STEP 4: monthly totals (mm/month)
    # -------------------------
    print("\n=== STEP 4: Aggregate precip to monthly totals (mm/month) ===")
    if args.resume and exists_all(f03_nc, f03_csv, f03_png):
        print("[RESUME] Step 4 outputs exist, skipping.")
        pr_mm_month = xr.open_dataset(f03_nc)[list(xr.open_dataset(f03_nc).data_vars)[0]]
    else:
        pr_mm_month = pr_mm_day.resample(time="MS").sum()
        pr_mm_month = pr_mm_month.astype("float32")
        pr_mm_month.name = "pr_mm_month"
        pr_mm_month.attrs["units"] = "mm/month"

        xr.Dataset({"pr_mm_month": pr_mm_month}).to_netcdf(f03_nc)
        save_csv_1d(pr_mm_month, f03_csv)
        plot_timeseries_1d(pr_mm_month, f03_png, "03 pr monthly totals (mm/month)", "mm/month")
        print("[SAVE]", f03_csv)
        print("[SAVE]", f03_nc)
        print("[PLOT]", f03_png)

    # -------------------------
    # STEP 5: daily PET (Hargreaves)
    # -------------------------
    print("\n=== STEP 5: Compute daily PET (Hargreaves) ===")
    if args.resume and exists_all(f04_nc, f04_csv, f04_png):
        print("[RESUME] Step 5 outputs exist, skipping.")
        pet_mm_day = xr.open_dataset(f04_nc)[list(xr.open_dataset(f04_nc).data_vars)[0]]
    else:
        if need_daily:
            tmin_cell = nearest_point_select(tasmin, args.lat, args.lon)
            tmax_cell = nearest_point_select(tasmax, args.lat, args.lon)
        else:
            # If we didn't load daily data, we must load them now for PET
            print("[INFO] Daily inputs not loaded; opening tasmin/tasmax to compute PET.")
            tasmin_paths = parse_comma_separated_paths(args.tasmin)
            tasmax_paths = parse_comma_separated_paths(args.tasmax)
            ds_tmin = xr.open_mfdataset(tasmin_paths, combine="by_coords")
            ds_tmax = xr.open_mfdataset(tasmax_paths, combine="by_coords")
            tmin_var = "tasmin" if "tasmin" in ds_tmin.data_vars else list(ds_tmin.data_vars)[0]
            tmax_var = "tasmax" if "tasmax" in ds_tmax.data_vars else list(ds_tmax.data_vars)[0]
            tmin_cell = nearest_point_select(ds_tmin[tmin_var], args.lat, args.lon)
            tmax_cell = nearest_point_select(ds_tmax[tmax_var], args.lat, args.lon)

        # Convert K->C if needed
        # Many ISIMIP tasmin/tasmax are in K
        if (tmin_cell.max() > 200) and (tmax_cell.max() > 200):
            tmin_c = tmin_cell - 273.15
            tmax_c = tmax_cell - 273.15
        else:
            tmin_c = tmin_cell
            tmax_c = tmax_cell

        pet_mm_day = hargreaves_pet_mm_day(tmin_c, tmax_c, args.lat).astype("float32")

        xr.Dataset({"pet_mm_day": pet_mm_day}).to_netcdf(f04_nc)
        save_csv_1d(pet_mm_day, f04_csv)
        plot_timeseries_1d(pet_mm_day, f04_png, "04 PET daily (Hargreaves) mm/day", "mm/day")
        print("[SAVE]", f04_csv)
        print("[SAVE]", f04_nc)
        print("[PLOT]", f04_png)

    # -------------------------
    # STEP 6: PET monthly totals (mm/month)
    # -------------------------
    print("\n=== STEP 6: Aggregate PET to monthly totals (mm/month) ===")
    if args.resume and exists_all(f05_nc, f05_csv, f05_png):
        print("[RESUME] Step 6 outputs exist, skipping.")
        pet_mm_month = xr.open_dataset(f05_nc)[list(xr.open_dataset(f05_nc).data_vars)[0]]
    else:
        pet_mm_month = pet_mm_day.resample(time="MS").sum()
        pet_mm_month = pet_mm_month.astype("float32")
        pet_mm_month.name = "pet_mm_month"
        pet_mm_month.attrs["units"] = "mm/month"

        xr.Dataset({"pet_mm_month": pet_mm_month}).to_netcdf(f05_nc)
        save_csv_1d(pet_mm_month, f05_csv)
        plot_timeseries_1d(pet_mm_month, f05_png, "05 PET monthly totals (mm/month)", "mm/month")
        print("[SAVE]", f05_csv)
        print("[SAVE]", f05_nc)
        print("[PLOT]", f05_png)

    # -------------------------
    # STEP 7: align monthly P and PET
    # -------------------------
    print("\n=== STEP 7: Align monthly P and PET ===")
    if args.resume and exists_all(f07_prpet_nc, f07_wb_nc):
        print("[RESUME] Step 7 outputs exist, skipping.")
    else:
        # ensure they are DataArrays
        pr_m = pr_mm_month
        pet_m = pet_mm_month
        pr_m, pet_m = xr.align(pr_m, pet_m, join="inner")

        xr.Dataset({"pr_mm_month": pr_m, "pet_mm_month": pet_m}).to_netcdf(f07_prpet_nc)
        xr.Dataset({"wb_mm_month": (pr_m - pet_m)}).to_netcdf(f07_wb_nc)

        print("[SAVE]", f07_prpet_nc)
        print("[SAVE]", f07_wb_nc)

    print("\n[DONE] Up to Step 7 complete.")

if __name__ == "__main__":
    main()

