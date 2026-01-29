
#!/usr/bin/env python3
"""
Reproduce SPEI (single grid cell) with checkpoints + plots after each step.

Workflow (daily inputs):
1) Load daily precip (pr) [kg m-2 s-1] and daily tasmin/tasmax [K or °C]
2) Select nearest grid cell to TARGET_LAT/LON
3) Convert pr to mm/day, then aggregate to monthly totals (mm/month)
4) Compute daily PET with Hargreaves-Samani (mm/day), then aggregate to monthly totals (mm/month)
5) Align monthly pr and monthly PET
6) Compute monthly water balance WB = P - PET (mm/month)
7) Compute rolling accumulation of WB over 'scale' months
8) Fit log-logistic distribution (scipy.stats.fisk) per calendar month on calibration period
   - If calibration values include <=0, shift by (-min + 1) like in your package
9) Transform CDF -> standard normal quantiles => SPEI
10) Save checkpoints (NetCDF + CSV) and plots (PNG)

NOTE:
- This is optimized for *one grid cell*, so it is easy to inspect and plot.
- It is a faithful reproduction of your repo's approach (spei.py + pet.py + utils.py).
"""

import argparse
import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats


# ----------------------------
# User defaults (can override via CLI)
# ----------------------------
TARGET_LAT_DEFAULT = 40.10
TARGET_LON_DEFAULT = -88.25


# ----------------------------
# I/O helpers
# ----------------------------
def _glob_multi(pattern: str):
    """
    Allow comma-separated glob patterns like:
      "/hist/pr*.nc,/ssp585/pr*.nc"
    """
    patterns = [p.strip() for p in pattern.split(",")]
    files = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if not matched:
            print(f"[WARN] No files for pattern: {p}")
        files.extend(matched)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def open_var(pattern: str, var: str | None = None, chunks=None) -> xr.DataArray:
    """
    Open a dataset from glob pattern(s) and return a DataArray variable.
    If var is None, uses the first data variable.
    """
    files = _glob_multi(pattern)
    print(f"[INFO] Opening {len(files)} files for {pattern}")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)

    if var is None:
        # choose first data var
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(f"No data variables found in files: {pattern}")
        var = data_vars[0]

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")

    return ds[var]


def select_cell(da: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
    """
    Select nearest grid cell and return time series (1D DataArray).
    """
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError("Expected 'lat' and 'lon' dimensions in DataArray.")
    cell = da.sel(lat=lat, lon=lon, method="nearest")
    cell = cell.drop_vars(
        [v for v in cell.coords if v not in cell.dims and v != "time"],
        errors="ignore"
    )
    return cell


def ensure_celsius(temp: xr.DataArray) -> xr.DataArray:
    """
    Convert Kelvin to °C if values look like Kelvin.
    Matches your package logic (mean > 100 -> Kelvin).
    """
    sample = temp.isel(time=0).values
    if np.nanmean(sample) > 100:
        temp = temp - 273.15
        temp.attrs["units"] = "degC"
    return temp


def convert_pr_to_mm_day(pr: xr.DataArray) -> xr.DataArray:
    """
    pr: kg m-2 s-1 == mm/s
    Convert to mm/day by *86400.
    Matches utils.convert_precip_units()
    """
    out = pr * 86400.0
    out.attrs["units"] = "mm/day"
    return out


# ----------------------------
# PET (Hargreaves) for ONE CELL
# ----------------------------
def ra_mm_day_for_lat_doy(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """
    Extraterrestrial radiation Ra, converted to equivalent mm/day.
    This mirrors the logic in your pet.py lookup (but scalar lat).

    Using FAO-56 style equations:
      Gsc = 0.0820 MJ m-2 min-1
      Ra (MJ m-2 day-1) = (24*60/pi)*Gsc*dr*(ws*sin(lat)*sin(decl) + cos(lat)*cos(decl)*sin(ws))
      Convert MJ m-2 day-1 to mm/day by /2.45
    """
    Gsc = 0.0820
    lat = np.deg2rad(lat_deg)

    doy = doy.astype(float)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)

    arccos_arg = -np.tan(lat) * np.tan(decl)
    arccos_arg = np.clip(arccos_arg, -1.0, 1.0)
    ws = np.arccos(arccos_arg)

    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat) * np.sin(decl) +
        np.cos(lat) * np.cos(decl) * np.sin(ws)
    )
    Ra_mm = Ra / 2.45
    return Ra_mm.astype(np.float64)


def pet_hargreaves_daily_cell(tmin_c: xr.DataArray, tmax_c: xr.DataArray, lat_deg: float) -> xr.DataArray:
    """
    Hargreaves-Samani PET for one grid cell, daily.

    PET = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
      where Ra in mm/day equivalent (as used in your code)
    Output: mm/day
    """
    # Ensure aligned times
    tmin_c, tmax_c = xr.align(tmin_c, tmax_c, join="inner")

    doy = tmin_c.time.dt.dayofyear.values
    Ra = ra_mm_day_for_lat_doy(lat_deg, doy)  # (n_time,)

    tmean = (tmin_c.values + tmax_c.values) / 2.0
    tr = np.maximum(tmax_c.values - tmin_c.values, 0.0)

    pet = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(tr)
    pet = np.maximum(pet, 0.0)

    out = xr.DataArray(pet, coords={"time": tmin_c.time}, dims=("time",))
    out.attrs["units"] = "mm/day"
    out.attrs["long_name"] = "Potential Evapotranspiration (Hargreaves) - cell"
    return out


# ----------------------------
# SPEI distribution fit/transform for ONE CELL
# ----------------------------
def spei_fit_transform_cell(
    wb_acc: xr.DataArray,
    calibration_period: tuple[int, int],
    min_samples: int | None = None,
) -> xr.DataArray:
    """
    Fit log-logistic distribution (scipy.stats.fisk) per calendar month on calibration period,
    then transform to standard normal quantiles (SPEI).

    This is the cell-level version of your spei.py logic, including shifting if values <= 0.

    Returns: xr.DataArray SPEI time series (same time index as wb_acc)
    """
    years = wb_acc.time.dt.year.values
    months = wb_acc.time.dt.month.values

    cal_mask = (years >= calibration_period[0]) & (years <= calibration_period[1])

    cal_years = calibration_period[1] - calibration_period[0] + 1
    if min_samples is None:
        min_samples = max(5, cal_years - 1)
        min_samples = min(min_samples, cal_years)

    x = wb_acc.values.astype(np.float64)
    spei = np.full_like(x, np.nan, dtype=np.float64)

    for m in range(1, 13):
        month_idx = np.where(months == m)[0]
        cal_idx = np.where((months == m) & cal_mask)[0]

        if len(month_idx) == 0:
            continue

        cal_vals = x[cal_idx]
        cal_vals = cal_vals[~np.isnan(cal_vals)]

        if len(cal_vals) < min_samples:
            continue

        # Shift so calibration data is strictly positive if needed (your approach)
        shift = 0.0
        if np.min(cal_vals) <= 0:
            shift = -np.min(cal_vals) + 1.0

        cal_shifted = cal_vals + shift

        # Fit Fisk (log-logistic)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                c, loc, scale = stats.fisk.fit(cal_shifted, floc=0)
            except Exception:
                continue

        # Transform all values for this month
        all_vals = x[month_idx]
        valid = ~np.isnan(all_vals)
        if not np.any(valid):
            continue

        all_shifted = all_vals[valid] + shift

        # Non-positive after shift -> tiny CDF
        cdf = np.full(all_shifted.shape, 1e-6, dtype=np.float64)
        pos = all_shifted > 0
        if np.any(pos):
            cdf[pos] = stats.fisk.cdf(all_shifted[pos], c, loc=0, scale=scale)

        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei_vals = stats.norm.ppf(cdf)

        # write back
        tmp = np.full(all_vals.shape, np.nan, dtype=np.float64)
        tmp[valid] = spei_vals
        spei[month_idx] = tmp

    out = xr.DataArray(spei, coords={"time": wb_acc.time}, dims=("time",))
    out.attrs["units"] = "1"
    out.attrs["long_name"] = f"SPEI (cell) calibrated {calibration_period[0]}-{calibration_period[1]}"
    return out


# ----------------------------
# Plot helpers (one plot per step)
# ----------------------------
def plot_series(da: xr.DataArray, out_png: str, title: str, ylabel: str):
    plt.figure()
    plt.plot(da.time.values, da.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[PLOT] {out_png}")


def save_checkpoint_cell(da: xr.DataArray, out_dir: str, name: str):
    """
    Save both CSV + NetCDF for easy inspection.
    """
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    df = pd.DataFrame({"time": pd.to_datetime(da.time.values), name: da.values})
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    # NetCDF
    nc_path = os.path.join(out_dir, f"{name}.nc")
    ds = xr.Dataset({name: da})
    ds.to_netcdf(nc_path)

    print(f"[SAVE] {csv_path}")
    print(f"[SAVE] {nc_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Reproduce SPEI for one grid cell with checkpoints+plots.")
    ap.add_argument("--pr", required=True, help="Glob pattern(s) to daily precip files (kg m-2 s-1). Comma-separated allowed.")
    ap.add_argument("--tasmin", required=True, help="Glob pattern(s) to daily tasmin files (K or °C).")
    ap.add_argument("--tasmax", required=True, help="Glob pattern(s) to daily tasmax files (K or °C).")
    ap.add_argument("--pr-var", default=None, help="Variable name for precip (default: first data var).")
    ap.add_argument("--tasmin-var", default=None, help="Variable name for tasmin.")
    ap.add_argument("--tasmax-var", default=None, help="Variable name for tasmax.")
    ap.add_argument("--lat", type=float, default=TARGET_LAT_DEFAULT, help="Target latitude (default: 52.5)")
    ap.add_argument("--lon", type=float, default=TARGET_LON_DEFAULT, help="Target longitude (default: 13.4)")
    ap.add_argument("--scale", type=int, default=3, help="SPEI accumulation scale in months (default: 3)")
    ap.add_argument("--calibration", default="1991-2020", help="Calibration period YYYY-YYYY (default: 1991-2020)")
    ap.add_argument("--outdir", default="spei_checkpoints_cell", help="Output folder for checkpoints and plots")
    ap.add_argument("--chunks", default=None, help="Optional dask chunks, e.g. 'time:365' or 'auto' (kept simple here)")

    args = ap.parse_args()

    # Parse calibration
    cal_parts = args.calibration.split("-")
    if len(cal_parts) != 2:
        raise ValueError("Calibration must be YYYY-YYYY")
    cal = (int(cal_parts[0]), int(cal_parts[1]))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # For 1-cell work, chunking usually isn't needed; keep it off unless you want it.
    chunks = None
    if args.chunks:
        if args.chunks.lower() == "auto":
            chunks = "auto"
        else:
            # minimal parser: "dim:val,dim:val"
            cd = {}
            for part in args.chunks.split(","):
                k, v = part.split(":")
                cd[k.strip()] = int(v.strip())
            chunks = cd

    print("\n=== STEP 1: Load daily inputs ===")
    pr = open_var(args.pr, var=args.pr_var, chunks=chunks)
    tmin = open_var(args.tasmin, var=args.tasmin_var, chunks=chunks)
    tmax = open_var(args.tasmax, var=args.tasmax_var, chunks=chunks)

    print("\n=== STEP 2: Select nearest grid cell ===")
    pr_cell = select_cell(pr, args.lat, args.lon)
    tmin_cell = select_cell(tmin, args.lat, args.lon)
    tmax_cell = select_cell(tmax, args.lat, args.lon)

    # Align daily time axis
    pr_cell, tmin_cell, tmax_cell = xr.align(pr_cell, tmin_cell, tmax_cell, join="inner")

    save_checkpoint_cell(pr_cell, outdir, "01_pr_raw_daily")
    plot_series(pr_cell, os.path.join(outdir, "01_pr_raw_daily.png"),
                title="Raw daily precipitation (cell)", ylabel="pr (native units)")

    print("\n=== STEP 3: Convert precip to mm/day ===")
    pr_mm_day = convert_pr_to_mm_day(pr_cell)
    save_checkpoint_cell(pr_mm_day, outdir, "02_pr_mm_day")
    plot_series(pr_mm_day, os.path.join(outdir, "02_pr_mm_day.png"),
                title="Daily precipitation converted to mm/day (cell)", ylabel="mm/day")

    print("\n=== STEP 4: Aggregate precip to monthly totals (mm/month) ===")
    # Monthly totals: sum mm/day over days in month
    pr_mm_month = pr_mm_day.resample(time="MS").sum()
    pr_mm_month.attrs["units"] = "mm/month"
    save_checkpoint_cell(pr_mm_month, outdir, "03_pr_mm_month")
    plot_series(pr_mm_month, os.path.join(outdir, "03_pr_mm_month.png"),
                title="Monthly precipitation totals (cell)", ylabel="mm/month")

    print("\n=== STEP 5: Compute daily PET (Hargreaves) ===")
    tmin_c = ensure_celsius(tmin_cell)
    tmax_c = ensure_celsius(tmax_cell)

    pet_mm_day = pet_hargreaves_daily_cell(tmin_c, tmax_c, lat_deg=args.lat)
    save_checkpoint_cell(pet_mm_day, outdir, "04_pet_mm_day")
    plot_series(pet_mm_day, os.path.join(outdir, "04_pet_mm_day.png"),
                title="Daily PET (Hargreaves) (cell)", ylabel="mm/day")

    print("\n=== STEP 6: Aggregate PET to monthly totals (mm/month) ===")
    pet_mm_month = pet_mm_day.resample(time="MS").sum()
    pet_mm_month.attrs["units"] = "mm/month"
    save_checkpoint_cell(pet_mm_month, outdir, "05_pet_mm_month")
    plot_series(pet_mm_month, os.path.join(outdir, "05_pet_mm_month.png"),
                title="Monthly PET totals (cell)", ylabel="mm/month")

    print("\n=== STEP 7: Align monthly P and PET ===")
    pr_mm_month, pet_mm_month = xr.align(pr_mm_month, pet_mm_month, join="inner")
    save_checkpoint_cell(pr_mm_month, outdir, "06_pr_mm_month_aligned")
    save_checkpoint_cell(pet_mm_month, outdir, "06_pet_mm_month_aligned")

    print("\n=== STEP 8: Water balance WB = P - PET (mm/month) ===")
    wb = pr_mm_month - pet_mm_month
    wb.attrs["units"] = "mm/month"
    wb.attrs["long_name"] = "Water balance (P - PET)"
    save_checkpoint_cell(wb, outdir, "07_wb_mm_month")
    plot_series(wb, os.path.join(outdir, "07_wb_mm_month.png"),
                title="Monthly water balance WB = P - PET (cell)", ylabel="mm/month")

    print(f"\n=== STEP 9: Rolling accumulation over {args.scale} months ===")
    wb_acc = wb.rolling(time=args.scale, min_periods=args.scale).sum()
    wb_acc.attrs["units"] = "mm"
    wb_acc.attrs["long_name"] = f"Accumulated water balance ({args.scale}-month)"
    save_checkpoint_cell(wb_acc, outdir, f"08_wb_acc_{args.scale:02d}m")
    plot_series(wb_acc, os.path.join(outdir, f"08_wb_acc_{args.scale:02d}m.png"),
                title=f"{args.scale}-month accumulated water balance (cell)", ylabel="mm")

    print("\n=== STEP 10: Fit log-logistic per month on calibration and transform to SPEI ===")
    spei = spei_fit_transform_cell(wb_acc, calibration_period=cal)
    spei.attrs["scale_months"] = args.scale
    spei.attrs["calibration_period"] = f"{cal[0]}-{cal[1]}"
    save_checkpoint_cell(spei, outdir, f"09_spei_{args.scale:02d}m")
    plot_series(spei, os.path.join(outdir, f"09_spei_{args.scale:02d}m.png"),
                title=f"SPEI-{args.scale} (cell), calibrated {cal[0]}-{cal[1]}", ylabel="SPEI (z-score)")

    print("\n=== DONE ===")
    print(f"Outputs written to: {outdir}")
    print("You now have 1 checkpoint + 1 plot per step.")


if __name__ == "__main__":
    main()

