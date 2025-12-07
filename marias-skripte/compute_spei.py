#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Standardized Precipitation–Evapotranspiration Index (SPEI)
using gridded monthly precipitation and PET (Penman) data.

Flexible accumulation windows (e.g., 2, 3, 6 months)
Zero-aware water balance (P − PET)
Log-logistic distribution fitting per calendar month and grid cell
Calibration period configurable (default: 1981–2010)
CF-compliant NetCDF output
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import fisk, norm  # fisk = log-logistic


# ---------- Utility functions ----------

def open_dataarray(path_dir, name_hint):
    """Open monthly NetCDF file matching a variable hint."""
    path_dir = Path(path_dir)
    nc_files = sorted(glob.glob(str(path_dir / "*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {path_dir}")
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    for v in ds.data_vars:
        if name_hint.lower() in v.lower():
            return ds[v]
    # fallback to first variable
    return list(ds.data_vars.values())[0]


def mmday_to_mmmonth(da):
    """Convert mm/day → mm/month using true month lengths."""
    days = da.time.dt.days_in_month
    return (da * days).assign_attrs(units="mm/month")


def rolling_sum(da, k):
    """k-month rolling sum (complete windows only)."""
    return da.rolling(time=k, min_periods=k).sum()


def _spei_1d(series, months, calib_mask, min_samples=15, eps=1e-6):
    """Fit log-logistic per calendar month and compute SPEI (1-D array)."""
    out = np.full_like(series, np.nan, dtype=np.float64)
    for m in range(1, 13):
        mask_m = months == m
        if not np.any(mask_m):
            continue
        x = series[mask_m]
        x_cal = series[mask_m & calib_mask]
        x_cal = x_cal[np.isfinite(x_cal)]
        if x_cal.size < min_samples:
            continue
        # Fit log-logistic (fisk) on calibration period
        try:
            c, loc, scale = fisk.fit(x_cal)
        except Exception:
            continue
        if not np.isfinite(c) or not np.isfinite(scale):
            continue
        # Evaluate CDF for all x in month m
        F = fisk.cdf(x, c, loc=loc, scale=scale)
        F = np.clip(F, eps, 1 - eps)
        out[mask_m] = norm.ppf(F)
    return out


def compute_spei(balance, time, cal_start, cal_end):
    """Vectorized SPEI computation for gridded xarray DataArray."""
    months = xr.DataArray(balance.time.dt.month, dims=["time"])
    calib = xr.DataArray(
        (balance.time >= np.datetime64(cal_start))
        & (balance.time <= np.datetime64(cal_end)),
        dims=["time"],
    )
    spei = xr.apply_ufunc(
        _spei_1d,
        balance,
        months,
        calib,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    return spei.assign_coords(time=balance.time)


# ---------- Main workflow ----------

def main():
    parser = argparse.ArgumentParser(description="Compute SPEI from monthly P and PET.")
    parser.add_argument("--root", required=True, help="Root folder containing Precip/ and PET/ subdirs.")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 3, 6], help="Accumulation windows (months).")
    parser.add_argument("--cal-start", default="1981-01-01", help="Calibration start date.")
    parser.add_argument("--cal-end", default="2010-12-31", help="Calibration end date.")
    parser.add_argument("--out", required=True, help="Output NetCDF file.")
    args = parser.parse_args()

    root = Path(args.root)

    # Load precipitation (mm/day)
    pr = open_dataarray(root / "Precip", "precip")
    pr = mmday_to_mmmonth(pr)

    # Load PET (mm/day)
    pet = open_dataarray(root / "PET", "PET")
    pet = mmday_to_mmmonth(pet)

    # Compute water balance
    wb = (pr - pet).assign_attrs(units="mm/month", long_name="P - PET (water balance)")

    # Chunk for Dask efficiency
    wb = wb.chunk({"time": -1, "lat": 25, "lon": 30})

    out = xr.Dataset()
    for k in args.scales:
        wb_acc = rolling_sum(wb, k)
        spei_k = compute_spei(wb_acc, wb.time, args.cal_start, args.cal_end)
        name = f"spei_{k:02d}"
        spei_k.name = name
        spei_k.attrs.update(
            {
                "long_name": "Standardized Precipitation–Evapotranspiration Index",
                "standard_name": "standardized_precipitation_evapotranspiration_index",
                "units": "1",
                "cell_methods": f"time: sum over {k} months then standardized",
                "distribution": "log-logistic (fisk)",
                "calibration_period": f"{args.cal_start} to {args.cal_end}",
            }
        )
        out[name] = spei_k

    out.attrs.update(
        {
            "title": "SPEI computed from monthly P and PET (Penman)",
            "source": "xarray + scipy (log-logistic fit)",
            "conventions": "CF-1.8",
            "history": f"Generated with accumulation windows {args.scales} months",
            "references": "Vicente-Serrano et al. (2010), Global SPEI dataset method.",
        }
    )

    comp = dict(zlib=True, complevel=4)
    out.to_netcdf(args.out, encoding={v: comp for v in out.data_vars})
    print(f"✅ SPEI written to {args.out} with variables: {', '.join(out.data_vars)}")


if __name__ == "__main__":
    main()

