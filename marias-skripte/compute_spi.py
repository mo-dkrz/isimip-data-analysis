#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Standardized Precipitation Index (SPI) on gridded monthly data.

- Loads monthly precip (mm/day) and converts to monthly totals (mm/month)
- Builds rolling k-month accumulations (e.g., 2, 3, 6)
- Fits a mixed distribution per calendar month and grid cell:
    P(X=0) = p0, and X|X>0 ~ Gamma(shape=k, scale=theta), with loc fixed to 0
- Transforms to Z via inverse standard normal CDF => SPI
- Writes CF-compliant NetCDF with variables spi_<k>

Designed for 0.5° CONUS monthly (1979–2018), but generally applicable.

Dependencies: xarray, numpy, scipy, pandas, dask (optional but recommended), netCDF4 or h5netcdf.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gamma as sp_gamma
from scipy.stats import norm as sp_norm


def find_precip_dataset(root: str) -> xr.DataArray:
    """
    Try to open precipitation from <root>/Precip/*.nc (preferred).
    If not found, fall back to Rainf + Snowf under corresponding folders.
    Returns DataArray with dims (time, lat, lon) in mm/day.
    """
    root = Path(root)

    # Preferred: Precip/*.nc
    precip_dir = root / "Precip"
    nc_candidates = sorted(glob.glob(str(precip_dir / "*.nc")))
    da = None
    if nc_candidates:
        dsP = xr.open_mfdataset(nc_candidates, combine="by_coords")
        # Try common variable names
        var_candidates = [v for v in dsP.data_vars]
        # Heuristic: pick the only var, or a var named like precip
        name = None
        for cand in ["pr", "precip", "precipitation", "__xarray_dataarray_variable__"]:
            if cand in var_candidates:
                name = cand
                break
        if name is None and len(var_candidates) == 1:
            name = var_candidates[0]
        if name is None:
            raise ValueError(f"Could not identify precip variable in {nc_candidates}")
        da = dsP[name]
        da.attrs.setdefault("units", "mm/day")

    if da is None:
        # Fallback: Rainf + Snowf
        rain_dir = root / "Rainf"
        snow_dir = root / "Snowf"
        rain_files = sorted(glob.glob(str(rain_dir / "*.nc")))
        snow_files = sorted(glob.glob(str(snow_dir / "*.nc")))
        if not rain_files or not snow_files:
            raise FileNotFoundError(
                "Could not find precipitation under Precip/*.nc or Rainf/*.nc + Snowf/*.nc"
            )
        dsR = xr.open_mfdataset(rain_files, combine="by_coords")
        dsS = xr.open_mfdataset(snow_files, combine="by_coords")
        # Try variable names
        rname = [v for v in dsR.data_vars][0]
        sname = [v for v in dsS.data_vars][0]
        da = (dsR[rname] + dsS[sname])
        da.attrs.setdefault("units", "mm/day")

    # Basic checks
    if "time" not in da.dims:
        raise ValueError("Input precipitation must have a 'time' dimension.")
    if not np.issubdtype(da.time.dtype, np.datetime64):
        # Try to decode if not already decoded
        da = xr.decode_cf(da.to_dataset(name="pr")).pr

    # Ensure canonical naming
    da = da.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in da.dims else da
    da = da.rename({"lonitude": "lon"}) if "lonitude" in da.dims else da
    return da


def to_monthly_totals_mm(da_mm_per_day: xr.DataArray) -> xr.DataArray:
    """Convert mm/day to mm/month using true days per month."""
    units = da_mm_per_day.attrs.get("units", "").lower().replace(" ", "")
    if "mm/day" not in units and "mm/d" not in units:
        # Be permissive but warn in attribute
        da_mm_per_day = da_mm_per_day.assign_attrs(note_units_warning="Units not mm/day; proceeding as mm/day.")
    days = da_mm_per_day["time"].dt.days_in_month
    out = (da_mm_per_day * days).assign_attrs(units="mm/month")
    return out


def rolling_sum_k(da: xr.DataArray, k: int) -> xr.DataArray:
    """k-month rolling sum with full window required (min_periods=k)."""
    return da.rolling(time=k, min_periods=k).sum()


def _spi_1d(accum_ts, month_index, calib_mask, min_pos_samples=15, eps=1e-6):
    """
    Core SPI transform for a single 1D time series (numpy arrays).
    - accum_ts: np.ndarray shape (T,), monthly accumulated precipitation (mm)
    - month_index: np.ndarray shape (T,), values 1..12
    - calib_mask: np.ndarray shape (T,), boolean mask for calibration times
    Returns SPI np.ndarray shape (T,)
    """
    T = accum_ts.shape[0]
    spi = np.full(T, np.nan, dtype=np.float64)

    # Work month-by-month to respect seasonality
    for m in range(1, 13):
        mask_m = (month_index == m)
        if not np.any(mask_m):
            continue

        x_all = accum_ts[mask_m]
        # Calibration subset for this month
        mask_cal = mask_m & calib_mask
        x_cal = accum_ts[mask_cal]

        # Handle missing-only month
        if x_cal.size == 0 or np.all(np.isnan(x_cal)):
            continue

        # Remove NaNs from calibration
        x_cal = x_cal[~np.isnan(x_cal)]

        if x_cal.size < 10:
            # Too few data to fit
            continue

        # Zero-inflation
        zeros = (x_cal <= 0) | np.isclose(x_cal, 0.0)
        p0 = zeros.mean()

        # Positive values for gamma fit
        pos = x_cal[~zeros]
        if pos.size < min_pos_samples:
            # Not enough positives to estimate gamma robustly
            continue

        # Fit gamma with loc fixed at 0: sp_gamma.fit(x, floc=0) returns (shape, loc=0, scale)
        try:
            shape, loc, scale = sp_gamma.fit(pos, floc=0)
            # Safety on parameters
            if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
                continue
        except Exception:
            continue

        # Evaluate CDF for all months m (including NaNs)
        x_eval = x_all.copy()
        F = np.full_like(x_eval, np.nan, dtype=np.float64)

        # Where x is NaN, remain NaN
        valid = ~np.isnan(x_eval)
        if not np.any(valid):
            # Assign back and continue
            spi[mask_m] = F
            continue

        xv = x_eval[valid]
        # Compute nonzero probability mass
        # Mixture CDF: F(x) = p0 + (1 - p0) * G(x) for x > 0
        # For x <= 0, F(x) = p0 * (optionally split mass). We'll set F= p0 * 0.5 if x==0 else small.
        G = sp_gamma.cdf(np.maximum(xv, 0.0), a=shape, loc=0.0, scale=scale)
        Fv = p0 + (1.0 - p0) * G
        # For exactly zero or negative (shouldn't be negative after accumulation), enforce F around p0
        zero_like = np.isclose(xv, 0.0)
        Fv[zero_like] = max(p0 * 0.5, eps)  # place at mid of the zero mass to avoid ppf(0)

        # Clip to avoid infinities in inverse normal
        Fv = np.clip(Fv, eps, 1.0 - eps)
        Zv = sp_norm.ppf(Fv)

        F[valid] = Zv
        spi[mask_m] = F

    return spi


def compute_spi_da(accum: xr.DataArray, time: xr.DataArray, cal_start: str, cal_end: str) -> xr.DataArray:
    """
    Wrapper to compute SPI for a gridded accumulation array using apply_ufunc.
    accum: DataArray (time, lat, lon)
    """
    # Prepare helper 1D arrays to pass alongside time
    months = xr.DataArray(accum["time"].dt.month, dims=["time"])
    calib = xr.DataArray(
        (accum["time"] >= np.datetime64(pd.to_datetime(cal_start, format="%Y-%m").date()))
        & (accum["time"] <= np.datetime64(pd.to_datetime(cal_end, format="%Y-%m").date()))
    , dims=["time"])

    spi = xr.apply_ufunc(
        _spi_1d,
        accum,
        months,
        calib,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    spi = spi.assign_coords(time=accum["time"]).transpose("time", ...)

    spi.attrs.update(
        {
            "long_name": "Standardized Precipitation Index",
            "standard_name": "standardized_precipitation_index",
            "units": "1",
            "comment": (
                "Per-calendar-month zero-inflated gamma fit on k-month accumulated precipitation; "
                "calibrated over [{:s} .. {:s}].".format(cal_start, cal_end)
            ),
        }
    )
    return spi


def main():
    ap = argparse.ArgumentParser(description="Compute SPI for multiple accumulation windows.")
    ap.add_argument("--root", required=True, help="Root data directory containing Precip/, Rainf/, Snowf/ subfolders.")
    ap.add_argument("--scales", nargs="+", type=int, default=[2, 3, 6], help="Accumulation windows in months.")
    ap.add_argument("--cal-start", default="1981-01", help="Calibration start YYYY-MM (default: 1981-01)")
    ap.add_argument("--cal-end", default="2010-12", help="Calibration end YYYY-MM (default: 2010-12)")
    ap.add_argument("--out", required=True, help="Output NetCDF file path.")
    ap.add_argument("--chunks", nargs=3, type=int, metavar=("T", "Y", "X"), default=[-1, 25, 30],
                    help="Dask chunk sizes for (time, lat, lon); use -1 for full chunk along a dim.")
    args = ap.parse_args()

    # Optionally enable dask if present on HPC
    try:
        import dask  # noqa: F401
        xr.set_options(use_bottleneck=True, keep_attrs=True)
    except Exception:
        pass

    # 1) Read precipitation (mm/day)
    pr_day = find_precip_dataset(args.root)

    # 2) Convert to mm/month totals
    pr_mon = to_monthly_totals_mm(pr_day)

    # 3) Chunk for performance
    t_chunk, y_chunk, x_chunk = args.chunks
    chunk_map = {}
    if "time" in pr_mon.dims and t_chunk is not None:
        chunk_map["time"] = t_chunk
    if "lat" in pr_mon.dims and y_chunk is not None:
        chunk_map["lat"] = y_chunk
    if "lon" in pr_mon.dims and x_chunk is not None:
        chunk_map["lon"] = x_chunk
    if chunk_map:
        pr_mon = pr_mon.chunk(chunk_map)

    # 4) Build output Dataset
    out = xr.Dataset()
    out_coords = dict(time=pr_mon.time, lat=pr_mon.lat, lon=pr_mon.lon)

    # 5) Compute SPI for each scale
    for k in args.scales:
        accum = rolling_sum_k(pr_mon, k).rename("pr{}_accum".format(k))
        spi_k = compute_spi_da(accum, pr_mon.time, args.cal_start, args.cal_end)
        vname = f"spi_{k:02d}"
        spi_k.name = vname
        # Attributes
        spi_k.attrs.update(
            {
                "cell_methods": f"time: sum over {k} months then standardized",
                "calibration_period": f"{args.cal_start} to {args.cal_end}",
                "distribution": "zero-inflated gamma (loc=0)",
                "eps_note": "CDF clipped to [1e-6, 1-1e-6] before inverse normal.",
            }
        )
        out[vname] = spi_k

    # 6) Copy coordinates and global attrs
    out = out.assign_coords(out_coords)
    out.attrs.update(
        dict(
            title="Standardized Precipitation Index (SPI), per-calendar-month gamma fit",
            conventions="CF-1.8",
            source="compute_spi.py (custom, xarray+scipy)",
            references="McKee et al. (1993); Guttman (1999)",
            history=f"Created by compute_spi.py; scales={args.scales}; calibration={args.cal_start}-{args.cal_end}",
        )
    )

    # 7) Save NetCDF
    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in out.data_vars}
    # Ensure time encoding is CF-compliant
    out["time"].encoding.update({"units": "days since 1900-01-01", "calendar": "proleptic_gregorian"})
    out.to_netcdf(args.out, encoding=encoding)
    print(f"✅ Wrote {args.out} with variables: {', '.join(out.data_vars)}")


if __name__ == "__main__":
    main()

