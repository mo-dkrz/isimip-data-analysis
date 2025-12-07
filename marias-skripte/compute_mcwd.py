#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute MCWD (Maximum Cumulative Water Deficit) from monthly precipitation and PET.

Definition (monthly, grid-cell-wise):
  WB_t = P_t(mm/month) - PET_t(mm/month)
  CWD_t = min(0, CWD_{t-1} + WB_t); CWD resets to 0 each hydrological year
  MCWD_k = rolling k-month minimum of CWD_t

Inputs:
  - Precip: monthly totals (mm/month) [already aggregated]
  - PET: Penman PET (mm/day) → converted to mm/month

Outputs:
  - mcwd_02, mcwd_06, mcwd_12 (mm)
  - CF-compliant NetCDF

Author: GPT-5 Climate Drought Workflow (2025)
"""

import argparse, glob, time
from pathlib import Path
import numpy as np
import xarray as xr


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def open_da(dirpath: Path) -> xr.DataArray:
    """Open first variable from NetCDF files in directory."""
    files = sorted(glob.glob(str(dirpath / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files in {dirpath}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds[list(ds.data_vars)[0]]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute MCWD (2,6,12 months) with annual reset.")
    ap.add_argument("--root", required=True, help="Folder with Precip/ and PET/ subdirs")
    ap.add_argument("--scales", nargs="+", type=int, default=[2, 6, 12], help="MCWD window lengths (months)")
    ap.add_argument("--out", required=True, help="Output NetCDF path")
    ap.add_argument("--progress-every", type=int, default=48, help="Print progress every N timesteps")
    ap.add_argument("--reset-month", type=int, default=10, help="Reset month (1=Jan, 10=Oct hydrological year)")
    args = ap.parse_args()

    root = Path(args.root)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    print("Loading precipitation and PET ...")
    P = open_da(root / "Precip")  # monthly totals (mm/month)
    E = open_da(root / "PET")     # mm/day → convert

    # Ensure time/lat/lon order
    P = P.transpose("time", "lat", "lon")
    E = E.transpose("time", "lat", "lon")

    # -----------------------------------------------------------------
    # Unit handling
    # -----------------------------------------------------------------
    print("Converting units: PET mm/day → mm/month; P used as-is ...")
    days = P.time.dt.days_in_month
    pr_mon = P.astype("float32").assign_attrs(units="mm/month")
    pet_mon = (E * days).astype("float32")
    pet_mon.attrs["units"] = "mm/month"

    # Water balance
    wb = (pr_mon - pet_mon).assign_attrs(units="mm/month", long_name="Water balance (P - PET)")

    # -----------------------------------------------------------------
    # Monthly CWD recursion with annual reset
    # -----------------------------------------------------------------
    print(f"Computing monthly CWD recursion with annual reset (month={args.reset_month}) ...")
    wb = wb.load()  # fully load into memory (~50–70 MB)
    T, Y, X = wb.sizes["time"], wb.sizes["lat"], wb.sizes["lon"]
    months = wb.time.dt.month.values
    wb_np = wb.values

    cwd_state = np.zeros((Y, X), dtype=np.float32)
    cwd_out = np.full((T, Y, X), np.nan, dtype=np.float32)

    t0 = time.time()
    for t in range(T):
        # reset each hydrological year (October default)
        if months[t] == args.reset_month:
            cwd_state[:] = 0.0

        w = wb_np[t]
        valid = np.isfinite(w)
        # accumulate deficit but allow recovery to 0
        cwd_state[valid] = np.minimum(cwd_state[valid] + w[valid], 0.0)

        slice_out = cwd_state.copy()
        slice_out[~valid] = np.nan
        cwd_out[t] = slice_out

        if (t + 1) % args.progress_every == 0 or (t + 1) == T:
            elapsed = time.time() - t0
            pct = 100.0 * (t + 1) / T
            eta = elapsed / (t + 1) * (T - (t + 1))
            print(f"  -> {t+1}/{T} months ({pct:4.1f}%) | elapsed {elapsed:6.1f}s | est. {eta:6.1f}s")

    cwd = xr.DataArray(
        cwd_out,
        coords=dict(time=wb.time, lat=wb.lat, lon=wb.lon),
        dims=("time", "lat", "lon"),
        name="cwd",
        attrs=dict(
            units="mm",
            long_name=f"Cumulative Water Deficit (annual reset month={args.reset_month})",
        ),
    )

    # -----------------------------------------------------------------
    # Rolling minima (MCWD)
    # -----------------------------------------------------------------
    out = xr.Dataset(coords=dict(time=wb.time, lat=wb.lat, lon=wb.lon))
    print("Computing MCWD rolling minima ...")
    for k in args.scales:
        vname = f"mcwd_{k:02d}"
        print(f"  - {vname} ({k}-month window)")
        mcwd_k = cwd.rolling(time=k, min_periods=k).min()
        mcwd_k.name = vname
        mcwd_k.attrs.update(
            units="mm",
            long_name=f"Maximum Cumulative Water Deficit (k={k} months, reset={args.reset_month})",
            cell_methods=f"time: cumulative deficit with annual reset; time: rolling {k}-month minimum",
            note="More negative = deeper/longer cumulative water deficit (≤ 0 by definition).",
        )
        out[vname] = mcwd_k

    out.attrs.update(
        title="MCWD from monthly P and PET (Penman)",
        method="WB=P(mm/month)−PET(mm/month); CWD_t=min(0,CWD_{t−1}+WB_t); reset each hydrological year; MCWD_k=rolling k-month minimum",
        aggregation_windows=str(args.scales),
        reset_month=args.reset_month,
        conventions="CF-1.8",
    )

    # -----------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------
    print(f"Writing to {args.out} ...")
    comp = dict(zlib=True, complevel=4)
    out.to_netcdf(args.out, encoding={v: comp for v in out.data_vars})
    print(f"✅ Wrote {args.out} with variables: {', '.join(out.data_vars)}")


if __name__ == "__main__":
    main()

