"""
Maximum Climatological Water Deficit (MCWD) calculation.

Reference:
    https://doi.org/10.1029/2006GL028946

Cumulative water deficit with annual reset, tracking the maximum
deficit reached during each period.
"""

from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from .utils import (
    convert_precip_units,
    load_data,
    save_netcdf,
)
from .pet import aggregate_pet_to_monthly


def compute_mcwd(
    precip: xr.DataArray,
    et: xr.DataArray,
    scale: int = 12,
    reset_month: int = 10,
) -> xr.DataArray:
    """
    Compute Maximum Climatological Water Deficit.

    CWD accumulates monthly deficits (P - ET when negative), resetting
    at the start of the hydrological year. MCWD is the most negative
    CWD reached over a rolling window.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    et : xr.DataArray
        Monthly evapotranspiration in mm/month (PET or fixed value)
    scale : int
        Rolling window in months for finding maximum deficit
    reset_month : int
        Month to reset CWD (1=Jan, 10=Oct for Southern Hemisphere)

    Returns
    -------
    xr.DataArray
        MCWD values (mm, more negative = worse deficit)
    """
    # Water balance (positive = surplus, negative = deficit)
    wb = precip - et

    # Compute cumulative water deficit with reset
    cwd = _compute_cwd_with_reset(wb, reset_month)

    # MCWD is the most negative CWD over rolling window
    # Rolling minimum (most negative value)
    mcwd = cwd.rolling(time=scale, min_periods=1).min()

    mcwd.attrs = {
        "units": "mm",
        "long_name": f"Maximum Climatological Water Deficit ({scale}-month)",
        "scale": scale,
        "reset_month": reset_month,
    }

    return mcwd


def _compute_cwd_with_reset(
    wb: xr.DataArray,
    reset_month: int,
) -> xr.DataArray:
    """
    Compute Cumulative Water Deficit with annual reset.

    CWD_t = min(0, CWD_{t-1} + WB_t)

    Resets to 0 at the start of the hydrological year.

    Parameters
    ----------
    wb : xr.DataArray
        Monthly water balance (P - ET)
    reset_month : int
        Month number to reset (1-12)

    Returns
    -------
    xr.DataArray
        Cumulative water deficit
    """
    # Stack spatial dimensions for iteration
    stacked = wb.stack(gridcell=("lat", "lon"))

    #process key: Load ALL data into numpy array ONCE
    print("  Loading data into memory...")
    data_array = stacked.values  # Shape: (n_times, n_cells)
    months = stacked.time.dt.month.values
    
    n_times, n_cells = data_array.shape

    # Output array
    cwd_out = np.full((n_times, n_cells), np.nan)

    # Process each grid cell with progress
    import time as _time
    _start = _time.time()
    
    for i in range(n_cells):
        wb_vals = data_array[:, i]
        cwd = np.zeros(n_times)

        for t in range(n_times):
            if np.isnan(wb_vals[t]):
                cwd[t] = np.nan
                continue

            # Reset at start of hydrological year
            if months[t] == reset_month:
                prev_cwd = 0
            elif t == 0:
                prev_cwd = 0
            else:
                prev_cwd = cwd[t - 1] if not np.isnan(cwd[t - 1]) else 0

            # CWD can only be zero or negative
            cwd[t] = min(0, prev_cwd + wb_vals[t])

        cwd_out[:, i] = cwd
        
        # progress bar every 5000 cells
        if (i + 1) % 5000 == 0:
            elapsed = _time.time() - _start
            rate = (i + 1) / elapsed
            remaining = (n_cells - i - 1) / rate
            print(f"    {i+1:,}/{n_cells:,} ({100*(i+1)/n_cells:.0f}%) - {remaining:.0f}s remaining", flush=True)
    
    print(f"    Done in {_time.time() - _start:.1f}s")

    # Rebuild DataArray
    cwd_stacked = xr.DataArray(
        cwd_out,
        dims=stacked.dims,
        coords=stacked.coords,
    )
    
    # Unstack
    cwd_result = cwd_stacked.unstack("gridcell")

    cwd_result.attrs = {
        "units": "mm",
        "long_name": "Cumulative Water Deficit",
    }

    return cwd_result


def compute_mcwd_fixed_et(
    precip: xr.DataArray,
    et_fixed: float = 100.0,
    scale: int = 12,
    reset_month: int = 10,
) -> xr.DataArray:
    """
    Compute MCWD with fixed monthly ET (Aragão et al. approach).

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    et_fixed : float
        Fixed monthly ET in mm/month (default 100, typical for tropical forests)
    scale : int
        Rolling window in months
    reset_month : int
        Month to reset CWD

    Returns
    -------
    xr.DataArray
        MCWD values
    """
    # Create ET array with same shape as precip
    et = xr.full_like(precip, et_fixed)

    return compute_mcwd(precip, et, scale=scale, reset_month=reset_month)


def compute_mcwd_multiscale(
    precip: xr.DataArray,
    et: xr.DataArray,
    scales: List[int],
    reset_month: int = 10,
) -> xr.Dataset:
    """
    Compute MCWD for multiple time scales.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    et : xr.DataArray
        Monthly ET in mm/month
    scales : list of int
        List of rolling window periods
    reset_month : int
        Month to reset CWD

    Returns
    -------
    xr.Dataset
        Dataset with MCWD for each scale
    """
    ds = xr.Dataset()

    for scale in scales:
        var_name = f"mcwd_{scale:02d}"
        print(f"Computing {var_name}...")
        ds[var_name] = compute_mcwd(
            precip,
            et,
            scale=scale,
            reset_month=reset_month,
        )

    ds.attrs = {
        "title": "Maximum Climatological Water Deficit",
        "institution": "ISIMIP Drought Indices",
        "source": "isimip-drought",
        "references": "Aragão et al. (2007); Malhi et al. (2009)",
    }

    return ds


def mcwd_from_files(
    precip_pattern: str,
    scales: List[int],
    output_path: str,
    et_fixed: Optional[float] = None,
    pet_pattern: Optional[str] = None,
    reset_month: int = 10,
    chunks: Optional[dict] = None,
) -> None:
    """
    Compute MCWD from NetCDF files and save output.

    Parameters
    ----------
    precip_pattern : str
        Glob pattern for precipitation files
    scales : list of int
        Rolling window periods
    output_path : str
        Output NetCDF path
    et_fixed : float, optional
        Fixed monthly ET value (mm/month). If None, uses pet_pattern.
    pet_pattern : str, optional
        Glob pattern for PET files (used if et_fixed is None)
    reset_month : int
        Month to reset CWD (1-12)
    chunks : dict, optional
        Dask chunks
    """
    print(f"Loading precipitation from: {precip_pattern}")
    pr = load_data(precip_pattern, chunks=chunks)
    pr_mm = convert_precip_units(pr)

    # Resample to monthly if daily
    if pr_mm.time.dt.dayofyear.max() > 12:
        print("Resampling precipitation to monthly...")
        pr_mm = pr_mm.resample(time="MS").sum()

    # Get ET
    if et_fixed is not None:
        print(f"Using fixed ET: {et_fixed} mm/month")
        et = xr.full_like(pr_mm, et_fixed)
    elif pet_pattern:
        print(f"Loading PET from: {pet_pattern}")
        et = load_data(pet_pattern, chunks=chunks)
        # Aggregate to monthly if daily
        if et.time.dt.dayofyear.max() > 12:
            print("Aggregating PET to monthly...")
            et = aggregate_pet_to_monthly(et)
        # Align times
        pr_mm, et = xr.align(pr_mm, et, join="inner")
    else:
        raise ValueError("Must provide either --et-fixed or --pet")

    # Compute MCWD
    print(f"Computing MCWD for scales: {scales}")
    ds = compute_mcwd_multiscale(
        pr_mm,
        et,
        scales=scales,
        reset_month=reset_month,
    )

    # Save
    print(f"Saving to: {output_path}")
    save_netcdf(ds, output_path)
    print("Done!")


def compute_annual_mcwd(
    precip: xr.DataArray,
    et: xr.DataArray,
    reset_month: int = 10,
) -> xr.DataArray:
    """
    Compute annual MCWD (classic approach).

    For each hydrological year, finds the most negative CWD reached.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation
    et : xr.DataArray
        Monthly ET
    reset_month : int
        Start of hydrological year

    Returns
    -------
    xr.DataArray
        Annual MCWD with one value per hydrological year
    """
    # Water balance
    wb = precip - et

    # Compute CWD with reset
    cwd = _compute_cwd_with_reset(wb, reset_month)

    # Group by hydrological year
    # Hydrological year starts at reset_month
    # e.g., if reset_month=10, Oct 2000 - Sep 2001 is hydro year 2001
    hydro_year = cwd.time.dt.year.where(
        cwd.time.dt.month < reset_month,
        cwd.time.dt.year + 1
    )

    # Find minimum (most negative) CWD per hydrological year
    annual_mcwd = cwd.groupby(hydro_year).min()
    annual_mcwd = annual_mcwd.rename({"year": "hydro_year"})

    annual_mcwd.attrs = {
        "units": "mm",
        "long_name": "Annual Maximum Climatological Water Deficit",
        "reset_month": reset_month,
    }

    return annual_mcwd