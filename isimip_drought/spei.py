"""
Standardized Precipitation Evapotranspiration Index (SPEI) calculation.

Reference:
    https://journals.ametsoc.org/doi/10.1175/2009JCLI2909.1

Uses log-logistic distribution fitted to water balance (P - PET) per calendar month.
Parallelized with joblib for multi-core processing.
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import stats

from .utils import (
    convert_precip_units,
    get_calibration_mask,
    load_data,
    rolling_sum,
    save_netcdf,
)
from .pet import (
    calc_pet_hargreaves,
    calc_pet_penman_monteith,
    calc_pet_thornthwaite,
    aggregate_pet_to_monthly,
)


def compute_spei(
    precip: xr.DataArray,
    pet: xr.DataArray,
    scale: int,
    calibration_period: Tuple[int, int] = (1991, 2020),
    min_samples: int = None,
) -> xr.DataArray:
    """
    Compute Standardized Precipitation Evapotranspiration Index.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    pet : xr.DataArray
        Monthly PET in mm/month
    scale : int
        Accumulation period in months
    calibration_period : tuple of int
        (start_year, end_year) for fitting distribution
    min_samples : int, optional
        Minimum samples required for fitting.
        Default: max(5, calibration_years - 1)

    Returns
    -------
    xr.DataArray
        SPEI values (dimensionless, standard normal)
    """
    # Auto-set min_samples based on calibration period length
    cal_years = calibration_period[1] - calibration_period[0] + 1
    if min_samples is None:
        min_samples = max(5, cal_years - 1)
        min_samples = min(min_samples, cal_years)
    
    # Water balance
    wb = precip - pet

    # Rolling sum for accumulation period
    wb_acc = rolling_sum(wb, scale)

    # Get calibration mask
    cal_mask = get_calibration_mask(wb_acc.time, calibration_period)

    # Load into memory and stack
    print("  Loading data into memory...")
    stacked = wb_acc.stack(gridcell=('lat', 'lon'))
    
    # process key: Load ALL data into numpy array ONCE
    data_array = stacked.values  # Shape: (n_times, n_cells)
    months = stacked.time.dt.month.values
    cal_mask_arr = cal_mask.values
    
    n_times, n_cells = data_array.shape
    
    # Output array
    spei_values = np.full((n_times, n_cells), np.nan)

    # Process each calendar month (serial - most reliable)
    for month in range(1, 13):
        month_mask = months == month
        cal_month_mask = month_mask & cal_mask_arr

        month_idx = np.where(month_mask)[0]
        cal_idx = np.where(cal_month_mask)[0]

        if len(month_idx) == 0 or len(cal_idx) < min_samples:
            continue

        print(f"  Month {month:2d}: fitting {n_cells} cells...", flush=True)
        import time as _time
        _start = _time.time()

        # Process all cells for this month
        for j in range(n_cells):
            cal_data = data_array[cal_idx, j]
            all_data = data_array[month_idx, j]
            
            result = _spei_loglogistic_fit_transform(cal_data, all_data, min_samples)
            spei_values[month_idx, j] = result
            
            # progress bar every 5000 cells to stdout
            if (j + 1) % 5000 == 0:
                elapsed = _time.time() - _start
                rate = (j + 1) / elapsed
                remaining = (n_cells - j - 1) / rate
                print(f"    {j+1:,}/{n_cells:,} ({100*(j+1)/n_cells:.0f}%) - {remaining:.0f}s remaining", flush=True)
        
        print(f"    Done in {_time.time() - _start:.1f}s")

    # Rebuild DataArray
    spei_stacked = xr.DataArray(
        spei_values,
        dims=stacked.dims,
        coords=stacked.coords,
    )
    spei = spei_stacked.unstack('gridcell')

    spei.attrs = {
        "units": "1",
        "long_name": f"Standardized Precipitation Evapotranspiration Index ({scale}-month)",
        "scale": scale,
        "calibration_period": f"{calibration_period[0]}-{calibration_period[1]}",
    }

    return spei


def _spei_loglogistic_fit_transform(
    cal_data: np.ndarray,
    all_data: np.ndarray,
    min_samples: int = 15,
) -> np.ndarray:
    """
    Fit log-logistic distribution on calibration data and transform all data.
    
    Parameters
    ----------
    cal_data : np.ndarray
        1D array of calibration period water balance
    all_data : np.ndarray
        1D array of all water balance to transform
    min_samples : int
        Minimum samples for fitting
        
    Returns
    -------
    np.ndarray
        SPEI values (same length as all_data)
    """
    # Handle NaN
    cal_valid = cal_data[~np.isnan(cal_data)]
    
    if len(cal_valid) < min_samples:
        return np.full(len(all_data), np.nan)
    
    # Shift data to positive if needed
    shift = 0
    if np.min(cal_valid) <= 0:
        shift = -np.min(cal_valid) + 1
    
    cal_shifted = cal_valid + shift
    
    # Fit log-logistic (Fisk distribution)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            c, loc, scale = stats.fisk.fit(cal_shifted, floc=0)
        except Exception:
            return np.full(len(all_data), np.nan)
    
    # Transform (vectorized)
    spei = np.full(len(all_data), np.nan)
    valid_mask = ~np.isnan(all_data)
    
    if not np.any(valid_mask):
        return spei
    
    all_shifted = all_data[valid_mask] + shift
    
    # Handle non-positive shifted values
    neg_mask = all_shifted <= 0
    cdf = np.full(len(all_shifted), 1e-6)
    
    pos_mask = ~neg_mask
    if np.any(pos_mask):
        cdf[pos_mask] = stats.fisk.cdf(all_shifted[pos_mask], c, loc=0, scale=scale)
    
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    spei[valid_mask] = stats.norm.ppf(cdf)
    
    return spei


def compute_spei_multiscale(
    precip: xr.DataArray,
    pet: xr.DataArray,
    scales: List[int],
    calibration_period: Tuple[int, int] = (1991, 2020),
    min_samples: int = None,
) -> xr.Dataset:
    """Compute SPEI for multiple time scales."""
    ds = xr.Dataset()

    for scale in scales:
        var_name = f"spei_{scale:02d}"
        print(f"Computing {var_name}...")
        ds[var_name] = compute_spei(
            precip, pet, scale=scale,
            calibration_period=calibration_period,
            min_samples=min_samples,
        )

    ds.attrs = {
        "title": "Standardized Precipitation Evapotranspiration Index",
        "institution": "ISIMIP Drought Indices",
        "source": "isimip-drought",
        "references": "Vicente-Serrano et al. (2010)",
    }

    return ds


def spei_from_files(
    precip_pattern: str,
    scales: List[int],
    output_path: str,
    calibration_period: Tuple[int, int] = (1991, 2020),
    pet_pattern: Optional[str] = None,
    pet_method: str = "hargreaves",
    tasmin_pattern: Optional[str] = None,
    tasmax_pattern: Optional[str] = None,
    tas_pattern: Optional[str] = None,
    hurs_pattern: Optional[str] = None,
    rsds_pattern: Optional[str] = None,
    sfcwind_pattern: Optional[str] = None,
    ps_pattern: Optional[str] = None,
    chunks: Optional[dict] = None,
) -> None:
    """Compute SPEI from NetCDF files and save output."""
    print(f"Loading precipitation from: {precip_pattern}")
    pr = load_data(precip_pattern, chunks=chunks)
    pr_mm = convert_precip_units(pr)

    if pr_mm.time.dt.dayofyear.max() > 12:
        print("Resampling precipitation to monthly...")
        pr_mm = pr_mm.resample(time="MS").sum()

    # Get PET
    if pet_pattern:
        print(f"Loading pre-computed PET from: {pet_pattern}")
        pet = load_data(pet_pattern, chunks=chunks)
    else:
        pet = _compute_pet_from_files(
            pet_method,
            tas_pattern=tas_pattern,
            tasmin_pattern=tasmin_pattern,
            tasmax_pattern=tasmax_pattern,
            hurs_pattern=hurs_pattern,
            rsds_pattern=rsds_pattern,
            sfcwind_pattern=sfcwind_pattern,
            ps_pattern=ps_pattern,
            chunks=chunks,
        )

    if pet.time.dt.dayofyear.max() > 12:
        print("Aggregating PET to monthly...")
        pet = aggregate_pet_to_monthly(pet)

    pr_mm, pet = xr.align(pr_mm, pet, join="inner")

    print(f"Computing SPEI for scales: {scales}")
    ds = compute_spei_multiscale(pr_mm, pet, scales=scales,
                                  calibration_period=calibration_period)

    print(f"Saving to: {output_path}")
    save_netcdf(ds, output_path)
    print("Done!")


def _compute_pet_from_files(
    method: str,
    tas_pattern: Optional[str] = None,
    tasmin_pattern: Optional[str] = None,
    tasmax_pattern: Optional[str] = None,
    hurs_pattern: Optional[str] = None,
    rsds_pattern: Optional[str] = None,
    sfcwind_pattern: Optional[str] = None,
    ps_pattern: Optional[str] = None,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Compute PET based on method."""

    if method == "thornthwaite":
        if not tas_pattern:
            raise ValueError("Thornthwaite requires --tas")
        print("Computing PET using Thornthwaite method...")
        tas = load_data(tas_pattern, chunks=chunks)
        tas_monthly = tas.resample(time="MS").mean()
        return calc_pet_thornthwaite(tas_monthly)

    elif method == "hargreaves":
        if not tasmin_pattern or not tasmax_pattern:
            raise ValueError("Hargreaves requires --tasmin and --tasmax")
        print("Computing PET using Hargreaves method...")
        tasmin = load_data(tasmin_pattern, chunks=chunks)
        tasmax = load_data(tasmax_pattern, chunks=chunks)
        return calc_pet_hargreaves(tasmin, tasmax)

    elif method == "penman":
        required = [tas_pattern, tasmin_pattern, tasmax_pattern,
                    hurs_pattern, rsds_pattern, sfcwind_pattern]
        if not all(required):
            raise ValueError(
                "Penman-Monteith requires --tas, --tasmin, --tasmax, "
                "--hurs, --rsds, --sfcwind"
            )
        print("Computing PET using Penman-Monteith method...")
        tas = load_data(tas_pattern, chunks=chunks)
        tasmin = load_data(tasmin_pattern, chunks=chunks)
        tasmax = load_data(tasmax_pattern, chunks=chunks)
        hurs = load_data(hurs_pattern, chunks=chunks)
        rsds = load_data(rsds_pattern, chunks=chunks)
        sfcwind = load_data(sfcwind_pattern, chunks=chunks)
        ps = load_data(ps_pattern, chunks=chunks) if ps_pattern else None

        return calc_pet_penman_monteith(
            tas, tasmin, tasmax, rsds, hurs, sfcwind, ps
        )

    else:
        raise ValueError(f"Unknown PET method: {method}")