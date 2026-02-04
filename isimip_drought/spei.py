"""
Standardized Precipitation Evapotranspiration Index (SPEI) calculation.

Reference:
    Vicente-Serrano et al. (2010): https://journals.ametsoc.org/doi/10.1175/2009JCLI2909.1
    Beguería et al. (2014): https://doi.org/10.1002/joc.3887

Uses 3-parameter log-logistic distribution fitted to water balance (P - PET) per calendar month.
Parameters estimated using unbiased Probability-Weighted Moments (PWM) method.
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import stats, special

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
    return_water_balance: bool = False,
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
    return_water_balance : bool, optional
        If True, return both SPEI and water balance as a Dataset

    Returns
    -------
    xr.DataArray or xr.Dataset
        SPEI values (dimensionless, standard normal)
        If return_water_balance=True, returns Dataset with 'spei' and 'wb' variables
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
    
    # Optionally return water balance as well
    if return_water_balance:
        wb_acc.attrs = {
            "units": "mm",
            "long_name": f"Water Balance P-PET ({scale}-month accumulation)",
            "scale": scale,
        }
        return xr.Dataset({"spei": spei, "wb": wb_acc})
    
    return spei


def _spei_loglogistic_fit_transform(
    cal_data: np.ndarray,
    all_data: np.ndarray,
    min_samples: int = 15,
) -> np.ndarray:
    """
    Fit 3-parameter log-logistic using unbiased PWM (Beguería et al. 2014).
    
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
    
    # Fit 3-parameter log-logistic using unbiased PWM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            alpha, beta, gamma = _fit_loglogistic_pwm(cal_valid)
        except Exception:
            return np.full(len(all_data), np.nan)
    
    # Transform all data to SPEI (vectorized)
    spei = np.full(len(all_data), np.nan)
    valid_mask = ~np.isnan(all_data)
    
    if not np.any(valid_mask):
        return spei
    
    # 3-parameter log-logistic CDF: F(x) = 1 / [1 + (α/(x-γ))^β]
    # Only defined for x > γ (location parameter)
    x = all_data[valid_mask]
    cdf = np.zeros(len(x))
    
    # For x > γ, calculate CDF normally
    safe_mask = x > gamma
    if np.any(safe_mask):
        cdf[safe_mask] = 1.0 / (1.0 + (alpha / (x[safe_mask] - gamma)) ** beta)
    
    # For x <= γ, CDF approaches 0 (extreme drought)
    # Set to small value to avoid ppf(-inf)
    cdf[~safe_mask] = 1e-6
    
    # Clip to avoid numerical issues at extremes
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    spei[valid_mask] = stats.norm.ppf(cdf)
    
    return spei


def _fit_loglogistic_pwm(data: np.ndarray) -> tuple:
    """
    Fit 3-parameter log-logistic using unbiased Probability-Weighted Moments.
    
    Uses the exact formulas from Vicente-Serrano et al. (2010) for SPEI.
    
    Reference: Vicente-Serrano et al. (2010) - A Multiscalar Drought Index 
    Sensitive to Global Warming: The Standardized Precipitation 
    Evapotranspiration Index. Journal of Climate, 23, 1696-1718.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of water balance data
        
    Returns
    -------
    tuple
        (alpha, beta, gamma) - scale, shape, location parameters
        Returns (nan, nan, nan) if fitting fails
    """
    from scipy import special
    
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    
    # Sort data
    x = np.sort(x)
    n = len(x)
    
    # Unbiased PWM estimator
    # Fi = (i - 0.35) / N
    i = np.arange(1, n + 1)
    Fi = (i - 0.35) / n
    
    # Calculate PWMs (b0, b1, b2) = (w0, w1, w2)
    w0 = np.mean(x)
    w1 = np.mean(x * (1 - Fi))
    w2 = np.mean(x * (1 - Fi) ** 2)
    
    # Vicente-Serrano et al. (2010) equations:
    # β = (2w₁ - w₀) / (6w₁ - w₀ - 6w₂)
    denom = 6.0 * w1 - w0 - 6.0 * w2
    if abs(denom) < 1e-10:
        return np.nan, np.nan, np.nan
    
    beta = (2.0 * w1 - w0) / denom
    
    # Validity check: log-logistic requires β > 0
    # Also need β > 1 for Γ(1-1/β) to be defined (since 1-1/β must be > 0)
    if not np.isfinite(beta) or beta <= 1.0:
        return np.nan, np.nan, np.nan
    
    # Gamma function term: Γ(1+1/β) × Γ(1-1/β)
    try:
        G = special.gamma(1.0 + 1.0 / beta) * special.gamma(1.0 - 1.0 / beta)
    except (ValueError, RuntimeWarning):
        return np.nan, np.nan, np.nan
    
    if not np.isfinite(G) or G == 0:
        return np.nan, np.nan, np.nan
    
    # α = [(w₀ - 2w₁) × β] / Γ(1+1/β)Γ(1-1/β)
    alpha = ((w0 - 2.0 * w1) * beta) / G
    
    # γ = w₀ - α × Γ(1+1/β)Γ(1-1/β)
    gamma = w0 - alpha * G
    
    # Validity check: scale parameter must be positive
    if not (np.isfinite(alpha) and np.isfinite(gamma)) or alpha <= 0:
        return np.nan, np.nan, np.nan
    
    return alpha, beta, gamma


def compute_spei_multiscale(
    precip: xr.DataArray,
    pet: xr.DataArray,
    scales: List[int],
    calibration_period: Tuple[int, int] = (1991, 2020),
    min_samples: int = None,
    include_water_balance: bool = True,
) -> xr.Dataset:
    """
    Compute SPEI for multiple time scales.
    
    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    pet : xr.DataArray
        Monthly PET in mm/month
    scales : List[int]
        List of accumulation periods in months
    calibration_period : tuple of int
        (start_year, end_year) for fitting distribution
    min_samples : int, optional
        Minimum samples required for fitting
    include_water_balance : bool, optional
        If True, include accumulated water balance (P-PET) for each scale
        
    Returns
    -------
    xr.Dataset
        Dataset with SPEI variables (spei_03, spei_06, etc.)
        and optionally water balance variables (wb_03, wb_06, etc.)
    """
    ds = xr.Dataset()

    for scale in scales:
        spei_var = f"spei_{scale:02d}"
        wb_var = f"wb_{scale:02d}"
        
        print(f"Computing {spei_var}...")
        result = compute_spei(
            precip, pet, scale=scale,
            calibration_period=calibration_period,
            min_samples=min_samples,
            return_water_balance=include_water_balance,
        )
        
        if include_water_balance:
            ds[spei_var] = result["spei"]
            ds[wb_var] = result["wb"]
        else:
            ds[spei_var] = result

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
    pet_method: str = "penman",
    tasmin_pattern: Optional[str] = None,
    tasmax_pattern: Optional[str] = None,
    tas_pattern: Optional[str] = None,
    hurs_pattern: Optional[str] = None,
    rsds_pattern: Optional[str] = None,
    sfcwind_pattern: Optional[str] = None,
    ps_pattern: Optional[str] = None,
    chunks: Optional[dict] = None,
    include_water_balance: bool = True,
) -> None:
    """
    Compute SPEI from NetCDF files and save output.
    
    Parameters
    ----------
    include_water_balance : bool, optional
        If True, include accumulated water balance (P-PET) variables in output.
        Default: True
    """
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
    ds = compute_spei_multiscale(
        pr_mm, pet, scales=scales,
        calibration_period=calibration_period,
        include_water_balance=include_water_balance,
    )

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