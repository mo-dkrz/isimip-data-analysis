"""
Standardized Precipitation Index (SPI) calculation.

Reference: 
    McKee, T.B., Doesken, N.J., & Kleist, J. (1993). 
    The relationship of drought frequency and duration
    to time scales. Proceedings of the 8th Conference
    on Applied Climatology, 179-184.

Uses zero-inflated Gamma distribution fitted per calendar month.
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


def compute_spi(
    precip: xr.DataArray,
    scale: int,
    calibration_period: Tuple[int, int] = (1991, 2020),
    min_samples: int = None,
) -> xr.DataArray:
    """
    Compute Standardized Precipitation Index.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    scale : int
        Accumulation period in months (e.g., 1, 3, 6, 12)
    calibration_period : tuple of int
        (start_year, end_year) for fitting distribution
    min_samples : int, optional
        Minimum positive samples required for fitting.
        Default: max(5, calibration_years - 1)

    Returns
    -------
    xr.DataArray
        SPI values (dimensionless, standard normal)
    """
    # Auto-set min_samples based on calibration period length
    cal_years = calibration_period[1] - calibration_period[0] + 1
    if min_samples is None:
        # Need at least 5 samples
        min_samples = max(5, cal_years - 1)
        min_samples = min(min_samples, cal_years)
    
    # Rolling sum for accumulation period
    precip_acc = rolling_sum(precip, scale)

    # Get calibration mask
    cal_mask = get_calibration_mask(precip_acc.time, calibration_period)

    # Load into memory and stack
    print("  Loading data into memory...")
    stacked = precip_acc.stack(gridcell=('lat', 'lon'))
    
    # prcess key: Load ALL data into numpy array ONCE
    data_array = stacked.values  # Shape: (n_times, n_cells)
    months = stacked.time.dt.month.values
    cal_mask_arr = cal_mask.values
    
    n_times, n_cells = data_array.shape
    
    # Prepare output
    spi_values = np.full((n_times, n_cells), np.nan)

    # Process each calendar month
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
            
            result = _spi_gamma_fit_transform(cal_data, all_data, min_samples)
            spi_values[month_idx, j] = result
            
            # progress bar every 5000 cells to stdout
            if (j + 1) % 5000 == 0:
                elapsed = _time.time() - _start
                rate = (j + 1) / elapsed
                remaining = (n_cells - j - 1) / rate
                print(f"    {j+1:,}/{n_cells:,} ({100*(j+1)/n_cells:.0f}%) - {remaining:.0f}s remaining", flush=True)
        
        print(f"    Done in {_time.time() - _start:.1f}s")

    # Rebuild DataArray
    spi_stacked = xr.DataArray(
        spi_values,
        dims=stacked.dims,
        coords=stacked.coords,
    )
    spi = spi_stacked.unstack('gridcell')

    spi.attrs = {
        "units": "1",
        "long_name": f"Standardized Precipitation Index ({scale}-month)",
        "scale": scale,
        "calibration_period": f"{calibration_period[0]}-{calibration_period[1]}",
    }

    return spi


def _spi_gamma_fit_transform(
    cal_data: np.ndarray,
    all_data: np.ndarray,
    min_samples: int = 15,
) -> np.ndarray:
    """
    Fit gamma distribution on calibration data and transform all data.
    
    Parameters
    ----------
    cal_data : np.ndarray
        1D array of calibration period values
    all_data : np.ndarray
        1D array of all values to transform
    min_samples : int
        Minimum positive samples for fitting
        
    Returns
    -------
    np.ndarray
        SPI values (same length as all_data)
    """
    # Handle NaN in calibration data
    cal_valid = cal_data[~np.isnan(cal_data)]
    
    if len(cal_valid) < min_samples:
        return np.full(len(all_data), np.nan)
    
    # Probability of zero (for zero-inflated gamma)
    n_zeros = np.sum(cal_valid <= 0)
    p_zero = n_zeros / len(cal_valid)
    
    # Fit gamma to positive values only
    pos_data = cal_valid[cal_valid > 0]
    
    if len(pos_data) < min_samples:
        return np.full(len(all_data), np.nan)
    
    # Fit gamma using scipy (MLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            shape, loc, scale = stats.gamma.fit(pos_data, floc=0)
        except Exception:
            return np.full(len(all_data), np.nan)
    
    # Transform all data to SPI (vectorized)
    spi = np.full(len(all_data), np.nan)
    valid_mask = ~np.isnan(all_data)
    
    # CDF for zeros
    zero_mask = valid_mask & (all_data <= 0)
    spi[zero_mask] = stats.norm.ppf(np.clip(p_zero * 0.5, 1e-6, 1-1e-6))
    
    # CDF for positive values
    pos_mask = valid_mask & (all_data > 0)
    if np.any(pos_mask):
        gamma_cdf = stats.gamma.cdf(all_data[pos_mask], shape, loc=0, scale=scale)
        cdf = p_zero + (1 - p_zero) * gamma_cdf
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spi[pos_mask] = stats.norm.ppf(cdf)
    
    return spi


def compute_spi_multiscale(
    precip: xr.DataArray,
    scales: List[int],
    calibration_period: Tuple[int, int] = (1991, 2020),
    min_samples: int = None,
) -> xr.Dataset:
    """
    Compute SPI for multiple time scales.

    Parameters
    ----------
    precip : xr.DataArray
        Monthly precipitation in mm/month
    scales : list of int
        List of accumulation periods (e.g., [1, 3, 6, 12])
    calibration_period : tuple of int
        (start_year, end_year) for fitting
    min_samples : int, optional
        Minimum positive samples for fitting.
        Default: auto-detected from calibration period.

    Returns
    -------
    xr.Dataset
        Dataset with SPI for each scale (spi_01, spi_03, etc.)
    """
    ds = xr.Dataset()

    for scale in scales:
        var_name = f"spi_{scale:02d}"
        print(f"Computing {var_name}...")
        ds[var_name] = compute_spi(
            precip,
            scale=scale,
            calibration_period=calibration_period,
            min_samples=min_samples,
        )

    ds.attrs = {
        "title": "Standardized Precipitation Index",
        "institution": "ISIMIP Drought Indices",
        "source": "isimip-drought",
        "references": "McKee et al. (1993); WMO (2012)",
    }

    return ds


def spi_from_files(
    precip_pattern: str,
    scales: List[int],
    output_path: str,
    calibration_period: Tuple[int, int] = (1991, 2020),
    chunks: Optional[dict] = None,
) -> None:
    """
    Compute SPI from NetCDF files and save output.

    Parameters
    ----------
    precip_pattern : str
        Glob pattern for precipitation files
    scales : list of int
        Accumulation periods
    output_path : str
        Output NetCDF path
    calibration_period : tuple of int
        Calibration years
    chunks : dict, optional
        Dask chunks
    """
    print(f"Loading precipitation from: {precip_pattern}")
    pr = load_data(precip_pattern, chunks=chunks)

    # Convert units
    print("Converting units...")
    pr_mm = convert_precip_units(pr)

    # Resample to monthly if needed
    if pr_mm.time.dt.dayofyear.max() > 12:
        print("Resampling to monthly...")
        pr_mm = pr_mm.resample(time="MS").sum()

    # Compute SPI
    print(f"Computing SPI for scales: {scales}")
    ds = compute_spi_multiscale(
        pr_mm,
        scales=scales,
        calibration_period=calibration_period,
    )

    # Save
    print(f"Saving to: {output_path}")
    save_netcdf(ds, output_path)
    print("Done!")