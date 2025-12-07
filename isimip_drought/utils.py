"""Shared utility functions for drought index calculations."""

import glob
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr


def load_data(
    pattern: str,
    variable: Optional[str] = None,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """
    Load NetCDF data from glob pattern(s).

    Parameters
    ----------
    pattern : str
        Glob pattern(s) for input files. Supports:
        - Single pattern: '/path/to/pr/*.nc'
        - Multiple patterns (comma-separated): '/path1/pr*.nc,/path2/pr*.nc'
    variable : str, optional
        Variable name to extract. If None, uses first data variable.
    chunks : dict, optional
        Chunk sizes for dask. Default: no chunking (load directly)

    Returns
    -------
    xr.DataArray
        Loaded data array
    """
    patterns = [p.strip() for p in pattern.split(',')]
    files = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if not matched:
            print(f"  Warning: No files found for pattern: {p}")
        files.extend(matched)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern(s): {pattern}")

    # Sort all files to ensure correct time ordering
    files = sorted(files)
    print(f"  Found {len(files)} files")

    # Open dataset - use native file chunking or no chunking
    if chunks is None:
        ds = xr.open_mfdataset(files, combine="by_coords")
    elif chunks == 'auto':
        ds = xr.open_mfdataset(files, chunks='auto', combine="by_coords")
    else:
        ds = xr.open_mfdataset(files, chunks=chunks, combine="by_coords")

    if variable is None:
        data_vars = [v for v in ds.data_vars if v not in ds.coords]
        if not data_vars:
            raise ValueError("No data variables found in dataset")
        variable = data_vars[0]

    return ds[variable]


def convert_precip_units(pr: xr.DataArray) -> xr.DataArray:
    """
    Convert precipitation from kg/m²/s to mm/day.

    ISIMIP data is in kg/m²/s (= mm/s), needs conversion to mm/day.
    Caller should then resample to monthly with .resample(time='MS').sum()

    Parameters
    ----------
    pr : xr.DataArray
        Precipitation in kg/m²/s

    Returns
    -------
    xr.DataArray
        Precipitation in mm/day
    """
    # kg/m²/s → mm/day (×86400)
    # Note: Do NOT multiply by days_in_month here!
    # The caller will use resample().sum() to get monthly totals
    pr_mm_day = pr * 86400
    pr_mm_day.attrs["units"] = "mm/day"
    return pr_mm_day


def convert_temp_units(temp: xr.DataArray) -> xr.DataArray:
    """
    Convert temperature from Kelvin to Celsius if needed.

    Parameters
    ----------
    temp : xr.DataArray
        Temperature (K or °C)

    Returns
    -------
    xr.DataArray
        Temperature in °C
    """
    # Check if likely in Kelvin (values > 100)
    sample = temp.isel(time=0).values
    if np.nanmean(sample) > 100:
        temp = temp - 273.15
        temp.attrs["units"] = "degC"
    return temp


def rolling_sum(da: xr.DataArray, window: int) -> xr.DataArray:
    """
    Calculate rolling sum over time dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    window : int
        Window size in time steps

    Returns
    -------
    xr.DataArray
        Rolling sum (NaN for first window-1 timesteps)
    """
    return da.rolling(time=window, min_periods=window).sum()


def get_calibration_mask(
    time: xr.DataArray,
    calibration_period: Tuple[int, int],
) -> xr.DataArray:
    """
    Create boolean mask for calibration period.

    Parameters
    ----------
    time : xr.DataArray
        Time coordinate
    calibration_period : tuple of int
        (start_year, end_year) inclusive

    Returns
    -------
    xr.DataArray
        Boolean mask (True for calibration period)
    """
    years = time.dt.year
    return (years >= calibration_period[0]) & (years <= calibration_period[1])


def save_netcdf(
    ds: xr.Dataset,
    output_path: str,
    compress: bool = True,
    complevel: int = 4,
) -> None:
    """
    Save dataset to NetCDF with CF conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to save
    output_path : str
        Output file path
    compress : bool
        Enable zlib compression
    complevel : int
        Compression level (1-9)
    """
    encoding = {}
    if compress:
        for var in ds.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": complevel,
                "dtype": "float32",
            }

    ds.to_netcdf(output_path, encoding=encoding)


def parse_calibration_period(cal_str: str) -> Tuple[int, int]:
    """
    Parse calibration period string.

    Parameters
    ----------
    cal_str : str
        Calibration period as 'YYYY-YYYY'

    Returns
    -------
    tuple of int
        (start_year, end_year)
    """
    parts = cal_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid calibration period format: {cal_str}. Use 'YYYY-YYYY'")
    return int(parts[0]), int(parts[1])


def get_lat_weights(lat: xr.DataArray) -> xr.DataArray:
    """
    Calculate area weights based on latitude.

    Parameters
    ----------
    lat : xr.DataArray
        Latitude coordinate

    Returns
    -------
    xr.DataArray
        Cosine weights normalized to sum to 1
    """
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.sum()
    return weights