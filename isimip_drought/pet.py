"""
Potential Evapotranspiration (PET) calculation methods.

Supports:
- Thornthwaite: Tas only
- Hargreaves: Tmin, Tmax
- Penman-Monteith (FAO-56): Full meteorological variables
"""

import time as _time

import numpy as np
import xarray as xr

from .utils import convert_temp_units


def _thornthwaite_day_correction(lat, month, days_in_month):
    """
    Calculate Thornthwaite day-length correction factor using sunset hour angle.

    Based on standard solar geometry: N = (24/pi) * Omegas
    where Omegas = arccos(-tan phi tan omega) is the sunset hour angle.

    Parameters
    ----------
    lat : xr.DataArray
        Latitude coordinate
    month : xr.DataArray
        Month of year (1-12)
    days_in_month : xr.DataArray
        Number of days in each month

    Returns
    -------
    xr.DataArray
        Correction factor
    """
    import xarray as xr

    # Convert latitude to radians
    lat_rad = np.deg2rad(lat)

    # Day of year at mid-month (15th day)
    # Approximate: Jan 15, Feb 15, etc.
    mid_month_doy = (month - 1) * 30.5 + 15

    # Solar declination (radians) using standard formula
    # omega = 0.409 * sin(2pi * J/365 - 1.39)
    # where J is day of year
    decl = 0.409 * np.sin(2 * np.pi * mid_month_doy / 365 - 1.39)

    # Sunset hour angle: Omegas = arccos(-tan phi tan omega)
    # Handle polar cases where sun doesn't set/rise
    cos_arg = -np.tan(lat_rad) * np.tan(decl)
    cos_arg = np.clip(cos_arg, -1.0, 1.0)  # Keep within valid arccos range
    omega_s = np.arccos(cos_arg)

    # Day length in hours: N = (24/pi) * Omegas
    N = (24 / np.pi) * omega_s

    # Thornthwaite correction factor
    # = (N/12) * (days_in_month/30)
    # where N/12 adjusts for actual vs standard (12hr) day length
    # and days_in_month/30 adjusts for month length
    correction = (N / 12) * (days_in_month / 30)

    return correction


def calc_pet_thornthwaite(tas: xr.DataArray) -> xr.DataArray:
    """
    Calculate PET using Thornthwaite method.

    Reference:
    https://www.jstor.org/stable/210739?origin=crossref

    Parameters
    ----------
    tas : xr.DataArray
        Mean air temperature (K or °C)

    Returns
    -------
    xr.DataArray
        PET in mm/month
    """
    print("  Loading data into memory...")
    _start = _time.time()
    tas = convert_temp_units(tas).load()
    print(f"    Done in {_time.time() - _start:.1f}s")

    print("  Computing Thornthwaite PET...")
    _start = _time.time()

    # Monthly mean temperature, clipped to positive
    tas_pos = tas.clip(min=0)

    # Annual heat index: I = sum((T/5)^1.514) for each month
    # Calculate per year then broadcast
    heat_index = ((tas_pos / 5) ** 1.514).groupby("time.year").sum("time")

    # Broadcast heat index back to monthly
    I = heat_index.sel(year=tas.time.dt.year).drop_vars("year")

    # Thornthwaite exponent
    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.79e-2 * I) + 0.49

    # Unadjusted PET (mm/month for 30-day month, 12-hour days)
    pet_unadj = 16 * ((10 * tas_pos / I) ** a)

    # Day length correction factor (proper latitude-based)
    lat = tas.lat
    month = tas.time.dt.month
    days_in_month = tas.time.dt.days_in_month

    # Calculate mean day length for each month and latitude
    # Using lookup table approach for efficiency
    correction_factor = _thornthwaite_day_correction(lat, month, days_in_month)

    pet = pet_unadj * correction_factor

    # Set negative temps to 0 PET
    pet = pet.where(tas > 0, 0)

    print(f"    Done in {_time.time() - _start:.1f}s")

    pet.attrs = {"units": "mm/month", "long_name": "Potential Evapotranspiration (Thornthwaite)"}
    return pet


def calc_pet_hargreaves(
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    lat: xr.DataArray = None,
) -> xr.DataArray:
    """
    Calculate PET using Hargreaves-Samani method.

    Parameters
    ----------
    tasmin : xr.DataArray
        Daily minimum temperature (K or °C)
    tasmax : xr.DataArray
        Daily maximum temperature (K or °C)
    lat : xr.DataArray, optional
        Latitude for extraterrestrial radiation. If None, uses coord from data.

    Reference:
        doi.org/10.13031/2013.26773
    Returns
    -------
    xr.DataArray
        PET in mm/day (aggregate to mm/month as needed)
    """
    # Load data into memory first (SPI pattern)
    print("  Loading data into memory...")
    _start = _time.time()
    tasmin = convert_temp_units(tasmin).load()
    tasmax = convert_temp_units(tasmax).load()
    print(f"    Done in {_time.time() - _start:.1f}s")

    print("  Computing Hargreaves PET...")
    _start = _time.time()

    if lat is None:
        lat = tasmin.lat

    # Precompute Ra lookup table (366 days x n_lats) - MUCH faster
    lat_vals = lat.values
    Ra_lookup = _precompute_ra_lookup(lat_vals)
    
    # Get Ra for each timestep using lookup
    doy = tasmin.time.dt.dayofyear.values
    Ra = Ra_lookup[doy - 1, :]  # Shape: (n_times, n_lats)
    
    # Broadcast to full grid (n_times, n_lats, n_lons)
    n_lons = len(tasmin.lon)
    Ra = np.broadcast_to(Ra[:, :, np.newaxis], (len(doy), len(lat_vals), n_lons))

    # Mean temperature
    tas = (tasmin.values + tasmax.values) / 2

    # Temperature range
    tr = np.maximum(tasmax.values - tasmin.values, 0)

    # Hargreaves equation
    # PET = 0.0023 * Ra * (T + 17.8) * sqrt(TR)
    pet_values = 0.0023 * Ra * (tas + 17.8) * np.sqrt(tr)
    pet_values = np.maximum(pet_values, 0)

    # Rebuild DataArray
    pet = xr.DataArray(
        pet_values,
        dims=tasmin.dims,
        coords=tasmin.coords,
    )

    print(f"    Done in {_time.time() - _start:.1f}s")

    pet.attrs = {"units": "mm/day", "long_name": "Potential Evapotranspiration (Hargreaves)"}
    return pet


def calc_pet_penman_monteith(
    tas: xr.DataArray,
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
    rsds: xr.DataArray,
    hurs: xr.DataArray,
    sfcwind: xr.DataArray,
    ps: xr.DataArray = None,
    lat: xr.DataArray = None,
) -> xr.DataArray:
    """
    Calculate PET using FAO-56 Penman-Monteith method.

    This is the standard reference ET method.

    Parameters
    ----------
    tas : xr.DataArray
        Mean air temperature (K or °C)
    tasmin : xr.DataArray
        Minimum air temperature (K or °C)
    tasmax : xr.DataArray
        Maximum air temperature (K or °C)
    rsds : xr.DataArray
        Downwelling shortwave radiation (W/m²)
    hurs : xr.DataArray
        Relative humidity (%)
    sfcwind : xr.DataArray
        Wind speed (m/s) - assumed at 10m, converted to 2m
    ps : xr.DataArray, optional
        Surface pressure (Pa). Default: 101325 Pa
    lat : xr.DataArray, optional
        Latitude for net radiation calc
    Reference:
    ----------
        https://www.fao.org/4/x0490e/x0490e00.htm
    Returns
    -------
    xr.DataArray
        PET in mm/day
    """
    if lat is None:
        lat = tas.lat
    lat_vals = lat.values
    
    # Precompute Ra lookup table ONCE (366 days x n_lats)
    print("  Precomputing Ra lookup table...")
    _start = _time.time()
    Ra_lookup = _precompute_ra_lookup(lat_vals)  # Shape: (366, n_lats)
    print(f"    Done in {_time.time() - _start:.1f}s")

    # Get coordinates for output
    coords = tas.coords
    dims = tas.dims
    n_times = len(tas.time)
    n_lats = len(lat_vals)
    n_lons = len(tas.lon)
    
    # Process in yearly chunks to manage memory
    years = np.unique(tas.time.dt.year.values)
    pet_chunks = []
    
    print(f"  Processing {len(years)} years...")
    _total_start = _time.time()
    
    for i, year in enumerate(years):
        _start = _time.time()
        
        # Select year
        year_mask = tas.time.dt.year == year
        
        # Load just this year into memory
        tas_y = convert_temp_units(tas.sel(time=year_mask)).values
        tasmin_y = convert_temp_units(tasmin.sel(time=year_mask)).values
        tasmax_y = convert_temp_units(tasmax.sel(time=year_mask)).values
        rsds_y = rsds.sel(time=year_mask).values
        hurs_y = hurs.sel(time=year_mask).values
        sfcwind_y = sfcwind.sel(time=year_mask).values
        
        if ps is not None:
            ps_y = ps.sel(time=year_mask).values
        else:
            ps_y = np.full_like(tas_y, 101325.0)
        
        # Get doy for this year
        doy_y = tas.sel(time=year_mask).time.dt.dayofyear.values
        n_days = len(doy_y)
        
        # Get Ra from lookup and broadcast to (n_days, n_lats, n_lons)
        Ra_y = Ra_lookup[doy_y - 1, :]  # (n_days, n_lats)
        Ra_y = np.broadcast_to(Ra_y[:, :, np.newaxis], (n_days, n_lats, n_lons))
        
        # Compute PET for this year (all numpy, fast)
        pet_y = _compute_penman_numpy(
            tas_y, tasmin_y, tasmax_y, rsds_y, hurs_y, sfcwind_y, ps_y, Ra_y
        )
        
        pet_chunks.append(pet_y)
        
        elapsed = _time.time() - _start
        print(f"    Year {year} ({i+1}/{len(years)}): {elapsed:.1f}s")
    
    # Concatenate all years
    print("  Concatenating results...")
    pet_values = np.concatenate(pet_chunks, axis=0)
    
    # Rebuild DataArray
    pet = xr.DataArray(
        pet_values,
        dims=dims,
        coords=coords,
    )
    
    print(f"  Total time: {_time.time() - _total_start:.1f}s")

    pet.attrs = {"units": "mm/day", "long_name": "Potential Evapotranspiration (FAO-56 Penman-Monteith)"}
    return pet


def _precompute_ra_lookup(lat_vals: np.ndarray) -> np.ndarray:
    """
    Precompute extraterrestrial radiation for all days of year and latitudes.
    
    Parameters
    ----------
    lat_vals : np.ndarray
        1D array of latitude values
        
    Returns
    -------
    np.ndarray
        Ra lookup table, shape (366, n_lats), in mm/day
    """
    Gsc = 0.0820  # Solar constant MJ/m²/min
    
    # Create lookup for days 1-366
    doy = np.arange(1, 367)  # (366,)
    
    # Latitude in radians
    lat_rad = np.deg2rad(lat_vals)  # (n_lats,)
    
    # Solar declination for each day
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)  # (366,)
    
    # Sunset hour angle: need to broadcast (366,) x (n_lats,)
    # arccos_arg[d, l] = -tan(lat_rad[l]) * tan(decl[d])
    arccos_arg = -np.outer(np.tan(decl), np.tan(lat_rad))  # (366, n_lats)
    arccos_arg = np.clip(arccos_arg, -1.0, 1.0)
    ws = np.arccos(arccos_arg)  # (366, n_lats)
    
    # Relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)  # (366,)
    
    # Extraterrestrial radiation
    # Ra[d, l] = f(dr[d], ws[d,l], lat_rad[l], decl[d])
    sin_lat = np.sin(lat_rad)  # (n_lats,)
    cos_lat = np.cos(lat_rad)  # (n_lats,)
    sin_decl = np.sin(decl)  # (366,)
    cos_decl = np.cos(decl)  # (366,)
    
    # Broadcast multiplication
    # term1[d, l] = ws[d, l] * sin_lat[l] * sin_decl[d]
    term1 = ws * sin_lat[np.newaxis, :] * sin_decl[:, np.newaxis]
    # term2[d, l] = cos_lat[l] * cos_decl[d] * sin(ws[d, l])
    term2 = cos_lat[np.newaxis, :] * cos_decl[:, np.newaxis] * np.sin(ws)
    
    Ra = (24 * 60 / np.pi) * Gsc * dr[:, np.newaxis] * (term1 + term2)
    
    # Convert MJ/m²/day to mm/day
    Ra_mm = Ra / 2.45
    
    return Ra_mm.astype(np.float32)


def _compute_penman_numpy(
    tas: np.ndarray,
    tasmin: np.ndarray,
    tasmax: np.ndarray,
    rsds: np.ndarray,
    hurs: np.ndarray,
    sfcwind: np.ndarray,
    ps: np.ndarray,
    Ra: np.ndarray,
) -> np.ndarray:
    """
    Compute Penman-Monteith PET using pure numpy (fast).
    
    All inputs should be numpy arrays with shape (n_times, n_lats, n_lons).
    Ra is precomputed extraterrestrial radiation in mm/day.
    """
    # Convert pressure Pa -> kPa
    P = ps / 1000.0

    # Wind speed: 10m -> 2m
    u2 = sfcwind * (4.87 / np.log(67.8 * 10 - 5.42))

    # Psychrometric constant (kPa/°C)
    gamma = 0.665e-3 * P

    # Saturation vapor pressure (kPa)
    es_min = 0.6108 * np.exp(17.27 * tasmin / (tasmin + 237.3))
    es_max = 0.6108 * np.exp(17.27 * tasmax / (tasmax + 237.3))
    es = (es_min + es_max) / 2

    # Actual vapor pressure from relative humidity
    ea = es * hurs / 100.0

    # Slope of saturation vapor pressure curve (kPa/°C)
    delta = 4098 * es / ((tas + 237.3) ** 2)

    # Net radiation
    Rs = rsds * 0.0864  # W/m² to MJ/m²/day
    Ra_mj = Ra * 2.45  # mm/day back to MJ/m²/day
    Rso = np.maximum(0.75 * Ra_mj, 1e-6)  # Clear-sky radiation
    
    Rns = (1 - 0.23) * Rs  # Net shortwave
    
    # Net longwave
    sigma = 4.903e-9
    tas_k = tas + 273.16
    cloud_factor = np.clip(1.35 * np.clip(Rs / Rso, 0.25, 1.0) - 0.35, 0, 1)
    humidity_factor = 0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))
    Rnl = sigma * (tas_k ** 4) * humidity_factor * cloud_factor
    
    Rn = Rns - Rnl

    # FAO-56 Penman-Monteith equation
    numerator = 0.408 * delta * Rn + gamma * (900 / (tas + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)

    pet = numerator / denominator
    pet = np.maximum(pet, 0)

    return pet.astype(np.float32)


def _extraterrestrial_radiation(lat: xr.DataArray, doy: xr.DataArray) -> xr.DataArray:
    """
    Calculate extraterrestrial radiation (Ra).
    
    Note: For better performance, use _precompute_ra_lookup() instead.
    """
    Gsc = 0.0820  # MJ/m²/min
    lat_rad = np.deg2rad(lat)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    arccos_arg = -np.tan(lat_rad) * np.tan(decl)
    arccos_arg = np.clip(arccos_arg, -1.0, 1.0)
    ws = np.arccos(arccos_arg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat_rad) * np.sin(decl) +
        np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
    )
    Ra_mm = Ra / 2.45
    return Ra_mm


def _net_radiation(
    rsds: xr.DataArray,
    tas: xr.DataArray,
    ea: xr.DataArray,
    lat: xr.DataArray,
    doy: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate net radiation (Rn) for Penman-Monteith.
    
    Note: For better performance, use _compute_penman_numpy() instead.
    """
    Rs = rsds * 0.0864
    Ra = _extraterrestrial_radiation(lat, doy) * 2.45
    Rso = np.maximum(0.75 * Ra, 1e-6)
    Rns = (1 - 0.23) * Rs
    sigma = 4.903e-9
    tas_k = tas + 273.16
    cloud_factor = 1.35 * (Rs / Rso).clip(min=0.25, max=1.0) - 0.35
    humidity_factor = 0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))
    Rnl = sigma * (tas_k ** 4) * humidity_factor * cloud_factor
    Rn = Rns - Rnl
    return Rn


def aggregate_pet_to_monthly(pet_daily: xr.DataArray) -> xr.DataArray:
    """
    Aggregate daily PET to monthly totals.

    Parameters
    ----------
    pet_daily : xr.DataArray
        Daily PET in mm/day

    Returns
    -------
    xr.DataArray
        Monthly PET in mm/month
    """
    pet_monthly = pet_daily.resample(time="MS").sum()
    pet_monthly.attrs = {
        "units": "mm/month",
        "long_name": pet_daily.attrs.get("long_name", "PET") + " (monthly)",
    }
    return pet_monthly