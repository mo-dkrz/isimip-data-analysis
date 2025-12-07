"""
Potential Evapotranspiration (PET) calculation methods.

Supports:
- Thornthwaite: Tas only
- Hargreaves: Tmin, Tmax
- Penman-Monteith (FAO-56): Full meteorological variables
"""

import numpy as np
import xarray as xr

from .utils import convert_temp_units


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
    tas = convert_temp_units(tas)

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

    # Day length correction factor (simplified, based on latitude)
    # Using 12 hours as baseline - for more accuracy, would need latitude
    days_in_month = tas.time.dt.days_in_month
    pet = pet_unadj * (days_in_month / 30)

    # Set negative temps to 0 PET
    pet = pet.where(tas > 0, 0)

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
    tasmin = convert_temp_units(tasmin)
    tasmax = convert_temp_units(tasmax)

    if lat is None:
        lat = tasmin.lat

    # Mean temperature
    tas = (tasmin + tasmax) / 2

    # Temperature range
    tr = tasmax - tasmin
    tr = tr.clip(min=0)

    # Day of year
    doy = tasmin.time.dt.dayofyear

    # Extraterrestrial radiation (Ra) in mm/day equivalent
    Ra = _extraterrestrial_radiation(lat, doy)

    # Hargreaves equation
    # PET = 0.0023 * Ra * (T + 17.8) * sqrt(TR)
    pet = 0.0023 * Ra * (tas + 17.8) * np.sqrt(tr)
    pet = pet.clip(min=0)

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
    # Unit conversions
    tas = convert_temp_units(tas)
    tasmin = convert_temp_units(tasmin)
    tasmax = convert_temp_units(tasmax)

    if lat is None:
        lat = tas.lat

    if ps is None:
        ps = xr.ones_like(tas) * 101325.0  # Standard pressure in Pa

    # Convert pressure Pa -> kPa
    P = ps / 1000.0

    # Wind speed: 10m -> 2m (logarithmic profile)
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

    # Net radiation calculation
    doy = tas.time.dt.dayofyear
    Rn = _net_radiation(rsds, tas, ea, lat, doy)

    # Soil heat flux (assume 0 for daily)
    G = 0

    # FAO-56 Penman-Monteith equation
    # Reference: Allen et al. (1998) FAO Irrigation and Drainage Paper 56
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (tas + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)

    pet = numerator / denominator
    pet = pet.clip(min=0)

    pet.attrs = {"units": "mm/day", "long_name": "Potential Evapotranspiration (FAO-56 Penman-Monteith)"}
    return pet


def _extraterrestrial_radiation(lat: xr.DataArray, doy: xr.DataArray) -> xr.DataArray:
    """
    Calculate extraterrestrial radiation (Ra).

    Parameters
    ----------
    lat : xr.DataArray
        Latitude in degrees
    doy : xr.DataArray
        Day of year

    Returns
    -------
    xr.DataArray
        Ra in mm/day equivalent
    """
    # Solar constant
    Gsc = 0.0820  # MJ/m²/min

    # Convert latitude to radians
    lat_rad = np.deg2rad(lat)

    # Solar declination
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

    # Sunset hour angle
    # Clip argument to [-1, 1] to avoid NaN from numerical precision at high latitudes
    arccos_arg = -np.tan(lat_rad) * np.tan(decl)
    arccos_arg = np.clip(arccos_arg, -1.0, 1.0)
    ws = np.arccos(arccos_arg)

    # Relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # Extraterrestrial radiation (MJ/m²/day)
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat_rad) * np.sin(decl) +
        np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
    )

    # Convert MJ/m²/day to mm/day (latent heat of vaporization ~2.45 MJ/kg)
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

    Parameters
    ----------
    rsds : xr.DataArray
        Downwelling shortwave radiation (W/m²)
    tas : xr.DataArray
        Air temperature (°C)
    ea : xr.DataArray
        Actual vapor pressure (kPa)
    lat : xr.DataArray
        Latitude
    doy : xr.DataArray
        Day of year

    Returns
    -------
    xr.DataArray
        Net radiation in MJ/m²/day
    """
    # Convert W/m² to MJ/m²/day
    Rs = rsds * 0.0864  # W/m² * 86400 s/day / 1e6 = MJ/m²/day

    # Extraterrestrial radiation
    Ra = _extraterrestrial_radiation(lat, doy) * 2.45  # Convert back to MJ/m²/day

    # Clear-sky radiation (simplified)
    Rso = 0.75 * Ra

    # Net shortwave radiation (albedo = 0.23 for reference crop)
    Rns = (1 - 0.23) * Rs

    # Net longwave radiation (Stefan-Boltzmann)
    sigma = 4.903e-9  # MJ/K⁴/m²/day
    tas_k = tas + 273.16

    # Cloudiness factor
    cloud_factor = 1.35 * (Rs / Rso).clip(min=0.25, max=1.0) - 0.35

    # Humidity factor
    humidity_factor = 0.34 - 0.14 * np.sqrt(ea)

    Rnl = sigma * (tas_k ** 4) * humidity_factor * cloud_factor

    # Net radiation
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