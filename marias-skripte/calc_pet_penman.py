#!/usr/bin/env python3
# calc_pet_penman.py
# Monthly FAO-56 Penman–Monteith PET [mm/day] from preprocessed WFDE5 CONUS data.

import os
import time
import numpy as np
import xarray as xr

DATA_DIR = "/p/projects/ou/labs/ai/mariafe/data/monthly"
OUT_DIR = os.path.join(DATA_DIR, "PET")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- FAO-56 helpers ----------------------------------------------------------
Gsc = 0.0820  # MJ m-2 min-1
sigma = 4.903e-9  # MJ K-4 m-2 day-1
ALBEDO = 0.23  # reference crop

def sat_vap_pressure(T_c):
    """Saturation vapor pressure [kPa] from T [°C]."""
    return 0.6108 * np.exp(17.27 * T_c / (T_c + 237.3))

def slope_vap_curve(T_c):
    """Slope of saturation vapor pressure curve Δ [kPa/°C] at T [°C]."""
    es = sat_vap_pressure(T_c)
    return 4098.0 * es / ((T_c + 237.3) ** 2)

def psychrometric_constant(P_kpa):
    """Psychrometric constant γ [kPa/°C] for pressure P [kPa]."""
    return 0.000665 * P_kpa

def specific_humidity_to_ea(q, P_kpa):
    """Actual vapor pressure ea [kPa] from specific humidity q [kg/kg] and P [kPa]."""
    return q * P_kpa / (0.622 + 0.378 * q)

def wind_10m_to_2m(u10):
    """Convert wind at 10 m to 2 m [m/s] (FAO-56)."""
    return u10 * (4.87 / np.log(67.8 * 10.0 - 5.42))

def doy_midmonth(time_index):
    t = xr.DataArray(time_index, dims=["time"])
    days = t.dt.days_in_month                        # 28–31
    offset = xr.apply_ufunc(np.minimum, days - 1, xr.DataArray(14))
    mid = t + offset.astype("timedelta64[D]")        # ~15th (caps to month length)
    return mid.dt.dayofyear


def extraterrestrial_radiation(lat_deg, J):
    """
    Ra [MJ m-2 day-1] using FAO-56 (lat in degrees, J=day of year).
    Works with xarray broadcasting over (lat, time).
    """
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * (J / 365.0))
    delta = 0.409 * np.sin(2.0 * np.pi * (J / 365.0) - 1.39)
    omega_s = xr.apply_ufunc(
        np.arccos,
        -np.tan(phi) * np.tan(delta),
        dask="allowed"
    )
    Ra = (24.0 * 60.0 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra

def clear_sky_radiation(Ra, z_m):
    """Rso [MJ m-2 day-1] using FAO-56: Rso = (0.75 + 2e-5 z) * Ra."""
    return (0.75 + 2.0e-5 * z_m) * Ra

def pressure_to_elevation(P_kpa):
    """Approx. elevation z [m] from pressure (US Std Atmosphere)."""
    # P [kPa]; sea level ~101.3 kPa
    return 44330.0 * (1.0 - (P_kpa / 101.3) ** 0.1903)

def net_radiation_fao56(Rs_MJ, Rso_MJ, Tmax_c, Tmin_c, ea_kpa):
    """
    Net radiation Rn [MJ m-2 day-1] using FAO-56:
      Rns = (1 - albedo) * Rs
      Rnl = sigma * ((TmaxK^4 + TminK^4)/2) * (0.34 - 0.14*sqrt(ea)) * (1.35*Rs/Rso - 1.35*0.35)
    with Rs/Rso capped to [0, 1.0+] (min 0.0).
    """
    Rns = (1.0 - ALBEDO) * Rs_MJ
    Tmax_K4 = (Tmax_c + 273.16) ** 4
    Tmin_K4 = (Tmin_c + 273.16) ** 4
    Rs_Rso = xr.where(Rso_MJ > 0, (Rs_MJ / Rso_MJ).clip(min=0.0), 0.0)
    Rnl = sigma * 0.5 * (Tmax_K4 + Tmin_K4) * (0.34 - 0.14 * np.sqrt(ea_kpa)) * (1.35 * Rs_Rso - 0.35)
    return Rns - Rnl

# ---- Main PET calculation ----------------------------------------------------
def calc_pet_penman(ds_t, ds_tmin, ds_tmax, ds_sw, ds_qair, ds_ps, ds_wind):
    T = ds_t["Tair"]        # °C
    Tn = ds_tmin["tmin"]    # °C
    Tx = ds_tmax["tmax"]    # °C
    Rs_W = ds_sw["SWdown"]  # W/m2 (monthly mean); convert to MJ/m2/day
    q = ds_qair["Qair"]     # kg/kg
    P = ds_ps["PSurf"]      # kPa (already converted)
    u10 = ds_wind["wind"]   # m/s at 10 m (WFDE5)

    # Convert/derive meteorological terms
    u2 = wind_10m_to_2m(u10)
    es = 0.5 * (sat_vap_pressure(Tx) + sat_vap_pressure(Tn))    # kPa
    ea = specific_humidity_to_ea(q, P).clip(min=0)               # kPa
    delta = slope_vap_curve(T)                                   # kPa/°C
    gamma = psychrometric_constant(P)                            # kPa/°C

    # Solar geometry (per lat & time)
    lat = ds_t["lat"]
    time_index = ds_t["time"]
    J = doy_midmonth(time_index)
    # Broadcast J to (time, lat, lon) via xarray
    J_b = xr.DataArray(J, dims=["time"]).broadcast_like(T)
    Ra = extraterrestrial_radiation(lat, J_b)                    # MJ/m2/day

    # Convert shortwave to MJ/m2/day
    Rs = Rs_W * 0.0864                                          # W->MJ/day
    z = pressure_to_elevation(P)
    Rso = clear_sky_radiation(Ra, z)
    Rn = net_radiation_fao56(Rs, Rso, Tx, Tn, ea)                # MJ/m2/day

    # Soil heat flux G ~ 0 for monthly
    G = 0.0

    # FAO-56 PM equation
    PET = (
        0.408 * delta * (Rn - G) + gamma * (900.0 / (T + 273.0)) * u2 * (es - ea)
    ) / (delta + gamma * (1.0 + 0.34 * u2))

    PET = PET.clip(min=0).where(np.isfinite(PET))
    PET.name = "PET"
    PET.attrs.update({
        "long_name": "Potential Evapotranspiration (FAO-56 Penman–Monteith)",
        "units": "mm/day",
        "method": "FAO-56 Penman–Monteith monthly",
        "albedo": ALBEDO,
        "source": "WFDE5 monthly (CONUS); inputs: Tair,tmin,tmax,SWdown,wind,Qair,PSurf",
    })
    return PET

# ---- I/O & driver ------------------------------------------------------------
def main():
    print("=== START PET (FAO-56 Penman–Monteith) ===")
    t0 = time.time()

    # Load monthly inputs
    ds_t    = xr.open_dataset(os.path.join(DATA_DIR, "Tair",  "Tair_monthly.nc"))
    ds_tn   = xr.open_dataset(os.path.join(DATA_DIR, "tmin",  "tmin_monthly.nc"))
    ds_tx   = xr.open_dataset(os.path.join(DATA_DIR, "tmax",  "tmax_monthly.nc"))
    ds_sw   = xr.open_dataset(os.path.join(DATA_DIR, "SWdown","SWdown_monthly.nc"))
    ds_qair = xr.open_dataset(os.path.join(DATA_DIR, "Qair",  "Qair_monthly.nc"))
    ds_ps   = xr.open_dataset(os.path.join(DATA_DIR, "PSurf", "PSurf_monthly.nc"))
    ds_wind = xr.open_dataset(os.path.join(DATA_DIR, "wind",  "wind_monthly.nc"))

    # Align to common time/space (just in case)
    T = ds_t["Tair"]
    Tn = ds_tn["tmin"].sel(time=T.time)
    Tx = ds_tx["tmax"].sel(time=T.time)
    SW = ds_sw["SWdown"].sel(time=T.time)
    QA = ds_qair["Qair"].sel(time=T.time)
    PS = ds_ps["PSurf"].sel(time=T.time)
    WN = ds_wind["wind"].sel(time=T.time)

    # Rebuild aligned datasets for function
    ds_t_al   = T.to_dataset(name="Tair")
    ds_tn_al  = Tn.to_dataset(name="tmin")
    ds_tx_al  = Tx.to_dataset(name="tmax")
    ds_sw_al  = SW.to_dataset(name="SWdown")
    ds_q_al   = QA.to_dataset(name="Qair")
    ds_ps_al  = PS.to_dataset(name="PSurf")
    ds_w_al   = WN.to_dataset(name="wind")

    PET = calc_pet_penman(ds_t_al, ds_tn_al, ds_tx_al, ds_sw_al, ds_q_al, ds_ps_al, ds_w_al)

    out_file = os.path.join(OUT_DIR, "PET_penman_monthly.nc")
    PET.to_netcdf(out_file)
    print(f"Saved: {out_file}")
    print(f"=== DONE in {(time.time()-t0)/60:.2f} min ===")

if __name__ == "__main__":
    main()


