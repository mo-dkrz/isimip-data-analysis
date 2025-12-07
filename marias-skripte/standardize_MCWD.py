#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:59:44 2025

@author: maria
"""
 

# %% Cell 1: imports + load MCWD
from pathlib import Path
import xarray as xr

BASE_DIR = Path.home() / "Documents/TUM/Thesis/WFDE5_CRU_CONUS/processed"
F_Mcwd   = BASE_DIR / "MCWD_conus_0p5deg_1979-2018_scales_2_6_12.nc"

MCWD = xr.open_dataset(F_Mcwd)
print(MCWD)


# %% Cell 2: compute anomalies + rename + save

# 1) monthly climatology
mcwd_clim = MCWD.groupby("time.month").mean("time")

# 2) anomalies (value - monthly mean)
mcwd_anom = MCWD.groupby("time.month") - mcwd_clim

print(type(mcwd_anom))
print(mcwd_anom)

# 3) make sure it's a Dataset (it should be, but just in case)
if isinstance(mcwd_anom, xr.DataArray):
    mcwd_anom = mcwd_anom.to_dataset(name="mcwd")

# 4) rename variables to *_anom
mcwd_anom = mcwd_anom.rename_vars({
    "mcwd_02": "mcwd_02_anom",
    "mcwd_06": "mcwd_06_anom",
    "mcwd_12": "mcwd_12_anom",
})

# 5) set attributes
mcwd_anom["mcwd_02_anom"].attrs["long_name"] = "MCWD anomaly (2-month)"
mcwd_anom["mcwd_06_anom"].attrs["long_name"] = "MCWD anomaly (6-month)"
mcwd_anom["mcwd_12_anom"].attrs["long_name"] = "MCWD anomaly (12-month)"
mcwd_anom.attrs["description"] = "MCWD anomalies relative to 1979-2018 monthly climatology"
mcwd_anom.attrs["history"] = "Computed by subtracting monthly climatology from raw MCWD dataset"

# 6) OPTIONALLY drop the 'month' coord before saving
#    (not required, but cleaner if you don't need it)
if "month" in mcwd_anom.coords:
    mcwd_anom = mcwd_anom.drop_vars("month")

# 7) save
OUT = BASE_DIR / "MCWD_conus_0p5deg_1979-2018_McwdAnomalies.nc"
mcwd_anom.to_netcdf(OUT)
print("Saved:", OUT)




# %% Cell 3: sanity checks on anomalies

# Use the SAME name you used above: mcwd_anom (lowercase)
print("Global mean over time & space (should ~0):")
for v in ["mcwd_02_anom", "mcwd_06_anom", "mcwd_12_anom"]:
    mean_val = mcwd_anom[v].mean("time").mean(("lat", "lon")).item()
    print(v, ":", mean_val)

print("\nMonthly climatology of anomalies (should be ~0 for each month):")
for v in ["mcwd_02_anom", "mcwd_06_anom", "mcwd_12_anom"]:
    clim_anom = mcwd_anom[v].groupby("time.month").mean("time")
    print(v, "min:", float(clim_anom.min()), "max:", float(clim_anom.max()))

# Spot-check a grid point
lat0, lon0 = 31.25, -100.25  # adjust to a valid grid point if needed
orig = MCWD.mcwd_06.sel(lat=lat0, lon=lon0, method="nearest")
anom = mcwd_anom.mcwd_06_anom.sel(lat=lat0, lon=lon0, method="nearest")

print("\nSpot check at lat, lon =", float(orig.lat), float(orig.lon))
print("Original mcwd_06 mean:", float(orig.mean()))
print("Anomaly mcwd_06_anom mean:", float(anom.mean()))






# %% 

MCWD_anom = mcwd_anom  # whatever variable name you use for the anomaly dataset



print(MCWD_anom.mcwd_02.mean("time").mean(("lat", "lon")).item())
print(MCWD_anom.mcwd_06.mean("time").mean(("lat", "lon")).item())
print(MCWD_anom.mcwd_12.mean("time").mean(("lat", "lon")).item())

for v in ["mcwd_02", "mcwd_06", "mcwd_12"]:
    clim_anom = MCWD_anom[v].groupby("time.month").mean("time")
    print(v, "monthly mean range:", float(clim_anom.min()), float(clim_anom.max()))
lat0, lon0 = 31.25, -100.25  # adjust to a grid point you know
orig = MCWD.mcwd_06.sel(lat=lat0, lon=lon0)
anom = MCWD_anom.mcwd_06.sel(lat=lat0, lon=lon0)

# time mean of anomaly should be ~0
print("orig mean:", float(orig.mean()))
print("anom mean:", float(anom.mean()))

mcwd_std = (MCWD.groupby("time.month") - mcwd_clim) / mcwd_std_dev



mcwd_std_dev = MCWD.groupby("time.month").std("time")

# %% 


mcwd_anom["mcwd_02_anom"].attrs["long_name"] = "MCWD anomaly (2-month)"
mcwd_anom["mcwd_06_anom"].attrs["long_name"] = "MCWD anomaly (6-month)"
mcwd_anom["mcwd_12_anom"].attrs["long_name"] = "MCWD anomaly (12-month)"
mcwd_anom.attrs["description"] = "MCWD anomalies relative to 1979-2018 monthly climatology"
mcwd_anom.attrs["history"] = "Computed by subtracting monthly climatology from raw MCWD dataset"





# %% 

OUT = BASE_DIR / "MCWD_conus_0p5deg_1979-2018_McwdAnomalies.nc"
mcwd_anom.to_netcdf(OUT)

print("Saved:", OUT)



# %% 
