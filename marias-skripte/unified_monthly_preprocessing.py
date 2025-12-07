#!/usr/bin/env python3
# =======================================================================
# unified_monthly_preprocessing.py
#
# PURPOSE
# -------
# Create a *single* preprocessing workflow that works for:
#   - Historical WFDE5 (e.g. /.../WFDE5_CRU/)
#   - ISIMIP3b future projections (e.g. /.../ssp126/GFDL-ESM4_conus/)
#
# The script:
#   1. Detects dataset type (WFDE5 vs ISIMIP) from filesystem structure.
#   2. Finds all relevant NetCDF files and determines which climate
#      variable each file contains.
#   3. Maps source variable names (WFDE5 / ISIMIP) to a unified naming:
#        pr, prsn, tas, tasmin, tasmax, rsds, rlds, sfcwind, huss, hurs, ps
#   4. Converts units to drought-index-friendly standards:
#        - P, P_snow: mm/day → later aggregated to mm/month
#        - Temperatures: K → °C
#        - Pressure: kept in Pa
#        - Radiation: W/m²
#        - Humidity: kg/kg (huss), % (hurs)
#   5. Aggregates daily → monthly:
#        - "sum" for precipitation
#        - "mean" for all other variables
#   6. Writes one monthly NetCDF per unified variable into OUTPUT_ROOT.
#
# These monthly files are suitable to compute:
#   - SPI (2/3/6 month) using monthly precipitation
#   - SPEI (2/3/6 month) using monthly P and PET
#   - MCWD using monthly P and PET
#
# This script does NOT compute PET / SPI / SPEI / MCWD yet.
# It only prepares consistent monthly inputs for them.
#
# =======================================================================

import os
import re
import time
import xarray as xr
import numpy as np
import pandas as pd

# =======================================================================
# 0. USER CONFIGURATION
# =======================================================================

# INPUT_ROOT:
#   - For WFDE5: e.g., "/p/projects/ou/labs/ai/mariafe/data/WFDE5_CRU"
#   - For ISIMIP: e.g., "/p/projects/ou/labs/ai/mariafe/data/ssp126/GFDL-ESM4_conus"
#
INPUT_ROOT = "/p/projects/ou/labs/ai/mariafe/data/WFDE5_CRU" ##HISTORIC
INPUT_ROOT = "/p/projects/ou/labs/ai/mariafe/data/ssp126/GFDL-ESM4_conus" ###ONE FUTURE

# Where monthly outputs will be stored (structure: OUTPUT_ROOT/<var>/<var>_monthly.nc)
OUTPUT_ROOT = os.path.join(INPUT_ROOT, "monthly")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Known WFDE5 variable folder names (for detection and mapping)
WFDE5_VAR_FOLDERS = {
    "Rainf", "Snowf", "Tair", "tmin", "tmax",
    "SWdown", "LWdown", "Qair", "PSurf", "wind"
}

# =======================================================================
# 1. VARIABLE NAME MAPPING
# =======================================================================
# We map *source* names (WFDE5 folder/variable names or ISIMIP variable names)
# to a *unified* variable name. This unified name is what we want to see in
# the final monthly datasets and will be used downstream for PET/SPI/SPEI/MCWD.
#
# Each entry:
#   key   = source variable (folder name, or var from filename/dataset)
#   value = (actual_var_name_in_file, unified_output_name)
#
# For WFDE5, folder "tmin" contains a data var "Tair" but it actually represents
# daily minimum T, so we map it to "tasmin".
# For ISIMIP, variable names are already CMIP-like (pr, tas, tasmax, ...).

VAR_NAME_MAP = {
    # --- Precipitation --------------------------------------------------
    # WFDE5
    "Rainf":  ("Rainf",  "pr"),       # rain flux → pr
    "Snowf":  ("Snowf",  "prsn"),     # snow flux → prsn
    # ISIMIP
    "pr":     ("pr",     "pr"),
    "prsn":   ("prsn",   "prsn"),

    # --- Temperature ----------------------------------------------------
    # WFDE5
    "Tair":   ("Tair",   "tas"),      # daily mean T
    "tmin":   ("Tair",   "tasmin"),   # daily minimum T (pre-calculated)
    "tmax":   ("Tair",   "tasmax"),   # daily maximum T (pre-calculated)
    # ISIMIP
    "tas":    ("tas",    "tas"),
    "tasmin": ("tasmin", "tasmin"),
    "tasmax": ("tasmax", "tasmax"),

    # --- Radiation ------------------------------------------------------
    # WFDE5
    "SWdown": ("SWdown", "rsds"),
    "LWdown": ("LWdown", "rlds"),
    # ISIMIP
    "rsds":   ("rsds",   "rsds"),
    "rlds":   ("rlds",   "rlds"),

    # --- Wind -----------------------------------------------------------
    # WFDE5
    "wind":     ("Wind",    "sfcwind"),  # folder "wind", var "Wind" inside
    "Wind":     ("Wind",    "sfcwind"),
    # ISIMIP
    "sfcwind":  ("sfcwind", "sfcwind"),

    # --- Humidity -------------------------------------------------------
    # WFDE5
    "Qair":  ("Qair",  "huss"),      # specific humidity
    # ISIMIP
    "huss":  ("huss",  "huss"),      # specific humidity
    "hurs":  ("hurs",  "hurs"),      # relative humidity (%)

    # --- Surface pressure -----------------------------------------------
    # WFDE5
    "PSurf": ("PSurf", "ps"),
    # ISIMIP
    "ps":    ("ps",    "ps"),
}


# =======================================================================
# 2. DATASET TYPE DETECTION
# =======================================================================
def detect_dataset_type(root):
    """
    Decide whether INPUT_ROOT looks like WFDE5 or ISIMIP.

    - WFDE5: we expect subdirectories named Rainf, Snowf, Tair, ...
    - ISIMIP: we expect many NetCDF files directly in the root with
              names like '..._ssp126_pr_conus_daily_2015_2020.nc'

    Returns:
        "WFDE5" or "ISIMIP"
    """
    entries = os.listdir(root)
    # If we see known WFDE5 variable folders, treat as WFDE5
    if any(e in WFDE5_VAR_FOLDERS and os.path.isdir(os.path.join(root, e)) for e in entries):
        print("Detected dataset type: WFDE5 (folder-based structure).")
        return "WFDE5"

    # If we see .nc files with 'sspXYZ' in their names, treat as ISIMIP
    for e in entries:
        if e.endswith(".nc") and "ssp" in e:
            print("Detected dataset type: ISIMIP (flat, scenario-based files).")
            return "ISIMIP"

    # Fallback: if there are .nc files directly in root and no WFDE5 folders
    if any(e.endswith(".nc") for e in entries):
        print("Detected dataset type: ISIMIP-like (flat .nc files).")
        return "ISIMIP"

    raise RuntimeError("Could not detect dataset type from INPUT_ROOT: {}".format(root))


# =======================================================================
# 3. FILE DISCOVERY
# =======================================================================
def discover_files_wfde5(root):
    """
    Discover WFDE5 daily files.

    WFDE5 structure:
        root/
            Rainf/*.nc
            Snowf/*.nc
            Tair/*.nc
            tmin/*.nc
            ...

    Returns:
        dict: { source_var_name : [list_of_file_paths] }
        where source_var_name is the folder name (Rainf, Snowf, tmin, ...)
    """
    var_files = {}
    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path) and sub in WFDE5_VAR_FOLDERS:
            files = sorted(f for f in os.listdir(sub_path) if f.endswith(".nc"))
            if files:
                var_files[sub] = [os.path.join(sub_path, f) for f in files]
    return var_files


def extract_isimip_var_from_filename(fname):
    """
    Extract ISIMIP variable name from a file name.

    Example filenames:
        gfdl-esm4_r1i1p1f1_w5e5_ssp126_tas_conus_daily_2015_2020.nc
                                          ^^^
    Pattern:
        ..._ssp[digits]_<var>_conus_daily_...

    Returns:
        variable name (str) or None if pattern not found.
    """
    m = re.search(r"ssp\d+_([a-zA-Z0-9]+)_conus_daily", fname)
    if m:
        return m.group(1)
    else:
        return None


def discover_files_isimip(root):
    """
    Discover ISIMIP daily files in a flat directory.

    Expect structure:
        root/
            model_scenario_var_conus_daily_y1_y2.nc
            ...

    Returns:
        dict: { source_var_name : [list_of_file_paths] }
        where source_var_name ~ pr, prsn, tas, tasmax, etc.
    """
    var_files = {}
    for f in sorted(os.listdir(root)):
        if not f.endswith(".nc"):
            continue
        var = extract_isimip_var_from_filename(f)
        if var is None:
            # non-standard file, skip
            continue
        var_files.setdefault(var, []).append(os.path.join(root, f))
    return var_files


# =======================================================================
# 4. UNIT CONVERSION
# =======================================================================
def convert_units(da, unified_name):
    """
    Convert DataArray units to standard units used for drought indices.

    unified_name : str
        The target unified variable name (pr, prsn, tas, rsds, ...)

    Rules:

    - Precipitation (pr, prsn):
        Input (WFDE5 & ISIMIP) is usually "kg m-2 s-1" (kg per m² per second).
        1 kg/m² ~= 1 mm water layer.
        We convert to mm/day by multiplying by 86400 (seconds/day).
        Monthly aggregation then sums mm/day to mm/month.

    - Temperature (tas, tasmin, tasmax):
        Input usually in K (Kelvin).
        We convert to °C by T(°C) = T(K) - 273.15.

    - Radiation (rsds, rlds):
        Typically in W m-2. We keep as W m-2 and take monthly means.

    - Surface pressure (ps):
        Keep in Pa (Pascal). Many PET formulations expect Pa or hPa;
        we choose Pa here consistently.

    - Humidity:
        huss = specific humidity [kg/kg]
        hurs = relative humidity [%]
        We only normalize units text where needed.

    Returns:
        DataArray with updated data and attrs['units'].
    """
    units = da.attrs.get("units", "").lower().strip()

    # --- Temperature: K → °C -------------------------------------------
    if unified_name in ["tas", "tasmin", "tasmax"]:
        if units in ["k", "kelvin"]:
            da = da - 273.15
        da.attrs["units"] = "°c"

    # --- Precipitation: kg m-2 s-1 → mm/day ----------------------------
    if unified_name in ["pr", "prsn"]:
        # We only transform if units indicate a flux
        if "kg m-2 s-1" in units or "kg m**-2 s**-1" in units:
            da = da * 86400.0
        # Now it's mm/day (numerically equivalent to kg/m²/day)
        da.attrs["units"] = "mm/day"

    # --- Radiation: keep W/m² ------------------------------------------
    if unified_name in ["rsds", "rlds"]:
        # Just normalize unit name
        if "w" in units and "m-2" in units:
            da.attrs["units"] = "w m-2"

    # --- Surface pressure: keep Pa -------------------------------------
    if unified_name == "ps":
        # normalize name to Pa if it looks like Pa
        if "pa" in units:
            da.attrs["units"] = "pa"

    # --- Humidity -------------------------------------------------------
    if unified_name == "huss":
        da.attrs["units"] = "kg/kg"
    if unified_name == "hurs":
        da.attrs["units"] = "%"

    return da


# =======================================================================
# 5. DAILY → MONTHLY AGGREGATION
# =======================================================================
def aggregate_to_monthly(da, unified_name):
    """
    Aggregate daily DataArray to monthly values.

    We use calendar-based resampling with frequency "MS"
    (Month Start) so each output point represents one month.

    Aggregation method:
        - "sum" for precipitation (pr, prsn) → mm/month
        - "mean" for all others (tas, tasmin, tasmax, rsds, rlds,
          sfcwind, ps, huss, hurs)

    Returns:
        DataArray with new time coordinate monthly.
    """
    if unified_name in ["pr", "prsn"]:
        # For precipitation, sum daily mm/day → mm/month
        da_month = da.resample(time="MS").sum(skipna=True)
    else:
        # For everything else, use monthly mean
        da_month = da.resample(time="MS").mean(skipna=True)

    return da_month


# =======================================================================
# 6. PROCESS ONE VARIABLE (LIST OF FILES)
# =======================================================================
def process_variable(source_var_name, file_list, dataset_type):
    """
    Process one climate variable (e.g. pr, tas, Rainf, tmin) from daily files,
    and write a unified monthly NetCDF.

    Args:
        source_var_name : str
            The variable name in folder/filename (e.g. "Rainf", "pr", "tasmin").
        file_list : list of str
            All NetCDF files containing this variable (e.g. different year blocks).
        dataset_type : "WFDE5" or "ISIMIP"
            Only used for logging; logic is independent of type.

    Steps:
      1. Look up actual variable name in file + unified output name via VAR_NAME_MAP.
      2. Loop through all files:
           - open dataset
           - select actual data variable
           - convert units
           - aggregate daily → monthly
         collect in a list.
      3. Concatenate all time-blocks into one long monthly series.
      4. Drop duplicate timestamps if any (just in case).
      5. Save to OUTPUT_ROOT/<unified_name>/<unified_name>_monthly.nc
    """
    if source_var_name not in VAR_NAME_MAP:
        print(f"  [SKIP] Source variable '{source_var_name}' not in VAR_NAME_MAP.")
        return

    actual_var, unified_name = VAR_NAME_MAP[source_var_name]

    print(f"\n=== Processing {source_var_name} ({dataset_type}) → unified '{unified_name}' ===")
    print(f"  Using actual data variable name: '{actual_var}'")

    # Output directory for this unified variable
    out_dir = os.path.join(OUTPUT_ROOT, unified_name)
    os.makedirs(out_dir, exist_ok=True)

    monthly_list = []

    for fpath in file_list:
        print(f"    Reading {os.path.basename(fpath)}")
        ds = xr.open_dataset(fpath)

        # Safety: check variable exists
        if actual_var not in ds.data_vars:
            print(f"      [WARN] '{actual_var}' not found in {fpath}. Found vars: {list(ds.data_vars)}")
            ds.close()
            continue

        da = ds[actual_var]

        # Convert units in a consistent way
        da = convert_units(da, unified_name)

        # Aggregate daily → monthly
        da_month = aggregate_to_monthly(da, unified_name)

        # Ensure consistent name for variable
        da_month = da_month.rename(unified_name)

        monthly_list.append(da_month)
        ds.close()

    if not monthly_list:
        print(f"  [WARN] No monthly data could be produced for {source_var_name}.")
        return

    # Concatenate along time
    combined = xr.concat(monthly_list, dim="time")

    # Remove any duplicate monthly timestamps (can happen at boundaries)
    combined = combined.sel(time=~combined.get_index("time").duplicated())

    # Save final monthly file
    out_file = os.path.join(out_dir, f"{unified_name}_monthly.nc")
    combined.to_netcdf(out_file)

    print(f"  [DONE] Saved monthly {unified_name} → {out_file}")
    print(f"         Time range: {str(combined.time.values[0])} → {str(combined.time.values[-1])}")


# =======================================================================
# 7. MAIN DRIVER
# =======================================================================
def main():
    start = time.time()

    print("\n========================================================")
    print("        UNIFIED DAILY → MONTHLY PREPROCESSING")
    print("========================================================")
    print(f"INPUT_ROOT:  {INPUT_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")

    dataset_type = detect_dataset_type(INPUT_ROOT)

    if dataset_type == "WFDE5":
        var_files = discover_files_wfde5(INPUT_ROOT)
    else:
        var_files = discover_files_isimip(INPUT_ROOT)

    if not var_files:
        print("No variables/files found to process. Check INPUT_ROOT.")
        return

    print("\nVariables discovered:")
    for k, v in var_files.items():
        print(f"  {k}: {len(v)} files")

    # Process each variable
    for source_var_name, file_list in var_files.items():
        process_variable(source_var_name, file_list, dataset_type)

    print("\n========================================================")
    print(f"ALL DONE. Total runtime: {(time.time() - start)/60:.2f} minutes")
    print("Monthly data are ready for PET/SPI/SPEI/MCWD calculation.")
    print("========================================================\n")


# =======================================================================
# 8. ENTRY POINT
# =======================================================================
if __name__ == "__main__":
    main()

