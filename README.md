# ISIMIP Drought Indices

A Python script to calculate drought indices from ISIMIP climate data.

This scripot calculates three widely-used drought indices:

| Index | Full Name | What It Measures |
|-------|-----------|------------------|
| **SPI** | Standardized Precipitation Index | Precipitation anomalies |
| **SPEI** | Standardized Precipitation Evapotranspiration Index | Water balance (precipitation minus evaporation) |
| **MCWD** | Maximum Climatological Water Deficit | Cumulative water stress (used for forest drought) |

**SPI and SPEI** are normalized indices where:
- **0** = normal conditions
- **Negative values** = drier than normal (drought)
- **Positive values** = wetter than normal

**MCWD** is in millimeters (mm), where more negative values indicate worse water deficit.

## Installation

```bash
python -m venv venv
source venv/bin/activate 

pip install git+https://github.com/mo-dkrz/isimip-drought
```

## Quick Start

### Command Line Interface

#### 1. Calculate SPI (precipitation only)

```bash
isimip-drought spi \
    --precip '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/pr*.nc' \
    -s 3 -s 6 -s 12 \
    --calibration 1991-2020 \
    --out spi_output.nc
```

**Options:**
- `--precip`: Path to precipitation files (supports wildcards like `*.nc`)
- `-s` or `--scale`: Accumulation period in months (can specify multiple)
- `--calibration`: Reference period for fitting the distribution (format: YYYY-YYYY)
- `--out`: Output filename

#### 2. Calculate SPEI (precipitation + temperature)

```bash
isimip-drought spei \
    --precip '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/pr*.nc' \
    --tasmin '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/tasmin*.nc' \
    --tasmax '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/tasmax*.nc' \
    --pet-method hargreaves \
    -s 3 -s 6 \
    --calibration 1991-2020 \
    --out spei_output.nc
```

**PET methods available:**
- `hargreaves`: Requires only Tmin and Tmax (recommended for most cases)
- `thornthwaite`: Requires only mean temperature (less accurate)
- `penman-monteith`: Most accurate, but requires humidity, radiation, wind, and pressure

#### 3. Calculate MCWD (water deficit tracking)

```bash
isimip-drought mcwd \
    --precip '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/pr*.nc' \
    --et-fixed 100 \
    -s 12 \
    --reset-month 10 \
    --out mcwd_output.nc
```

**Options:**
- `--et-fixed`: Fixed monthly evapotranspiration in mm (100 mm/month is typical for tropical forests)
- `--reset-month`: Month to reset the cumulative deficit (10 = October, start of wet season in Amazon)

#### 4. Calculate PET separately

```bash
isimip-drought pet \
    --method hargreaves \
    --tasmin '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/tasmin*.nc' \
    --tasmax '/data/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/tasmax*.nc' \
    --out pet_output.nc
```

### Python API

```python
import xarray as xr
from isimip_drought import compute_spi, compute_spei, compute_mcwd
from isimip_drought.utils import convert_precip_units
from isimip_drought.pet import calc_pet_hargreaves

# Load your data
pr = xr.open_mfdataset('pr*.nc')['pr']
tasmin = xr.open_mfdataset('tasmin*.nc')['tasmin']
tasmax = xr.open_mfdataset('tasmax*.nc')['tasmax']

# Convert precipitation: ISIMIP uses kg/m²/s, we need mm/day then sum to monthly
pr_mm = convert_precip_units(pr) 
pr_monthly = pr_mm.resample(time='MS').sum()

# Calculate SPI
spi_3month = compute_spi(pr_monthly, scale=3, calibration_period=(1991, 2020))

# Calculate PET (daily) and aggregate to monthly
pet_daily = calc_pet_hargreaves(tasmin, tasmax)
pet_monthly = pet_daily.resample(time='MS').sum()

# Calculate SPEI
spei_3month = compute_spei(pr_monthly, pet_monthly, scale=3, calibration_period=(1991, 2020))

# Calculate MCWD
mcwd = compute_mcwd(pr_monthly, pet_monthly, scale=12, reset_month=10)

# Save results
spi_3month.to_netcdf('spi_output.nc')
```

## Running on HPC

### Interactive Mode (Login Node)

> [!Note] 
> The login node is the computer you connect to when you SSH into an HPC cluster. 
> It's shared by all users and meant only for small tasks like editing files or submitting jobs—not for heavy computation.

For quick tests with small data subsets, you can run directly on the login node:

```bash

module load anaconda

python -m venv venv
source venv/bin/activate

# quick test
isimip-drought spi --precip 'small_couns.nc' -s 3 --out test.nc
```

### Batch Mode (Compute Nodes)

> [!Note] 
> For large datasets, submit a job to the compute nodes using SLURM. 
> This ensures your job gets dedicated resources and doesn't slow down the login node for other users.

Create a job script (e.g., `run_drought.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=drought_indices
#SBATCH --partition=compute
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=drought_%j.log

# Load modules
module load anaconda

# Activate environment
source /path/to/your/venv/bin/activate

# Run the calculation
isimip-drought spei \
    --precip '/data/ISIMIP3b/.../pr*.nc' \
    --tasmin '/data/ISIMIP3b/.../tasmin*.nc' \
    --tasmax '/data/ISIMIP3b/.../tasmax*.nc' \
    --pet-method hargreaves \
    -s 3 -s 6 -s 12 \
    --calibration 1991-2020 \
    --out /work/username/spei_output.nc
```

Submit with:
```bash
chmod +x run_drought.sh
sbatch run_drought.sh
```

Check with:

```bash
squeue -u $USER
tail -f drought_*.log
```
