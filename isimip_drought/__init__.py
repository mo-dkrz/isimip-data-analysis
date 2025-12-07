"""
ISIMIP Drought Indices.

Compute SPI, SPEI, and MCWD drought indices from ISIMIP data.

Example usage:
    from isimip_drought import compute_spi, compute_spei, compute_mcwd
    from isimip_drought.pet import calc_pet_hargreaves

    import xarray as xr
    precip = xr.open_dataset('precip.nc')['pr']
    spi = compute_spi(precip, scale=3, calibration_period=(1991, 2020))
"""

from .spi import compute_spi, compute_spi_multiscale, spi_from_files
from .spei import compute_spei, compute_spei_multiscale, spei_from_files
from .mcwd import (
    compute_mcwd,
    compute_mcwd_fixed_et,
    compute_mcwd_multiscale,
    compute_annual_mcwd,
    mcwd_from_files,
)
from .pet import (
    calc_pet_thornthwaite,
    calc_pet_hargreaves,
    calc_pet_penman_monteith,
    aggregate_pet_to_monthly,
)
from .utils import (
    load_data,
    convert_precip_units,
    convert_temp_units,
    save_netcdf,
)

__version__ = "0.1.0"

__all__ = [
    # SPI
    "compute_spi",
    "compute_spi_multiscale",
    "spi_from_files",
    # SPEI
    "compute_spei",
    "compute_spei_multiscale",
    "spei_from_files",
    # MCWD
    "compute_mcwd",
    "compute_mcwd_fixed_et",
    "compute_mcwd_multiscale",
    "compute_annual_mcwd",
    "mcwd_from_files",
    # PET
    "calc_pet_thornthwaite",
    "calc_pet_hargreaves",
    "calc_pet_penman_monteith",
    "aggregate_pet_to_monthly",
    # Utils
    "load_data",
    "convert_precip_units",
    "convert_temp_units",
    "save_netcdf",
]
