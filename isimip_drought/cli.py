"""
CLI for isimip-drought incidies.

Usage:
    isimip-drought spi --precip /path/to/pr/*.nc --scales 3 6 12 --out spi.nc
    isimip-drought spei --precip /path/to/pr/*.nc --tasmin ... --scales 3 6 12 --out spei.nc
    isimip-drought mcwd --precip /path/to/pr/*.nc --et-fixed 100 --out mcwd.nc
    isimip-drought pet --method hargreaves --tasmin ... --tasmax ... --out pet.nc
"""

import click

from .spi import spi_from_files
from .spei import spei_from_files
from .mcwd import mcwd_from_files
from .pet import (
    calc_pet_hargreaves,
    calc_pet_penman_monteith,
    calc_pet_thornthwaite,
)
from .utils import load_data, save_netcdf, parse_calibration_period


@click.group()
@click.version_option(version="0.1.0")
def main():
    """ISIMIP Drought Indices Calculator.

    Compute SPI, SPEI, and MCWD drought indices from ISIMIP data.
    """
    pass


@main.command()
@click.option(
    "--precip", "-p",
    required=True,
    help="Glob pattern for precipitation files (e.g., '/path/to/pr/*.nc')"
)
@click.option(
    "--scales", "-s",
    required=True,
    multiple=True,
    type=int,
    help="Accumulation periods in months (can specify multiple: -s 3 -s 6 -s 12)"
)
@click.option(
    "--out", "-o",
    required=True,
    help="Output NetCDF file path"
)
@click.option(
    "--calibration", "-c",
    default="1991-2020",
    help="Calibration period as YYYY-YYYY (default: 1991-2020)"
)
@click.option(
    "--chunks",
    default=None,
    help="Dask chunks as 'lat:50,lon:50' (default: time:-1,lat:50,lon:50)"
)
def spi(precip, scales, out, calibration, chunks):
    """Compute Standardized Precipitation Index (SPI).

    Example:
        isimip-drought spi -p '/data/pr/*.nc' -s 3 -s 6 -s 12 -o spi.nc
    """
    cal_period = parse_calibration_period(calibration)
    chunk_dict = _parse_chunks(chunks)

    spi_from_files(
        precip_pattern=precip,
        scales=list(scales),
        output_path=out,
        calibration_period=cal_period,
        chunks=chunk_dict,
    )


@main.command()
@click.option(
    "--precip", "-p",
    required=True,
    help="Glob pattern for precipitation files (e.g., '/path/to/pr/*.nc')"
)
@click.option(
    "--scales", "-s",
    required=True,
    multiple=True,
    type=int,
    help="Accumulation periods in months"
)
@click.option(
    "--out", "-o",
    required=True,
    help="Output NetCDF file path"
)
@click.option(
    "--calibration", "-c",
    default="1991-2020",
    help="Calibration period as YYYY-YYYY"
)
@click.option(
    "--pet",
    default=None,
    help="Glob pattern for pre-computed PET files"
)
@click.option(
    "--pet-method",
    type=click.Choice(["thornthwaite", "hargreaves", "penman"]),
    default="hargreaves",
    help="PET calculation method (default: hargreaves)"
)
@click.option("--tas", default=None, help="Glob pattern for mean temperature files e.g., '/path/to/tas/*.nc'")
@click.option("--tasmin", default=None, help="Glob pattern for min temperature files e.g., '/path/to/tasmin/*.nc'")
@click.option("--tasmax", default=None, help="Glob pattern for max temperature files e.g., '/path/to/tasmax/*.nc'")
@click.option("--hurs", default=None, help="Glob pattern for relative humidity files e.g., '/path/to/hurs/*.nc'")
@click.option("--rsds", default=None, help="Glob pattern for shortwave radiation files e.g., '/path/to/rsds/*.nc'")
@click.option("--sfcwind", default=None, help="Glob pattern for wind speed files e.g., '/path/to/sfcwind/*.nc'")
@click.option("--ps", default=None, help="Glob pattern for surface pressure files e.g., '/path/to/ps/*.nc'")
@click.option("--chunks", default=None, help="Dask chunks as 'lat:50,lon:50'")
def spei(precip, scales, out, calibration, pet, pet_method,
         tas, tasmin, tasmax, hurs, rsds, sfcwind, ps, chunks):
    """Compute Standardized Precipitation Evapotranspiration Index (SPEI).

    PET can be provided directly with --pet, or calculated using one of:

    \b
    Hargreaves (default): --tasmin --tasmax
    Thornthwaite: --tas
    Penman-Monteith: --tas --tasmin --tasmax --hurs --rsds --sfcwind [--ps]

    Example:
        isimip-drought spei -p '/data/pr/*.nc' --tasmin '/data/tasmin/*.nc' \\
            --tasmax '/data/tasmax/*.nc' -s 3 -s 6 -o spei.nc
    """
    cal_period = parse_calibration_period(calibration)
    chunk_dict = _parse_chunks(chunks)

    spei_from_files(
        precip_pattern=precip,
        scales=list(scales),
        output_path=out,
        calibration_period=cal_period,
        pet_pattern=pet,
        pet_method=pet_method,
        tas_pattern=tas,
        tasmin_pattern=tasmin,
        tasmax_pattern=tasmax,
        hurs_pattern=hurs,
        rsds_pattern=rsds,
        sfcwind_pattern=sfcwind,
        ps_pattern=ps,
        chunks=chunk_dict,
    )


@main.command()
@click.option(
    "--precip", "-p",
    required=True,
    help="Glob pattern for precipitation files (e.g., '/path/to/pr/*.nc')"
)
@click.option(
    "--scales", "-s",
    required=True,
    multiple=True,
    type=int,
    help="Rolling window periods in months (e.g., -s 6 -s 12)"
)
@click.option(
    "--out", "-o",
    required=True,
    help="Output NetCDF file path"
)
@click.option(
    "--et-fixed",
    type=float,
    default=None,
    help="Fixed monthly ET in mm/month (e.g., 100 for tropical forests)"
)
@click.option(
    "--pet",
    default=None,
    help="Glob pattern for PET files (alternative to --et-fixed)"
)
@click.option(
    "--reset-month",
    type=int,
    default=10,
    help="Month to reset CWD (1-12, default: 10 for October)"
)
@click.option("--chunks", default=None, help="Dask chunks as 'lat:50,lon:50'")
def mcwd(precip, scales, out, et_fixed, pet, reset_month, chunks):
    """Compute Maximum Climatological Water Deficit (MCWD).

    Provide ET as either:
    - --et-fixed: Constant monthly ET (e.g., 100 mm/month for Amazon)
    - --pet: Pre-computed PET files

    Example:
        isimip-drought mcwd -p '/data/pr/*.nc' --et-fixed 100 -s 6 -s 12 -o mcwd.nc
    """
    if et_fixed is None and pet is None:
        raise click.UsageError("Must provide either --et-fixed or --pet")

    chunk_dict = _parse_chunks(chunks)

    mcwd_from_files(
        precip_pattern=precip,
        scales=list(scales),
        output_path=out,
        et_fixed=et_fixed,
        pet_pattern=pet,
        reset_month=reset_month,
        chunks=chunk_dict,
    )


@main.command()
@click.option(
    "--method", "-m",
    type=click.Choice(["thornthwaite", "hargreaves", "penman"]),
    required=True,
    help="PET calculation method"
)
@click.option(
    "--out", "-o",
    required=True,
    help="Output NetCDF file path"
)
@click.option("--tas", default=None, help="Glob pattern for mean temperature files e.g., '/path/to/tas/*.nc'")
@click.option("--tasmin", default=None, help="Glob pattern for min temperature files e.g., '/path/to/tasmin/*.nc'")
@click.option("--tasmax", default=None, help="Glob pattern for max temperature files e.g., '/path/to/tasmax/*.nc'")
@click.option("--hurs", default=None, help="Glob pattern for relative humidity files e.g., '/path/to/hurs/*.nc'")
@click.option("--rsds", default=None, help="Glob pattern for shortwave radiation files e.g., '/path/to/rsds/*.nc'")
@click.option("--sfcwind", default=None, help="Glob pattern for wind speed files e.g., '/path/to/sfcwind/*.nc'")
@click.option("--ps", default=None, help="Glob pattern for surface pressure files e.g., '/path/to/ps/*.nc'")
@click.option("--chunks", default=None, help="Dask chunks as 'lat:50,lon:50'")
def pet(method, out, tas, tasmin, tasmax, hurs, rsds, sfcwind, ps, chunks):
    """Compute Potential Evapotranspiration (PET).

    \b
    Methods and required inputs:
    - thornthwaite: --tas
    - hargreaves: --tasmin --tasmax
    - penman: --tas --tasmin --tasmax --hurs --rsds --sfcwind [--ps]

    Example:
        isimip-drought pet -m hargreaves --tasmin '/data/tasmin/*.nc' \\
            --tasmax '/data/tasmax/*.nc' -o pet.nc
    """
    import xarray as xr

    chunk_dict = _parse_chunks(chunks)

    if method == "thornthwaite":
        if not tas:
            raise click.UsageError("Thornthwaite requires --tas")
        print(f"Loading temperature from: {tas}")
        tas_data = load_data(tas, chunks=chunk_dict)
        # Resample to monthly
        tas_monthly = tas_data.resample(time="MS").mean()
        print("Computing PET (Thornthwaite)...")
        pet_data = calc_pet_thornthwaite(tas_monthly)

    elif method == "hargreaves":
        if not tasmin or not tasmax:
            raise click.UsageError("Hargreaves requires --tasmin and --tasmax")
        print(f"Loading tasmin from: {tasmin}")
        tasmin_data = load_data(tasmin, chunks=chunk_dict)
        print(f"Loading tasmax from: {tasmax}")
        tasmax_data = load_data(tasmax, chunks=chunk_dict)
        print("Computing PET (Hargreaves)...")
        pet_data = calc_pet_hargreaves(tasmin_data, tasmax_data)

    elif method == "penman":
        required = [tas, tasmin, tasmax, hurs, rsds, sfcwind]
        if not all(required):
            raise click.UsageError(
                "Penman-Monteith requires --tas, --tasmin, --tasmax, "
                "--hurs, --rsds, --sfcwind"
            )
        print("Loading variables for Penman-Monteith...")
        tas_data = load_data(tas, chunks=chunk_dict)
        tasmin_data = load_data(tasmin, chunks=chunk_dict)
        tasmax_data = load_data(tasmax, chunks=chunk_dict)
        hurs_data = load_data(hurs, chunks=chunk_dict)
        rsds_data = load_data(rsds, chunks=chunk_dict)
        sfcwind_data = load_data(sfcwind, chunks=chunk_dict)
        ps_data = load_data(ps, chunks=chunk_dict) if ps else None

        print("Computing PET (Penman-Monteith)...")
        pet_data = calc_pet_penman_monteith(
            tas_data, tasmin_data, tasmax_data,
            rsds_data, hurs_data, sfcwind_data, ps_data
        )

    # Create dataset
    ds = xr.Dataset({"pet": pet_data})
    ds.attrs = {
        "title": f"Potential Evapotranspiration ({method})",
        "institution": "ISIMIP Drought Indices Package",
        "source": "isimip-drought",
    }

    print(f"Saving to: {out}")
    save_netcdf(ds, out)
    print("Done!")


def _parse_chunks(chunks_str):
    """Parse chunk string like 'lat:50,lon:50' to dict, or None for auto."""
    if not chunks_str or chunks_str.lower() == 'auto':
        # xarray auto detect
        return None

    chunk_dict = {}
    for part in chunks_str.split(","):
        key, val = part.split(":")
        val = val.strip()
        chunk_dict[key.strip()] = -1 if val == '-1' else int(val)

    return chunk_dict


if __name__ == "__main__":
    main()