import xarray as xr
from pathlib import Path

indir = Path("SPEI_Check_Single_Cell")

pr_ds  = xr.open_dataset(indir/"03_pr_mm_month.nc")
pet_ds = xr.open_dataset(indir/"05_pet_mm_month.nc")

# grab the only variable in each file (robust to variable name differences)
pr_var  = list(pr_ds.data_vars)[0]
pet_var = list(pet_ds.data_vars)[0]

pr  = pr_ds[pr_var]
pet = pet_ds[pet_var]

# align monthly series
pr, pet = xr.align(pr, pet, join="inner")

# save aligned monthly P and PET (and optionally water balance)
xr.Dataset({"pr_mm_month": pr, "pet_mm_month": pet}).to_netcdf(indir/"07_pr_pet_aligned_mm_month.nc")
xr.Dataset({"wb_mm_month": (pr - pet)}).to_netcdf(indir/"07_wb_mm_month.nc")

print("[SAVE]", indir/"07_pr_pet_aligned_mm_month.nc")
print("[SAVE]", indir/"07_wb_mm_month.nc")
print("pr time:", pr.time.values[0], "->", pr.time.values[-1], "n=", pr.sizes["time"])
print("pet time:", pet.time.values[0], "->", pet.time.values[-1], "n=", pet.sizes["time"])





