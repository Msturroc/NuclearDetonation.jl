"""
Download ERA5 model-level data for ETEX-1 via NCAR RDA OPeNDAP.

Server-side spatial subsetting — only downloads grid points we need.
~2 GB total instead of ~60 GB.
"""

import xarray as xr
import numpy as np
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

DAYS = list(range(23, 28))  # Oct 23-27
LAT_MAX, LAT_MIN = 63.0, 40.0  # slice order: descending
LON_WEST, LON_EAST = 352.0, 28.0  # -8E = 352 in 0-360 coords

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ERA5_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ML_BASE = ("https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633006/"
           "e5.oper.an.ml/199410/")
SFC_BASE = ("https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
            "e5.oper.an.sfc/199410/")

VARS = {
    "t": ("0_5_0_0_0_t", "regn320sc", "T"),
    "u": ("0_5_0_2_2_u", "regn320uv", "U"),
    "v": ("0_5_0_2_3_v", "regn320uv", "V"),
}


def subset_domain(da):
    """Select European domain, handling prime meridian crossing."""
    # West of prime meridian: 352-360, east: 0-28
    p1 = da.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_WEST, 360))
    p2 = da.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(0, LON_EAST))
    return xr.concat([p1, p2], dim="longitude")


def download_sp():
    """Download surface pressure."""
    out = os.path.join(OUTPUT_DIR, "era5_etex_sp.nc")
    if os.path.exists(out):
        print("  SP: exists, skipping")
        return
    url = SFC_BASE + "e5.oper.an.sfc.128_134_sp.ll025sc.1994100100_1994103123.nc"
    t0 = time.time()
    ds = xr.open_dataset(url)
    var = [v for v in ds.data_vars if v.upper() == "SP"][0]
    sp = subset_domain(ds[var].sel(time=slice("1994-10-23", "1994-10-27T23:00")))
    sp.load().to_dataset(name="sp").to_netcdf(out)
    ds.close()
    print(f"  SP: {os.path.getsize(out)/1e6:.1f} MB in {time.time()-t0:.0f}s")


def download_ml_chunk(day, h0, var_key):
    """Download one 6-hour model-level chunk."""
    h1 = h0 + 5
    out = os.path.join(OUTPUT_DIR, f"era5_ml_{var_key}_{day:02d}_{h0:02d}.nc")
    if os.path.exists(out):
        return True

    code, grid, var_name = VARS[var_key]
    fname = f"e5.oper.an.ml.{code}.{grid}.199410{day:02d}{h0:02d}_199410{day:02d}{h1:02d}.nc"
    url = ML_BASE + fname

    try:
        ds = xr.open_dataset(url)
        sub = subset_domain(ds[var_name]).load()

        # Also grab hybrid coefficients from the first file
        out_ds = sub.to_dataset(name=var_key)
        if "a_model" in ds and "b_model" in ds:
            out_ds["ap"] = ds["a_model"].load()
            out_ds["b"] = ds["b_model"].load()

        out_ds.to_netcdf(out)
        ds.close()
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        if os.path.exists(out):
            os.remove(out)
        return False


if __name__ == "__main__":
    print("="*60)
    print("NCAR RDA OPeNDAP — ETEX ERA5 Model Levels")
    print(f"Domain: {LAT_MIN}-{LAT_MAX}N, {LON_WEST}(={LON_WEST-360:.0f})-{LON_EAST}E")
    print("Server-side subsetting, ~2 GB total")
    print("="*60)

    print("\n--- Surface Pressure ---")
    download_sp()

    print("\n--- Model Levels (t, u, v x 137 levels) ---")
    ok, total = 0, 0
    for day in DAYS:
        t_day = time.time()
        for h0 in range(0, 24, 6):
            for var_key in ["t", "u", "v"]:
                total += 1
                t0 = time.time()
                tag = f"Oct {day} {h0:02d}Z {var_key}"
                if download_ml_chunk(day, h0, var_key):
                    f = os.path.join(OUTPUT_DIR, f"era5_ml_{var_key}_{day:02d}_{h0:02d}.nc")
                    mb = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
                    print(f"  {tag}: {mb:.1f} MB ({time.time()-t0:.0f}s)")
                    ok += 1
                else:
                    print(f"  {tag}: FAILED")
        print(f"  Day {day} done in {time.time()-t_day:.0f}s")

    print(f"\n{'='*60}")
    print(f"Complete: {ok}/{total} files")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)
