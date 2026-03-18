"""
Download ERA5 model-level data for ETEX-1 from Google Cloud ARCO-ERA5.

Model levels (u, v, t): from ARCO-ERA5 on Google Cloud (fast, no queue)
Surface pressure (sp): from CDS API (single-level data is fast, no tape)

Output: NetCDF files compatible with merge_era5_etex.jl
"""

import xarray as xr
import numpy as np
import os
import sys
import time
import cdsapi

sys.stdout.reconfigure(line_buffering=True)

# --- CONFIGURATION ---
DATES = ['1994-10-23', '1994-10-24', '1994-10-25', '1994-10-26', '1994-10-27']

LAT_MIN, LAT_MAX = 40.0, 63.0
LON_MIN, LON_MAX = -8.0, 28.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ERA5_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ML_STORE = 'gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1'
STORAGE_OPTS = dict(token='anon')


def download_ml_day(date_str, ds_ml):
    """Download one day of model-level u, v, t from ARCO-ERA5."""

    ml_file = os.path.join(OUTPUT_DIR, f'era5_{date_str.replace("-","")}_00-23_ml.nc')

    if os.path.exists(ml_file):
        size_gb = os.path.getsize(ml_file) / 1e9
        print(f"  {date_str} ML: exists ({size_gb:.2f} GB), skipping")
        return True

    time_slice = slice(f'{date_str}T00:00', f'{date_str}T23:00')

    # Longitude selection (handle wrap)
    if LON_MIN < 0:
        lon_west = np.arange(360 + LON_MIN, 360, 0.25)
        lon_east = np.arange(0, LON_MAX + 0.25, 0.25)
        lon_vals = np.concatenate([lon_west, lon_east])
    else:
        lon_vals = np.arange(LON_MIN, LON_MAX + 0.25, 0.25)

    print(f"  {date_str} ML: selecting u, v, t ...")
    t0 = time.time()
    try:
        ml_sub = ds_ml[['u_component_of_wind', 'v_component_of_wind', 'temperature']].sel(
            time=time_slice,
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=lon_vals,
        )
        print(f"    Selection ready ({time.time()-t0:.0f}s), downloading ...")
        ml_data = ml_sub.compute()
        print(f"    Downloaded ({time.time()-t0:.0f}s), writing NetCDF ...")

        # Rename to CDS raw format
        ml_data = ml_data.rename({
            'u_component_of_wind': 'u',
            'v_component_of_wind': 'v',
            'temperature': 't',
            'hybrid': 'model_level',
            'time': 'valid_time',
        })

        # Convert lon 0-360 -> -180..180
        lons = ml_data.longitude.values.copy()
        lons[lons > 180] -= 360
        ml_data = ml_data.assign_coords(longitude=lons).sortby('longitude')

        ml_data.to_netcdf(ml_file)
        dt = time.time() - t0
        size_gb = os.path.getsize(ml_file) / 1e9
        print(f"    Done: {size_gb:.2f} GB in {dt:.0f}s")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()
        if os.path.exists(ml_file):
            os.remove(ml_file)
        return False


def download_sfc_all(dates):
    """Download surface pressure for all dates from CDS API (fast, on-disk data)."""

    sfc_file = os.path.join(OUTPUT_DIR, f'era5_etex_sfc.nc')

    if os.path.exists(sfc_file):
        print(f"  Surface file exists, skipping")
        return True

    print(f"  Downloading surface pressure from CDS API ...")
    t0 = time.time()

    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'surface_pressure',
                'year': '1994',
                'month': '10',
                'day': [d.split('-')[2] for d in dates],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
                'data_format': 'netcdf',
            },
            sfc_file
        )
        dt = time.time() - t0
        size_mb = os.path.getsize(sfc_file) / 1e6
        print(f"    Done: {size_mb:.1f} MB in {dt:.0f}s")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()
        if os.path.exists(sfc_file):
            os.remove(sfc_file)
        return False


if __name__ == "__main__":
    print("="*60)
    print("ERA5 Download — ETEX Release 1")
    print(f"Domain: {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E")
    print("Model levels: ARCO-ERA5 on Google Cloud (fast)")
    print("Surface: CDS API single-levels (fast, on-disk)")
    print("="*60)

    # --- 1. Model-level data from Google Cloud ---
    print("\nOpening ARCO-ERA5 model-level store ...")
    t0 = time.time()
    ds_ml = xr.open_zarr(ML_STORE, consolidated=True, storage_options=STORAGE_OPTS)
    print(f"  Opened in {time.time()-t0:.0f}s")

    ml_success = 0
    for date_str in DATES:
        print(f"\n--- {date_str} ---")
        if download_ml_day(date_str, ds_ml):
            ml_success += 1

    # --- 2. Surface pressure from CDS API ---
    print(f"\n{'='*40}")
    print("Surface pressure (CDS API)")
    print("="*40)
    sfc_ok = download_sfc_all(DATES)

    print(f"\n{'='*60}")
    print(f"Model levels: {ml_success}/{len(DATES)} days")
    print(f"Surface: {'OK' if sfc_ok else 'FAILED'}")
    print(f"Files in: {OUTPUT_DIR}")
    print("-> Next: Run merge_era5_etex.jl to create SNAP-ready files")
    print("="*60)
