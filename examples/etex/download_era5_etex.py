"""
ERA5 Download Script — ETEX Release 1 (European Tracer Experiment)

This downloads ERA5 data in TWO separate calls per day:
1. Model-level 3D variables (t, u, v) on all 137 hybrid levels
2. Surface 2D variables (sp, 10u, 10v, tp)

Files are downloaded separately and will be merged using Julia.

ETEX Release 1
  23 October 1994, 16:00 UTC — 24 October 1994, 03:50 UTC
  Tracer: 340 kg PMCH from Monterfil, Brittany, France (48.058N, 2.008W)
  Release height: ~8 m AGL
  168 sampling stations across 17 European countries
"""

import cdsapi
import os

# --- CONFIGURATION ---
# 5 days: release day + 4 days of transport across Europe
DATES_TO_PROCESS = ['1994-10-23', '1994-10-24', '1994-10-25', '1994-10-26', '1994-10-27']

# Tighter domain: stations span 42.6-61.0N, -4.4 to 26.1E
# Add buffer for inflow meteorology
AREA = '63/-8/40/28'  # [North, West, South, East]
GRID = '0.5/0.5'  # Coarser grid to keep file sizes manageable

# Output directory (symlinked to external drive)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ERA5_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All 137 model levels (required by transport code)
MODEL_LEVELS = "/".join([str(i) for i in range(1, 138)])

# All 24 hours
ALL_HOURS = "/".join([f'{h:02d}:00' for h in range(24)])

c = cdsapi.Client()

# --- DOWNLOAD FUNCTION ---
def download_era5_day(date_str):
    """
    Download ERA5 data for a full day (24 hourly timesteps).

    Makes TWO API calls:
    1. Model-level 3D fields (t, u, v)
    2. Surface 2D fields (sp, u10, v10, tp)
    """
    year, month, day = date_str.split('-')

    model_level_file = os.path.join(OUTPUT_DIR, f'era5_etex_{year}{month}{day}_ml.nc')
    surface_file = os.path.join(OUTPUT_DIR, f'era5_etex_{year}{month}{day}_sfc.nc')

    if os.path.exists(model_level_file) and os.path.exists(surface_file):
        print(f"\n  {date_str}: Both files already exist, skipping")
        return (model_level_file, surface_file)

    print(f"\n{'='*60}")
    print(f"Processing: {date_str}")
    print(f"{'='*60}")

    # === DOWNLOAD 1: Model-level 3D variables ===
    if not os.path.exists(model_level_file):
        print(f"\n[1/2] Downloading model-level 3D fields (t, u, v)...")
        print(f"  Domain: {AREA}, Grid: {GRID}, Levels: 1-137")
        try:
            c.retrieve(
                'reanalysis-era5-complete',
                {
                    'class': 'ea',
                    'date': date_str,
                    'expver': '1',
                    'levtype': 'ml',
                    'levelist': MODEL_LEVELS,
                    'param': '130/131/132',  # t, u, v (skip omega to save space)
                    'step': '0',
                    'stream': 'oper',
                    'type': 'an',
                    'area': AREA,
                    'grid': GRID,
                    'time': ALL_HOURS,
                    'format': 'netcdf',
                },
                model_level_file
            )
            size_gb = os.path.getsize(model_level_file) / 1e9
            print(f"  Downloaded model-level data ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    else:
        print(f"  Model-level file already exists: {model_level_file}")

    # === DOWNLOAD 2: Surface variables ===
    if not os.path.exists(surface_file):
        print(f"\n[2/2] Downloading surface fields (sp, u10, v10, tp)...")
        try:
            c.retrieve(
                'reanalysis-era5-complete',
                {
                    'class': 'ea',
                    'date': date_str,
                    'expver': '1',
                    'levtype': 'sfc',
                    'param': '134/165/166/228',  # sp, u10, v10, tp
                    'step': '0',
                    'stream': 'oper',
                    'type': 'an',
                    'area': AREA,
                    'grid': GRID,
                    'time': ALL_HOURS,
                    'format': 'netcdf',
                },
                surface_file
            )
            size_mb = os.path.getsize(surface_file) / 1e6
            print(f"  Downloaded surface data ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    else:
        print(f"  Surface file already exists: {surface_file}")

    return (model_level_file, surface_file)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ERA5 Download — ETEX Release 1")
    print("23-27 Oct 1994, domain 40-63N, 8W-28E")
    print("Grid: 0.5 x 0.5 deg, 137 model levels")
    print("="*60)

    success_count = 0

    for date_str in DATES_TO_PROCESS:
        result = download_era5_day(date_str)
        if result:
            success_count += 1

    print("\n" + "="*60)
    print(f"Download complete: {success_count}/{len(DATES_TO_PROCESS)} days successful")
    print("="*60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\n-> Next: Run merge_era5_etex.jl to combine into SNAP-ready files\n")
