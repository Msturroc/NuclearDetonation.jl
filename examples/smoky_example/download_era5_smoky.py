"""
ERA5 Download Script for SNAP — Shot Smoky (Plumbbob, 44 kT)

This downloads ERA5 data in TWO separate calls:
1. Model-level 3D variables (t, u, v, omega) on all 137 hybrid levels
2. Surface 2D variables (sp, 10u, 10v, tp)

Files are downloaded separately and will be merged using Julia.

Shot Smoky
  Operation Plumbbob, 31 August 1957
  Yield: 44 kT (tower shot, ~213 m HOB)
  Location: NTS Area 2, approx 37.177°N 116.046°W
  Detonation: ~12:00 UTC (05:00 PDT)
"""

import cdsapi
import os

# --- CONFIGURATION ---
DATES_TO_PROCESS = ['1957-08-31', '1957-09-01', '1957-09-02']
AREA = '41/-118/35/-109'  # [North, West, South, East]
GRID = '0.28125/0.28125'  # ERA5 native resolution

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ERA5_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All 137 model levels
MODEL_LEVELS = "/".join([str(i) for i in range(1, 138)])

c = cdsapi.Client()

# --- DOWNLOAD FUNCTION ---
def download_era5_for_snap(date_str, hour_start, hour_end):
    """
    Download ERA5 data for SNAP in proper format.

    Makes TWO API calls:
    1. Model-level 3D fields (t, u, v, omega)
    2. Surface 2D fields (sp, 10u, 10v, tp)
    """
    year, month, day = date_str.split('-')
    time_label = f"{hour_start:02d}-{hour_end:02d}"

    # Time steps to download
    time_period = [f'{h:02d}:00' for h in range(hour_start, hour_end + 1)]

    # Output files
    model_level_file = os.path.join(OUTPUT_DIR, f'era5_{year}{month}{day}_{time_label}_ml.nc')
    surface_file = os.path.join(OUTPUT_DIR, f'era5_{year}{month}{day}_{time_label}_sfc.nc')

    # Check if both files already exist
    if os.path.exists(model_level_file) and os.path.exists(surface_file):
        print(f"\n  {date_str} {time_label}: Both files already exist, skipping")
        return (model_level_file, surface_file)

    print(f"\n{'='*60}")
    print(f"Processing: {date_str} hours {time_label}")
    print(f"{'='*60}")

    # === DOWNLOAD 1: Model-level 3D variables ===
    if not os.path.exists(model_level_file):
        print(f"\n[1/2] Downloading model-level 3D fields (t, u, v, omega)...")
        try:
            c.retrieve(
                'reanalysis-era5-complete',
                {
                    'class': 'ea',
                    'date': date_str,
                    'expver': '1',
                    'levtype': 'ml',
                    'levelist': MODEL_LEVELS,
                    'param': '130/131/132/135',  # t, u, v, omega
                    'step': '0',
                    'stream': 'oper',
                    'type': 'an',
                    'area': AREA,
                    'grid': GRID,
                    'time': '/'.join(time_period),
                    'format': 'netcdf',
                },
                model_level_file
            )
            print(f"  Downloaded model-level data")
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
                    'time': '/'.join(time_period),
                    'format': 'netcdf',
                },
                surface_file
            )
            print(f"  Downloaded surface data")
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    else:
        print(f"  Surface file already exists: {surface_file}")

    # === SUCCESS ===
    print(f"\n  Model-level file: {model_level_file}")
    print(f"  Surface file: {surface_file}")

    return (model_level_file, surface_file)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ERA5 Download for SNAP — Shot Smoky (Plumbbob, 44 kT)")
    print("31 Aug – 2 Sep 1957, domain 35-41N 118-109W")
    print("="*60)

    success_count = 0
    total_count = 0

    # Download all 8 × 3-hour chunks per day for 4 days = 32 chunks
    for date_str in DATES_TO_PROCESS:
        for hour_start in range(0, 24, 3):
            hour_end = hour_start + 2
            total_count += 1
            result = download_era5_for_snap(date_str, hour_start, hour_end)
            if result:
                success_count += 1

    print("\n" + "="*60)
    print(f"Download complete: {success_count}/{total_count} time periods successful")
    print("="*60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("Model-level files: era5_YYYYMMDD_HH-HH_ml.nc")
    print("Surface files: era5_YYYYMMDD_HH-HH_sfc.nc")
    print("\n-> Next: Run merge_era5_smoky.jl to combine into SNAP-ready files\n")
