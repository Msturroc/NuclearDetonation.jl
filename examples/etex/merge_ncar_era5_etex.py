"""
Merge NCAR RDA ERA5 files into SNAP-ready format for NuclearDetonation.jl.

Input: per-variable 6-hour chunks (era5_ml_{t,u,v}_{day}_{hour}.nc) + SP file
Output: 3-hour combined files (era5_etex_YYYYMMDD_HH-HH_snap.nc) with:
  - Dims: longitude, latitude, hybrid, time
  - Vars: x_wind_ml, y_wind_ml, air_temperature_ml, surface_air_pressure, ap, b
"""

import xarray as xr
import numpy as np
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ERA5_data')
DAYS = list(range(23, 28))


def find_ml_file(var, day, hour):
    """Find the 6-hour file containing the given hour."""
    h0 = (hour // 6) * 6  # 0, 6, 12, or 18
    return os.path.join(DATA_DIR, f"era5_ml_{var}_{day:02d}_{h0:02d}.nc")


def merge_chunk(day, h_start, h_end, sp_ds):
    """Merge t, u, v + sp for a 3-hour window into SNAP format."""
    tag = f"199410{day:02d}_{h_start:02d}-{h_end:02d}"
    out_file = os.path.join(DATA_DIR, f"era5_etex_{tag}_snap.nc")

    if os.path.exists(out_file):
        print(f"  {tag}: exists, skipping")
        return True

    hours = list(range(h_start, h_end + 1))
    time_sel = [np.datetime64(f"1994-10-{day:02d}T{h:02d}:00") for h in hours]

    try:
        # Load t, u, v
        t_file = find_ml_file("t", day, h_start)
        u_file = find_ml_file("u", day, h_start)
        v_file = find_ml_file("v", day, h_start)

        ds_t = xr.open_dataset(t_file).sel(time=time_sel)
        ds_u = xr.open_dataset(u_file).sel(time=time_sel)
        ds_v = xr.open_dataset(v_file).sel(time=time_sel)

        # Get hybrid coefficients (ap in Pa, b dimensionless)
        ap = ds_t["ap"].values.astype(np.float32)
        b = ds_t["b"].values.astype(np.float32)

        lat = ds_t.latitude.values
        lon = ds_t.longitude.values

        # Interpolate SP onto ML grid
        sp_chunk = sp_ds["sp"].sel(time=time_sel)
        sp_interp = sp_chunk.interp(
            latitude=lat, longitude=lon, method="linear"
        ).values.astype(np.float32)

        # Build output dataset — match Smoky dim order: (time, hybrid, latitude, longitude)
        # Input dims from NCAR: (time, level, latitude, longitude)
        n_time = len(hours)
        t_vals = np.nan_to_num(ds_t["t"].values, nan=0.0).astype(np.float32)  # (time, level, lat, lon)
        u_vals = np.nan_to_num(ds_u["u"].values, nan=0.0).astype(np.float32)
        v_vals = np.nan_to_num(ds_v["v"].values, nan=0.0).astype(np.float32)
        sp_vals = np.nan_to_num(sp_interp, nan=101325.0).astype(np.float32)   # (time, lat, lon)

        out = xr.Dataset(
            {
                "air_temperature_ml": (["time", "hybrid", "latitude", "longitude"], t_vals),
                "x_wind_ml": (["time", "hybrid", "latitude", "longitude"], u_vals),
                "y_wind_ml": (["time", "hybrid", "latitude", "longitude"], v_vals),
                "surface_air_pressure": (["time", "latitude", "longitude"], sp_vals),
                "ap": (["hybrid"], ap),
                "b": (["hybrid"], b),
            },
            coords={
                "longitude": np.where(lon > 180, lon - 360, lon).astype(np.float64),
                "latitude": lat.astype(np.float64),
                "hybrid": np.arange(1, 138, dtype=np.float32),
                "time": ds_t.time.values,
            },
        )

        out["ap"].attrs = {"long_name": "hybrid A coefficient at layer midpoints", "units": "Pa"}
        out["b"].attrs = {"long_name": "hybrid B coefficient at layer midpoints", "units": "1"}
        out.attrs["title"] = f"ERA5 data for ETEX Release 1 ({tag})"
        out.attrs["source"] = "ERA5 reanalysis via NCAR RDA"
        out.attrs["vertical_coordinate"] = "hybrid sigma-pressure (ECMWF L137)"

        # Disable _FillValue so Julia loads as plain Float32 (not Union{Missing,Float32})
        encoding = {v: {"_FillValue": None} for v in out.data_vars}
        out.to_netcdf(out_file, encoding=encoding)

        ds_t.close()
        ds_u.close()
        ds_v.close()

        size_mb = os.path.getsize(out_file) / 1e6
        return size_mb

    except Exception as e:
        print(f"  {tag}: ERROR - {e}")
        import traceback; traceback.print_exc()
        if os.path.exists(out_file):
            os.remove(out_file)
        return False


if __name__ == "__main__":
    print("="*60)
    print("Merging NCAR ERA5 files into SNAP format")
    print("="*60)

    # Load SP once
    print("Loading surface pressure ...")
    sp_ds = xr.open_dataset(os.path.join(DATA_DIR, "era5_etex_sp.nc"))
    print(f"  SP loaded: {dict(sp_ds.sizes)}")

    ok = 0
    total = 0
    for day in DAYS:
        for h_start in range(0, 24, 3):
            h_end = h_start + 2
            total += 1
            t0 = time.time()
            result = merge_chunk(day, h_start, h_end, sp_ds)
            if result:
                if isinstance(result, float):
                    print(f"  199410{day:02d}_{h_start:02d}-{h_end:02d}: {result:.1f} MB ({time.time()-t0:.0f}s)")
                ok += 1

    sp_ds.close()

    print(f"\n{'='*60}")
    print(f"Complete: {ok}/{total} snap files")
    snap_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_snap.nc")])
    print(f"Files: {len(snap_files)}")
    if snap_files:
        print(f"  First: {snap_files[0]}")
        print(f"  Last:  {snap_files[-1]}")
    print("="*60)
