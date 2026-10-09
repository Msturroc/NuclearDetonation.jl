#!/usr/bin/env python3
"""
bundle_rivm_contours.py — RIVM intercomparison GIS bundler.

Reads the CF-1.8 dose NetCDFs written by cmaes_calibration.jl's NC_EXPORT block
(variable `dose_rate_mR_hr` on a WGS84 longitude x latitude grid) and produces,
for each file, iso-dose CONTOUR POLYGONS at the agreed RIVM dose levels:

    [1, 4, 10, 40, 100, 400, 1000] mR/hr

Each level's enclosed region is emitted as a (multi)polygon in EPSG:4326 and
written to BOTH an ESRI Shapefile and a GeoPackage (per input file), plus a
single combined GeoPackage `rivm_all_contours.gpkg` holding every test/stage.

If a NetCDF also carries a time-of-arrival field `toa_hours` (not currently
exported), TOA contours at [1,2,4,8,12,24,48] h are added too; absent that
variable the TOA step is silently skipped.

Usage:
    python bundle_rivm_contours.py <dir-or-glob> [--out <dir>]

    # default: scan ./rivm_intercomparison for *_dose_pre.nc / *_dose_post.nc
    python bundle_rivm_contours.py rivm_intercomparison

Requires (in .gisvenv): geopandas, shapely, matplotlib, netCDF4.
Run with the project venv, e.g.:
    /home/marc/NuclearDetonation.jl/.gisvenv/bin/python \
        examples/calibration_us_tests/bundle_rivm_contours.py rivm_intercomparison
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

DOSE_LEVELS = [1.0, 4.0, 10.0, 40.0, 100.0, 400.0, 1000.0]   # mR/hr
TOA_LEVELS = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]          # hours
CRS = "EPSG:4326"
FILL = -9999.0


def _read_field(ds, varname):
    """Return (lon, lat, Z[lat,lon]) for a (longitude, latitude) NetCDF var, or None."""
    if varname not in ds.variables:
        return None
    lon = np.asarray(ds.variables["longitude"][:], dtype=float)
    lat = np.asarray(ds.variables["latitude"][:], dtype=float)
    raw = np.asarray(ds.variables[varname][:], dtype=float)  # (longitude, latitude)
    # NC_EXPORT writes dims (longitude, latitude); contour wants Z[row=lat, col=lon].
    if raw.shape == (len(lon), len(lat)):
        Z = raw.T
    elif raw.shape == (len(lat), len(lon)):
        Z = raw
    else:
        raise ValueError(f"{varname}: shape {raw.shape} != lon/lat {(len(lon), len(lat))}")
    Z = np.where((Z == FILL) | ~np.isfinite(Z), np.nan, Z)
    return lon, lat, Z


def _contour_polygons(lon, lat, Z, level):
    """Iso-level enclosed region as a (multi)polygon, or None. Rings -> polygons,
    dissolved with unary_union. Dose plumes are simple so hole loss is negligible;
    buffer(0) repairs self-intersections."""
    if Z is None or not np.isfinite(Z).any() or np.nanmax(Z) < level:
        return None
    fig = plt.figure()
    try:
        cs = plt.contour(lon, lat, Z, levels=[level])
        segs = cs.allsegs[0] if cs.allsegs else []
        polys = []
        for seg in segs:
            if len(seg) >= 3:
                p = Polygon(seg)
                if not p.is_valid:
                    p = p.buffer(0)
                if (not p.is_empty) and p.area > 0:
                    polys.append(p)
    finally:
        plt.close(fig)
    if not polys:
        return None
    geom = unary_union(polys)
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])
    return geom


def process_file(path, out_dir):
    """Build per-level contour polygons for one NetCDF; return list of records."""
    base = os.path.splitext(os.path.basename(path))[0]   # e.g. trinity_dose_post
    parts = base.split("_dose_")
    test = parts[0]
    stage = parts[1] if len(parts) > 1 else ""           # pre | post | ""
    records = []
    with Dataset(path) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        dose = _read_field(ds, "dose_rate_mR_hr")
        if dose is None:
            print(f"  ! {base}: no dose_rate_mR_hr — skipped")
            return records
        lon, lat, Zdose = dose
        for lvl in DOSE_LEVELS:
            geom = _contour_polygons(lon, lat, Zdose, lvl)
            if geom is not None:
                records.append(dict(test=test, stage=stage, field="dose",
                                    level=lvl, unit="mR/hr", geometry=geom))
        toa = _read_field(ds, "toa_hours")
        if toa is not None:
            lon, lat, Ztoa = toa
            for lvl in TOA_LEVELS:
                geom = _contour_polygons(lon, lat, Ztoa, lvl)
                if geom is not None:
                    records.append(dict(test=test, stage=stage, field="toa",
                                        level=lvl, unit="hours", geometry=geom))
    if not records:
        print(f"  ! {base}: no contours at any level (max dose "
              f"{np.nanmax(Zdose):.3g} mR/hr) — skipped")
        return records
    gdf = gpd.GeoDataFrame(records, crs=CRS)
    shp = os.path.join(out_dir, base + ".shp")
    gpkg = os.path.join(out_dir, base + ".gpkg")
    gdf.to_file(shp, driver="ESRI Shapefile")
    gdf.to_file(gpkg, driver="GPKG")
    nd = sum(r["field"] == "dose" for r in records)
    print(f"  ok {base}: {nd} dose levels -> {os.path.basename(shp)} + .gpkg")
    return records


def main():
    ap = argparse.ArgumentParser(description="Bundle RIVM iso-dose contour polygons.")
    ap.add_argument("target", nargs="?", default="rivm_intercomparison",
                    help="directory of *_dose_*.nc, or a glob")
    ap.add_argument("--out", default=None, help="output dir (default: alongside inputs)")
    args = ap.parse_args()

    if os.path.isdir(args.target):
        files = sorted(glob.glob(os.path.join(args.target, "*_dose_*.nc")))
        default_out = args.target
    else:
        files = sorted(glob.glob(args.target))
        default_out = os.path.dirname(files[0]) if files else "."
    if not files:
        print(f"No *_dose_*.nc files found at {args.target}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or default_out
    os.makedirs(out_dir, exist_ok=True)
    print(f"Bundling {len(files)} NetCDF file(s) -> {out_dir}")

    all_records = []
    for path in files:
        all_records.extend(process_file(path, out_dir))

    if all_records:
        combined = gpd.GeoDataFrame(all_records, crs=CRS)
        combined_path = os.path.join(out_dir, "rivm_all_contours.gpkg")
        combined.to_file(combined_path, driver="GPKG")
        print(f"Combined {len(all_records)} contour features -> "
              f"{os.path.basename(combined_path)}  (crs={combined.crs})")
    else:
        print("No contour features produced.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
