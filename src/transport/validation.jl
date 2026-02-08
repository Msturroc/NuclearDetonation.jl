# SNAP: Severe Nuclear Accident Programme
# Validation Module - GeoJSON contour loading, UTM conversion, and FMS scoring
#
# Provides tools for comparing model output against digitised historical contours
# from nuclear test fallout patterns.

using JSON3
using LinearAlgebra
using Statistics: mean

# ============================================================================
# Coordinate Conversion: UTM Zone 11 (EPSG:26711) to WGS84
# ============================================================================

"""
    UTMZone

Parameters for a UTM zone projection.
"""
struct UTMZone
    zone::Int
    hemisphere::Symbol  # :north or :south
    k0::Float64         # Scale factor at central meridian
    false_easting::Float64
    false_northing::Float64
end

# NAD27 UTM Zone 11N (EPSG:26711) - used by Nancy contour data
const UTM_ZONE_11N_NAD27 = UTMZone(11, :north, 0.9996, 500000.0, 0.0)

# Clarke 1866 ellipsoid (used by NAD27)
const CLARKE_1866_A = 6378206.4  # Semi-major axis (m)
const CLARKE_1866_F = 1.0 / 294.978698214  # Flattening
const CLARKE_1866_B = CLARKE_1866_A * (1 - CLARKE_1866_F)  # Semi-minor axis
const CLARKE_1866_E2 = (CLARKE_1866_A^2 - CLARKE_1866_B^2) / CLARKE_1866_A^2  # First eccentricity squared
const CLARKE_1866_EP2 = (CLARKE_1866_A^2 - CLARKE_1866_B^2) / CLARKE_1866_B^2  # Second eccentricity squared

"""
    utm_to_latlon(easting::Float64, northing::Float64, zone::UTMZone) -> (lat, lon)

Convert UTM coordinates to latitude/longitude (WGS84).

Uses the Clarke 1866 ellipsoid for NAD27 compatibility.
Based on Snyder's "Map Projections - A Working Manual" (USGS Professional Paper 1395).
"""
function utm_to_latlon(easting::Float64, northing::Float64, zone::UTMZone=UTM_ZONE_11N_NAD27)
    # Central meridian for the zone
    λ0 = deg2rad(-180.0 + (zone.zone - 0.5) * 6.0)

    # Remove false easting/northing
    x = easting - zone.false_easting
    y = northing - zone.false_northing

    # Constants
    a = CLARKE_1866_A
    e2 = CLARKE_1866_E2
    e = sqrt(e2)
    ep2 = CLARKE_1866_EP2
    k0 = zone.k0

    # Footprint latitude
    M = y / k0
    μ = M / (a * (1 - e2/4 - 3*e2^2/64 - 5*e2^3/256))

    e1 = (1 - sqrt(1 - e2)) / (1 + sqrt(1 - e2))

    φ1 = μ + (3*e1/2 - 27*e1^3/32) * sin(2*μ) +
              (21*e1^2/16 - 55*e1^4/32) * sin(4*μ) +
              (151*e1^3/96) * sin(6*μ) +
              (1097*e1^4/512) * sin(8*μ)

    # Convergence
    C1 = ep2 * cos(φ1)^2
    T1 = tan(φ1)^2
    N1 = a / sqrt(1 - e2 * sin(φ1)^2)
    R1 = a * (1 - e2) / (1 - e2 * sin(φ1)^2)^1.5
    D = x / (N1 * k0)

    # Latitude
    φ = φ1 - (N1 * tan(φ1) / R1) * (
        D^2/2 -
        (5 + 3*T1 + 10*C1 - 4*C1^2 - 9*ep2) * D^4/24 +
        (61 + 90*T1 + 298*C1 + 45*T1^2 - 252*ep2 - 3*C1^2) * D^6/720
    )

    # Longitude
    λ = λ0 + (D - (1 + 2*T1 + C1) * D^3/6 +
              (5 - 2*C1 + 28*T1 - 3*C1^2 + 8*ep2 + 24*T1^2) * D^5/120) / cos(φ1)

    lat = rad2deg(φ)
    lon = rad2deg(λ)

    return (lat, lon)
end

"""
    convert_utm_coords(coords, zone::UTMZone) -> Vector{Tuple{Float64,Float64}}

Convert a vector/array of UTM [easting, northing] pairs to (lat, lon) tuples.
Works with both standard Julia Vectors and JSON3 arrays.
"""
function convert_utm_coords(coords, zone::UTMZone=UTM_ZONE_11N_NAD27)
    result = Tuple{Float64,Float64}[]
    for c in coords
        easting = Float64(c[1])
        northing = Float64(c[2])
        push!(result, utm_to_latlon(easting, northing, zone))
    end
    return result
end


# ============================================================================
# GeoJSON Data Structures
# ============================================================================

"""
    DoseRateContour

A single dose rate contour polygon (or multipolygon) at a specific threshold.
"""
struct DoseRateContour
    dose_rate_mR_hr::Float64              # Dose rate threshold (mR/hr)
    polygons::Vector{Vector{Tuple{Float64,Float64}}}  # List of polygons, each is list of (lat, lon)
end

"""
    TOAContour

A time-of-arrival contour line at a specific hour post-detonation.
"""
struct TOAContour
    hour::Float64                          # Hours post-detonation
    lines::Vector{Vector{Tuple{Float64,Float64}}}  # List of linestrings, each is list of (lat, lon)
end

"""
    NancyObservations

Complete set of digitised observations from the Nancy nuclear test.
"""
struct NancyObservations
    dose_rate_contours::Vector{DoseRateContour}
    toa_contours::Vector{TOAContour}
    # Test metadata
    detonation_lat::Float64
    detonation_lon::Float64
    yield_kt::Float64
    hob_m::Float64
    detonation_utc::String  # ISO 8601 format
end


# ============================================================================
# GeoJSON Parsing
# ============================================================================

"""
    load_doserate_geojson(filepath::String) -> Vector{DoseRateContour}

Load dose rate contours from a GeoJSON file (EPSG:26711 UTM Zone 11N).
Converts coordinates to WGS84 lat/lon.
"""
function load_doserate_geojson(filepath::String)
    json_str = read(filepath, String)
    geojson = JSON3.read(json_str)

    contours = DoseRateContour[]

    for feature in geojson.features
        # Extract dose rate from properties
        props = feature.properties
        dose_rate = Float64(props.doserate)

        geom = feature.geometry
        polygons = Vector{Tuple{Float64,Float64}}[]

        if geom.type == "MultiPolygon"
            # MultiPolygon: array of polygons, each polygon is array of rings
            for polygon in geom.coordinates
                for ring in polygon
                    latlon_coords = convert_utm_coords(ring)
                    push!(polygons, latlon_coords)
                end
            end
        elseif geom.type == "Polygon"
            # Polygon: array of rings (outer ring + holes)
            for ring in geom.coordinates
                latlon_coords = convert_utm_coords(ring)
                push!(polygons, latlon_coords)
            end
        end

        push!(contours, DoseRateContour(dose_rate, polygons))
    end

    return contours
end

"""
    load_toa_geojson(filepath::String) -> Vector{TOAContour}

Load time-of-arrival contours from a GeoJSON file (EPSG:26711 UTM Zone 11N).
Converts coordinates to WGS84 lat/lon.
"""
function load_toa_geojson(filepath::String)
    json_str = read(filepath, String)
    geojson = JSON3.read(json_str)

    contours = TOAContour[]

    for feature in geojson.features
        # Extract hour from properties
        props = feature.properties
        hour = Float64(props.hour)

        geom = feature.geometry
        lines = Vector{Tuple{Float64,Float64}}[]

        if geom.type == "MultiLineString"
            for line in geom.coordinates
                latlon_coords = convert_utm_coords(line)
                push!(lines, latlon_coords)
            end
        elseif geom.type == "LineString"
            latlon_coords = convert_utm_coords(geom.coordinates)
            push!(lines, latlon_coords)
        end

        push!(contours, TOAContour(hour, lines))
    end

    return contours
end

"""
    load_nancy_observations(contour_dir::String) -> NancyObservations

Load all Nancy test observation data from the digitised contour directory.

# Arguments
- `contour_dir`: Path to Nancy_exposurerate_contours_digitised/geoJSON/

# Returns
- NancyObservations struct with dose rate and TOA contours
"""
function load_nancy_observations(contour_dir::String)
    doserate_file = joinpath(contour_dir, "Nancy_doserate_contours.geojson")
    toa_file = joinpath(contour_dir, "Nancy_TOA.geojson")

    dose_contours = load_doserate_geojson(doserate_file)
    toa_contours = load_toa_geojson(toa_file)

    # Nancy test metadata
    return NancyObservations(
        dose_contours,
        toa_contours,
        37.0956,      # Detonation latitude
        -116.1028,    # Detonation longitude
        24.0,         # Yield (kT)
        91.0,         # Height of burst (m)
        "1953-03-24T13:10:00Z"  # Detonation time UTC
    )
end

"""
    load_nancy_observations() -> NancyObservations

Convenience overload that loads Nancy observations from the package's bundled data directory.

# Example
```julia
using NuclearDetonation
obs = NuclearDetonation.Transport.load_nancy_observations()
```
"""
function load_nancy_observations()
    contour_dir = joinpath(pkgdir(NuclearDetonation), "data", "nancy_observations")
    load_nancy_observations(contour_dir)
end


# ============================================================================
# Rasterisation: Convert Polygons to Grid Masks
# ============================================================================

"""
    point_in_polygon(lat::Float64, lon::Float64, polygon::Vector{Tuple{Float64,Float64}}) -> Bool

Test if a point (lat, lon) is inside a polygon using ray casting algorithm.
"""
function point_in_polygon(lat::Float64, lon::Float64, polygon::Vector{Tuple{Float64,Float64}})
    n = length(polygon)
    inside = false

    j = n
    for i in 1:n
        yi, xi = polygon[i]
        yj, xj = polygon[j]

        if ((yi > lat) != (yj > lat)) &&
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end

    return inside
end

"""
    rasterise_contour(contour::DoseRateContour,
                      lat_grid::Vector{Float64},
                      lon_grid::Vector{Float64}) -> BitMatrix

Convert a dose rate contour polygon to a binary grid mask.

# Arguments
- `contour`: DoseRateContour with polygon boundaries
- `lat_grid`: Vector of latitude values (grid cell centres)
- `lon_grid`: Vector of longitude values (grid cell centres)

# Returns
- BitMatrix where true indicates grid cell is inside the contour
"""
function rasterise_contour(contour::DoseRateContour,
                           lat_grid::Vector{Float64},
                           lon_grid::Vector{Float64})
    ny = length(lat_grid)
    nx = length(lon_grid)
    mask = falses(nx, ny)

    for polygon in contour.polygons
        for j in 1:ny
            for i in 1:nx
                if point_in_polygon(lat_grid[j], lon_grid[i], polygon)
                    mask[i, j] = true
                end
            end
        end
    end

    return mask
end

"""
    rasterise_all_contours(contours::Vector{DoseRateContour},
                           lat_grid::Vector{Float64},
                           lon_grid::Vector{Float64}) -> Dict{Float64, BitMatrix}

Rasterise all dose rate contours to grid masks.

# Returns
- Dictionary mapping dose rate threshold to binary mask
"""
function rasterise_all_contours(contours::Vector{DoseRateContour},
                                 lat_grid::Vector{Float64},
                                 lon_grid::Vector{Float64})
    masks = Dict{Float64, BitMatrix}()

    for contour in contours
        mask = rasterise_contour(contour, lat_grid, lon_grid)

        # If same dose rate already exists, combine with OR
        if haskey(masks, contour.dose_rate_mR_hr)
            masks[contour.dose_rate_mR_hr] .|= mask
        else
            masks[contour.dose_rate_mR_hr] = mask
        end
    end

    return masks
end


# ============================================================================
# FMS Scoring: Figure of Merit in Space
# ============================================================================

"""
    FMSResult

Result of a Figure of Merit in Space calculation for a single threshold.
"""
struct FMSResult
    fms_score::Float64           # Overall Figure of Merit in Space (0-1)
    intersection_area::Float64   # Area where both agree (km²)
    union_area::Float64          # Total area covered by either (km²)
    overprediction::Float64      # Area model predicts but obs doesn't (km²)
    underprediction::Float64     # Area obs shows but model doesn't (km²)
    threshold_mR_hr::Float64     # Dose rate threshold used
end

"""
    TOAResult

Result of time-of-arrival comparison.
"""
struct TOAResult
    mean_arrival_error_hours::Float64   # Mean error in arrival time (hours)
    max_arrival_error_hours::Float64    # Maximum error in arrival time
    arrival_errors::Vector{Float64}     # Individual errors for each TOA contour
    hours_compared::Vector{Float64}     # Hours at which comparison was made
end

"""
    ValidationScore

Combined validation score for APMC model comparison.
"""
struct ValidationScore
    dose_fms::Float64            # Mean FMS across dose rate thresholds (0-1)
    toa_score::Float64           # Mean arrival time accuracy (0-1)
    combined::Float64            # Weighted combination for APMC
    fms_results::Vector{FMSResult}  # Individual FMS results per threshold
    toa_result::Union{TOAResult, Nothing}
end

"""
    compute_fms(model_mask::BitMatrix, obs_mask::BitMatrix,
                threshold::Float64, cell_area_km2::Float64) -> FMSResult

Compute Figure of Merit in Space between model and observation masks.

FMS = Area(Intersection) / Area(Union)

# Arguments
- `model_mask`: Binary mask of model prediction (above threshold)
- `obs_mask`: Binary mask of observations (above threshold)
- `threshold`: Dose rate threshold (mR/hr) for labelling
- `cell_area_km2`: Area of each grid cell (km²)

# Returns
- FMSResult with score and area statistics
"""
function compute_fms(model_mask::BitMatrix, obs_mask::BitMatrix,
                     threshold::Float64, cell_area_km2::Float64)
    # Compute intersection and union
    intersection = model_mask .& obs_mask
    union_mask = model_mask .| obs_mask

    intersection_cells = sum(intersection)
    union_cells = sum(union_mask)

    # Compute areas
    intersection_area = intersection_cells * cell_area_km2
    union_area = union_cells * cell_area_km2

    # Overprediction: model says yes, obs says no
    overprediction_cells = sum(model_mask .& .!obs_mask)
    overprediction = overprediction_cells * cell_area_km2

    # Underprediction: obs says yes, model says no
    underprediction_cells = sum(.!model_mask .& obs_mask)
    underprediction = underprediction_cells * cell_area_km2

    # FMS score (0 if no area in union)
    fms = union_cells > 0 ? intersection_area / union_area : 0.0

    return FMSResult(fms, intersection_area, union_area,
                     overprediction, underprediction, threshold)
end

"""
    compute_multi_threshold_fms(model_dose_rate::Matrix{Float64},
                                 obs_contours::Dict{Float64, BitMatrix},
                                 lat_grid::Vector{Float64},
                                 lon_grid::Vector{Float64}) -> Vector{FMSResult}

Compute FMS for multiple dose rate thresholds.

# Arguments
- `model_dose_rate`: Model predicted dose rate grid (mR/hr)
- `obs_contours`: Dictionary of observation masks by threshold
- `lat_grid`, `lon_grid`: Grid coordinates

# Returns
- Vector of FMSResult for each threshold
"""
function compute_multi_threshold_fms(model_dose_rate::Matrix{Float64},
                                      obs_contours::Dict{Float64, BitMatrix},
                                      lat_grid::Vector{Float64},
                                      lon_grid::Vector{Float64})
    results = FMSResult[]

    # Compute approximate cell area (using mean latitude)
    mean_lat = mean(lat_grid)
    dlat = length(lat_grid) > 1 ? abs(lat_grid[2] - lat_grid[1]) : 0.1
    dlon = length(lon_grid) > 1 ? abs(lon_grid[2] - lon_grid[1]) : 0.1

    # Cell dimensions in km (approximate)
    R_EARTH_KM = 6371.0
    cell_height_km = dlat * (π / 180) * R_EARTH_KM
    cell_width_km = dlon * (π / 180) * R_EARTH_KM * cos(deg2rad(mean_lat))
    cell_area_km2 = cell_height_km * cell_width_km

    # Compute FMS for each threshold
    for (threshold, obs_mask) in obs_contours
        # Create model mask at this threshold
        model_mask = model_dose_rate .>= threshold

        fms_result = compute_fms(model_mask, obs_mask, threshold, cell_area_km2)
        push!(results, fms_result)
    end

    # Sort by threshold for consistent ordering
    sort!(results, by=r -> r.threshold_mR_hr)

    return results
end

"""
    compute_toa_score(model_snapshots::Vector{Matrix{Float64}},
                      snapshot_times_hours::Vector{Float64},
                      obs_toa::Vector{TOAContour},
                      lat_grid::Vector{Float64},
                      lon_grid::Vector{Float64};
                      threshold_fraction::Float64=0.01) -> TOAResult

Compute time-of-arrival accuracy by comparing when model plume reaches
TOA contour locations vs observed arrival times.

# Arguments
- `model_snapshots`: Vector of dose rate grids at different times
- `snapshot_times_hours`: Hours post-detonation for each snapshot
- `obs_toa`: Observed TOA contours
- `lat_grid`, `lon_grid`: Grid coordinates
- `threshold_fraction`: Fraction of max dose rate to consider "arrived"

# Returns
- TOAResult with arrival time errors
"""
function compute_toa_score(model_snapshots::Vector{Matrix{Float64}},
                           snapshot_times_hours::Vector{Float64},
                           obs_toa::Vector{TOAContour},
                           lat_grid::Vector{Float64},
                           lon_grid::Vector{Float64};
                           threshold_fraction::Float64=0.01)
    errors = Float64[]
    hours = Float64[]

    # Get maximum dose rate across all snapshots for threshold
    max_dose = maximum(maximum.(model_snapshots))
    threshold = max_dose * threshold_fraction

    for toa_contour in obs_toa
        obs_hour = toa_contour.hour

        # Sample points along the TOA contour
        sample_points = Tuple{Float64,Float64}[]
        for line in toa_contour.lines
            # Sample every 3rd point to reduce computation
            for i in 1:3:length(line)
                push!(sample_points, line[i])
            end
        end

        if isempty(sample_points)
            continue
        end

        # Find model arrival time at each sample point
        model_arrivals = Float64[]

        for (plat, plon) in sample_points
            # Find nearest grid cell
            j = argmin(abs.(lat_grid .- plat))
            i = argmin(abs.(lon_grid .- plon))

            # Find first time when dose exceeds threshold
            arrival_time = Inf
            for (t_idx, snapshot) in enumerate(model_snapshots)
                if i <= size(snapshot, 1) && j <= size(snapshot, 2)
                    if snapshot[i, j] >= threshold
                        arrival_time = snapshot_times_hours[t_idx]
                        break
                    end
                end
            end

            if isfinite(arrival_time)
                push!(model_arrivals, arrival_time)
            end
        end

        # Compute mean model arrival time for this contour
        if !isempty(model_arrivals)
            mean_model_arrival = mean(model_arrivals)
            error = abs(mean_model_arrival - obs_hour)
            push!(errors, error)
            push!(hours, obs_hour)
        end
    end

    if isempty(errors)
        return TOAResult(Inf, Inf, Float64[], Float64[])
    end

    return TOAResult(mean(errors), maximum(errors), errors, hours)
end

"""
    compute_combined_score(fms_results::Vector{FMSResult},
                           toa_result::Union{TOAResult, Nothing};
                           dose_weight::Float64=0.7,
                           toa_weight::Float64=0.3,
                           max_toa_error_hours::Float64=6.0) -> ValidationScore

Compute combined validation score for APMC.

# Arguments
- `fms_results`: Vector of FMS results for each dose rate threshold
- `toa_result`: TOA comparison result (optional)
- `dose_weight`: Weight for dose rate FMS (default: 0.7)
- `toa_weight`: Weight for TOA score (default: 0.3)
- `max_toa_error_hours`: Error at which TOA score = 0 (default: 6 hours)

# Returns
- ValidationScore with combined score (0-1, higher is better)
"""
function compute_combined_score(fms_results::Vector{FMSResult},
                                 toa_result::Union{TOAResult, Nothing}=nothing;
                                 dose_weight::Float64=0.7,
                                 toa_weight::Float64=0.3,
                                 max_toa_error_hours::Float64=6.0)
    # Mean FMS across all thresholds
    if isempty(fms_results)
        dose_fms = 0.0
    else
        dose_fms = mean(r.fms_score for r in fms_results)
    end

    # TOA score: convert error to 0-1 score (0 error = 1.0, max error = 0.0)
    if isnothing(toa_result) || isinf(toa_result.mean_arrival_error_hours)
        toa_score = 0.0
        # Adjust weights if no TOA data
        effective_dose_weight = 1.0
        effective_toa_weight = 0.0
    else
        toa_score = max(0.0, 1.0 - toa_result.mean_arrival_error_hours / max_toa_error_hours)
        effective_dose_weight = dose_weight
        effective_toa_weight = toa_weight
    end

    # Normalise weights
    total_weight = effective_dose_weight + effective_toa_weight
    combined = (effective_dose_weight * dose_fms + effective_toa_weight * toa_score) / total_weight

    return ValidationScore(dose_fms, toa_score, combined, fms_results, toa_result)
end

"""
    compute_validation_score(model_dose_rate::Matrix{Float64},
                              nancy_obs::NancyObservations,
                              lat_grid::Vector{Float64},
                              lon_grid::Vector{Float64};
                              model_snapshots::Union{Nothing, Vector{Matrix{Float64}}}=nothing,
                              snapshot_times_hours::Union{Nothing, Vector{Float64}}=nothing,
                              dose_weight::Float64=0.7,
                              toa_weight::Float64=0.3) -> ValidationScore

High-level function to compute complete validation score against Nancy observations.

# Arguments
- `model_dose_rate`: Model H+12 dose rate grid (mR/hr)
- `nancy_obs`: Nancy observations loaded from GeoJSON
- `lat_grid`, `lon_grid`: Model grid coordinates
- `model_snapshots`: Optional time series for TOA comparison
- `snapshot_times_hours`: Hours for each snapshot
- `dose_weight`, `toa_weight`: Weights for combined score

# Returns
- ValidationScore with all metrics
"""
function compute_validation_score(model_dose_rate::Matrix{Float64},
                                   nancy_obs::NancyObservations,
                                   lat_grid::Vector{Float64},
                                   lon_grid::Vector{Float64};
                                   model_snapshots::Union{Nothing, Vector{Matrix{Float64}}}=nothing,
                                   snapshot_times_hours::Union{Nothing, Vector{Float64}}=nothing,
                                   dose_weight::Float64=0.7,
                                   toa_weight::Float64=0.3)
    # Rasterise observation contours to model grid
    obs_masks = rasterise_all_contours(nancy_obs.dose_rate_contours, lat_grid, lon_grid)

    # Compute FMS for dose rates
    fms_results = compute_multi_threshold_fms(model_dose_rate, obs_masks, lat_grid, lon_grid)

    # Compute TOA score if time series provided
    toa_result = nothing
    if !isnothing(model_snapshots) && !isnothing(snapshot_times_hours)
        toa_result = compute_toa_score(model_snapshots, snapshot_times_hours,
                                       nancy_obs.toa_contours, lat_grid, lon_grid)
    end

    # Compute combined score
    return compute_combined_score(fms_results, toa_result;
                                  dose_weight=dose_weight, toa_weight=toa_weight)
end


# ============================================================================
# APMC Distance Function
# ============================================================================

"""
    apmc_distance(validation_score::ValidationScore) -> Float64

Convert validation score to APMC distance (lower is better).

Distance = 1 - combined_score

# Arguments
- `validation_score`: ValidationScore from compute_validation_score

# Returns
- Distance value in [0, 1] where 0 is perfect match
"""
function apmc_distance(validation_score::ValidationScore)
    return 1.0 - validation_score.combined
end


# ============================================================================
# Utility Functions
# ============================================================================

"""
    contour_bounds(contours::Vector{DoseRateContour}) -> (lat_min, lat_max, lon_min, lon_max)

Get bounding box of all dose rate contours.
"""
function contour_bounds(contours::Vector{DoseRateContour})
    lat_min, lat_max = Inf, -Inf
    lon_min, lon_max = Inf, -Inf

    for contour in contours
        for polygon in contour.polygons
            for (lat, lon) in polygon
                lat_min = min(lat_min, lat)
                lat_max = max(lat_max, lat)
                lon_min = min(lon_min, lon)
                lon_max = max(lon_max, lon)
            end
        end
    end

    return (lat_min, lat_max, lon_min, lon_max)
end

"""
    suggest_grid(nancy_obs::NancyObservations; resolution_km::Float64=2.0)
        -> (lat_grid, lon_grid)

Suggest a grid for model output that covers the observation contours.

# Arguments
- `nancy_obs`: Nancy observations
- `resolution_km`: Desired grid resolution (km)

# Returns
- (lat_grid, lon_grid) vectors for model output
"""
function suggest_grid(nancy_obs::NancyObservations; resolution_km::Float64=2.0, buffer_fraction::Float64=0.1)
    lat_min, lat_max, lon_min, lon_max = contour_bounds(nancy_obs.dose_rate_contours)

    # Add buffer around observation contours
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_min -= buffer_fraction * lat_range
    lat_max += buffer_fraction * lat_range
    lon_min -= buffer_fraction * lon_range
    lon_max += buffer_fraction * lon_range

    # Convert resolution to degrees (approximate)
    mean_lat = (lat_min + lat_max) / 2
    dlat = resolution_km / 111.0  # ~111 km per degree latitude
    dlon = resolution_km / (111.0 * cos(deg2rad(mean_lat)))

    lat_grid = collect(lat_min:dlat:lat_max)
    lon_grid = collect(lon_min:dlon:lon_max)

    return (lat_grid, lon_grid)
end

"""
    print_validation_summary(score::ValidationScore)

Print a human-readable summary of validation results.
"""
function print_validation_summary(score::ValidationScore)
    println("="^60)
    println("VALIDATION SUMMARY")
    println("="^60)
    println()
    println("Combined Score: $(round(score.combined, digits=3)) (higher is better)")
    println("APMC Distance:  $(round(apmc_distance(score), digits=3)) (lower is better)")
    println()
    println("Dose Rate FMS:  $(round(score.dose_fms, digits=3))")
    println()
    println("FMS by Threshold:")
    for r in score.fms_results
        println("  $(r.threshold_mR_hr) mR/hr: FMS=$(round(r.fms_score, digits=3)), " *
                "Over=$(round(r.overprediction, digits=1)) km², Under=$(round(r.underprediction, digits=1)) km²")
    end
    println()
    if !isnothing(score.toa_result)
        println("TOA Score:      $(round(score.toa_score, digits=3))")
        println("Mean TOA Error: $(round(score.toa_result.mean_arrival_error_hours, digits=2)) hours")
    else
        println("TOA Score:      N/A (no time series provided)")
    end
    println("="^60)
end
