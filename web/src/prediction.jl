# Impact prediction using pre-trained XGBoost models
#
# For each nuclear power plant site, a gradient-boosted tree classifier
# (XGBoost) was trained on 2.2 million HYSPLIT atmospheric dispersion
# simulations (2011-2023) to predict whether radioactive fallout would
# reach Ireland given current meteorological conditions.
#
# Feature vector: 201 features
#   - 14 met variables x 7 statistics x 2 regions (Ireland + overall domain) = 196
#   - 5 manual features (start_hour, duration, height, dayofyear, month)

module Prediction

using Dates
using Statistics
using XGBoost

export load_prediction_models!, predict_impact, extract_features

# --- Constants ---

# NPP site names (lowercase, matching model filenames)
const SITE_NAMES = ["wylfa", "heysham", "hinkley", "flamanville", "paluel", "sizewell"]

# Ireland bounding box for regional statistics
const IRELAND_LON_MIN = -10.47472
const IRELAND_LON_MAX = -6.01306
const IRELAND_LAT_MIN = 51.44555
const IRELAND_LAT_MAX = 55.37999

# Overall domain bounding box
const OVERALL_LON_MIN = -10.5
const OVERALL_LON_MAX = 2.0
const OVERALL_LAT_MIN = 46.0
const OVERALL_LAT_MAX = 56.0

# The 14 variables used by the XGBoost models (surface + 1000 hPa only).
# Order matches Julia Dict iteration order from the training pipeline's
# ARL_summary.jl, which determines CSV row order and hence feature position.
const VARIABLE_SPECS = [
    # (name_in_feature_vector, ARL_level, ARL_variable_name)
    ("PRSS",     0.0,    "PRSS"),
    ("RELH1000", 1000.0, "RELH"),
    ("UWND1000", 1000.0, "UWND"),
    ("TPP1",     0.0,    "TPP1"),
    ("HGTS1000", 1000.0, "HGTS"),
    ("WWND1000", 1000.0, "WWND"),
    ("CAPE",     0.0,    "CAPE"),
    ("TEMP1000", 1000.0, "TEMP"),
    ("U10M",     0.0,    "U10M"),
    ("VWND1000", 1000.0, "VWND"),
    ("V10M",     0.0,    "V10M"),
    ("PBLH",     0.0,    "PBLH"),
    ("T02M",     0.0,    "T02M"),
    ("LTHF",     0.0,    "LTHF"),
]

const N_VARS = length(VARIABLE_SPECS)  # 14
const N_STATS = 7
const N_WEATHER_FEATURES = N_VARS * N_STATS * 2  # 196 (Ireland + overall)
const N_MANUAL_FEATURES = 5
const N_TOTAL_FEATURES = N_WEATHER_FEATURES + N_MANUAL_FEATURES  # 201

# --- Model storage ---

const MODELS = Dict{String, Booster}()

"""
    load_prediction_models!(models_dir::String)

Load pre-trained XGBoost models for all NPP sites from JSON files.
"""
function load_prediction_models!(models_dir::String)
    empty!(MODELS)
    for site in SITE_NAMES
        path = joinpath(models_dir, "$site.json")
        if isfile(path)
            MODELS[site] = XGBoost.load(Booster, path)
            println("  Loaded prediction model: $site")
        else
            @warn "Prediction model not found: $path"
        end
    end
    println("  $(length(MODELS))/$(length(SITE_NAMES)) prediction models loaded")
    nothing
end

# --- Statistics ---

function safe_skewness(x::AbstractVector{<:Real})
    n = length(x)
    n < 3 && return 0.0
    m = mean(x)
    v = var(x; corrected=false)
    v < 1e-30 && return 0.0
    s = sum((xi - m)^3 for xi in x) / n
    s / v^1.5
end

function safe_kurtosis(x::AbstractVector{<:Real})
    n = length(x)
    n < 4 && return 0.0
    m = mean(x)
    v = var(x; corrected=false)
    v < 1e-30 && return 0.0
    k = sum((xi - m)^4 for xi in x) / n
    k / v^2 - 3.0
end

"""
    compute_stats(values::AbstractVector) -> Vector{Float64}

Compute 7 summary statistics: mean, variance, min, max, median, skewness, kurtosis.
"""
function compute_stats(values::AbstractVector{<:Real})
    isempty(values) && return zeros(7)
    Float64[
        mean(values),
        var(values; corrected=false),
        minimum(values),
        maximum(values),
        median(values),
        safe_skewness(values),
        safe_kurtosis(values),
    ]
end

# --- Feature extraction ---

"""
    region_mask(lats, lons, lat_min, lat_max, lon_min, lon_max)

Return linear indices of grid points within the bounding box.
"""
function region_indices(lats::Vector{Float64}, lons::Vector{Float64},
                        lat_min::Float64, lat_max::Float64,
                        lon_min::Float64, lon_max::Float64)
    lat_idx = findall(lat_min .<= lats .<= lat_max)
    lon_idx = findall(lon_min .<= lons .<= lon_max)
    (lon_idx, lat_idx)
end

"""
    extract_region_values(field::Matrix{Float32}, lon_idx, lat_idx)

Extract values from the 2D field (nx x ny) within the given index ranges.
"""
function extract_region_values(field::Matrix{Float32}, lon_idx, lat_idx)
    vec(Float64.(field[lon_idx, lat_idx]))
end

"""
    extract_features(arl_files::Vector{String}, date::Date, hour::Int;
                     release_duration=48.0, release_height=100.0) -> Vector{Float64}

Build the 201-element feature vector from ARL meteorological data for a given date/hour.
"""
function extract_features(arl_files::Vector{String}, date::Date, hour::Int;
                          release_duration::Float64=48.0,
                          release_height::Float64=100.0)
    # Find the ARL file covering this date
    arl = nothing
    for f in arl_files
        a = Main.ARLReader.read_arl(f)
        arl_date = Main.ARLReader.get_date(a)
        if arl_date <= date <= arl_date + Day(a.n_timesteps ÷ 8)
            arl = a
            break
        end
    end
    isnothing(arl) && error("No ARL file found covering $date")

    lats = arl.grid.lats
    lons = arl.grid.lons

    # Pre-compute region indices
    ire_lon, ire_lat = region_indices(lats, lons,
        IRELAND_LAT_MIN, IRELAND_LAT_MAX, IRELAND_LON_MIN, IRELAND_LON_MAX)
    ovr_lon, ovr_lat = region_indices(lats, lons,
        OVERALL_LAT_MIN, OVERALL_LAT_MAX, OVERALL_LON_MIN, OVERALL_LON_MAX)

    day = Dates.day(date)

    # Compute 7 statistics for each of the 14 variables, for both regions.
    # Store as a 14x7 matrix (rows=variables, cols=statistics) per region,
    # matching the training pipeline's readdlm(csv) -> vec() layout.
    ireland_matrix = zeros(Float64, N_VARS, N_STATS)
    overall_matrix = zeros(Float64, N_VARS, N_STATS)

    for (vi, (_, level, varname)) in enumerate(VARIABLE_SPECS)
        if Main.ARLReader.has_variable(arl, level, varname)
            field = Main.ARLReader.load_field(arl, day, hour, level, varname)
        else
            field = zeros(Float32, length(lons), length(lats))
        end
        ireland_matrix[vi, :] = compute_stats(extract_region_values(field, ire_lon, ire_lat))
        overall_matrix[vi, :] = compute_stats(extract_region_values(field, ovr_lon, ovr_lat))
    end

    # vec() in Julia is column-major: all vars' means, then all vars' variances, etc.
    ireland_vec = vec(ireland_matrix)  # 98 elements
    overall_vec = vec(overall_matrix)  # 98 elements

    # Manual features
    hour_index = Float64(hour)
    doy = Float64(Dates.dayofyear(date))
    mon = Float64(Dates.month(date))

    features = vcat(ireland_vec, overall_vec,
                    Float64[hour_index, release_duration, release_height, doy, mon])

    # Replace any NaN with 0
    replace!(features, NaN => 0.0)

    features
end

"""
    predict_impact(site::String, features::Vector{Float64}) -> NamedTuple

Run the XGBoost model for the given site and return prediction.
Returns (impact::Bool, probability::Float64, site::String).
"""
function predict_impact(site::String, features::Vector{Float64})
    site_lower = lowercase(site)
    haskey(MODELS, site_lower) || error("No model loaded for site '$site'")

    booster = MODELS[site_lower]
    dm = DMatrix(reshape(features, 1, :))
    prob = XGBoost.predict(booster, dm)[1]
    impact = prob >= 0.5

    (impact=impact, probability=round(prob; digits=4), site=site_lower)
end

"""
    predict_from_arl(site::String, arl_dir::String, date::Date, hour::Int;
                     release_duration=48.0, release_height=100.0) -> NamedTuple

Convenience function: extract features from ARL files and run prediction.
"""
function predict_from_arl(site::String, arl_dir::String, date::Date, hour::Int;
                          release_duration::Float64=48.0,
                          release_height::Float64=100.0)
    arl_files = sort(filter(f -> endswith(f, ".ARL") || endswith(f, ".arl"),
                            readdir(arl_dir; join=true)))
    isempty(arl_files) && error("No ARL files found in $arl_dir")

    features = extract_features(arl_files, date, hour;
                                release_duration=release_duration,
                                release_height=release_height)
    predict_impact(site, features)
end

end # module Prediction
