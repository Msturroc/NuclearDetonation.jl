
module Overpressure

using ..Utilities
using Interpolations

export brode_overpressure, DNA_static_overpressure, DNA_dynamic_pressure, soviet_overpressure, r_soviet_overpressure

# Utility functions
scale_range(bomb_yield, ground_range) = ground_range / (bomb_yield^(1.0 / 3))
scale_height(bomb_yield, burst_height) = burst_height / (bomb_yield^(1.0 / 3))

# Brode model
function _brode(z, r, y)
    a(z) = 1.22 - ((3.908 * z^2) / (1 + 810.2 * z^5))
    b(z) = 2.321 + ((6.195 * z^18) / (1 + 1.113 * z^18)) - ((0.03831 * z^17) / (1 + 0.02415 * z^17)) + (0.6692 / (1 + 4164 * z^8))
    c(z) = 4.153 - ((1.149 *  z^18) / (1 + 1.641 * z^18)) - (1.1 / (1 + 2.771 * z^2.5))
    d(z) = -4.166 + ((25.76 * z^1.75) / (1 + 1.382 * z^18)) + ((8.257 * z) / (1 + 3.219 * z))
    e(z) = 1 - ((0.004642 * z^18) / (1 + 0.003886 * z^18))
    f(z) = 0.6096 + ((2.879 * z^9.25) / (1 + 2.359 * z^14.5)) - ((17.5 * z^2) / (1 + 71.66 * z^3))
    g(z) = 1.83 + ((5.361 * z^2) / (1 + 0.3139 * z^6))
    h(z, r, y) = ((8.808 * z^1.5) / (1 + 154.5 * z^3.5)) - ((0.2905 + 64.67 * z^5) / (1 + 441.5 * z^5)) - ((1.389 * z) / (1 + 49.03 * z^5)) + ((1.094 * r^2) / ((781.2 - (123.4 * r) + (37.98 * r^1.5) + r^2) * (1 + (2 * y))))
    j(y) = ((0.000629 * y^4) / (3.493e-9 + y^4)) - ((2.67 * y^2) / (1 + (1e7 * y^4.3)))
    k(y) = 5.18 + ((0.2803 * y^3.5) / (3.788e-6 + y^4))
    return (10.47 / r^a(z)) + (b(z) / r^c(z)) + ((d(z) * e(z)) / (1 + (f(z) * r^g(z)))) + h(z, r, y) + (j(y) / r^k(y))
end

function _brodeop(bomb_yield, ground_range, burst_height)
    z = (burst_height / ground_range)
    y = scale_height(bomb_yield, burst_height)
    x = scale_range(bomb_yield, ground_range)
    r = (x^2 + y^2)^0.5
    return _brode(z, r, y)
end

function brode_overpressure(y, r, h; yunits="kT", dunits="m", opunits="kg/cm^2")
    yld = convert_units(y, yunits, "kT")
    ground_range = convert_units(r, dunits, "kilofeet")
    height = convert_units(h, dunits, "kilofeet")
    op = _brodeop(yld, ground_range, height)
    return convert_units(op, "psi", opunits)
end

# DNA model
function _altitude_t(alt)
    if 0 <= alt < 11000
        return 1 - (2 * 10^9)^-0.5 * alt
    elseif 11000 <= alt < 20000
        return 0.7535 * (1 + (2.09 * 10^-7) * alt)
    elseif alt >= 20000
        return 0.684 *  (1 + (5.16 * 10^-6) * alt)
    end
end

function _altitude_p(alt)
    if 0 <= alt < 11000
        return _altitude_t(alt)^5.3
    elseif 11000 <= alt < 20000
        return 1.6^0.5 * (1 + (2.09 * 10^-7) * alt)^-754 
    elseif alt >= 20000
        return 1.4762 *  (1 + (5.16 * 10^-6) * alt)^-33.6
    end
end

_altitude_sp(alt) = _altitude_p(alt)
_altitude_sd(alt) = _altitude_sp(alt)^(-1.0/3)
_altitude_st(alt) = _altitude_sd(alt) * _altitude_t(alt)^-0.5

_altitude_speed_of_sound(alt) = (340.5 * _altitude_sd(alt)) / _altitude_st(alt)

function _DNA1kTfreeairop(r)
    return (3.04 * 10^11)/r^3 + (1.13 * 10^9)/r^2 + (7.9 * 10^6)/(r * (log(r / 445.42 + 3 * exp(sqrt(r / 445.42) / -3.0)))^0.5)
end

function _DNAfreeairpeakop(r, y, alt)
    r1 = r / (_altitude_sd(alt) * y^(1.0/3))
    return _DNA1kTfreeairop(r1) * _altitude_sp(alt)
end

_shock_strength(op) = op / 101325 + 1

function _shock_gamma(op)
    xi = _shock_strength(op)
    t = 10^-12 * xi^6
    z = log(xi) - (0.47 * t) / (100 + t)
    return 1.402 - (3.4 * 10^-4 * z^4) / (1+ 2.22 * 10^-5 * z^6)
end

_shock_mu(g) = (g + 1) / (g - 1)

function _mass_density_ratio(op)
    xi = _shock_strength(op)
    mu = _shock_mu(_shock_gamma(op))
    return (1 + mu * xi) / (5.975 + xi)
end

function _DNA1kTfreeairdyn(r)
    op = _DNA1kTfreeairop(r)
    return 0.5 * op * (_mass_density_ratio(op) - 1)
end

function _DNAfreeairpeakdyn(r, y, alt)
    r1 = r / (_altitude_sd(alt) * y^(1.0/3))
    return _DNA1kTfreeairdyn(r1) * _altitude_sp(alt)
end

function _scaledfreeairblastwavetoa(r)
    r2 = r * r
    return (r2 * (6.7 + r)) / (7.12e6 + 7.32e4 * r + 340.5 * r2)
end

function _freeairblastwavetoa(r, y, alt)
    return _scaledfreeairblastwavetoa(r) * _altitude_st(alt) * y^(1.0/3)
end

function _normal_reflection_factor(op)
    g = _shock_gamma(op)
    n = _mass_density_ratio(op)
    return 2 + ((g + 1) * (n - 1)) / 2
end

function _peak_particle_mach_number(pfree)
    n = _mass_density_ratio(pfree)
    return ((pfree * (1 - (1 /n))) / 142000)^0.5
end

function _shock_front_mach_number(pfree)
    n = _mass_density_ratio(pfree)
    vc = _peak_particle_mach_number(pfree)
    return vc / (1 - 1 / n)
end

function _scale_slant_range(r, y, alt)
    sgr = r / y^(1.0/3)
    shob = alt / y^(1.0/3)
    return sqrt(sgr^2 + shob^2)
end

function _regular_mach_merge_angle(r, y, alt)
    pfree = _DNA1kTfreeairop(_scale_slant_range(r, y, alt))
    t = 340 / pfree^0.55
    u = 1 / (7782 * pfree^0.7 + 0.9)
    return atan(1 / (t + u))
end

function _merge_region_width(r, y, alt)
    pfree = _DNA1kTfreeairop(_scale_slant_range(r, y, alt))
    t = 340 / pfree^0.55
    w = 1 / (7473 * pfree^0.5 + 6.6)
    v = 1 / (647 * pfree^0.8 + w)
    return atan(1 / (t + v))
end

function _regular_mach_switching_parameter(r, y, alt)
    sgr = r / y^(1.0/3)
    shob = alt / y^(1.0/3)
    alpha = atan(shob / sgr)
    s = (alpha - _regular_mach_merge_angle(r, y, alt)) / _merge_region_width(r, y, alt)
    s0 = max(min(s, 1), -1)
    return 0.5 * (sin(0.5 * pi * s0) + 1)
end

function _p_mach(r, y, alt)
    sgr = r / y^(1.0/3)
    shob = alt / y^(1.0/3)
    alpha = atan(shob / sgr)
    a = min(3.7 - 0.94 * log(sgr), 0.7)
    b = 0.77 * log(sgr) - 3.8 - 18 / sgr
    c = max(a, b)
    return _DNA1kTfreeairop(sgr / 2^(1.0/3)) / (1 - c * sin(alpha))
end

function _p_reg(r, y, alt)
    sgr = r / y^(1.0/3)
    pfree = _DNA1kTfreeairop(_scale_slant_range(r, y, alt))
    shob = alt / y^(1.0/3)
    alpha = atan(shob / sgr)
    r_n = 2 + ((_shock_gamma(pfree) + 1) * (_mass_density_ratio(pfree) - 1)) / 2
    f = pfree / 75842
    d = (f^6 * (1.2 + 0.07 * f^0.5) ) / (f^6 + 1)
    return pfree * ((r_n - 2) * sin(alpha)^d + 2)
end

function _DNAairburstpeakop(r, y, alt)
    sigma = _regular_mach_switching_parameter(r, y, alt)
    if sigma == 0
        return _p_mach(r, y, alt)
    elseif 0 < sigma < 1
        return _p_reg(r, y, alt) * sigma + _p_mach(r, y, alt) * (1 - sigma)
    elseif sigma == 1
        return _p_reg(r, y, alt)
    end
end

function _DNAairburstpeakdyn(r, y, alt)
    pair = _DNAairburstpeakop(r, y, alt)
    sigma = _regular_mach_switching_parameter(r, y, alt)
    sgr = r / y^(1.0/3)
    shob = alt / y^(1.0/3)
    alpha = atan(shob / sgr)
    n_q = _mass_density_ratio(pair)
    return 0.5 * pair * (n_q - 1) * (1 - (sigma * sin(alpha)^2))
end

function _scaledmachstemformationrange(y, alt)
    shob = alt / y^(1.0/3)
    return shob^2.5 / 5822 + 2.09 * shob^0.75
end

function _slantrangescalingfactor(r, y, alt)
    sgr = r / y^(1.0/3)
    x_m = _scaledmachstemformationrange(y, alt)
    if sgr <= x_m
        return 1
    else
        return 1.26 - 0.26 * (x_m / sgr)
    end
end

function _airburstblastwavetoa(r, y, alt)
    v = _slantrangescalingfactor(r, y, alt)
    r1 = _scale_slant_range(r, y, alt) / v
    ta_air = _scaledfreeairblastwavetoa(r1)
    return ta_air * y^(1.0/3) * v
end

function DNA_static_overpressure(y, r, h; yunits="kT", dunits="m", opunits="kg/cm^2")
    yld = convert_units(y, yunits, "kT")
    gr = convert_units(r, dunits, "m")
    height = convert_units(h, dunits, "m")
    op = _DNAairburstpeakop(gr, yld, height)
    return convert_units(op, "Pa", opunits)
end

function DNA_dynamic_pressure(y, r, h; yunits="kT", dunits="m", opunits="kg/cm^2")
    yld = convert_units(y, yunits, "kT")
    gr = convert_units(r, dunits, "m")
    height = convert_units(h, dunits, "m")
    dyn = _DNAairburstpeakdyn(gr, yld, height)
    return convert_units(dyn, "Pa", opunits)
end

# Soviet overpressure models
include("soviet_overpressure_data.jl")
include("soviet_overpressure_funcs.jl")

end
