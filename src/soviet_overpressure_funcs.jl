# Soviet overpressure model functions

# Linear interpolation in log10 space
function lerp10(h, h1, h2, o1, o2)
    """Returns 10^o, where o is the linear interpolation of value h between (h1, o1) and (h2, o2)."""
    o_interp = o1 + (h - h1) * (o2 - o1) / (h2 - h1)
    return 10^o_interp
end

# Interpolation functions for each dataset
function _soviet_mach_sh20(range)
    if 0 <= range <= _soviet_mach_sh20x[end]
        # Linear interpolation using Interpolations.jl
        itp = linear_interpolation(_soviet_mach_sh20x, _soviet_mach_sh20y)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet Mach sh20 model"))
    end
end

function _soviet_mach_sh12(range)
    if 0 <= range <= _soviet_mach_sh12x[end]
        itp = linear_interpolation(_soviet_mach_sh12x, _soviet_mach_sh12y)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet Mach sh12 model"))
    end
end

function _soviet_mach_sh7(range)
    if _soviet_mach_sh7x[1] <= range <= _soviet_mach_sh7x[end]
        itp = linear_interpolation(_soviet_mach_sh7x, _soviet_mach_sh7y)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet Mach sh7 model"))
    end
end

function _soviet_nomach_sh20(range)
    if 0 <= range <= _soviet_nomach_sh20x[end]
        itp = linear_interpolation(_soviet_nomach_sh20x, _soviet_nomach_sh20y)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet No-Mach sh20 model"))
    end
end

function _soviet_nomach_sh12(range)
    if 0 <= range <= _soviet_nomach_sh12x[end]
        itp = linear_interpolation(_soviet_nomach_sh12x, _soviet_nomach_sh12y)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet No-Mach sh12 model"))
    end
end

function _soviet_nomach_sh7(range)
    if _soviet_nomach_sh7x[1] <= range <= _soviet_nomach_sh7x[end]
        itp = linear_interpolation(_soviet_nomach_sh7x, _soviet_nomach_sh7y)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet No-Mach sh7 model"))
    end
end

function _soviet_ground(range)
    if _soviet_groundx[1] <= range <= _soviet_groundx[end]
        itp = linear_interpolation(_soviet_groundx, _soviet_groundy)
        return itp(range)
    else
        throw(ValueOutsideGraphError("Range $range outside valid bounds for Soviet ground model"))
    end
end

# Main Soviet overpressure functions
function _sovietmach(scale_height, ground_range)
    if 120 <= scale_height <= 200
        return lerp10(scale_height, 120, 200, _soviet_mach_sh12(ground_range), _soviet_mach_sh20(ground_range))
    elseif 70 <= scale_height < 120
        return lerp10(scale_height, 70, 120, _soviet_mach_sh7(ground_range), _soviet_mach_sh12(ground_range))
    elseif 0 <= scale_height < 70
        return lerp10(scale_height, 0, 70, _soviet_ground(ground_range), _soviet_mach_sh7(ground_range))
    else
        throw(ValueOutsideGraphError("Scale height $scale_height outside valid bounds (0-200m)"))
    end
end

function _sovietnomach(scale_height, ground_range)
    if 120 < scale_height <= 200
        return lerp10(scale_height, 120, 200, _soviet_nomach_sh12(ground_range), _soviet_nomach_sh20(ground_range))
    elseif 70 <= scale_height <= 120
        return lerp10(scale_height, 70, 120, _soviet_nomach_sh7(ground_range), _soviet_nomach_sh12(ground_range))
    elseif 0 <= scale_height < 70
        return lerp10(scale_height, 0, 70, _soviet_ground(ground_range), _soviet_nomach_sh7(ground_range))
    else
        throw(ValueOutsideGraphError("Scale height $scale_height outside valid bounds (0-200m)"))
    end
end

# Reverse (inverse) functions for finding range given overpressure
# Create reversed arrays for inverse interpolation
const _rsoviet_mach_sh20x = reverse(_soviet_mach_sh20x)
const _rsoviet_mach_sh20y = reverse(_soviet_mach_sh20y)
const _rsoviet_mach_sh12x = reverse(_soviet_mach_sh12x)
const _rsoviet_mach_sh12y = reverse(_soviet_mach_sh12y)
const _rsoviet_mach_sh7x = reverse(_soviet_mach_sh7x)
const _rsoviet_mach_sh7y = reverse(_soviet_mach_sh7y)
const _rsoviet_nomach_sh20x = reverse(_soviet_nomach_sh20x)
const _rsoviet_nomach_sh20y = reverse(_soviet_nomach_sh20y)
const _rsoviet_nomach_sh12x = reverse(_soviet_nomach_sh12x)
const _rsoviet_nomach_sh12y = reverse(_soviet_nomach_sh12y)
const _rsoviet_nomach_sh7x = reverse(_soviet_nomach_sh7x)
const _rsoviet_nomach_sh7y = reverse(_soviet_nomach_sh7y)
const _rsoviet_groundx = reverse(_soviet_groundx)
const _rsoviet_groundy = reverse(_soviet_groundy)

function _rsoviet_mach_sh20(overpressure)
    if _rsoviet_mach_sh20y[1] <= overpressure <= _rsoviet_mach_sh20y[end]
        itp = linear_interpolation(_rsoviet_mach_sh20y, _rsoviet_mach_sh20x)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsoviet_mach_sh12(overpressure)
    if _rsoviet_mach_sh12y[1] <= overpressure <= _rsoviet_mach_sh12y[end]
        itp = linear_interpolation(_rsoviet_mach_sh12y, _rsoviet_mach_sh12x)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsoviet_mach_sh7(overpressure)
    if _rsoviet_mach_sh7y[1] <= overpressure <= _rsoviet_mach_sh7y[end]
        itp = linear_interpolation(_rsoviet_mach_sh7y, _rsoviet_mach_sh7x)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsoviet_nomach_sh20(overpressure)
    if _rsoviet_nomach_sh20y[1] <= overpressure <= _rsoviet_nomach_sh20y[end]
        itp = linear_interpolation(_rsoviet_nomach_sh20y, _rsoviet_nomach_sh20x)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsoviet_nomach_sh12(overpressure)
    if _rsoviet_nomach_sh12y[1] <= overpressure <= _rsoviet_nomach_sh12y[end]
        itp = linear_interpolation(_rsoviet_nomach_sh12y, _rsoviet_nomach_sh12x)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsoviet_nomach_sh7(overpressure)
    if _rsoviet_nomach_sh7y[1] <= overpressure <= _rsoviet_nomach_sh7y[end]
        itp = linear_interpolation(_rsoviet_nomach_sh7y, _rsoviet_nomach_sh7x)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsoviet_ground(overpressure)
    if _rsoviet_groundy[1] <= overpressure <= _rsoviet_groundy[end]
        itp = linear_interpolation(_rsoviet_groundy, _rsoviet_groundx)
        return itp(overpressure)
    else
        throw(ValueOutsideGraphError("Overpressure $overpressure outside valid bounds"))
    end
end

function _rsovietnomach(scale_height, overpressure)
    logop = log10(overpressure)
    if scale_height >= 120 && overpressure > 2.975
        # Kludge for high overpressures above scale_height = 200
        l = x -> log10(_sovietnomach(scale_height, x))
        distances = [Float64(x*10) for x in range(10, 0, step=-1)]
        logvals = [l(d) for d in distances]
        itp = linear_interpolation(logvals, distances)
        return itp(logop)
    elseif 120 <= scale_height <= 200
        r12 = _rsoviet_nomach_sh12(logop)
        r20 = _rsoviet_nomach_sh20(logop)
        return r12 + (scale_height - 120) * (r20 - r12) / (200 - 120)
    elseif 70 <= scale_height < 120
        r7 = _rsoviet_nomach_sh7(logop)
        r12 = _rsoviet_nomach_sh12(logop)
        return r7 + (scale_height - 70) * (r12 - r7) / (120 - 70)
    elseif 0 <= scale_height < 70
        r0 = _rsoviet_ground(logop)
        r7 = _rsoviet_nomach_sh7(logop)
        return r0 + (scale_height - 0) * (r7 - r0) / (70 - 0)
    else
        throw(ValueOutsideGraphError("Scale height $scale_height outside valid bounds"))
    end
end

function _rsovietmach(scale_height, overpressure)
    logop = log10(overpressure)
    if scale_height >= 120 && overpressure > 2.2336
        # Kludge for high overpressures above scale_height = 200
        l = x -> log10(_sovietmach(scale_height, x))
        distances = [Float64(x*10) for x in range(17, 0, step=-1)]
        logvals = [l(d) for d in distances]
        itp = linear_interpolation(logvals, distances)
        return itp(logop)
    elseif 120 <= scale_height <= 200
        r12 = _rsoviet_mach_sh12(logop)
        r20 = _rsoviet_mach_sh20(logop)
        return r12 + (scale_height - 120) * (r20 - r12) / (200 - 120)
    elseif 70 <= scale_height < 120
        r7 = _rsoviet_mach_sh7(logop)
        r12 = _rsoviet_mach_sh12(logop)
        return r7 + (scale_height - 70) * (r12 - r7) / (120 - 70)
    elseif 0 <= scale_height < 70
        r0 = _rsoviet_ground(logop)
        r7 = _rsoviet_mach_sh7(logop)
        return r0 + (scale_height - 0) * (r7 - r0) / (70 - 0)
    else
        throw(ValueOutsideGraphError("Scale height $scale_height outside valid bounds"))
    end
end

# Public API functions
function soviet_overpressure(y, r, h; thermal_layer=true, yunits="kT", dunits="m", opunits="kg/cm^2")
    """Estimate peak static overpressure at radius r from the epicenter based on the
    graphs in the 1987 Soviet military publication _Iadernoe oruzhie: Posobie dlia ofitserov_.

    The most interesting feature of this model is that it provides for cases
    in which the Mach stem is suppressed by a thermal layer. This phenomenon was
    considered a largely theoretical 'second-order effect' among most US NWE researchers,
    but was observed in extreme forms in the USSR's atmospheric nuclear tests, leading
    them to conclude it would occur in many real-world military scenarios. To use a
    Soviet model with a Mach stem present, set the parameter thermal_layer to false."""
    yld = convert_units(y, yunits, "kT")
    gr = convert_units(r, dunits, "m")
    height = convert_units(h, dunits, "m")
    sr = scale_range(yld, gr)
    sh = scale_height(yld, height)
    if thermal_layer
        result = _sovietnomach(sh, sr)
    else
        result = _sovietmach(sh, sr)
    end
    return convert_units(result, "kg/cm^2", opunits)
end

function r_soviet_overpressure(y, op, h; thermal_layer=true, yunits="kT", dunits="m", opunits="kg/cm^2")
    """Estimate the radius from the epicenter at which peak static overpressure op
    will be experienced based on graphs in the 1987 Soviet military publication
    _Iadernoe oruzhie: Posobie dlia ofitserov_.

    The most interesting feature of this model is that it provides for cases
    in which the Mach stem is suppressed by a thermal layer. This phenomenon was
    considered a largely theoretical 'second-order effect' among most US NWE researchers,
    but was observed in extreme forms in the USSR's atmospheric nuclear tests, leading
    them to conclude it would occur in many real-world military scenarios. To use a
    Soviet model with a Mach stem present, set the parameter thermal_layer to false."""
    yld = convert_units(y, yunits, "kT")
    height = convert_units(h, dunits, "m")
    sh = scale_height(yld, height)
    overp = convert_units(op, opunits, "kg/cm^2")
    if thermal_layer
        range_m = _rsovietnomach(sh, overp)
    else
        range_m = _rsovietmach(sh, overp)
    end
    return convert_units(range_m, "m", dunits) * y^(1.0/3.0)
end
