
module Utilities

export ValueOutsideGraphError, UnknownUnitError, convert_units, dict_reverse

struct ValueOutsideGraphError <: Exception
    value::Any
end

struct UnknownUnitError <: Exception
    value::Any
end

function Base.showerror(io::IO, e::ValueOutsideGraphError)
    print(io, "ValueOutsideGraphError: ", e.value, " is outside the graph bounds.")
end

function Base.showerror(io::IO, e::UnknownUnitError)
    print(io, "UnknownUnitError: The unit conversion (", e.value[1], " to ", e.value[2], ") is unknown.")
end

function convert_units(v, unitsfrom, unitsto)
    if unitsfrom == unitsto
        return v
    # yield
    elseif unitsfrom == "kT" && unitsto == "MT"
        return v / 1000.0
    elseif unitsfrom == "MT" && unitsto == "kT"
        return v * 1000.0
    # distance
    elseif unitsfrom == "m" && unitsto == "kilofeet"
        return v / 304.8
    elseif unitsfrom == "m" && unitsto == "km"
        return v / 1000.0
    elseif unitsfrom == "km" && unitsto == "m"
        return v * 1000.0
    elseif unitsfrom == "kilofeet" && unitsto == "m"
        return 304.8 * v
    elseif unitsfrom == "yards" && unitsto == "m"
        return v / 1.09361
    elseif unitsfrom == "m" && unitsto == "yards"
        return v * 1.09361
    elseif unitsfrom == "ft" && unitsto == "m"
        return v * 0.3048
    elseif unitsfrom == "m" && unitsto == "ft"
        return v / 0.3048
    elseif unitsfrom == "kilofeet" && unitsto == "km"
        return convert_units(v, "kilofeet", "m") / 1000.0
    elseif unitsfrom == "kilofeet" && unitsto == "mi"
        return v / 5.28
    elseif unitsfrom == "mi" && unitsto == "km"
        return v * 1.60934
    elseif unitsfrom == "km" && unitsto == "mi"
        return v / 1.60934
    elseif unitsfrom == "km" && unitsto == "kilofeet"
        return v / 0.3048
    elseif unitsfrom == "yards" && unitsto == "meters"
        return v * 0.9144
    elseif unitsfrom == "yards" && unitsto == "km"
        return v * 0.0009144
    elseif unitsfrom == "meters" && unitsto == "yards"
        return v / 0.9144
    elseif unitsfrom == "km" && unitsto == "yards"
        return v / 0.0009144
    #pressure
    elseif unitsfrom == "psi" && unitsto == "kg/cm^2"
        return v * 0.070307
    elseif unitsfrom == "kg/cm^2" && unitsto == "psi"
        return v / 0.070307
    elseif unitsfrom == "MPa" && unitsto == "psi"
        return v * 145.037738
    elseif unitsfrom == "psi" && unitsto == "MPa"
        return v / 145.037738
    elseif unitsfrom == "kg/cm^2" && unitsto == "MPa"
        return convert_units(convert_units(v, "kg/cm^2", "psi"), "psi", "MPa")
    elseif unitsfrom == "MPa" && unitsto == "kg/cm^2"
        return convert_units(convert_units(v, "psi", "kg/cm^2"), "MPa", "psi")
    elseif unitsfrom == "Pa"
        return convert_units(v, "MPa", unitsto) / 1e6
    elseif unitsto == "Pa"
        return convert_units(v, unitsfrom, "MPa") * 1e6
    # speed
    elseif unitsfrom == "m/s" && unitsto == "mph"
        return v * 2.23694
    elseif unitsfrom == "mph" && unitsto == "m/s"
        return v / 2.23694
    elseif unitsfrom == "m/s" && unitsto == "km/h"
        return v * 3.6
    elseif unitsfrom == "km/h" && unitsto == "m/s"
        return v / 3.6
    elseif unitsfrom == "mph" && unitsto == "km/h"
        return v * 1.60934
    elseif unitsfrom == "km/h" && unitsto == "mph"
        return v / 1.60934
    # wind shear
    elseif unitsfrom == "m/s-km" && unitsto == "mph/kilofoot"
        return v * 0.13625756613945836
    # dose
    elseif unitsfrom == "Roentgen" && unitsto == "Sv"
        return v / 100.0
    else
        throw(UnknownUnitError((unitsfrom, unitsto)))
    end
end

function dict_reverse(d)
    new_dict = Dict()
    for (k, v) in d
        new_dict[k] = reverse(v)
    end
    return new_dict
end

end
