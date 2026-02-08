module Fallout

using ..Utilities
using SpecialFunctions
using Distributions
using QuadGK
using CoordinateTransformations
using StaticArrays

export WSEG10, g, phi, D_Hplus1, fallouttoa, dose

struct WSEG10
    translation::Translation
    wd::Number
    yld::Number
    ff::Number
    wind::Number
    shear::Number
    tob::Number
    H_c::Number
    s_0::Number
    s_02::Number
    s_h::Number
    T_c::Number
    L_0::Number
    L_02::Number
    s_x2::Number
    s_x::Number
    L_2::Number
    L::Number
    n::Number
    a_1::Number
end

function WSEG10(gzx, gzy, yld, ff, wind, wd, shear; tob=0, dunits="km", wunits="km/h", shearunits="m/s-km", yunits="kT")
    translation = Translation(-convert_units(gzx, dunits, "mi"), -convert_units(gzy, dunits, "mi"))
    yld_MT = convert_units(yld, yunits, "MT")
    wind_mph = convert_units(wind, wunits, "mph")
    shear_mph_kf = convert_units(shear, shearunits, "mph/kilofoot")

    d = log(yld_MT) + 2.42
    H_c = 44 + 6.1 * log(yld) - 0.205 * abs(d) * d
    lnyield = log(yld_MT)
    s_0 = exp(0.7 + lnyield / 3 - 3.25 / (4.0 + (lnyield + 5.4)^2))
    s_02 = s_0^2
    s_h = 0.18 * H_c
    T_c = 1.0573203 * (12 * (H_c / 60) - 2.5 * (H_c / 60)^2) * (1 - 0.5 * exp(-1 * (H_c / 25)^2))
    L_0 = wind_mph * T_c
    L_02 = L_0^2
    s_x2 = s_02 * (L_02 + 8 * s_02) / (L_02 + 2 * s_02)
    s_x = sqrt(s_x2)
    L_2 = L_02 + 2 * s_x2
    L = sqrt(L_2)
    n = (ff * L_02 + s_x2) / (L_02 + 0.5 * s_x2)
    a_1 = 1 / (1 + ((0.001 * H_c * wind_mph) / s_0))

    WSEG10(translation, wd, yld_MT, ff, wind_mph, shear_mph_kf, tob, H_c, s_0, s_02, s_h, T_c, L_0, L_02, s_x2, s_x, L_2, L, n, a_1)
end

function g(model::WSEG10, x)
    return exp(-(abs(x) / model.L)^model.n) / (model.L * SpecialFunctions.gamma(1 + 1 / model.n))
end

function phi(model::WSEG10, x)
    w = (model.L_0 / model.L) * (x / (model.s_x * model.a_1))
    return cdf(Normal(), w)
end

function D_Hplus1(model::WSEG10, x, y; dunits="km", doseunits="Sv")
    rx_ry = model.translation(SVector(convert_units(x, dunits, "mi"), convert_units(y, dunits, "mi")))
    rotr = LinearMap(Angle2d(deg2rad(-model.wd + 270))) # Create a rotation matrix
    rx, ry = rotr(rx_ry)

    f_x = model.yld * 2e6 * phi(model, rx) * g(model, rx) * model.ff
    s_y = sqrt(model.s_02 + ((8 * abs(rx + 2 * model.s_x) * model.s_02) / model.L) + (2 * (model.s_x * model.T_c * model.s_h * model.shear)^2 / model.L_2) + (((rx + 2 * model.s_x) * model.L_0 * model.T_c * model.s_h * model.shear)^2 / model.L^4))
    a_2 = 1 / (1 + ((0.001 * model.H_c * model.wind) / model.s_0) * (1 - cdf(Normal(), 2 * x / model.wind)))
    f_y = exp(-0.5 * (ry / (a_2 * s_y))^2) / (2.5066282746310002 * s_y)
    return convert_units(f_x * f_y, "Roentgen", doseunits)
end

function fallouttoa(model::WSEG10, x)
    T_1 = 1.0
    return sqrt(0.25 + (model.L_02 * (x + 2 * model.s_x2) * model.T_c^2) / (model.L_2 * (model.L_02 + 0.5 * model.s_x2)) + ((2 * model.s_x2 * T_1^2) / (model.L_02 + 0.5 * model.s_x)))
end

function dose(model::WSEG10, x, y; dunits="km", doseunits="Sv")
    rx_ry = model.translation(SVector(convert_units(x, dunits, "mi"), convert_units(y, dunits, "mi")))
    rotr = LinearMap(Angle2d(deg2rad(-model.wd + 270)))
    rx, ry = rotr(rx_ry)

    t_a = fallouttoa(model, rx)
    bio = exp(-(0.287 + 0.52 * log(t_a / 31.6) + 0.04475 * log((t_a / 31.6)^2)))
    return D_Hplus1(model, x, y, dunits=dunits, doseunits=doseunits) * bio
end

end