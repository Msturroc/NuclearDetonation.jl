# Tests for numerical_config.jl: config factories, dispatch, and solver kwargs
#
# These exercise the lightweight glue between user-facing config and OrdinaryDiffEq.

using Test
using OrdinaryDiffEq

const _Tx_n = NuclearDetonation.Transport

@testset "numerical_config: factories and dispatch" begin

    @testset "NumericalConfig defaults" begin
        cfg = _Tx_n.NumericalConfig{Float64}()
        @test cfg.interpolation_order == _Tx_n.LinearInterp
        @test cfg.ode_solver_type == :Euler
        @test cfg.fixed_dt == 300.0
        @test cfg.reltol > 0 && cfg.abstol > 0
        @test cfg.name == "default"
    end

    @testset "ERA5NumericalConfig defaults" begin
        cfg = _Tx_n.ERA5NumericalConfig{Float64}()
        @test cfg.interpolation_order == _Tx_n.LinearInterp
        @test cfg.ode_solver_type == :Euler
        @test cfg.turbulence == _Tx_n.OrnsteinUhlenbeck
        @test cfg.store_turbulent_velocities == true
        @test cfg.fixed_dt == 300.0
    end

    @testset "ValidationMode / ModernMode factories" begin
        v = _Tx_n.ValidationMode(120.0)
        @test v.ode_solver_type == :Euler
        @test v.fixed_dt == 120.0
        @test v.interpolation_order == _Tx_n.ReferenceInterp
        @test v.name == "validation"

        v_lin = _Tx_n.ValidationMode(60.0; name="lin", use_reference_interp=false)
        @test v_lin.interpolation_order == _Tx_n.LinearInterp
        @test v_lin.name == "lin"

        m = _Tx_n.ModernMode(Float64)
        @test m.ode_solver_type == :AutoTsit5
        @test m.interpolation_order == _Tx_n.CubicInterp
        @test m.fixed_dt === nothing  # adaptive

        m_fixed = _Tx_n.ModernMode(Float64; fixed_dt=200.0, solver=:Tsit5,
                                  interpolation=_Tx_n.LinearInterp,
                                  reltol=1e-4, abstol=1e-6, name="custom")
        @test m_fixed.fixed_dt == 200.0
        @test m_fixed.ode_solver_type == :Tsit5
        @test m_fixed.interpolation_order == _Tx_n.LinearInterp
        @test m_fixed.reltol == 1e-4
        @test m_fixed.abstol == 1e-6
        @test m_fixed.name == "custom"
    end

    @testset "ERA5ValidationMode / ERA5ModernMode factories" begin
        v = _Tx_n.ERA5ValidationMode(150.0)
        @test v.ode_solver_type == :Euler
        @test v.fixed_dt == 150.0
        @test v.turbulence == _Tx_n.OrnsteinUhlenbeck
        @test v.interpolation_order == _Tx_n.LinearInterp

        m = _Tx_n.ERA5ModernMode(Float64)
        @test m.ode_solver_type == :Tsit5
        @test m.interpolation_order == _Tx_n.CubicInterp
        @test m.fixed_dt == 300.0

        m_adaptive = _Tx_n.ERA5ModernMode(Float64; fixed_dt=nothing,
                                          solver=:AutoTsit5,
                                          turbulence=_Tx_n.HannaTurbulence)
        @test m_adaptive.fixed_dt === nothing
        @test m_adaptive.ode_solver_type == :AutoTsit5
        @test m_adaptive.turbulence == _Tx_n.HannaTurbulence
    end

    @testset "get_ode_solver dispatch" begin
        cfg_e = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:Euler)
        cfg_t = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:Tsit5)
        cfg_a = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:AutoTsit5)
        @test _Tx_n.get_ode_solver(cfg_e) isa Euler
        @test _Tx_n.get_ode_solver(cfg_t) isa Tsit5
        @test _Tx_n.get_ode_solver(cfg_a) isa Tsit5

        cfg_bad = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:Unknown)
        @test_throws ErrorException _Tx_n.get_ode_solver(cfg_bad)

        # ERA5 dispatch mirrors NumericalConfig dispatch.
        ecfg_e = _Tx_n.ERA5NumericalConfig{Float64}(ode_solver_type=:Euler)
        ecfg_t = _Tx_n.ERA5NumericalConfig{Float64}(ode_solver_type=:Tsit5)
        @test _Tx_n.get_ode_solver(ecfg_e) isa Euler
        @test _Tx_n.get_ode_solver(ecfg_t) isa Tsit5
        @test_throws ErrorException _Tx_n.get_ode_solver(
            _Tx_n.ERA5NumericalConfig{Float64}(ode_solver_type=:Bogus))
    end

    @testset "get_interpolation_scheme dispatch" begin
        cfg_lin = _Tx_n.NumericalConfig{Float64}(interpolation_order=_Tx_n.LinearInterp)
        cfg_cub = _Tx_n.NumericalConfig{Float64}(interpolation_order=_Tx_n.CubicInterp)
        @test _Tx_n.get_interpolation_scheme(cfg_lin) !== nothing
        @test _Tx_n.get_interpolation_scheme(cfg_cub) !== nothing

        # ReferenceInterp is not handled by Interpolations.jl scheme dispatch.
        cfg_ref = _Tx_n.NumericalConfig{Float64}(interpolation_order=_Tx_n.ReferenceInterp)
        @test_throws ErrorException _Tx_n.get_interpolation_scheme(cfg_ref)

        # ERA5 path
        ecfg_lin = _Tx_n.ERA5NumericalConfig{Float64}(interpolation_order=_Tx_n.LinearInterp)
        ecfg_cub = _Tx_n.ERA5NumericalConfig{Float64}(interpolation_order=_Tx_n.CubicInterp)
        @test _Tx_n.get_interpolation_scheme(ecfg_lin) !== nothing
        @test _Tx_n.get_interpolation_scheme(ecfg_cub) !== nothing
        ecfg_ref = _Tx_n.ERA5NumericalConfig{Float64}(interpolation_order=_Tx_n.ReferenceInterp)
        @test_throws ErrorException _Tx_n.get_interpolation_scheme(ecfg_ref)
    end

    @testset "get_solve_kwargs dispatch" begin
        # Euler: fixed dt, non-adaptive
        cfg_e = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:Euler, fixed_dt=120.0)
        kw = _Tx_n.get_solve_kwargs(cfg_e)
        @test kw[:dt] == 120.0
        @test kw[:adaptive] == false

        # Tsit5: fixed dt, non-adaptive
        cfg_t = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:Tsit5, fixed_dt=60.0)
        kw_t = _Tx_n.get_solve_kwargs(cfg_t)
        @test kw_t[:dt] == 60.0
        @test kw_t[:adaptive] == false

        # AutoTsit5 with fixed_dt as dtmax
        cfg_a = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:AutoTsit5,
                                              fixed_dt=300.0,
                                              reltol=1e-5, abstol=1e-7)
        kw_a = _Tx_n.get_solve_kwargs(cfg_a)
        @test kw_a[:adaptive] == true
        @test kw_a[:reltol] == 1e-5
        @test kw_a[:abstol] == 1e-7
        @test kw_a[:dtmax] == 300.0

        # AutoTsit5 with no dtmax
        cfg_an = _Tx_n.NumericalConfig{Float64}(ode_solver_type=:AutoTsit5, fixed_dt=nothing)
        kw_an = _Tx_n.get_solve_kwargs(cfg_an)
        @test kw_an[:adaptive] == true
        @test !haskey(kw_an, :dtmax)

        # ERA5 dispatch parity
        ecfg_e = _Tx_n.ERA5NumericalConfig{Float64}(ode_solver_type=:Euler, fixed_dt=200.0)
        kw_ee = _Tx_n.get_solve_kwargs(ecfg_e)
        @test kw_ee[:dt] == 200.0
        @test kw_ee[:adaptive] == false

        ecfg_a = _Tx_n.ERA5NumericalConfig{Float64}(ode_solver_type=:AutoTsit5,
                                                   fixed_dt=400.0,
                                                   reltol=1e-4, abstol=1e-6)
        kw_ea = _Tx_n.get_solve_kwargs(ecfg_a)
        @test kw_ea[:adaptive] == true
        @test kw_ea[:dtmax] == 400.0
    end

    @testset "create_numerical_config / create_era5_numerical_config" begin
        @test _Tx_n.create_numerical_config(:validation).ode_solver_type == :Euler
        @test _Tx_n.create_numerical_config(:modern).ode_solver_type == :AutoTsit5
        @test_throws ErrorException _Tx_n.create_numerical_config(:nonsense)

        @test _Tx_n.create_era5_numerical_config(:baseline).ode_solver_type == :Euler
        @test _Tx_n.create_era5_numerical_config(:validation).ode_solver_type == :Euler
        @test _Tx_n.create_era5_numerical_config(:modern).ode_solver_type == :Tsit5
        @test _Tx_n.create_era5_numerical_config(:enhanced).ode_solver_type == :Tsit5
        @test_throws ErrorException _Tx_n.create_era5_numerical_config(:bogus)
    end
end

println("✓ All numerical_config tests passed!")
