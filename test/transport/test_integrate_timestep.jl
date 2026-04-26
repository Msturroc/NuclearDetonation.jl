# Tests for integrate_timestep! in orchestration.jl: end-to-end advection of a single particle
# through one timestep, with all turbulence/deposition/decay branches disabled or empty.

using Test
using StaticArrays

const _Tx_it = NuclearDetonation.Transport

# Build a uniform meteorology fixture suitable for advection: 5 m/s eastward wind,
# realistic hybrid coordinates, populated geopotential heights.
function _make_advect_met(; nx=12, ny=12, nk=6, T_K=288.15, ps_hpa=1013.25, u_ms=5.0)
    met = MeteoFields(nx, ny, nk; T=Float64)
    met.u1 .= u_ms; met.u2 .= u_ms
    met.v1 .= 0.0;  met.v2 .= 0.0
    met.w1 .= 0.0;  met.w2 .= 0.0
    met.t1 .= T_K;  met.t2 .= T_K
    met.ps1 .= ps_hpa; met.ps2 .= ps_hpa

    # Half-level pressures, surface→top.
    p_half = collect(range(ps_hpa, 50.0; length=nk + 1))
    met.ahalf .= p_half; met.bhalf .= 0.0
    met.alevel .= [0.5 * (p_half[k] + p_half[k+1]) for k in 1:nk]
    met.blevel .= 0.0
    met.p1 .= reshape(met.alevel, 1, 1, nk)
    met.p2 .= reshape(met.alevel, 1, 1, nk)

    # Ascending sigma 0 (top) → 1 (surface).
    met.vlevel .= collect(range(0.0, 1.0; length=nk))
    met.vhalf  .= collect(range(0.0, 1.0; length=nk + 1))

    # Plausible geopotential heights for an isothermal atmosphere — surface 0, top ~20 km.
    h_levels = collect(range(20_000.0, 0.0; length=nk))  # decreasing from top to surface
    for k in 1:nk
        met.hlevel1[:, :, k] .= h_levels[k]
        met.hlevel2[:, :, k] .= h_levels[k]
    end

    met.garea .= 1.0; met.xm .= 1.0; met.ym .= 1.0
    return met
end

function _make_advect_state(; nx_dom=10, ny_dom=10, nz=5, lon_min=0.0, lon_max=10.0,
                              lat_min=0.0, lat_max=10.0)
    dx, dy = 1000.0, 1000.0
    hlevel = [0.0, 100.0, 500.0, 1000.0, 2000.0]
    xm = ones(Float64, nx_dom, ny_dom)
    ym = ones(Float64, nx_dom, ny_dom)

    t_start = _Tx_it.DateTime(2025, 1, 1, 0)
    t_end   = _Tx_it.DateTime(2025, 1, 1, 6)
    dt_output = _Tx_it.Duration(0, 1, 0, 0)
    dt_met    = _Tx_it.Duration(0, 1, 0, 0)

    domain = _Tx_it.SimulationDomain(nx_dom, ny_dom, nz, dx, dy, hlevel, xm, ym,
                                     t_start, t_end, dt_output, dt_met;
                                     lon_min=lon_min, lon_max=lon_max,
                                     lat_min=lat_min, lat_max=lat_max)

    sources = _Tx_it.ReleaseSource{Float64}[]  # no auto-released particles
    component_names = ["Cs137"]
    decay_params = [_Tx_it.DecayParams{Float64}(kdecay=_Tx_it.NoDecay)]

    state = _Tx_it.initialize_simulation(domain, sources, component_names, decay_params)

    # Add a single particle near the centre at sigma=0.5 (mid-altitude).
    pos  = SVector{3,Float64}(5.0, 5.0, 0.5)
    vel  = SVector{3,Float64}(0.0, 0.0, 0.0)
    mass = [1.0e10]
    _Tx_it.add_particle!(state.ensemble, pos, vel, mass, 0.0)
    return state
end

@testset "integrate_timestep! advection-only smoke" begin

    @testset "Single particle advects under uniform u-wind" begin
        met = _make_advect_met(; u_ms=5.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)
        state = _make_advect_state()
        @test length(state.ensemble.particles) == 1
        x0 = state.ensemble.positions[1][1]
        y0 = state.ensemble.positions[1][2]

        psc = _Tx_it.ParticleSizeConfig()
        # apply_dry_deposition default true, but dry_enabled=false flag in the call below
        # will gate it off.
        dc = _Tx_it.DepositionConfig{Float64}()
        decay = [_Tx_it.DecayParams{Float64}(kdecay=_Tx_it.NoDecay)]
        sc = _Tx_it.SimulationConfig{Float64}(verbose=false)
        oc = _Tx_it.OutputConfig(trace_enabled=false)

        dt = 600.0
        n_dep = _Tx_it.integrate_timestep!(
            state, winds, dt, psc, dc, decay, sc;
            advection_enabled=true,
            settling_enabled=false,
            dry_enabled=false,
            wet_enabled=false,
            output_config=oc,
        )

        # No deposition / decay → all particles still active.
        @test n_dep == 0
        @test length(state.ensemble.particles) == 1
        # Position should have moved in +x (uniform eastward wind).
        x1 = state.ensemble.positions[1][1]
        y1 = state.ensemble.positions[1][2]
        @test x1 != x0
        @test x1 > x0
        @test isapprox(y1, y0; atol=1e-8)
    end

    @testset "Deposition setup blocks execute with empty size_bins" begin
        # Even with empty ParticleSizeConfig.size_bins, the dry_enabled=true path exercises
        # the dry_active && !use_simple_deposition guard and the temperature_field /
        # pressure_field / precip_field fill loops.
        met = _make_advect_met(; u_ms=5.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)
        state = _make_advect_state()
        psc = _Tx_it.ParticleSizeConfig()
        dc = _Tx_it.DepositionConfig{Float64}(apply_wet_deposition=true)
        decay = [_Tx_it.DecayParams{Float64}(kdecay=_Tx_it.NoDecay)]
        sc = _Tx_it.SimulationConfig{Float64}(verbose=false)
        oc = _Tx_it.OutputConfig(trace_enabled=false)

        n_dep = _Tx_it.integrate_timestep!(
            state, winds, 600.0, psc, dc, decay, sc;
            advection_enabled=true,
            settling_enabled=false,
            dry_enabled=true,
            wet_enabled=true,
            output_config=oc,
        )
        @test n_dep == 0
        @test length(state.ensemble.particles) == 1
    end

    @testset "Reference stepping branch (Heun)" begin
        met = _make_advect_met(; u_ms=5.0)
        winds = create_wind_interpolants(met, 0.0, 3600.0;
                                         lon_min=0.0, lon_max=10.0,
                                         lat_min=0.0, lat_max=10.0)
        state = _make_advect_state()
        psc = _Tx_it.ParticleSizeConfig()
        dc = _Tx_it.DepositionConfig{Float64}()
        decay = [_Tx_it.DecayParams{Float64}(kdecay=_Tx_it.NoDecay)]
        sc = _Tx_it.SimulationConfig{Float64}(verbose=false, use_reference_stepping=true)
        oc = _Tx_it.OutputConfig(trace_enabled=false)

        x0 = state.ensemble.positions[1][1]
        n_dep = _Tx_it.integrate_timestep!(
            state, winds, 600.0, psc, dc, decay, sc;
            advection_enabled=true,
            settling_enabled=false,
            dry_enabled=false,
            wet_enabled=false,
            output_config=oc,
        )
        @test n_dep == 0
        x1 = state.ensemble.positions[1][1]
        @test x1 > x0
    end
end

println("✓ All integrate_timestep! tests passed!")
