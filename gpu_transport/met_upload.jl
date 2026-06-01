#!/usr/bin/env julia
# Met upload — load Nancy MET_CACHE into device-resident Float32 tapes.
# =====================================================================
# Each met window (1 hour) becomes one `GpuMetWindow`. The window holds
# 4D (x, y, σ, 2) tapes for u, v, w, t, p, hlevel, plus 3D (x, y, 2) tapes
# for ps and hbl. Sigma levels are sorted ascending (matches host shadow's
# convention) and replicated as a 1D CuArray per window for the kernel.
#
# All entries are Float32 — matches host_shadow_v2.jl's WindTapeF32 layout
# byte-for-byte.

using CUDA
using NuclearDetonation
using NuclearDetonation.Transport

struct GpuMetWindow
    u::CuArray{Float32,4}        # (nx, ny, nk, 2)
    v::CuArray{Float32,4}
    w::CuArray{Float32,4}
    t::CuArray{Float32,4}
    p::CuArray{Float32,4}
    hlevel::CuArray{Float32,4}
    ps::CuArray{Float32,3}       # (nx, ny, 2)
    hbl::CuArray{Float32,3}
    z_grid::CuArray{Float32,1}   # nk
    nx::Int
    ny::Int
    nk::Int
end

function load_nancy_gpu_windows()
    file_range_start = CACHE_START_FILE
    file_range_end = CACHE_END_FILE

    windows = GpuMetWindow[]
    for file_idx in file_range_start:file_range_end
        n_windows = 0
        for k in keys(MET_CACHE)
            if k[1] == file_idx
                n_windows = max(n_windows, k[2])
            end
        end
        n_windows = max(0, n_windows - 1)
        n_windows == 0 && continue

        for window_idx in 1:n_windows
            mf = MET_CACHE[(file_idx, window_idx)]
            push!(windows, build_gpu_window(mf))
        end
    end
    return windows
end

function build_gpu_window(mf::Transport.MeteoFields)
    nx, ny, nk = mf.nx, mf.ny, mf.nk
    z = copy(mf.vlevel)
    perm = sortperm(z)
    z_sorted = Float32.(z[perm])
    for i in 2:nk
        if z_sorted[i] <= z_sorted[i-1]
            z_sorted[i] = z_sorted[i-1] + Float32(eps(Float32) * 10)
        end
    end
    permuted = perm != collect(1:nk)

    pull3 = (a1, a2, default::Float32) -> begin
        out = Array{Float32,4}(undef, nx, ny, nk, 2)
        if permuted
            @views begin
                out[:, :, :, 1] .= a1[:, :, perm]
                out[:, :, :, 2] .= a2[:, :, perm]
            end
        else
            out[:, :, :, 1] .= a1
            out[:, :, :, 2] .= a2
        end
        replace!(v -> isnan(v) ? default : v, out)
        out
    end

    u = pull3(mf.u1, mf.u2, 0.0f0)
    v = pull3(mf.v1, mf.v2, 0.0f0)
    w = pull3(mf.w1, mf.w2, 0.0f0)
    t = pull3(mf.t1, mf.t2, 288.15f0)
    replace!(val -> val < 100.0f0 ? 288.15f0 : val, t)
    p = pull3(mf.p1, mf.p2, 1013.25f0)
    replace!(val -> val == 0.0f0 ? 1013.25f0 : val, p)
    h = pull3(mf.hlevel1, mf.hlevel2, 9999.0f0)

    ps = Array{Float32,3}(undef, nx, ny, 2)
    ps[:, :, 1] .= mf.ps1
    ps[:, :, 2] .= mf.ps2
    replace!(val -> isnan(val) ? 1013.25f0 : val, ps)

    hbl = Array{Float32,3}(undef, nx, ny, 2)
    hbl[:, :, 1] .= mf.hbl1
    hbl[:, :, 2] .= mf.hbl2
    replace!(val -> isnan(val) ? 0.0f0 : val, hbl)

    return GpuMetWindow(
        CuArray(u), CuArray(v), CuArray(w), CuArray(t),
        CuArray(p), CuArray(h),
        CuArray(ps), CuArray(hbl),
        CuArray(z_sorted),
        nx, ny, nk,
    )
end
