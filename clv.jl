module CLV

export clv

using LinearAlgebra
using DynamicalSystems
using StaticArrays

# --- QR with positive diagonal, THIN (m×n with n ≪ m) -----------------------
@inline function _thin_qr_pos(Z::AbstractMatrix)
    F = qr(Z)                                 # no pivoting
    m, n = size(Z)
    Q = Matrix(F.Q)[:, 1:n]                   # thin Q
    R = Matrix(F.R)[1:n, 1:n]                 # thin R
    @inbounds for j in 1:n
        if R[j,j] < 0
            @views R[j, :] .*= -1
            @views Q[:, j] .*= -1
        end
    end
    return Q, R
end

"""
    clv(ds; nclv, dt, nstore, nspend_att=1000, nspend_fwd=1000, nspend_bkw=1000;
        reltol=1e-9, abstol=1e-9)

Compute Covariant Lyapunov Vectors for a `DynamicalSystem` `ds` using
the Ginelli two-pass algorithm.

Returns `(Gamma, Gs, xs, ts)` where:
- `Gamma[i]::Matrix{Float64}` is `N×nclv`: physical CLVs at stored time `i`.
- `Gs[i]::Matrix{Float64}` is `N×nclv`: the Gram–Schmidt basis at time `i`.
- `xs[i]`: state at time `i` (same type as system state).
- `ts[i]::Float64`: time stamp, spaced by `dt` from the start of the stored window.

Notes:
- `reltol` and `abstol` are accepted for API compatibility but are not used by
  `step!` from DynamicalSystems.jl; tolerances are controlled when constructing `ds`.
"""
function clv(
    ds::DynamicalSystem;
    nclv::Integer,
    dt::Real,
    nstore::Integer,
    nspend_att::Integer = 1000,
    nspend_fwd::Integer = 1000,
    nspend_bkw::Integer = 1000,
    reltol::Real = 1e-9,    # accepted, unused
    abstol::Real = 1e-9,    # accepted, unused
)
    # --- setup ---------------------------------------------------------------
    m  = DynamicalSystems.dimension(ds)
    n  = Int(nclv)
    n  > m && error("nclv ($n) cannot exceed system dimension ($m).")

    # Burn-in to the attractor with a fresh random state
    u0_proto = DynamicalSystems.get_state(ds)
    u0 = typeof(u0_proto)(rand(m))
    DynamicalSystems.reinit!(ds, u0)
    for _ in 1:nspend_att
        step!(ds, dt)
    end

    # Initialize GS basis for tangent integrator
    Q0 = u0_proto isa SVector ? SMatrix{m,n}(I) : Matrix{Float64}(I, m, n)
    tint = DynamicalSystems.tangent_integrator(ds, Q0)

    # --- forward transient: converge GS directions --------------------------
    for _ in 1:nspend_fwd
        step!(tint, dt)
        Z = DynamicalSystems.get_deviations(tint)     # m×n
        Q, _ = _thin_qr_pos(Z)
        DynamicalSystems.set_deviations!(tint, Q)
    end

    # --- continue forward: store R everywhere and Q on the kept window ------
    total = nstore + nspend_bkw
    R_hist = Vector{Matrix{Float64}}(undef, total)    # n×n upper-triangular
    G_hist_keep = Vector{Matrix{Float64}}(undef, nstore)  # m×n
    x_hist_full = Vector{typeof(u0_proto)}(undef, total)
    t_hist_full = Vector{Float64}(undef, total)

    for i in 1:total
        step!(tint, dt)
        Z = DynamicalSystems.get_deviations(tint)     # J*Q_prev
        Q, R = _thin_qr_pos(Z)
        DynamicalSystems.set_deviations!(tint, Q)

        if i <= nstore
            G_hist_keep[i] = Q
        end
        R_hist[i] = R
        x_hist_full[i] = DynamicalSystems.get_state(tint)
        t_hist_full[i] = i*dt
    end

    # --- backward pass on coefficient matrices C ----------------------------
    C = Matrix{Float64}(I, n, n)  # any nonsingular upper-triangular works

    # discard backward-transient by evolving C back over the tail
    for i in total:-1:(nstore+1)
        C = UpperTriangular(R_hist[i]) \ C
        @inbounds for j in 1:n
            s = norm(@view C[:, j])
            C[:, j] ./= s
        end
    end

    # now traverse the kept window, assembling physical CLVs V = Q*C
    Gamma = Vector{Matrix{Float64}}(undef, nstore)
    Gs    = Vector{Matrix{Float64}}(undef, nstore)
    xs    = Vector{typeof(u0_proto)}(undef, nstore)
    ts    = Vector{Float64}(undef, nstore)

    for i in nstore:-1:1
        Q = G_hist_keep[i]
        Gamma[i] = Q * C
        Gs[i]    = Q
        xs[i]    = x_hist_full[i]
        ts[i]    = t_hist_full[i]

        C = UpperTriangular(R_hist[i]) \ C
        @inbounds for j in 1:n
            s = norm(@view C[:, j])
            C[:, j] ./= s
        end
    end

    return Gamma, Gs, xs, ts
end

end # module
