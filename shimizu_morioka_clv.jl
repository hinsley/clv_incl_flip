using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("clv.jl")
using .CLV
using DynamicalSystems
using LinearAlgebra
using GLMakie
using Random
using StaticArrays

# 1) System
# Shimizu–Morioka system (standard form):
# \dot{x} = y
# \dot{y} = x - λ y - x z
# \dot{z} = -α z + x^2
function shimizu_morioka_eom!(du, u, p, t)
    λ, α, s = p
    x, y, z = u
    # Scaled system: u = s * v, so du/dt = s * f(u/s).
    xs = x / s
    ys = y / s
    zs = z / s
    du[1] = s * ys
    du[2] = s * (xs - λ * ys - xs * zs)
    du[3] = s * (-α * zs + xs^2)
    return nothing
end

# Parameter set selector.
# pseudohyperbolic = true  → Lorenz-like (pseudohyperbolic) regime.
# pseudohyperbolic = false → quasiattractor (quasistochastic) regime.
pseudohyperbolic = true

if pseudohyperbolic
    λ = 0.85   # ≈ Lorenz-like per literature
    α  = 0.50
else
    λ = 0.555   # ≈ quasistochastic regime near onset
    α  = 0.45
end

state_scale = 20.0
u0 = SVector{3, Float64}(state_scale * 0.1, 0.0, 0.0)
p  = (λ, α, state_scale)
ds = ContinuousDynamicalSystem(shimizu_morioka_eom!, u0, p)

# 2) Parameters
nclv       = 3
t_renorm   = 0.05
dt_plot    = 1e-2
nstore     = Int(1e4)
nspend_att = Int(1e5)
nspend_fwd = Int(1e5)
nspend_bkw = Int(1e5)

Random.seed!(0)

# 3) Compute CLVs
println("Computing Covariant Lyapunov Vectors…")
Gamma, Gs, xs, ts = clv(ds;
    nclv       = nclv,
    dt         = t_renorm,
    nstore     = nstore,
    nspend_att = nspend_att,
    nspend_fwd = nspend_fwd,
    nspend_bkw = nspend_bkw,
    reltol     = 1e-9,
    abstol     = 1e-9,
)
println("Done. Stored $(length(Gamma)) samples.")

# 4) Background trajectory for plotting
u0_s   = xs[1]
ds_bg  = ContinuousDynamicalSystem(shimizu_morioka_eom!, u0_s, p)
DynamicalSystems.reinit!(ds_bg, u0_s)

t_full = ts[end] - ts[1]
K      = max(0, Int(round(t_full / dt_plot)))
trajectory_dense = Vector{Point3f}(undef, K + 1)
trajectory_dense[1] = Point3f(u0_s)
for k in 1:K
    step!(ds_bg, dt_plot)
    trajectory_dense[k+1] = Point3f(DynamicalSystems.get_state(ds_bg))
end

# 5) Plot attractor and CLVs at stored points
fig = Figure(size=(900, 900))
regime_str = pseudohyperbolic ? "pseudohyperbolic (Lorenz-like)" : "quasiattractor"
ax = Axis3(fig[1,1],
    title="Shimizu–Morioka (λ=$(round(λ, digits=3)), α=$(round(α, digits=3))) — " * regime_str,
    xlabel="x", ylabel="y", zlabel="z",
)

# Time series plot of θ(t) below the 3D plot.
ax2 = Axis(fig[2,1],
    title="Minimum angle: CLV3 vs span{CLV1,CLV2}",
    xlabel="t", ylabel="θ (rad)",
    yticks=0:π/4:π, ytickformat=xs -> ["0", "π/4", "π/2", "3π/4", "π"][1:length(xs)]
)

lines!(ax, trajectory_dense, linewidth=0.3)

stride     = 20
scale      = 2.5
clv_colors = (:red, :orange, :green)

origins = Point3f.(xs[1:stride:end])

# Compute min angle θ(t) between CLV 3 and span{CLV 1, CLV 2}.
times_rel = [t - ts[1] for t in ts]
angles = Vector{Float64}(undef, length(Gamma))
for i in 1:length(Gamma)
    v1 = Gamma[i][:, 1]
    v2 = Gamma[i][:, 2]
    v3 = Gamma[i][:, 3]

    # Orthonormal basis of span{v1, v2} via QR.
    A = Matrix(hcat(v1, v2))
    F = qr(A)
    Q = Matrix(F.Q)[:, 1:2]

    # Projection of v3 onto the plane and corresponding angle.
    proj_v3 = Q * (Q' * v3)
    cos_alpha = norm(proj_v3) / norm(v3)
    cos_alpha = clamp(cos_alpha, 0.0, 1.0)
    angles[i] = acos(cos_alpha)
end

for j in 1:nclv
    dirs = [Vec3f(scale .* normalize(Gamma[i][:, j])) for i in 1:stride:length(Gamma)]
    arrows!(ax, origins[1:length(dirs)], dirs;
            color=clv_colors[j], linewidth=0.01,
            tipradius=0.01, tiplength=0.03, label="CLV $j")
end

axislegend(ax)

lines!(ax2, times_rel, angles, color=:blue, linewidth=1.5)
scr = display(fig)        # returns a GLMakie.Screen in a terminal run
if scr !== nothing        # VSCode/Pluto may return nothing
    wait(scr)             # blocks until you close the window
else
    println("Press Enter to exit…"); readline()  # fallback
end
