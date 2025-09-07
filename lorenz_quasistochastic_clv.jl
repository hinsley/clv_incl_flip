using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("clv.jl")
using .CLV
using DynamicalSystems
using LinearAlgebra
using GLMakie
using Random

# 1) System
ds = Systems.lorenz(ρ=45.0)   # ContinuousDynamicalSystem, N = 3, ρ = 45

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
ds_bg  = Systems.lorenz(ρ=45.0)
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
ax = Axis3(fig[1,1],
    title="Lorenz Attractor (ρ=45) with Covariant Lyapunov Vectors",
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
