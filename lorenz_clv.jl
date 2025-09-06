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
ds = Systems.lorenz()   # ContinuousDynamicalSystem, N = 3

# 2) Parameters
nclv       = 3
t_renorm   = 1.0
dt_plot    = 1e-2
nstore     = Int(1e4)
nspend_att = Int(3e3)
nspend_fwd = Int(3e4)
nspend_bkw = Int(3e3)

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
    reltol     = 1e-9,   # accepted by clv for API compatibility
    abstol     = 1e-9,
)
println("Done. Stored $(length(Gamma)) samples.")

# 4) Background trajectory for plotting
# Avoid DifferentialEquations here. Use DynamicalSystems step! so types match.
u0_s   = xs[1]                         # typically an SVector
ds_bg  = Systems.lorenz()
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
    title="Lorenz Attractor with Covariant Lyapunov Vectors",
    xlabel="x", ylabel="y", zlabel="z",
)

lines!(ax, trajectory_dense, linewidth=0.3)

stride     = 200
scale      = 4.0
clv_colors = (:red, :orange, :green)

origins = Point3f.(xs[1:stride:end])

for j in 1:nclv
    dirs = [Vec3f(scale .* normalize(Gamma[i][:, j])) for i in 1:stride:length(Gamma)]
    arrows!(ax, origins[1:length(dirs)], dirs;
            color=clv_colors[j], linewidth=0.01,
            tipradius=0.01, tiplength=0.03, label="CLV $j")
end

axislegend(ax)
scr = display(fig)        # returns a GLMakie.Screen in a terminal run
if scr !== nothing        # VSCode/Pluto may return nothing
    wait(scr)             # blocks until you close the window
else
    println("Press Enter to exit…"); readline()  # fallback
end
