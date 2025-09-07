using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("clv.jl")
include("Plant_model.jl")
using .CLV
using .Plant
using DynamicalSystems
using LinearAlgebra
using GLMakie
using Random
using StaticArrays
using OrdinaryDiffEq

# 1) Configurable parameters
# You can modify `gh` to change the hyperpolarization-activated current conductance (p[4]).
# You can also modify Δx (p[16]) and ΔCa (p[17]).
gh = 0.005 # Float64(Plant.default_params[4])
τh = Float64(Plant.default_params[14])
ΔCa = -40.0
Δx  = -1.0

# 2) System
# Build initial condition by inserting y between x and n. We use y0 = hinf(V0).
begin
    x0  = Float64(Plant.default_state[1])
    n0  = Float64(Plant.default_state[2])
    h0  = Float64(Plant.default_state[3])
    Ca0 = Float64(Plant.default_state[4])
    V0  = Float64(Plant.default_state[5])
    y0  = Plant.hinf(V0)

    u0 = @SVector Float64[x0, y0, n0, h0, Ca0, V0]
end

# Update parameters with configured gh (index 4), Δx (index 16), and ΔCa (index 17).
p = collect(Float64.(Plant.default_params))
p[4] = gh
p[14] = τh
p[16] = Δx
p[17] = ΔCa

ds = ContinuousDynamicalSystem(Plant.melibeNew!, u0, p; diffeq=(
    alg = Vern9(),
    abstol = 1e-12,
    reltol = 1e-12,
))

# 3) Parameters
nclv       = 6
t_renorm   = 1e-2
dt_plot    = 1e-2
nstore     = Int(1e5)
nspend_att = Int(1e5)
nspend_fwd = Int(1e5)
nspend_bkw = Int(1e5)

Random.seed!(0)

# 4) Compute CLVs
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

# 5) Background trajectory for plotting (projected to 3D as (Ca, x, V))
u0_s   = xs[1]
ds_bg  = ContinuousDynamicalSystem(Plant.melibeNew!, u0_s, p; diffeq=(
    alg = Vern9(),
    abstol = 1e-12,
    reltol = 1e-12,
))

t_full = ts[end] - ts[1]
K      = max(0, Int(round(t_full / dt_plot)))
trajectory_dense = Vector{Point3f}(undef, K + 1)
trajectory_dense[1] = Point3f(u0_s[5], u0_s[1], u0_s[6])
for k in 1:K
    step!(ds_bg, dt_plot)
    u_bg = DynamicalSystems.get_state(ds_bg)
    trajectory_dense[k+1] = Point3f(u_bg[5], u_bg[1], u_bg[6])
end

# 6) Plot attractor and projected CLVs
fig = Figure(size=(900, 900))
ax = Axis3(fig[1,1],
    title="Plant Model with Covariant Lyapunov Vectors",
    xlabel="Ca", ylabel="x", zlabel="V",
)

# Time series plot of θ(t) below the 3D plot.
ax2 = Axis(fig[2,1],
    title="Minimum angle: span{CLV1, CLV2} vs span{CLV3…6}",
    xlabel="t", ylabel="θ (rad)",
    yticks=0:π/4:π, ytickformat=xs -> ["0", "π/4", "π/2", "3π/4", "π"][1:length(xs)]
)

# Make the θ(t) subplot shorter and the state-space plot taller.
rowsize!(fig.layout, 1, Relative(0.8))
rowsize!(fig.layout, 2, Relative(0.2))

lines!(ax, trajectory_dense, linewidth=0.3)

scale      = 0.06
V_scale    = 100
n_arrows   = 100000   # Number of arclength-sampled points to draw CLVs at.
clv_colors = (:red, :orange, :green, :blue, :purple, :brown)
min_proj_norm = 1e-8

# Compute arclength in displayed (Ca, x, V/V_scale) space and pick n_arrows points.
Nkeep = length(Gamma)
speeds = Vector{Float64}(undef, Nkeep)
for i in 1:Nkeep
    du = Plant.melibeNew(xs[i], p, ts[i])
    # (Ca, x, V/V_scale) speed for visual arclength sampling.
    speeds[i] = norm(SVector(du[5], du[1], du[6]/V_scale))
end

arc = zeros(Float64, Nkeep)
for i in 2:Nkeep
    dt_i = ts[i] - ts[i-1]
    arc[i] = arc[i-1] + 0.5 * (speeds[i-1] + speeds[i]) * dt_i
end

total_arc = arc[end]
targets = range(0, stop=total_arc, length=n_arrows)
sel_indices = Int[]
for a in targets
    idx = searchsortedfirst(arc, a)
    if 1 <= idx <= Nkeep
        push!(sel_indices, idx)
    end
end
unique!(sel_indices)

# Origins in 3D: (Ca, x, V)
origins_sel = [Point3f(u[5], u[1], u[6]) for u in (xs[i] for i in sel_indices)]

# 7) Compute subspace angle θ(t): span{Γ₁,Γ₂} vs span{Γ₃,…,Γ₆} in full state space
times_rel = [t - ts[1] for t in ts]
angles = Vector{Float64}(undef, length(Gamma))
for i in 1:length(Gamma)
    A = Matrix(Gamma[i][:, 1:2])
    B = Matrix(Gamma[i][:, 3:6])

    QA = Matrix(qr(A).Q)[:, 1:2]
    QB = Matrix(qr(B).Q)[:, 1:4]

    C = QA' * QB
    svals = svdvals(C)
    sigma_max = maximum(svals)
    sigma_max = clamp(sigma_max, 0.0, 1.0)
    angles[i] = acos(sigma_max)
end

# real_j = [1, 0, 2, 3, 4, 5] # Hack: Since we don't use y it is skipped when gh=0.
real_j = 1:nclv
for j in 1:nclv
    origins_j = Point3f[]
    dirs_j = Vec3f[]
    for (k, i) in enumerate(sel_indices)
        vfull = Gamma[i][:, j]
        v3d = @SVector Float64[vfull[5], vfull[1], vfull[6]/V_scale]
        nv = norm(v3d)
        if nv >= min_proj_norm
            v3n = (v3d ./ nv) .* scale
            push!(dirs_j, Vec3f(Float32(v3n[1]), Float32(v3n[2]), Float32(v3n[3]*V_scale)))
            push!(origins_j, origins_sel[k])
        end
    end
    if !isempty(dirs_j)
        arrows!(ax, origins_j, dirs_j;
                color=clv_colors[real_j[j]], linewidth=0.1 * scale,
                tipradius=0.1 * scale, tiplength=0.3 * scale, label="CLV $(real_j[j])")
    end
end

axislegend(ax)

lines!(ax2, times_rel, angles, color=:blue, linewidth=1.5)

scr = display(fig)
if scr !== nothing
    wait(scr)
else
    println("Press Enter to exit…"); readline()
end
