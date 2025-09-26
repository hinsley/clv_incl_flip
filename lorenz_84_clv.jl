using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("clv.jl")
using .CLV
using DynamicalSystems
using LinearAlgebra
using GLMakie
using UMAP
using Random
using StaticArrays

# 0) Lorenz '84 vector field
function lorenz84!(du, u, p, t)
    x, y, z = u
    a = p.a
    b = p.b
    F = p.F
    G = p.G

    du[1] = -y^2 - z^2 - a * x + a * F
    du[2] = x * y - b * x * z - y + G
    du[3] = b * x * y + x * z - z
    return nothing
end

# 1) System
params = (a = 0.25, b = 4.0, F = 8.0, G = 1.0)
u0 = SVector{3}(0.1, 0.0, 0.0)
ds = ContinuousDynamicalSystem(lorenz84!, u0, params)

# 2) Parameters
nclv       = 3
t_renorm   = 0.05
dt_plot    = 3e-3
nstore     = Int(3e4)
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
ds_bg  = ContinuousDynamicalSystem(lorenz84!, u0_s, params)
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
fig = Figure(size=(1400, 900))
ax = Axis3(fig[1, 1:3],
    title="Lorenz '84 Attractor with Covariant Lyapunov Vectors",
    xlabel="x", ylabel="y", zlabel="z",
)

# Time series plot of θ(t) below the 3D plot.
ax2 = Axis(fig[2,1],
    title="Minimum angle: CLV3 vs span{CLV1,CLV2}",
    xlabel="t", ylabel="θ (rad)",
    yticks=0:π/4:π, ytickformat=xs -> ["0", "π/4", "π/2", "3π/4", "π"][1:length(xs)]
)
ax_rec = Axis(fig[2,2],
    title="Recurrence of local minima",
    xlabel="θ_min(t_i)", ylabel="θ_min(t_{i+1})"
)
ax_rec_max = Axis(fig[2,3],
    title="Recurrence of local maxima",
    xlabel="θ_max(t_i)", ylabel="θ_max(t_{i+1})"
)
rowsize!(fig.layout, 1, Relative(0.75))
rowsize!(fig.layout, 2, Relative(0.25))
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.3))
colsize!(fig.layout, 3, Relative(0.2))

lines!(ax, trajectory_dense, linewidth=0.3)

stride     = 10
scale      = 0.3
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

local_min_indices = Int[]
local_max_indices = Int[]
for i in 2:(length(angles)-1)
    if angles[i] <= angles[i-1] && angles[i] <= angles[i+1]
        push!(local_min_indices, i)
    end
    if angles[i] >= angles[i-1] && angles[i] >= angles[i+1]
        push!(local_max_indices, i)
    end
end

function recurrence_plot!(ax_plot, values; color=:dodgerblue, fallback="Not enough extrema")
    if length(values) >= 2
        recurrence_x = values[1:end-1]
        recurrence_y = values[2:end]

        min_lim = min(minimum(recurrence_x), minimum(recurrence_y))
        max_lim = max(maximum(recurrence_x), maximum(recurrence_y))
        span = max_lim - min_lim
        padding = span > 0 ? 0.05 * span : 0.01
        lower = min_lim - padding
        upper = max_lim + padding

        lines!(ax_plot, [lower, upper], [lower, upper]; color=:gray, linestyle=:dot)
        scatter!(ax_plot, recurrence_x, recurrence_y; color=color, markersize=6)
        xlims!(ax_plot, lower, upper)
        ylims!(ax_plot, lower, upper)
    else
        text!(ax_plot, Point2f(0.5, 0.5), fallback;
              align=(:center, :center), space=:relative)
    end
end

local_min_values = angles[local_min_indices]
recurrence_plot!(ax_rec, local_min_values; color=:dodgerblue, fallback="Not enough minima")

local_max_values = angles[local_max_indices]
recurrence_plot!(ax_rec_max, local_max_values; color=:tomato, fallback="Not enough maxima")

for j in 1:nclv
    dirs = [Vec3f(scale .* normalize(Gamma[i][:, j])) for i in 1:stride:length(Gamma)]
    arrows!(ax, origins[1:length(dirs)], dirs;
            color=clv_colors[j], linewidth=0.01,
            tipradius=0.01, tiplength=0.03, label="CLV $j")
end

axislegend(ax)

lines!(ax2, times_rel, angles, color=:blue, linewidth=1.5)
scr = display(fig)        # returns a GLMakie.Screen in a terminal run
# if scr !== nothing        # VSCode/Pluto may return nothing
#     wait(scr)             # blocks until you close the window
# else
#     println("Press Enter to exit…"); readline()  # fallback
# end

# --- UMAP embedding of (x, Γ1, Γ2, Γ3) -------------------------------------------------
println("Building 12D representation for UMAP…")
Γs_flat = Matrix{Float64}(undef, 12, length(Gamma))
for (i, Γ) in enumerate(Gamma)
    x = xs[i]
    Γ1 = Γ[:, 1]
    Γ2 = Γ[:, 2]
    Γ3 = Γ[:, 3]
    Γs_flat[:, i] = vcat(x, Γ1, Γ2, Γ3)
end

println("Running UMAP (12 → 2)…")
embedding = UMAP.umap(Γs_flat; n_neighbors=30, min_dist=0.1)

fig_umap = Figure(size=(800, 600))
ax_umap = Axis(fig_umap[1, 1],
    title="UMAP of Lorenz '84 CLV state",
    xlabel="UMAP₁", ylabel="UMAP₂"
)

line_x = Float64[]
line_y = Float64[]
for i in 1:(size(embedding, 2) - 1)
    dx = embedding[1, i+1] - embedding[1, i]
    dy = embedding[2, i+1] - embedding[2, i]
    if hypot(dx, dy) <= 2.0
        push!(line_x, embedding[1, i], embedding[1, i+1], NaN)
        push!(line_y, embedding[2, i], embedding[2, i+1], NaN)
    end
end
if !isempty(line_x)
    lines!(ax_umap, line_x, line_y; color=:black, linewidth=0.5)
end
scatter!(ax_umap, embedding[1, :], embedding[2, :]; color=:gray, markersize=4)
scatter!(ax_umap, [embedding[1, 1]], [embedding[2, 1]]; color=:red, markersize=12)
scatter!(ax_umap, [embedding[1, end]], [embedding[2, end]]; color=:blue, markersize=12)

display(fig_umap)
