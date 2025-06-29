using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("clv.jl")
using .CLV
using DynamicalSystems
using LinearAlgebra
using GLMakie
using DifferentialEquations
using Random
using StaticArrays

# 1. Set up the Lorenz system.
ds = Systems.lorenz()
prob = ODEProblem(ds)

# 2. Set parameters for the CLV computation.
nclv = 3       # We want to compute all 3 CLVs for the 3D system.
dt = 0.01      # Time step for integration.
nstore = 2000  # Number of points to store CLVs at.
nspend_att = 4000
nspend_fwd = 4000
nspend_bkw = 4000

# Set a seed for reproducibility. This is crucial.
Random.seed!(0)

# 3. Compute the trajectory for plotting, matching the one inside clv_lu.
# We must replicate the initial state and burn-in period of clv_lu.
u0 = SVector{3}(rand(3))
t_burn_in = dt * (nspend_att + nspend_fwd)
prob_burn_in = remake(prob, u0 = u0, tspan = (0.0, t_burn_in))
sol_burn_in = solve(prob_burn_in)
u_start = sol_burn_in.u[end]

# Now, compute the full trajectory needed for the forward and backward passes.
t_full = dt * (nstore + nspend_bkw)
prob_full = remake(prob, u0 = u_start, tspan = (0.0, t_full))
# Get the continuous solution, which can be used as an interpolant.
full_trajectory_solution = solve(prob_full)
# The trajectory for plotting is just the first `nstore` points.
trajectory = [full_trajectory_solution(t) for t in 0:dt:(nstore*dt)]

# 4. Compute the CLVs. Must reset the seed to get the same u0 inside clv_lu.
println("Computing Covariant Lyapunov Vectors for the Lorenz system...")
Random.seed!(1234)
Gamma = clv_lu(
    ds;
    nclv = nclv,
    dt = dt,
    nstore = nstore,
    nspend_att = nspend_att,
    nspend_fwd = nspend_fwd,
    nspend_bkw = nspend_bkw,
    trajectory_interpolant = full_trajectory_solution,
)

println("Finished computing CLVs.")
println("Size of the resulting Gamma array of matrices: ", size(Gamma))
println("Each element of Gamma is a ", size(Gamma[1]), " matrix containing the CLVs at that trajectory point.")

# 5. Plot the results using GLMakie.
println("Plotting attractor and CLVs...")

# Create the figure and a 3D axis
fig = Figure(size = (800, 800))
ax = Axis3(
    fig[1, 1],
    title = "Lorenz Attractor with Covariant Lyapunov Vectors",
    xlabel = "x",
    ylabel = "y",
    zlabel = "z",
)

# Plot the attractor trajectory.
# Makie works well with vectors of Point3f.
traj_points = [Point3f(p) for p in trajectory]
lines!(ax, traj_points, color = :blue, linewidth = 1.0, label = "Lorenz Attractor")

# Plot the CLVs on top of the attractor.
# We'll only plot them every `stride` steps to avoid cluttering the plot.
stride = 100
scale = 4.0 # Scaling factor for the vectors to make them visible.
clv_colors = [:red, :orange, :green] # Colors for the different CLVs.

# To create a legend, we plot each CLV type as a separate series.
for j = 1:nclv
    origins = [Point3f(trajectory[i]) for i in 1:stride:nstore]
    directions = [Vec3f(scale .* normalize(Gamma[i][:, j])) for i in 1:stride:nstore]
    arrows!(
        ax,
        origins,
        directions,
        color = clv_colors[j],
        linewidth = 0.01,
        arrowsize = 0.03,
        label = "CLV $j",
    )
end

# Add a legend to the plot.
axislegend(ax)

# Display the plot.
display(fig)

println("Plotting complete. Please check the plot window.")