module CLV

export clv_lu

using LinearAlgebra
using DifferentialEquations
using DynamicalSystems
using StaticArrays

function _null_vector(A::AbstractMatrix)
    # The null vector is found using the singular value decomposition.
    # It corresponds to the right singular vector associated with the smallest singular value.
    U, S, V = svd(A)
    return V[:, end]
end

"""
    clv_lu(ds::DynamicalSystem; kwargs...) -> Vector{Matrix{Float64}}

Compute the covariant Lyapunov vectors (CLVs) of a dynamical system `ds` using the LU decomposition method.

# Keyword Arguments
- `nclv::Integer`: The number of covariant Lyapunov vectors to compute.
- `dt::Real`: The time step for the integration and orthogonalization.
- `nstore::Integer`: The number of trajectory points where the CLVs are computed and stored.
- `nspend_att::Integer = 1000`: Number of initial steps to discard to ensure the trajectory is on the attractor.
- `nspend_fwd::Integer = 1000`: Number of forward steps to align the tangent vectors with the Lyapunov directions.
- `nspend_bkw::Integer = 1000`: Number of backward steps to align the transposed vectors.
- `trajectory_interpolant`: A function to interpolate the trajectory for backward integration.

# Returns
- `Gamma::Vector{Matrix{Float64}}`: A vector of matrices. Each matrix `Gamma[i]` is of size `(m, nclv)` and its columns are the `nclv` covariant Lyapunov vectors at the `i`-th point of the stored trajectory.
"""
function clv_lu(
    ds::DynamicalSystem;
    nclv::Integer,
    dt::Real,
    nstore::Integer,
    nspend_att::Integer = 1000,
    nspend_fwd::Integer = 1000,
    nspend_bkw::Integer = 1000,
    trajectory_interpolant
)
    # Get the dimension of the system.
    m = DynamicalSystems.dimension(ds)
    if nclv > m
        error("nclv cannot be larger than the system dimension.")
    end

    # *** ARRIVE AT THE ATTRACTOR ***
    # Start with a random initial condition and evolve it to be on the attractor.
    u0_prototype = DynamicalSystems.get_state(ds)
    # Create a random u0 of the same type as the system's state vector.
    u0 = typeof(u0_prototype)(rand(m))
    DynamicalSystems.reinit!(ds, u0)
    step!(ds, dt * nspend_att)

    # *** PRELIMINARY STAGE ***
    # Evolve a set of random orthogonal vectors to align them with the forward Lyapunov directions.
    Q_rand = SMatrix{m, nclv}(rand(m, nclv))
    Q = qr(Q_rand).Q
    tangent_integrator = DynamicalSystems.tangent_integrator(ds, Q)

    for _ = 1:nspend_fwd
        step!(tangent_integrator, dt)
        Q_new = DynamicalSystems.get_deviations(tangent_integrator)
        Q = qr(Q_new).Q
        DynamicalSystems.set_deviations!(tangent_integrator, Q)
    end

    # *** STAGE A-B ***
    # Evolve forward and store the trajectory and the orthonormalized vectors (PhiMns).
    traj = Vector{typeof(u0)}(undef, nstore + nspend_bkw)
    PhiMns = Vector{typeof(Q)}(undef, nstore)

    for i = 1:nstore
        step!(tangent_integrator, dt)
        Q_new = DynamicalSystems.get_deviations(tangent_integrator)
        Q = qr(Q_new).Q
        DynamicalSystems.set_deviations!(tangent_integrator, Q)
        traj[i] = DynamicalSystems.get_state(tangent_integrator)
        PhiMns[i] = Q
    end

    # *** STAGE B-C ***
    # Continue evolving the trajectory forward to have points for the backward integration.
    for i = 1:nspend_bkw
        step!(tangent_integrator, dt)
        traj[nstore+i] = DynamicalSystems.get_state(tangent_integrator)
    end

    # Create the result array.
    Gamma = Vector{Matrix{Float64}}(undef, nstore)

    if nclv == 1
        for i = 1:nstore
            Gamma[i] = PhiMns[i]
        end
        return Gamma
    end

    # *** STAGE C-B ***
    # Evolve a new set of random orthogonal vectors backward in time.
    # We use a mutable Matrix for the backward pass for compatibility with the in-place ODE solver.
    Q_rand = rand(m, nclv - 1)
    Q = Matrix(qr(Q_rand).Q)
    J = DynamicalSystems.jacobian(ds)
    p = ds.p0

    # Define the adjoint dynamics using the provided trajectory interpolant.
    function adjoint_rhs!(du, u, p_ode, t)
        J_u = J(trajectory_interpolant(t), p, t)
        mul!(du, -J_u', u)
    end

    # Set up and solve the backward ODE problem.
    t_start_bwd = dt * (nstore + nspend_bkw)
    t_end_bwd = dt * nstore
    prob_bwd = ODEProblem(adjoint_rhs!, Q, (t_start_bwd, t_end_bwd))
    # We want to save the solution at each `dt` step of the backward pass.
    sol_bwd = solve(prob_bwd, Tsit5(), saveat = -dt)
    
    # The backward-evolved Q matrices are the time-reversed solution.
    Q_bwd = reverse(sol_bwd.u)

    # *** STAGE B-A ***
    # Now, iterate backward through the stored trajectory and compute the CLVs.
    for i = 1:nstore
        # The Q matrix for this point is from the backward solution.
        Q = qr(Q_bwd[i]).Q 

        P = Q' * PhiMns[nstore-i+1]
        
        current_gamma = zeros(m, nclv)
        current_gamma[:, 1] = PhiMns[nstore-i+1][:, 1]

        for j = 2:nclv
            a = _null_vector(P[1:j - 1, 1:j])
            current_gamma[:, j] = PhiMns[nstore-i+1][:, 1:j] * a
        end
        Gamma[nstore-i+1] = current_gamma
    end

    return Gamma
end

end # module CLV