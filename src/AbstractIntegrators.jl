"""
    AbstractIntegratorBackend

Top-level abstract type for all geodesic integrators in KerrGeodesics.jl.
"""
abstract type AbstractIntegratorBackend end

"""
    AbstractStateLessCustomIntegrator <: AbstractIntegratorBackend

Abstract type for custom-built integrators optimized for GPU execution and 
specific geodesic integration needs in curved spacetime.

# Required Interface

All concrete subtypes `x<:AbstractStateLessCustomIntegrator` MUST implement:

- `max_timesteps(x)::Int` - Maximum number of integration steps before forced termination
- `geodesic_step(state::SVector{8,T}, x, args...) -> StepResult{T}` - Advances geodesic by one timestep
- 'initialize_state(state::SVector{8},T}, x, inverse_metric; kwargs) -> SVector{N, T} - May involve things like noormalization, scaling raised components, etc.

where `state = [x0, x1, x2, x3, v0, v1, v2, v3]` represents position and lowered four-velocity.

# Performance Notes

Implementations should be allocation-free and GPU-compatible (isbitstype structs only).
"""
abstract type AbstractStateLessCustomIntegrator <: AbstractIntegratorBackend end

"""
    AbstractHeureticIntegrator <: AbstractStateLessCustomIntegrator

Integrators that use heuristic-based adaptive timestep control via an 
`AbstractHeureticStepScaler`. The timestep and termination conditions are 
delegated to the associated step scaler.

# Unified API

All `AbstractHeureticIntegrator` subtypes provide:
- `scaler(x)` - Returns the step scaler
- `metric(x)` - Returns the spacetime metric
- `max_timesteps(x)` - Delegates to `max_timesteps(scaler(x))`
- `geodesic_step(state::SVector{8,T}, x, dtc_cache) -> StepResult{T}` - Advances geodesic by one timestep, using the cache derived from the step scaler.
"""
abstract type AbstractHeureticIntegrator <: AbstractStateLessCustomIntegrator end

# Unified API implementations
scaler(x::AbstractHeureticIntegrator) = x.stepscaler
metric(x::AbstractHeureticIntegrator) = x.metric
max_timesteps(x::AbstractHeureticIntegrator) = max_timesteps(scaler(x))

"""
    RK4Heuretic{T, U<:AbstractHeureticStepScaler} <: AbstractHeureticIntegrator

Fourth-order Runge-Kutta integrator with heuristic adaptive timestep control.
Uses the classical RK4 method with weights [1/6, 1/3, 1/3, 1/6].

# Fields
- `metric::KerrMetric{T}` - The Kerr spacetime metric
- `stepscaler::U` - Step size controller and termination condition provider

# Constructor
    RK4Heuretic(metric::KerrMetric{T}, stepscaler::AbstractHeureticStepScaler{T})
# Example
```julia
metric = KerrMetric(1.0f0, 0.8f0)
scaler = HorizonHeureticScaler(0.5f0, metric, 0.02f0, 0.05f0, 0.025f0, 15f0, 60f0, 10000)
integrator = RK4Heuretic(metric, scaler)
```
"""
struct RK4Heuretic{T,U<:AbstractHeureticStepScaler} <: AbstractHeureticIntegrator
    metric::KerrMetric{T}
    stepscaler::U
end

"""
    RK2Heuretic{T, U<:AbstractHeureticStepScaler} <: AbstractHeureticIntegrator

Second-order Runge-Kutta (midpoint) integrator with heuristic adaptive timestep control.
Faster but less accurate than RK4; useful for quick previews or less demanding scenarios.

# Fields
- `metric::KerrMetric{T}` - metric instance
- `stepscaler::U` - Step size controller and termination condition provider

# Constructor
    RK2Heuretic(metric::KerrMetric{T}, stepscaler::AbstractHeureticStepScaler{T})

# Example
```julia
metric = KerrMetric(1.0f0, 0.8f0)
scaler = HorizonHeureticScaler(0.5f0, metric, 0.02f0, 0.05f0, 0.025f0, 15f0, 60f0, 10000)
integrator = RK2Heuretic(metric, scaler)
```
"""
struct RK2Heuretic{T,U<:AbstractHeureticStepScaler} <: AbstractHeureticIntegrator
    metric::KerrMetric{T}
    stepscaler::U
end

"""
    AdamMoultonHeuretic{T, U<:AbstractHeureticStepScaler, N} <: AbstractHeureticIntegrator

Second-order implicit Adam-Moulton scheme. Note that it is NOT sympletic and calibration shows that it only improved over RK2 near the horizon.

In order to be GPU compatible, this integrator carries out a fixed number (N) fixed-point iterations and performs no error checks. 

# Fields
- `metric::KerrMetric{T}` - metric instance
- `stepscaler::U` - Step size controller and termination condition provider

# Constructor
    AdamMoultonHeuretic(metric::KerrMetric{T}, stepscaler::AbstractHeureticStepScaler{T}, N::Int)

# Example
```julia
metric = KerrMetric(1.0f0, 0.8f0)
scaler = HorizonHeureticScaler(0.5f0, metric, 0.02f0, 0.05f0, 0.025f0, 15f0, 60f0, 10000)
integrator = AdamMoultonHeuretic(metric, scaler)
```
"""

struct AdamMoultonHeuretic{T,U<:AbstractHeureticStepScaler,N} <: AbstractHeureticIntegrator
    metric::KerrMetric{T}
    stepscaler::U
end

"""
    SplitHamiltonianHeuretic{T, U<:AbstractHeureticStepScaler} <: AbstractHeureticIntegrator

Second-order symplectic integrator based on a split-Hamiltonian (extended phase space) method.
Tracks a physical geodesic and a shadow copy simultaneously, coupling them via a harmonic
oscillator term with frequency `ω`. This construction allows the use of a time-reversible,
symplectic splitting scheme that conserves a modified Hamiltonian over long integrations.

The method follows the scheme described in https://arxiv.org/pdf/1609.02212, using
a BACBA (A-B-C-B-A) symmetric composition of substeps, where:
- **A**: advances the physical trajectory under the original geodesic Hamiltonian
- **B**: advances the shadow copy under the original geodesic Hamiltonian
- **C**: couples physical and shadow copies via a harmonic oscillator rotation

The state vector has length 16: the first 8 components `[x0, x1, x2, x3, v0, v1, v2, v3]`
hold the physical trajectory, and the next 8 hold the shadow copy initialized identically.

Compared to `RK2Heuretic` and `RK4Heuretic`, this integrator exhibits bounded (oscillatory)
Hamiltonian drift rather than secular drift, making it better suited for long integrations.
In practice, calibration shows the improvement is most significant away from the horizon,
where the heuristic timestep remains large and explicit methods accumulate error faster.

# Fields
- `metric::KerrMetric{T}` - The Kerr spacetime metric
- `stepscaler::U` - Step size controller and termination condition provider
- `omega::T` - Coupling frequency ω of the harmonic oscillator between physical and shadow copies.
  Larger values tighten the coupling but may require smaller timesteps for stability.

# Constructor
    SplitHamiltonianHeuretic(metric::KerrMetric{T}, stepscaler::AbstractHeureticStepScaler{T}, omega::T = T(0.2))

# Example
```julia
metric = KerrMetric(1.0f0, 0.5f0)
scaler = HorizonHeureticScaler(0.5f0, metric, 0.001f0, 0.0f0, 0.0f0, 15f0, 60f0, 80000)
integrator = SplitHamiltonianHeuretic(metric, scaler, 1.0f0)
```

# Notes
- Returns a `DuplicatedStepResult{T}` rather than `StepResult{T}`; use `state(result)` to
  extract only the physical trajectory.
- The shadow copy state is discarded for rendering but is needed to maintain the symplectic
  structure across steps, so `initialize_state` must be used to set up the full 16-component
  state before integration.
- Because termination is evaluated from a finite-difference approximation of the derivative
  `(new_state - old_state) / dt`, the escape/redshift conditions are O(dt) accurate rather
  than using an analytic derivative as in the Runge-Kutta integrators.
- Best used with a constant timestep scaler (zero `a1`, `a2` in `HorizonHeureticScaler`)
  to preserve the symplectic property. Adaptive stepping breaks time-reversibility.
"""

struct SplitHamiltonianHeuretic{T,U<:AbstractHeureticStepScaler} <: AbstractHeureticIntegrator
    metric::KerrMetric{T}
    stepscaler::U
    omega::T
end

# Convenience constructors with type inference
function RK4Heuretic(metric::KerrMetric{T}, stepscaler::U) where {T,U<:AbstractHeureticStepScaler{T}}
    return RK4Heuretic{T,U}(metric, stepscaler)
end

function RK2Heuretic(metric::KerrMetric{T}, stepscaler::U) where {T,U<:AbstractHeureticStepScaler{T}}
    return RK2Heuretic{T,U}(metric, stepscaler)
end

function AdamMoultonHeuretic(metric::KerrMetric{T}, stepscaler::U, N::Int) where {T,U<:AbstractHeureticStepScaler{T}}
    return AdamMoultonHeuretic{T,U,N}(metric, stepscaler)
end

function SplitHamiltonianHeuretic(metric::KerrMetric{T}, stepscaler::U, omega=T(0.2)) where {T,U<:AbstractHeureticStepScaler{T}}
    return SplitHamiltonianHeuretic{T,U}(metric, stepscaler, omega)
end

#default behaviour for non-split state
function initialize_state(state, integrator::AbstractHeureticIntegrator, inverse_metric; norm_raised=true)
    if norm_raised
        x0, x1, x2, x3, v0, v1, v2, v3 = state
        w0, w1, w2, w3 = mult_by_metric(inverse_metric, (v0, v1, v2, v3))

        v0, v1, v2, v3 = v0 / w0, v1 / w0, v2 / w0, v3 / w0
        return @SVector [x0, x1, x2, x3, v0, v1, v2, v3]
    end
    return state
end

function initialize_state(state, integrator::SplitHamiltonianHeuretic, inverse_metric; norm_raised=true)
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    if norm_raised
        w0, w1, w2, w3 = mult_by_metric(inverse_metric, (v0, v1, v2, v3))

        v0, v1, v2, v3 = v0 / w0, v1 / w0, v2 / w0, v3 / w0
        return @SVector [x0, x1, x2, x3, v0, v1, v2, v3, x0, x1, x2, x3, v0, v1, v2, v3]
    end
    return @SVector [x0, x1, x2, x3, v0, v1, v2, v3, x0, x1, x2, x3, v0, v1, v2, v3]
end

"""
    geodesic_step(state::SVector{8,T}, integrator::RK4Heuretic{T}) -> StepResult{T}

Advances the geodesic state by one RK4 timestep.

# Arguments
- `state::SVector{8,T}` - Current state `[x0, x1, x2, x3, v0, v1, v2, v3]`
- `integrator::RK4Heuretic{T}` - RK4 integrator with metric and step scaler

# Returns
`StepResult{T}` containing:
- `state::SVector{8,T}` - New state after timestep
- `is_escaped::Bool` - Whether ray has escaped to infinity
- `is_redshifted::Bool` - Whether ray has crossed event horizon

# Implementation Details
The timestep `dt` and cached quantities are obtained from `get_dt(state, metric, scaler)`.
Termination conditions are evaluated using the final derivative estimate and cached values.
"""
function geodesic_step(state, integrator::RK4Heuretic{T}, dtc_cache) where T
    dtcontrol = scaler(integrator)
    lmetric = metric(integrator)
    @fastmath begin
        dstate_1, geom_cache = calculate_differential_and_geom(state, lmetric)

        dt, cache = get_dt(state, geom_cache, lmetric, dtcontrol)

        dt_half = dt * T(0.5)

        input_2 = @. state + dt_half * dstate_1
        dstate_2 = calculate_differential(input_2, lmetric)

        input_3 = @. state + dt_half * dstate_2
        dstate_3 = calculate_differential(input_3, lmetric)

        input_4 = @. state + dt * dstate_3
        dstate_4 = calculate_differential(input_4, lmetric)

        renorm_6 = 1 / T(6)
        renorm_3 = 1 / T(3)
        dstate = @. renorm_6 * (dstate_1 + dstate_4) + renorm_3 * (dstate_2 + dstate_3)

        escap = is_escaped(state, dstate, cache, dtc_cache, dtcontrol)
        redshift = is_redshifted(state, dstate, cache, dtc_cache, dtcontrol)
        newstate = @. state + dt * dstate
    end

    return StepResult(newstate, escap, redshift)
end

"""
    geodesic_step(state::SVector{8,T}, integrator::RK2Heuretic{T}) -> StepResult{T}

Advances the geodesic state by one RK2 (midpoint method) timestep.

# Arguments
- `state::SVector{8,T}` - Current state `[x0, x1, x2, x3, v0, v1, v2, v3]`
- `integrator::RK2Heuretic{T}` - RK2 integrator with metric and step scaler

# Returns
`StepResult{T}` containing:
- `state::SVector{8,T}` - New state after timestep
- `is_escaped::Bool` - Whether ray has escaped to infinity
- `is_redshifted::Bool` - Whether ray has crossed event horizon

# Implementation Details
Uses only two derivative evaluations (at current point and midpoint) for speed.
Termination conditions use the midpoint derivative `dstate_2` as the best available estimate.
"""
function geodesic_step(state, integrator::RK2Heuretic{T}, dtc_cache) where T
    dtcontrol = scaler(integrator)
    lmetric = metric(integrator)

    @fastmath begin
        dstate_1, geom_cache = calculate_differential_and_geom(state, lmetric)
        dt, cache = get_dt(state, geom_cache, lmetric, dtcontrol)

        dt_half = dt * T(0.5)

        input_2 = @. state + dt_half * dstate_1
        dstate_2 = calculate_differential(input_2, lmetric)

        newstate = @. state + dt * dstate_2

        escap = is_escaped(state, dstate_2, cache, dtc_cache, dtcontrol)
        redshift = is_redshifted(state, dstate_2, cache, dtc_cache, dtcontrol)
    end

    return StepResult(newstate, escap, redshift)
end

"""
    geodesic_step(state::SVector{8,T}, integrator::AdamMoultonHeuretic{T,U,N}) -> StepResult{T}

Advances the geodesic state by one Adam-Moluton timestep.

# Arguments
- `state::SVector{8,T}` - Current state `[x0, x1, x2, x3, v0, v1, v2, v3]`
- `integrator::AdamMolutonHeuretic{T,U,N}` - Adam-Moluton integrator with metric and step scaler, using a fixed number (N) iterations

# Returns
`StepResult{T}` containing:
- `state::SVector{8,T}` - New state after timestep
- `is_escaped::Bool` - Whether ray has escaped to infinity
- `is_redshifted::Bool` - Whether ray has crossed event horizon

# Implementation Details
The timestep `dt` and cached quantities are obtained from `get_dt(state, metric, scaler)`.
Termination conditions are evaluated using the final derivative estimate and cached values.
"""
@generated function geodesic_step(state::SVector{8,T}, integrator::AdamMoultonHeuretic{T,U,N}, dtc_cache) where {T,U,N}
    quote
        dtcontrol = scaler(integrator)
        lmetric = metric(integrator)

        @fastmath begin
            dstate_n, geom_cache = calculate_differential_and_geom(state, lmetric)
            dt, cache = get_dt(state, geom_cache, lmetric, dtcontrol)

            u_current = @. state + dt * dstate_n

            #could do the usual way of appending expressions
            #Can this cause issues near the horizon? I dont think so, but worth keeping in mind.
            Base.Cartesian.@nexprs $N i -> begin

                midpoint = @. (state + u_current) * T(0.5)


                dstate_mid = calculate_differential(midpoint, lmetric)


                u_current = @. state + dt * dstate_mid
            end

            newstate = u_current


            final_midpoint = @. (state + newstate) * T(0.5)
            dstate_final = calculate_differential(final_midpoint, lmetric)

            escap = is_escaped(newstate, dstate_final, cache, dtc_cache, dtcontrol)
            redshift = is_redshifted(newstate, dstate_final, cache, dtc_cache, dtcontrol)
        end

        return StepResult(newstate, escap, redshift)
    end
end


#helper functions for split hamiltonian

@inline function inner_A_ham_step(q, p, x, y, metric, dt::T, w::T) where T
    lstate, geom = calculate_differential_and_geom(q, y, metric)
    dx, dp = lstate
    return @fastmath @. q, p + dt * dp, x + dt * dx, y, geom
end

@inline function inner_B_ham_step(q, p, x, y, metric, dt::T, w::T) where T
    lstate, geom = calculate_differential_and_geom(x, p, metric)
    dq, dy = lstate
    return @fastmath @. q + dt * dq, p, x, y + dt * dy, geom
end

#see:
#https://arxiv.org/pdf/1609.02212
@inline function inner_C_ham_step(q, p, x, y, metric, dt::T, w::T) where T

    infec = 2 * w * dt
    cos_term = @fastmath cos(infec)
    sin_term = @fastmath sin(infec)

    qpx = @. q + x
    ppy = @. p + y

    qmx = @. q - x
    pmy = @. p - y

    new_q = @fastmath @. T(0.5) * (qpx + cos_term * qmx + sin_term * pmy)
    new_p = @fastmath @. T(0.5) * (ppy - sin_term * qmx + cos_term * pmy)
    new_x = @fastmath @. T(0.5) * (qpx - cos_term * qmx - sin_term * pmy)
    new_y = @fastmath @. T(0.5) * (ppy + sin_term * qmx - cos_term * pmy)

    return new_q, new_p, new_x, new_y, nothing
end

#explicit, second order sympletic step
@inline function separated_ham_2nd_step(q, p, x, y, metric, dt::T, w::T) where T
    dthalf = T(0.5) * dt
    q, p, x, y, _ = inner_A_ham_step(q, p, x, y, metric, dthalf, w)
    q, p, x, y, _ = inner_B_ham_step(q, p, x, y, metric, dthalf, w)
    q, p, x, y, _ = inner_C_ham_step(q, p, x, y, metric, dt, w)
    q, p, x, y, _ = inner_B_ham_step(q, p, x, y, metric, dthalf, w)
    q, p, x, y, geom = inner_A_ham_step(q, p, x, y, metric, dthalf, w)
    return q, p, x, y, geom
end

# Should be:
"""
    geodesic_step(state::SVector{16,T}, integrator::SplitHamiltonianHeuretic{T,U}, dtc_cache) -> DuplicatedStepResult{T}
...
- `integrator::SplitHamiltonianHeuretic{T,U}` - Integrator with metric, step scaler, and coupling frequency ω
...
"""
function geodesic_step(state::SVector{16,T}, integrator::SplitHamiltonianHeuretic{T,U}, dtc_cache) where {T,U}

    dtcontrol = scaler(integrator)
    lmetric = metric(integrator)
    w = integrator.omega

    q = state[SVector(1:4...)]
    p = state[SVector(5:8...)]

    x = state[SVector(9:12...)]
    y = state[SVector(13:16...)]


    dt, cache = get_dt(state[SVector(1:8...)], lmetric, dtcontrol)
    nq, np, nx, ny, cache = separated_ham_2nd_step(q, p, x, y, lmetric, dt, w)

    old_phys_state = vcat(q, p)
    new_phys_state = vcat(nq, np)
    new_shadow_state = vcat(nx, ny)

    #not ideal - could extract midvale for O(dt^2) approx?
    dstate = @fastmath @. (new_phys_state - old_phys_state) / dt
    escap = is_escaped(new_phys_state, dstate, cache, dtc_cache, dtcontrol)
    redshift = is_redshifted(new_phys_state, dstate, cache, dtc_cache, dtcontrol)

    newstate = vcat(new_phys_state, new_shadow_state)
    return DuplicatedStepResult(newstate, escap, redshift)
end


"""
    AbstractStateFullCustomIntegrator <: AbstractIntegratorBackend

Abstract type for custom-built integrators optimized for GPU execution and 
specific geodesic integration needs in curved spacetime, that make use of caching.

# Required Interface

All concrete subtypes `x<:AbstractStateLessCustomIntegrator` MUST implement:

- `max_timesteps(x)::Int` - Maximum number of integration steps before forced termination
- `geodesic_step(state::SVector{N,T}, x, current_cache, args...) -> StepResult{T}, cache` - Advances geodesic by one timestep, given a cache.
- `geodesic_step(state::SVector{N,T}, x, current_cache::Nothing, args...) -> StepResult{T}, cache` - Initial geodesic step.

where `state = [x0, x1, x2, x3, v0, v1, v2, v3]` represents position and lowered four-velocity.

# Performance Notes

Implementations should be allocation-free and GPU-compatible (isbitstype structs only). The behaviour of the "cache" is left as an implementation detail, 
but it MUST be stack allocated and type infereble for Adapt.jl

"""
abstract type AbstractStateFullCustomIntegrator <: AbstractIntegratorBackend end

#WIP