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
- `geodesic_step(state::SVector{8,T}, x) -> StepResult{T}` - Advances geodesic by one timestep

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
struct RK4Heuretic{T, U <: AbstractHeureticStepScaler} <: AbstractHeureticIntegrator
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
struct RK2Heuretic{T, U <: AbstractHeureticStepScaler} <: AbstractHeureticIntegrator
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

struct AdamMoultonHeuretic{T, U <: AbstractHeureticStepScaler, N} <: AbstractHeureticIntegrator
    metric::KerrMetric{T}
    stepscaler::U
end

# Convenience constructors with type inference
function RK4Heuretic(metric::KerrMetric{T}, stepscaler::U) where {T, U <: AbstractHeureticStepScaler{T}}
    return RK4Heuretic{T, U}(metric, stepscaler)
end

function RK2Heuretic(metric::KerrMetric{T}, stepscaler::U) where {T, U <: AbstractHeureticStepScaler{T}}
    return RK2Heuretic{T, U}(metric, stepscaler)
end

function AdamMoultonHeuretic(metric::KerrMetric{T}, stepscaler::U, N::Int) where {T, U <: AbstractHeureticStepScaler{T}}
    return AdamMoultonHeuretic{T, U, N}(metric, stepscaler)
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
function geodesic_step(state, integrator::RK4Heuretic{T}) where T
    dtcontrol = scaler(integrator)
    lmetric = metric(integrator)
    @fastmath begin
        dt, cache = get_dt(state, lmetric, dtcontrol)
        
        dstate_1 = calculate_differential(state, lmetric)
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
        newstate = @. state + dt * dstate
        
        escap = is_escaped(state, dstate, cache, dtcontrol)
        redshift = is_redshifted(state, dstate, cache, dtcontrol)
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
function geodesic_step(state, integrator::RK2Heuretic{T}) where T
    dtcontrol = scaler(integrator)
    lmetric = metric(integrator)
    
    @fastmath begin
        dt, cache = get_dt(state, lmetric, dtcontrol)
        
        dstate_1 = calculate_differential(state, lmetric)
        dt_half = dt * T(0.5)
        
        input_2 = @. state + dt_half * dstate_1
        dstate_2 = calculate_differential(input_2, lmetric)
        
        newstate = @. state + dt * dstate_2
        
        escap = is_escaped(state, dstate_2, cache, dtcontrol)
        redshift = is_redshifted(state, dstate_2, cache, dtcontrol)
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
@generated function geodesic_step(state::SVector{8,T}, integrator::AdamMoultonHeuretic{T,U,N}) where {T,U,N}
    quote
        dtcontrol = scaler(integrator)
        lmetric = metric(integrator)
        
        @fastmath begin
            dt, cache = get_dt(state, lmetric, dtcontrol)
            
            
            dstate_n = calculate_differential(state, lmetric)
            
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
            
            escap = is_escaped(newstate, dstate_final, cache, dtcontrol)
            redshift = is_redshifted(newstate, dstate_final, cache, dtcontrol)
        end
        
        return StepResult(newstate, escap, redshift)
    end
end
"""
    AbstractStateFullCustomIntegrator <: AbstractIntegratorBackend

Abstract type for custom-built integrators optimized for GPU execution and 
specific geodesic integration needs in curved spacetime, that make use of caching.

# Required Interface

All concrete subtypes `x<:AbstractStateLessCustomIntegrator` MUST implement:

- `max_timesteps(x)::Int` - Maximum number of integration steps before forced termination
- `geodesic_step(state::SVector{N,T}, x, current_cache) -> StepResult{T}, cache` - Advances geodesic by one timestep, given a cache.
- `geodesic_step(state::SVector{N,T}, x, current_cache::Nothing) -> StepResult{T}, cache` - Initial geodesic step.

where `state = [x0, x1, x2, x3, v0, v1, v2, v3]` represents position and lowered four-velocity.

# Performance Notes

Implementations should be allocation-free and GPU-compatible (isbitstype structs only). The behaviour of the "cache" is left as an implementation detail, 
but it MUST be stack allocated and type infereble for Adapt.jl

"""
abstract type AbstractStateFullCustomIntegrator <: AbstractIntegratorBackend end