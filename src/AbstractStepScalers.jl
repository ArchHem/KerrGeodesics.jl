"""
    All members, x<:AbstractHeureticStepScaler{T} must implement:
    
    get_dt(state, input_cache, metric, x) -> (T, cache) where cache is type-stable, non-heap-allocated. Input cache is returned by calculate_differential_and_geom
    get_dt(state, metric, x) -> (T, cache) where cache is type-stable, non-heap-allocated.
    is_redshifted(state, dstate, cache, x)::Bool
    is_escaped(state, dstate, cache, x)::Bool
    max_timesteps(x)::Int
"""

abstract type AbstractHeureticStepScaler{T} end

"""
    HorizonHeureticScaler{T} <: AbstractHeureticStepScaler{T}

Adaptive timestep controller using a quadratic heuristic based on distance from 
the event horizon. Provides both timestep scaling and termination conditions for 
geodesic integration in Kerr spacetime.

# Fields
- `max::T` - Maximum allowed timestep (upper bound for stability)
- `event_horizon::T` - Radius of the outer event horizon: `rₕ = M + √(M² - a²)`
- `a0::T` - Constant term in quadratic timestep formula
- `a1::T` - Linear coefficient in quadratic timestep formula
- `a2::T` - Quadratic coefficient in quadratic timestep formula
- `redshift_stop::T` - Threshold for temporal velocity component (v^0) to detect horizon crossing
- `r_stop::T` - Radial distance threshold for escape to infinity
- `maxtimesteps::Int` - Maximum integration steps before forced termination

# Timestep Heuristic

The timestep is computed as:
```
dt = min(a₀ + a₁(r - rₕ) + a₂(r - rₕ)², max)
```

where `r` is the Boyer-Lindquist radial coordinate and `rₕ` is the event horizon radius.
# Termination Conditions

- **Escaped**: `r > r_stop` - Ray has reached "infinity" (far from black hole)
- **Redshifted**: `v^0 > redshift_stop` - Ray's temporal component indicates horizon crossing

The redshift condition detects when a null ray's time component becomes anomalously 
large, signaling it has gotten near the event horizon and is undergoing a significant shift.

# Constructor
    HorizonHeureticScaler(max::T, metric::KerrMetric{T}, a0::T, a1::T, a2::T, 
                          redshift_stop::T, r_stop::T, maxtimesteps::Int) where T

"""
struct HorizonHeureticScaler{T} <: AbstractHeureticStepScaler{T}
    max::T
    event_horizon::T
    a0::T
    a1::T
    a2::T
    redshift_stop::T
    r_stop::T
    maxtimesteps::Int
end

function HorizonHeureticScaler(max::T, metric::KerrMetric{T}, a0::T, a1::T, a2::T, redshift_stop::T, r_stop::T, maxtimesteps::Int) where T
    event_horizon = sqrt(metric.M^2-metric.a^2) + metric.M
    return HorizonHeureticScaler{T}(max, event_horizon, a0, a1, a2, redshift_stop, r_stop, maxtimesteps)
end

@inline function max_timesteps(s::HorizonHeureticScaler{T}) where T
    return s.maxtimesteps
end

@inline function get_dt(state, input_cache, metric::KerrMetric{T}, s::HorizonHeureticScaler{T}) where T
    r = input_cache
    @fastmath begin
        diff = r - s.event_horizon
        dt_primal = s.a0 + s.a1 * (diff) + s.a2 * diff * diff
        dt = min(dt_primal, s.max)
    end
    return (-dt, (r))
end

@inline function get_dt(state, metric::KerrMetric{T}, s::HorizonHeureticScaler{T}) where T
    @inbounds x0, x1, x2, x3 = state[1], state[2], state[3], state[4]
    @fastmath begin
        r = sqrt(yield_r2(x0, x1, x2, x3, metric))
        diff = r - s.event_horizon
        dt_primal = s.a0 + s.a1 * (diff) + s.a2 * diff * diff
        dt = min(dt_primal, s.max)
    end
    return (-dt, (r))
end

@inline function is_redshifted(state, dstate, cache, s::HorizonHeureticScaler{T}) where T
    #use heuretic of temporal component, assumes that ray was normalized to u0 = 1 initially
    return dstate[1] > s.redshift_stop
end

@inline function is_escaped(state, dstate, cache, s::HorizonHeureticScaler{T}) where T
    r = cache[1]
    return r > s.r_stop
end

