abstract type AbstractHeureticStepScaler{T} end

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

@inline function get_dt(state, metric::KerrMetric{T}, s::HorizonHeureticScaler{T}) where T
    @inbounds x0, x1, x2, x3, _, _, _, _ = state
    @fastmath begin
        r = sqrt(yield_r2(x0, x1, x2, x3, metric))
        diff = r - s.event_horizon
        dt_primal = s.a0 + s.a1 * (diff) + s.a2 * diff * diff
        dt = min(dt_primal, s.max)
    end
    return dt
end

@inline function get_dt(r, s::HorizonHeureticScaler{T}) where T
    @fastmath begin
        diff = r - s.event_horizon
        dt_primal = s.a0 + s.a1 * (diff) + s.a2 * diff * diff
        dt = min(dt_primal, s.max)
    end
    return dt
end



