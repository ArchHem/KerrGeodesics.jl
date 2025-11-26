#store metric information
"""
    KerrMetric{T}

    Represenets a Kerr blackhole with "mass" M and normalized spin a.
    Note that expected behaviour is onlgy guaranteed for abs(a) < M
"""
struct KerrMetric{T}
    M::T
    a::T
end

#This compues specialized, heuretical timesteps in the Kerr spacetime.

#Use kerr's papers on EH in KS coordinates;

#r2 - 2mr + a^2 = 0 for the EH.

#The outer horizon has r = sqrt(m^2-a^2) + m

#i.e. r^2 = 2*m^2 - a^2 + 2 m * sqrt(m^2 - a^2)



struct TimeStepScaler{T}
    max::T
    event_horizon::T
    a0::T
    a1::T
    a2::T
    redshift_stop::T
    r_stop::T
    maxtimesteps::Int
end

function TimeStepScaler(max::T, metric::KerrMetric{T}, a0::T, a1::T, a2::T, redshift_stop::T, r_stop::T, maxtimesteps::Int) where T
    event_horizon = sqrt(metric.M^2-metric.a^2) + metric.M
    return TimeStepScaler{T}(max, event_horizon, a0, a1, a2, redshift_stop, r_stop, maxtimesteps)
end

@inline function get_dt(r::T, s::TimeStepScaler{T}) where T

    diff = r - s.event_horizon
    dt_primal = s.a0 + s.a1 * (diff) + s.a2 * diff * diff
    dt = min(dt_primal, s.max)
    return dt
end

struct SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

end

function SubStruct(V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks)
    return SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}()
end