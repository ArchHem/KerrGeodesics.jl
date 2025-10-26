#store metric information
struct KerrMetric{T}
    M::T
    a::T
end

#This compues specialized, heuretical timesteps in the Kerr spacetime.
#It uses the following heuretic:
#If r2 > threshold, dt = min(flat + r2 * outer_scaling, max)
#If r2 < threshold, dt = min(max, flat + inner_scaling/r2)
#To presevre continuity at the threshold, we enforce:
# threshold * outer_scaling = inner_scaling / threshold

struct TimeStepScaler{T}
    max::T
    flat::T
    outer_scaling::T
    inner_scaling::T
    threshold::T
    redshift_stop::T
    r_stop::T
    maxtimesteps::Int
end

function TimeStepScaler(max::T, flat::T, outer_scaling::T, threshold::T, redshift_stop::T, r_stop::T, maxtimesteps::Int) where T
    inner_scaling = threshold * threshold * outer_scaling
    return TimeStepScaler{T}(max, flat, outer_scaling, inner_scaling, threshold, redshift_stop, r_stop, maxtimesteps)
end

@inline function get_dt(r2::T, s::TimeStepScaler{T}) where T
    dt = ifelse(r2 > s.threshold, min(s.max, s.flat + s.outer_scaling * r2), min(s.max, s.flat + s.inner_scaling / r2))
    return dt
end

struct SubStruct{V, H}

end

function SubStruct(V, H)
    return SubStruct{V, H}()
end