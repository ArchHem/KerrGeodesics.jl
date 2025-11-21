#store metric information
struct KerrMetric{T}
    M::T
    a::T
end

#This compues specialized, heuretical timesteps in the Kerr spacetime.
#It uses the following heuretic:
#If r2 > threshold, dt = min(2 * outer_scaling, max)
#If r2 < threshold, dt = flat
#To presevre continuity at the threshold, we enforce

#Use kerr's papers on EH in KS coordinates;

#r2 - 2mr + a^2 = 0 for the EH.

#The outer horizon has r = sqrt(m^2-a^2) + m

#i.e. r^2 = 2*m^2 - a^2 + 2 m * sqrt(m^2 - a^2)

struct TimeStepScaler{T}
    max::T
    flat::T
    outer_scaling::T
    threshold::T
    redshift_stop::T
    r_stop::T
    maxtimesteps::Int
end

function TimeStepScaler(max::T, flat::T, threshold::T, redshift_stop::T, r_stop::T, maxtimesteps::Int) where T
    outer_scaling = flat/threshold
    return TimeStepScaler{T}(max, flat, outer_scaling, threshold, redshift_stop, r_stop, maxtimesteps)
end

@inline function get_dt(r2::T, s::TimeStepScaler{T}) where T
    dt = ifelse(r2 > s.threshold, min(s.max, s.outer_scaling * r2), min(s.max, s.flat))
    return dt
end

struct SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

end

function SubStruct(V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks)
    return SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}()
end