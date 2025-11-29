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

struct StepResult{T}
    state::StaticVector{8, T}
    isterminated::Bool
end

function StepResult(state, status)
    T = eltype(state)
    return StepResult{T}(SVector{8, T}(state), status)
end