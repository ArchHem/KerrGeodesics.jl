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
    state::SVector{8, T}
    isterminated::Bool
end

@inline function StepResult(state::SVector{8, T}, status::Bool) where T
    return StepResult{T}(state, status)
end

state(x::StepResult) = x.state
isterminated(x::StepResult) = x.isterminated