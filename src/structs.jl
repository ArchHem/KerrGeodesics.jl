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
    is_escaped::Bool
    is_redshifted::Bool
end

@inline function StepResult(state::SVector{8, T}, is_escaped::Bool, is_redshifted::Bool) where T
    return StepResult{T}(state, is_escaped, is_redshifted)
end

state(x::StepResult) = x.state
isredshifted(x::StepResult) = x.is_redshifted
isescaped(x::StepResult) = x.is_escaped
isterminated(x::StepResult) = isescaped(x) || isredshifted(x)