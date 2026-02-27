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

abstract type AbstractStepResult{T} end

struct StepResult{T} <: AbstractStepResult{T}
    state::SVector{8, T}
    is_escaped::Bool
    is_redshifted::Bool
end

#@inline function StepResult(state::SVector{8, T}, is_escaped::Bool, is_redshifted::Bool) where T
#    return StepResult{T}(state, is_escaped, is_redshifted)
#end

struct DuplicatedStepResult{T} <: AbstractStepResult{T}
    state::SVector{16, T}
    is_escaped::Bool
    is_redshifted::Bool
end

state_length(x::AbstractStepResult) = length(x.state)

full_state(x::AbstractStepResult) = x.state
state(x::StepResult) = x.state
#for duplicated systems, return the first state as the physical truth.
#The first 8 elements store the physical state, and the rest store the rest.
state(x::DuplicatedStepResult) = @inbounds x.state[SVector(1:8...)]
shadow_state(x::DuplicatedStepResult) = @inbounds x.state[SVector(9:16...)]
isredshifted(x::AbstractStepResult) = x.is_redshifted
isescaped(x::AbstractStepResult) = x.is_escaped
isterminated(x::AbstractStepResult) = isescaped(x) || isredshifted(x)