abstract type AbstractIntegratorBackend end

abstract type AbstractDEIntegrator <: AbstractIntegratorBackend end

abstract type AbstractCustomIntegrator <: AbstractIntegratorBackend end #for all of our custom needs

abstract type AbstractHeureticIntegrator <: AbstractCustomIntegrator end #uses some dt scaling heuretic

#concrete implementations

struct RK4Heuretic{T} <: AbstractHeureticIntegrator
    metric::KerrMetric{T} 
    stepscaler::AbstractHeureticStepScaler{T}
end
struct RK2Heuretic{T} <: AbstractHeureticIntegrator
    metric::KerrMetric{T} 
    stepscaler::AbstractHeureticStepScaler{T}
end