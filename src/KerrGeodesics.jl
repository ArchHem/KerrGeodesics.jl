module KerrGeodesics

using LoopVectorization, KernelAbstractions, FastDifferentiation
include("./structs.jl")
include("./GeodesicOps.jl")
include("./KADiffOperators.jl")

export calculate_innerprod!, calculate_differentials_backward!, KerrMetric, BatchInfo, TimeStepScaler, calculate_differential!
end
