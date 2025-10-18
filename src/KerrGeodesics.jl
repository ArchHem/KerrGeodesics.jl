module KerrGeodesics

using LoopVectorization, KernelAbstractions, FastDifferentiation
include("./structs.jl")
include("./CPUDiffOperators.jl")

export calculate_innerprod!, calculate_differentials_backward!, KerrMetric, BatchInfo
end
