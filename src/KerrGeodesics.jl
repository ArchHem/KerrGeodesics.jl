module KerrGeodesics

using LoopVectorization, Enzyme, FastDifferentiation
include("./structs.jl")
include("./DiffOperators.jl")

export calculate_innerprod!, calculate_differentials_backward!, KerrMetric, BatchInfo
end
