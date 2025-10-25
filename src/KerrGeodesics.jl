module KerrGeodesics

using LoopVectorization, KernelAbstractions, FastDifferentiation, Adapt, LinearAlgebra, StaticArrays
include("./AlgebraicUtils.jl")
include("./structs.jl")
include("./GeodesicOps.jl")
include("./CameraUtils.jl")
include("./KADiffOperators.jl")

export KerrMetric, TimeStepScaler, calculate_differential!, PinHoleCamera
end
