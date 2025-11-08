module KerrGeodesics

using KernelAbstractions, ForwardDiff, Adapt, LinearAlgebra, StaticArrays, Colors, Images, VideoIO
include("./AlgebraicUtils.jl")
include("./structs.jl")
include("./CameraUtils.jl")
include("./GeodesicOps.jl")
include("./CameraFunctions.jl")
include("./RenderKernels.jl")
include("./KADiffOperators.jl")
include("./Utils.jl")

export KerrMetric, TimeStepScaler, PinHoleCamera, 
    SubStruct, ensemble_ODE_RK4!, propegate_camera_chain, 
    render_output, PinHoleCamera, integrate_single_geodesic!
end
