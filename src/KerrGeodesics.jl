module KerrGeodesics

using KernelAbstractions, Adapt, LinearAlgebra, StaticArrays, Colors, Images, VideoIO


include("./AlgebraicUtils.jl")
include("./structs.jl")
include("./PhysicsUtils.jl")
include("./RenderStructs.jl")
include("./AbstractStepScalers.jl")
include("./AbstractIntegrators.jl") 
include("./Interpolants.jl")
include("./CameraUtils.jl")
include("./GeodesicOps.jl")
include("./CameraFunctions.jl")
include("./RenderKernels.jl")
include("./Utils.jl")

export KerrMetric, HorizonHeureticScaler, PinHoleCamera, 
    SubStruct, propegate_camera_chain, 
    render_output, integrate_single_geodesic!, NearestInterpolant, BiLinearInterpolant,
    RK4Heuretic, RK2Heuretic
end
