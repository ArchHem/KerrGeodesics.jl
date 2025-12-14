include("../src/KerrGeodesics.jl")
using .KerrGeodesics
using StaticArrays

N_timesteps = 1000
a = 0.5f0
const test_metric = KerrMetric{Float32}(1.0f0, a)
const test_dtc = HorizonHeureticScaler(0.5f0, test_metric, 0.001f0, 0.025f0, 0.025f0, 15f0, 60f0, N_timesteps)
const test_states = Dict(
    "out_of_plane_horizon_hitter" => @SVector[0.0f0, 5.0f0, 1.0f0, 2.0f0, -1f0, 1.0f0, 0.f0, 0.0f0],
    "in_plane_horizon_hitter" => @SVector[0.0f0, 5.0f0, 1.0f0, 0.0f0, -1f0, 1.0f0, 0.f0, 0.0f0],
    "flyby_in_plane" => @SVector[0.0f0, 5.0f0, 5.0f0, 0.0f0, -0.1f0, 1.0f0, 0.f0, 0.0f0],
    "flyby_out_of_plane" => @SVector[0.0f0, 5.0f0, 5.0f0, 5.0f0, -0.1f0, 1.0f0, 0.f0, -1.0f0]
)