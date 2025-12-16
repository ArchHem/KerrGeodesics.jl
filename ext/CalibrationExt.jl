module CalibrationExt

using KerrGeodesics, GLMakie, StaticArrays, NonlinearSolve, Interpolations
using KerrGeodesics: KerrMetric, HorizonHeureticScaler

N_timesteps = 1000
a = 0.5f0
const test_metric = KerrMetric{Float32}(1.0f0, a)
const test_dtc = HorizonHeureticScaler(0.5f0, test_metric, 0.001f0, 0.025f0, 0.025f0, 15f0, 60f0, N_timesteps)
const test_states = Dict(
    "out_of_plane_horizon_hitter" => @SVector[0.0f0, 5.0f0, 1.0f0, 2.0f0, -1f0, 1.0f0, 0.f0, 0.0f0],
    "in_plane_horizon_hitter" => @SVector[0.0f0, 5.0f0, 1.0f0, 0.0f0, -1f0, 1.0f0, 0.f0, 0.0f0],
    "flyby_in_plane" => @SVector[0.0f0, 5.0f0, 5.0f0, 0.0f0, -0.1f0, 1.0f0, 0.f0, 0.0f0],
    "flyby_out_of_plane" => @SVector[0.0f0, 5.0f0, 5.0f0, 5.0f0, -0.1f0, 1.0f0, 0.f0, -1.0f0],
    "in_plane_close_horizon" => [0.0f0, 2.34276f0, 0.6914759f0, 0.0f0, -1.0f0, 10.295938f0, 2.0629196f0, 0.0f0]
)

export test_metric, test_dtc, test_states
end