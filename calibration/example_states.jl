using StaticArrays

const test_states = Dict(
    "out_of_plane_horizon_hitter" => @SVector[0.0f0, 5.0f0, 1.0f0, 2.0f0, -1f0, 1.0f0, 0.f0, 0.0f0],
    "in_plane_horizon_hitter" => @SVector[0.0f0, 5.0f0, 1.0f0, 0.0f0, -1f0, 1.0f0, 0.f0, 0.0f0],
    "flyby_in_plane" => @SVector[0.0f0, 5.0f0, 5.0f0, 0.0f0, -0.1f0, 1.0f0, 0.f0, 0.0f0],
    "flyby_out_of_plane" => @SVector[0.0f0, 5.0f0, 5.0f0, 5.0f0, -0.1f0, 1.0f0, 0.f0, -1.0f0]
)