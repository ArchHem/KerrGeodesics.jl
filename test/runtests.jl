include("../src/KerrGeodesics.jl")
using .KerrGeodesics
using Test, Random

@testset "CameraGeneration" begin
    
    #test camera generation
    position = [0.f0, 12.f0, 0.0f0, 0.0f0]
    pointing = [0.f0, -1.f0, 0.0f0, 0.0f0]
    upwards = [0.f0, 0.0f0, 0.0f0, 1.0f0]
    veloc = [-1.f0, 0.f0, 0.f0, 0.f0]

    Ny = 1600
    Nx = 3200
    angle_y = Float32(pi/12)
    angle_x = Float32(pi/6)

    metric = KerrMetric{Float32}(1f0, 0.2f0)

    example_camera = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, Nx, Ny)

    #check orthognaility of generated directions.
    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, 
        example_camera.lowered_pointing, 
        example_camera.lowered_rightward), 
        0.0f0, atol = 1f0^-7)
    
    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, 
        example_camera.lowered_pointing, 
        example_camera.lowered_upward), 
        0.0f0, atol = 1f0^-7)
    
    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, 
        example_camera.lowered_pointing, 
        example_camera.lowered_velocity), 
        0.0f0, atol = 1f0^-7)

    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, 
        example_camera.lowered_velocity, 
        example_camera.lowered_rightward), 
        0.0f0, atol = 1f0^-7)

    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, 
        example_camera.lowered_velocity, 
        example_camera.lowered_upward), 
        0.0f0, atol = 1f0^-7)

    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, 
        example_camera.lowered_upward, 
        example_camera.lowered_rightward), 
        0.0f0, atol = 1f0^-7)

    test_vec = KerrGeodesics.generate_camera_ray(0.2f0, 0.6f0, example_camera)

    @test isapprox(KerrGeodesics.yield_innerprod(example_camera.inverse_metric_tpl, test_vec...), 
        0.0f0, atol = 1f0^-7)

    

end

@testset "UtilTests" begin

    #test consistency of pixel mapping

    NWarp = 20
    MWarp = 80
    J = 4
    I = 8
    maxframes = 1000

    st = SubStruct(I, J, NWarp, MWarp)

    Random.seed!(1234)
    i = rand(1:NWarp * I)
    j = rand(1:MWarp * J)
    k = rand(1:maxframes)

    warp_index, lane_index = KerrGeodesics.video_index_to_array_index(i, j, k, st)

    i_n, j_n, k_n = KerrGeodesics.array_index_to_video_index(warp_index, lane_index, st)

    @test i_n == i
    @test j_n == j
    @test k_n == k

    #backwards test 
    maxwarp = NWarp * MWarp * maxframes
    maxlane = I * J

    testlane = rand(1:maxlane)
    testwarp = rand(1:maxwarp)

    i_, j_, k_ = KerrGeodesics.array_index_to_video_index(testwarp, testlane, st)

    new_warp, new_lane = KerrGeodesics.video_index_to_array_index(i_, j_, k_, st)

    @test new_warp == testwarp
    @test new_lane == testlane


end

@testset "Geometry" begin
    #choose some random positions
    metric = KerrMetric{Float32}(1f0, 0.2f0)

    r2 = KerrGeodesics.yield_r2
end
