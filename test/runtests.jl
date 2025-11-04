include("../src/KerrGeodesics.jl")
using .KerrGeodesics
using Test, Random, Enzyme

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
    a = 0.2f0
    metric = KerrMetric{Float32}(1f0, a)

    x0 = 1.0f0
    x1 = 4.0f0 + randn(Float32)
    x2 = 3.0f0 + randn(Float32)
    x3 = 6.0f0 + randn(Float32)

    r2 = KerrGeodesics.yield_r2(x0, x1, x2, x3, metric)

    #implicit form of r2...
    @test isapprox((x1^2 + x2^2)/(r2 + a^2) + (x3^2/r2), 1.0f0)


end

#This for some reason (likely paranoid Enzyme tracing) can not be run inside a testblock.

a = 0.2f0
metric = KerrMetric{Float32}(1f0, a)

function yield_hamiltonian(res, xvec, vvec, lmetric)
    x0, x1, x2, x3 = xvec
    v0, v1, v2, v3 = vvec
    mmetric = KerrGeodesics.yield_inverse_metric(x0, x1, x2, x3, lmetric)
    H = KerrGeodesics.yield_innerprod(mmetric, v0, v1, v2, v3) / 2
    res[1] = H
    return nothing
end

#the EOM are the same for non-geodesics as well... so we can just choose some random numbers.

x0 = 1.0f0
x1 = 6.2f0
x2 = 7.4f0
x3 = 3.2f0

v0 = -0.2f0
v1 = 1.2f0
v2 = -0.1f0
v3 = 0.2f0

pos = [x0, x1, x2, x3]
vel = [v0, v1, v2, v3]

dpos = zeros(Float32, 4)
dvel = zeros(Float32, 4)

H = zeros(Float32, 1)
dH = ones(Float32, 1)

Enzyme.autodiff(Reverse, yield_hamiltonian, DuplicatedNoNeed(H, dH), Duplicated(pos, dvel), Duplicated(vel, dpos), Const(metric))

diff_tpl = KerrGeodesics.calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric)
dvel = -1 .* dvel

local_metric_tpl = KerrGeodesics.yield_inverse_metric(x0, x1, x2, x3, metric)
raised_fourveloc = Float32[KerrGeodesics.mult_by_metric(local_metric_tpl, (v0, v1, v2, v3))...]

dpos_d = Float32[diff_tpl[i] for i in 1:4]
dvec_d = Float32[diff_tpl[i] for i in 5:8]

@testset "EoM" begin
    #We will use Enzyme to test the EOM
    

    @test isapprox(dpos_d, dpos)
    @test isapprox(dvec_d, dvel)
    #test from metric definition, independent of FD generated function.
    @test isapprox(raised_fourveloc, dpos)

    
end
