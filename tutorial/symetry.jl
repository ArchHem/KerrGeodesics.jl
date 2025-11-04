include("../src/KerrGeodesics.jl")
using .KerrGeodesics, Plots


position = [0.f0, 40.f0, 0.0f0, 0.0f0]
pointing = [0.f0, -1.f0, 0.0f0, 0.0f0]
upwards = [0.f0, 0.0f0, 0.0f0, 1.0f0]
veloc = [-1.f0, 0.f0, 0.f0, 0.f0]


Ny = 100
Nx = 100
angle_y = Float32(pi/6)
angle_x = Float32(pi/6)

#schwarschild blackhole
metric = KerrMetric{Float32}(1f0, 0.0f0)

camera = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, Nx, Ny)

N = 10000
dtc = TimeStepScaler(0.25f0, 0.025f0, 60f0^2, 3f0, 6400f0, N)

state_1 = KerrGeodesics.generate_camera_ray(Float32(0.), Float32(0.5), camera)
true_initial_state_1 = [camera.position..., state_1...]
buffer_1 = zeros(Float32, 8, N)
integrate_single_geodesic!(buffer_1, true_initial_state_1, dtc, metric, norm = 0.0f0, null = true)

state_2 = KerrGeodesics.generate_camera_ray(Float32(1.0), Float32(0.5), camera)
true_initial_state_2 = [camera.position..., state_2...]
buffer_2 = zeros(Float32, 8, N)
integrate_single_geodesic!(buffer_2, true_initial_state_2, dtc, metric, norm = 0.0f0, null = true)

state_3 = KerrGeodesics.generate_camera_ray(Float32(0.5), Float32(0.), camera)
true_initial_state_3 = [camera.position..., state_3...]
buffer_3 = zeros(Float32, 8, N)
integrate_single_geodesic!(buffer_3, true_initial_state_3, dtc, metric, norm = 0.0f0, null = true)

state_4 = KerrGeodesics.generate_camera_ray(Float32(0.5), Float32(1.), camera)
true_initial_state_4 = [camera.position..., state_4...]
buffer_4 = zeros(Float32, 8, N)
integrate_single_geodesic!(buffer_4, true_initial_state_4, dtc, metric, norm = 0.0f0, null = true)

#if the integrator is correct, 

p1 = plot(buffer_1[2, :], buffer_1[4, :])
plot!(p1, buffer_2[2, :], buffer_2[4, :])

plot!(p1, buffer_3[2, :], buffer_3[3, :])
plot!(p1, buffer_4[2, :], buffer_4[3, :])


#so, why could this be?

x0, x1, x2, x3 = 0.0f0, 40.0f0, 5.0f0, 0.0f0
metric = KerrMetric{Float32}(1f0, 0.0f0)

#these yield components in order of u00, u10, u20, u30, u11, u21, u31, u22, u32, u33
m1 = KerrGeodesics.yield_inverse_metric(x0, x1, x2, x3, metric)
m2 = KerrGeodesics.yield_inverse_metric(x0, x1, x3, x2, metric)