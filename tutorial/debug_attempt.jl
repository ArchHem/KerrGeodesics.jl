include("../src/KerrGeodesics.jl")
using .KerrGeodesics, Plots

plotly()
st = SubStruct(8, 4, 200, 800)
position = [0.f0, 12.f0, 0.0f0, 0.0f0]
pointing = [0.f0, -1.f0, 0.0f0, 0.0f0]
upwards = [0.f0, 0.0f0, 0.0f0, 1.0f0]
veloc = [-1.f0, 0.f0, 0.f0, 0.f0]

#need to align dimensions with ST for best results.
Ny = 1600
Nx = 3200
angle_y = Float32(pi/12)
angle_x = Float32(pi/6)

metric = KerrMetric{Float32}(1f0, 0.0f0)

example_camera = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, Nx, Ny)

N = 1000
dtc = TimeStepScaler(0.1f0, 0.025f0, 60f0^2, 2000f0, 6400f0, N)

#generate top-most pixel:

state_1 = KerrGeodesics.generate_camera_ray(Float32(1/1600), Float32(1/3200), example_camera)

true_initital_state_1 = [example_camera.position..., state_1...]

buffer_1 = zeros(Float32, 8, N)

integrate_single_geodesic!(buffer_1, true_initital_state_1, dtc, metric, norm = 0.0f0, null = true)

trajectory = plot(buffer_1[2, :], buffer_1[3, :], buffer_1[4, :])

state_2 = KerrGeodesics.generate_camera_ray(Float32(1600/1600), Float32(1/3200), example_camera)

true_initital_state_2 = [example_camera.position..., state_2...]

buffer_2 = zeros(Float32, 8, N)

integrate_single_geodesic!(buffer_2, true_initital_state_2, dtc, metric, norm = 0.0f0, null = true)

plot!(trajectory,buffer_2[2, :], buffer_2[3, :], buffer_2[4, :])

state_3 = KerrGeodesics.generate_camera_ray(Float32(1/1600), Float32(3200/3200), example_camera)

true_initital_state_3 = [example_camera.position..., state_3...]

buffer_3 = zeros(Float32, 8, N)

integrate_single_geodesic!(buffer_3, true_initital_state_3, dtc, metric, norm = 0.0f0, null = true)

plot!(trajectory,buffer_3[2, :], buffer_3[3, :], buffer_3[4, :])

state_4 = KerrGeodesics.generate_camera_ray(Float32(1600/1600), Float32(3200/3200), example_camera)

true_initital_state_4 = [example_camera.position..., state_4...]

buffer_4 = zeros(Float32, 8, N)

integrate_single_geodesic!(buffer_4, true_initital_state_4, dtc, metric, norm = 0.0f0, null = true)

plot!(trajectory,buffer_4[2, :], buffer_4[3, :], buffer_4[4, :])