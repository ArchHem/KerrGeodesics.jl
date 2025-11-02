include("../src/KerrGeodesics.jl")
using .KerrGeodesics, Plots

plotly()
st = SubStruct(8, 4, 200, 800)
position = [0.f0, 40.f0, 0.0f0, 0.0f0]
pointing = [0.f0, -1.f0, 0.0f0, 0.0f0]
upwards = [0.f0, 0.0f0, 0.0f0, 1.0f0]
veloc = [-1.f0, 0.f0, 0.f0, 0.f0]

#need to align dimensions with ST for best results.
Ny = 1600
Nx = 3200
angle_y = Float32(pi/12)
angle_x = Float32(pi/6)

metric = KerrMetric{Float32}(1f0, 0.9f0)

example_camera = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, Nx, Ny)

N = 10000
dtc = TimeStepScaler(0.25f0, 0.025f0, 60f0^2, 3f0, 6400f0, N)

function integrate_and_plot!(trajectory, camera, metric, dtc, y_frac, x_frac, N; plot_first=false)
    state = KerrGeodesics.generate_camera_ray(Float32(y_frac), Float32(x_frac), camera)
    true_initial_state = [camera.position..., state...]
    buffer = zeros(Float32, 8, N)
    integrate_single_geodesic!(buffer, true_initial_state, dtc, metric, norm = 0.0f0, null = true)
    
    if plot_first
        return plot(buffer[2, :], buffer[3, :], buffer[4, :]), buffer
    else
        plot!(trajectory, buffer[2, :], buffer[3, :], buffer[4, :])
        return buffer
    end
end

trajectory, b1 = integrate_and_plot!(nothing, example_camera, metric, dtc, 1/1600, 1/3200, N, plot_first=true)

for j in LinRange(0, 1, 30)
    for i in LinRange(0, 1, 30)
        local_buffer = integrate_and_plot!(trajectory, example_camera, metric, dtc, i, j, N)
    end
end


trajectory