include("../src/KerrGeodesics.jl")
using .KerrGeodesics, KernelAbstractions, Metal, Images

backend = MetalBackend()
texture_path = joinpath(pwd(), "example_cs", "tracker.png")
bckg = load(texture_path)

bckg_fp32 = RGB{Float32}.(bckg)

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

dtc = TimeStepScaler(0.5f0, 0.02f0, 30f0^2, 2000f0, 6400f0, 10000)
interim = propegate_camera_chain([example_camera], st, dtc, metric, backend)

res = render_output(interim, st, bckg_fp32, backend, 1)
