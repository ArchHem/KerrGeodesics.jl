include("../src/KerrGeodesics.jl")
using .KerrGeodesics, KernelAbstractions, Metal, Images

backend = MetalBackend()
texture_path = joinpath(pwd(), "example_cs", "tracker.png")
bckg = load(texture_path)

bckg_fp32 = RGB{Float32}.(bckg)

st = SubStruct(8, 4, 200, 800)
position = [0.f0, 40.f0, 0.0f0, 0.0f0]
pointing = [0.f0, -1.f0, 0.0f0, 0.0f0]
upwards = [0.f0, 0.0f0, 0.0f0, 1.0f0]
veloc = [-1.f0, 0.f0, 0.f0, 0.f0]


#need to align dimensions with ST for best results.
Ny = 1600
Nx = 3200
angle_y = Float32(pi/4)
angle_x = Float32(pi/2)

metric = KerrMetric{Float32}(1f0, 0.0f0)

example_camera = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, Nx, Ny)

N = 10000
dtc = TimeStepScaler(0.25f0, 0.025f0, 60f0^2, 15f0, 6400f0, N)
interim = propegate_camera_chain([example_camera], st, dtc, metric, backend)

res = render_output(interim, st, bckg_fp32, backend, 1)
