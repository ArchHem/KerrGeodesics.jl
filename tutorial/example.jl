include("../src/KerrGeodesics.jl")
using .KerrGeodesics, KernelAbstractions, Metal, Images

backend = MetalBackend()
texture_path = joinpath(pwd(), "example_cs", "QUASI_CS.png")
bckg = load(texture_path)

bckg_fp32 = RGB{Float32}.(bckg)

st = SubStruct(8, 4, 2, 4, 50, 50)

veloc = [-1.f0, 0.f0, 0.f0, 0.f0]

angle_y = Float32(pi/2)
angle_x = Float32(pi/2)

metric = KerrMetric{Float32}(1f0, 0.8f0)

n_frames = 360
camera_chain = Vector{PinHoleCamera{Float32}}(undef, n_frames)
T = 40f0

for (idx, θ) in enumerate(LinRange(0.f0, 2.f0 * Float32(π), n_frames))
    x1 = 40.f0 * cos(θ)
    x3 = 40.f0 * sin(θ)
    t = Float32(idx * T / n_frames)
    position = [t, x1, 0.f0, x3] * (exp(-t * 0.05f0))
    
    pointing_unnorm = [0.f0, -x1, 0.f0, -x3]
    norm = sqrt(x1^2 + x3^2)
    pointing = pointing_unnorm ./ norm
    
    upwards = [0.f0, sin(θ), 0.f0, -cos(θ)]
    
    camera_chain[idx] = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, st)
end

N = 10000
dtc = TimeStepScaler(0.5f0, metric, 0.02f0, 0.05f0, 0.025f0, 15f0, 60f0, N)


interim = propegate_camera_chain(camera_chain, st, dtc, metric, backend)

res = render_output(interim, st, bckg_fp32, backend, 30)

