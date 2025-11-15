include("../src/KerrGeodesics.jl")
ENV["METAL_CAPTURE_ENABLED"] = 1
ENV["METAL_DEVICE_WRAPPER_TYPE"] = 1
using .KerrGeodesics, KernelAbstractions, Metal, Images

backend = MetalBackend()
texture_path = joinpath(pwd(), "example_cs", "imaginary_LEO.png")
bckg = load(texture_path)

bckg_fp32 = RGB{Float32}.(bckg)

st = SubStruct(8, 4, 200, 400)

veloc = [-1.f0, 0.f0, 0.f0, 0.f0]

Ny = 1600
Nx = 1600
angle_y = Float32(pi/2)
angle_x = Float32(pi/2)

metric = KerrMetric{Float32}(1f0, 0.8f0)

n_frames = 4
camera_chain = Vector{PinHoleCamera{Float32}}(undef, n_frames)

for (idx, θ) in enumerate(LinRange(0.f0, 2.f0 * Float32(π), n_frames))
    x1 = 40.f0 * cos(θ)
    x3 = 40.f0 * sin(θ)
    position = [0.f0, x1, 0.f0, x3]
    
    pointing_unnorm = [0.f0, -x1, 0.f0, -x3]
    norm = sqrt(x1^2 + x3^2)
    pointing = pointing_unnorm ./ norm
    
    upwards = [0.f0, sin(θ), 0.f0, -cos(θ)]
    
    camera_chain[idx] = PinHoleCamera(position, veloc, pointing, upwards, metric, angle_x, angle_y, Nx, Ny)
end

N = 10000

dtc = TimeStepScaler(0.25f0, 0.025f0, 60f0^2, 10f0, 6400f0, N)

Metal.@capture begin

    interim = propegate_camera_chain(camera_chain, st, dtc, metric, backend)

    res = render_output(interim, st, bckg_fp32, backend, 2)

end
