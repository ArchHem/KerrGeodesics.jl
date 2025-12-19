using KerrGeodesics, KernelAbstractions, Metal, GLMakie, BenchmarkTools
using Statistics, Printf, Colors, Images

backend = MetalBackend()

a = 0.85f0
metric = KerrMetric{Float32}(1.0f0, a)
dtc = HorizonHeureticScaler(0.5f0, metric, 0.02f0, 0.05f0, 0.025f0, 15f0, 60f0, 10000)

veloc = [-1.f0, 0.f0, 0.f0, 0.f0]
angle_y = Float32(pi/2)
angle_x = Float32(pi/2)

n_frames = 10

function create_camera_chain(st, n_frames, metric, angle_x, angle_y, veloc)
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
    
    return camera_chain
end

function full_job(st, camera_chain, integrator, backend, bckg)
    result = propegate_camera_chain(camera_chain, st, integrator, backend)
    frames = render_frames(result, st, bckg, backend)
    return frames
end

function benchmark_layout(st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, integrator, n_frames, metric, angle_x, angle_y, veloc, backend) where {V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}
    camera_chain = create_camera_chain(st, n_frames, metric, angle_x, angle_y, veloc)
    texture_path = joinpath(pwd(), "example_cs", "QUASI_CS.png")
    bckg = load(texture_path)
    bckg_fp32 = RGB{Float32}.(bckg)
    #render cost of propegation + texture composition.
    
    bench = @benchmark begin
        full_job($st, $camera_chain, $integrator, $backend, $bckg_fp32)
    end samples=20 seconds=60
    
    Ny = V * MicroNWarps * NBlocks
    Nx = H * MicroMWarps * MBlocks
    total_pixels = Nx * Ny * n_frames
    
    return bench, total_pixels
end

layouts = [
    ("Hierarchical\n(8×4, 2×4, 64×64)", SubStruct(8, 4, 2, 4, 64, 64)),
    ("Larger Hierarchical\n(8×4, 4×4, 32×64)", SubStruct(8, 4, 4, 4, 32, 64)),
    ("Column-major\n(32×1, 4×1, 8x1024)", SubStruct(32, 1, 4, 1, 8, 1024)),
    ("Square tiles\n(16×16, 1×1, 64×64)", SubStruct(16, 16, 1, 1, 64, 64)),
    ("Large tiles\n(32x8, 1x1, 32x128)", SubStruct(32, 8, 1, 1, 32, 128))
]

integrators = [
    ("RK2", RK2Heuretic(metric, dtc))
]

results = []

for (int_name, integrator) in integrators
    println("\nBenchmarking $int_name...")
    
    for (layout_name, st) in layouts
        println("  Layout: $(replace(layout_name, "\n" => " "))")
        
        bench, total_pixels = benchmark_layout(st, integrator, n_frames, metric, angle_x, angle_y, veloc, backend)
        
        mean_time = mean(bench).time / 1e9
        time_per_pixel = mean_time / total_pixels * 1e6
        
        push!(results, (
            integrator = int_name,
            layout = layout_name,
            mean_time = mean_time,
            time_per_pixel = time_per_pixel
        ))
    end
end

fig = Figure(size = (1400, 600))

rk2_results = filter(r -> r.integrator == "RK2", results)

layout_names = [r.layout for r in rk2_results]
rk2_times = [r.time_per_pixel for r in rk2_results]

rk2_baseline = rk2_times[1]
rk2_speedup = rk2_baseline ./ rk2_times

ax1 = GLMakie.Axis(fig[1, 1],
           xlabel = "Memory Layout",
           ylabel = "Time per Pixel (μs)",
           title = "Absolute Performance",
           xticks = (1:length(layout_names), layout_names),
           xticklabelrotation = π/6)

barplot!(ax1, 1:length(layout_names), rk2_times, 
         color = :palegreen)

ax2 = GLMakie.Axis(fig[1, 2],
           xlabel = "Memory Layout",
           ylabel = "Speedup (relative to hierarchical)",
           title = "Relative Performance",
           xticks = (1:length(layout_names), layout_names),
           xticklabelrotation = π/6)

barplot!(ax2, 1:length(layout_names), rk2_speedup, 
         color = :palegreen)

hlines!(ax2, [1.0], color = :gray, linestyle = :dash, linewidth = 2)

Label(fig[0, :], "Memory Layout Performance Comparison - RK2 - Metal Backend, a = $(round(metric.a,digits = 2))", 
      fontsize = 20, font = :bold)

display(fig)

println("\nResults Summary:")
println("="^60)
println("\nRK2:")
for (idx, r) in enumerate(rk2_results)
    speedup = rk2_baseline / r.time_per_pixel
    @printf("  %-35s: %6.3f μs/pixel (%.2fx)\n", 
            replace(r.layout, "\n" => " "), r.time_per_pixel, speedup)
end
save("exhibits/layout_comparison.png", fig, dpi = 600)