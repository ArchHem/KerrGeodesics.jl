include("../src/KerrGeodesics.jl")
using .KerrGeodesics, StaticArrays
using GLMakie

a = 0.5f0
const metric = KerrMetric{Float32}(1.0f0, a)
start_state = @SVector [0.0f0, 2.0f0, 0.0f0, 0.0f0, -0.1f0, 0.0f0, 1.f0, 1.0f0]
N_timesteps = 1000
dtc = HorizonHeureticScaler(0.5f0, metric, 0.001f0, 0.025f0, 0.025f0, 15f0, 60f0, N_timesteps)

integrator = RK4Heuretic(metric, dtc)
buffer = zeros(Float32, 8, N_timesteps)

li, isr = integrate_single_geodesic!(buffer, start_state, integrator, norm = 0.0f0, null = true)
Hs = map(x -> KerrGeodesics.yield_hamiltonian(x, metric), eachcol(@view buffer[:, 1:li]))

x = buffer[2, 1:li]
y = buffer[3, 1:li]
z = buffer[4, 1:li]

H0 = Hs[1]
ΔH = Hs .- H0

fig = Figure(size = (1400, 600))

ax1 = Axis3(fig[1, 1], 
    xlabel = "x",
    ylabel = "y", 
    zlabel = "z",
    title = "Trajectory",
    aspect = :equal
)

lines!(ax1, x, y, z, color = :blue, linewidth = 2)
scatter!(ax1, [x[1]], [y[1]], [z[1]], color = :green, markersize = 15)
scatter!(ax1, [x[end]], [y[end]], [z[end]], color = :red, markersize = 15)

ax2 = Axis(fig[1, 2],
    xlabel = "t",
    ylabel = "ΔH",
    title = "Hamiltonian Drift"
)

t = buffer[1, 1:li]
lines!(ax2, t, ΔH, color = :blue, linewidth = 2)

fig

