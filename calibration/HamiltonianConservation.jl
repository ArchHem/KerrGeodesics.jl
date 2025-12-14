include("../src/KerrGeodesics.jl")
using .KerrGeodesics, StaticArrays

a = 0.5f0
const metric = KerrMetric{Float32}(1.0f0, a)
#initial state - will get normalized
start_state = @SVector [0.0f0, 5.0f0, 1.0f0, 2.0f0, -1f0, 1.0f0, 0.f0, 0.0f0]
N_timesteps = 1000
dtc = HorizonHeureticScaler(0.5f0, metric, 0.001f0, 0.025f0, 0.025f0, 15f0, 60f0, N_timesteps)

Moulton_order = 4
integrator_AM = AdamMoultonHeuretic(metric, dtc, Moulton_order)
integrator_RK = RK2Heuretic(metric, dtc)
buffer = zeros(Float32, 8, N_timesteps)

li, isr = integrate_single_geodesic!(buffer, start_state, integrator_RK, norm = 0.0f0, null = true)
Hs_RK = map(x -> KerrGeodesics.yield_hamiltonian(x, metric), eachcol(@view buffer[:, 1:li]))

li, isr = integrate_single_geodesic!(buffer, start_state, integrator_AM, norm = 0.0f0, null = true)
Hs_AM = map(x -> KerrGeodesics.yield_hamiltonian(x, metric), eachcol(@view buffer[:, 1:li]))