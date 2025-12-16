using GLMakie, NonlinearSolve
using LinearAlgebra, Printf, StaticArrays
using KerrGeodesics

const CalibrationExt = Base.get_extension(KerrGeodesics, :CalibrationExt)

const test_states = CalibrationExt.test_states
const test_metric = CalibrationExt.test_metric
const test_dtc = CalibrationExt.test_dtc
const N_timesteps = CalibrationExt.N_timesteps
const constant_timesteps = 80000

const test_dtc_constant = HorizonHeureticScaler(
    test_dtc.max,
    test_metric,
    test_dtc.a0,
    0.0f0,
    0.0f0,
    test_dtc.redshift_stop,
    test_dtc.r_stop,
    constant_timesteps
)

function prep_state(start_state, integrator)
    x0, x1, x2, x3, v0, v1, v2, v3 = start_state
    T = typeof(x0)
    local_metric = KerrGeodesics.metric(integrator)
    metric_tpl = KerrGeodesics.yield_inverse_metric(x0, x1, x2, x3, local_metric)
    w0, w1, w2, w3 = KerrGeodesics.mult_by_metric(metric_tpl, (v0, v1, v2, v3))
    v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0
    v0, v1, v2, v3 = KerrGeodesics.normalize_fourveloc(metric_tpl, v0, v1, v2, v3, norm = T(0), null = true)
    return @SVector [x0, x1, x2, x3, v0, v1, v2, v3]
end

function compute_hamiltonian_drift(integrator, start_state)
    buffer = zeros(Float32, 8, KerrGeodesics.scaler(integrator).maxtimesteps)
    prepared_state = prep_state(start_state, integrator)
    li, isr = KerrGeodesics.integrate_single_geodesic!(buffer, prepared_state, integrator, norm = 0.0f0, null = true, do_norm = false)
    Hs = map(x -> KerrGeodesics.yield_hamiltonian(x, test_metric), eachcol(@view buffer[:, 1:li]))
    t = buffer[1, 1:li]
    H0 = Hs[1]
    ΔH = Hs .- H0
    return t, ΔH, li
end

integrators_adaptive = [
    ("RK2", RK2Heuretic(test_metric, test_dtc)),
    ("RK4", RK4Heuretic(test_metric, test_dtc)),
    ("Adam-Moulton (N=4)", AdamMoultonHeuretic(test_metric, test_dtc, 4))
]

integrators_constant = [
    ("RK2", RK2Heuretic(test_metric, test_dtc_constant)),
    ("RK4", RK4Heuretic(test_metric, test_dtc_constant)),
    ("Adam-Moulton (N=4)", AdamMoultonHeuretic(test_metric, test_dtc_constant, 4))
]

state_keys = collect(keys(test_states))
num_states = length(state_keys)
colors = Makie.wong_colors()
color_map = Dict(zip(state_keys, colors[1:num_states]))

fig = Figure(size = (1800, 1200))

for (idx, (integrator_name, integrator)) in enumerate(integrators_adaptive)
    ax = Axis(fig[1, idx],
              xlabel = "Timelike coordinate (t)",
              ylabel = "ΔH",
              title = integrator_name * " (Adaptive)",
              yscale = log10)
    
    for state_key in state_keys
        start_state = test_states[state_key]
        t, ΔH, li = compute_hamiltonian_drift(integrator, start_state)
        lines!(ax, t, abs.(ΔH), color = color_map[state_key], linewidth = 2, label = string(state_key))
    end
    
    hlines!(ax, [eps(Float32)], color = :gray, linestyle = :dash, linewidth = 1)
end

for (idx, (integrator_name, integrator)) in enumerate(integrators_constant)
    ax = Axis(fig[2, idx],
              xlabel = "Timelike coordinate (t)",
              ylabel = "ΔH",
              title = integrator_name * " (Constant dt)",
              yscale = log10)
    
    for state_key in state_keys
        start_state = test_states[state_key]
        t, ΔH, li = compute_hamiltonian_drift(integrator, start_state)
        lines!(ax, t, abs.(ΔH), color = color_map[state_key], linewidth = 2, label = string(state_key))
    end
    
    hlines!(ax, [eps(Float32)], color = :gray, linestyle = :dash, linewidth = 1)
end

Legend(fig[1:2, 4], [LineElement(color = color_map[k], linewidth = 2) for k in state_keys],
       [string(k) for k in state_keys], "Starting States", framevisible = true)

Label(fig[0, :], "Hamiltonian Conservation: Adaptive vs Constant Timestep", fontsize = 24, font = :bold)

display(fig)