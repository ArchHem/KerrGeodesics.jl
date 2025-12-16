using GLMakie, NonlinearSolve, Interpolations
using LinearAlgebra, Printf, StaticArrays
using KerrGeodesics

const CalibrationExt = Base.get_extension(KerrGeodesics, :CalibrationExt)

const test_states = CalibrationExt.test_states
const test_metric = CalibrationExt.test_metric
const test_dtc = CalibrationExt.test_dtc
const N_timesteps = CalibrationExt.N_timesteps

const test_dtc_constant = HorizonHeureticScaler(
    test_dtc.max,
    test_metric,
    test_dtc.a0,
    0.0f0,
    0.0f0,
    test_dtc.redshift_stop,
    test_dtc.r_stop,
    N_timesteps
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

function compare_solutions(integrator_adaptive, integrator_constant, start_state)
    buffer_a = zeros(Float32, 8, N_timesteps)
    buffer_c = zeros(Float32, 8, N_timesteps)
    
    prepared_state_a = prep_state(start_state, integrator_adaptive)
    prepared_state_c = prep_state(start_state, integrator_constant)
    
    li_a, _ = KerrGeodesics.integrate_single_geodesic!(buffer_a, prepared_state_a, integrator_adaptive, 
                                                        norm = 0.0f0, null = true, do_norm = false)
    li_c, _ = KerrGeodesics.integrate_single_geodesic!(buffer_c, prepared_state_c, integrator_constant, 
                                                        norm = 0.0f0, null = true, do_norm = false)
    
    t_a = buffer_a[1, 1:li_a]
    t_c = buffer_c[1, 1:li_c]
    
    itp_a = [linear_interpolation(t_a, buffer_a[i, 1:li_a]) for i in 1:8]
    itp_c = [linear_interpolation(t_c, buffer_c[i, 1:li_c]) for i in 1:8]
    
    t_start = max(t_a[1], t_c[1])
    t_end = min(t_a[end], t_c[end])
    t_common = range(t_start, t_end, length=1000)
    
    diffs = [norm([itp_a[i](t) - itp_c[i](t) for i in 1:8]) for t in t_common]
    
    return collect(t_common), diffs
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

fig = Figure(size = (1800, 600))

for (idx, ((name_a, int_a), (name_c, int_c))) in enumerate(zip(integrators_adaptive, integrators_constant))
    ax = Axis(fig[1, idx],
              xlabel = "Timelike coordinate (t)",
              ylabel = "||Δstate||",
              title = name_a,
              yscale = log10)
    
    for state_key in state_keys
        start_state = test_states[state_key]
        t_common, diffs = compare_solutions(int_a, int_c, start_state)
        lines!(ax, t_common, diffs, color = color_map[state_key], linewidth = 2, label = string(state_key))
    end
end

Legend(fig[1, 4], [LineElement(color = color_map[k], linewidth = 2) for k in state_keys],
       [string(k) for k in state_keys], "Starting States", framevisible = true)

Label(fig[0, :], "Solution Difference: Adaptive vs Constant Timestep", fontsize = 24, font = :bold)

display(fig)