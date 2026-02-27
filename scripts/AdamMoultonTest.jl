using GLMakie, NonlinearSolve, BenchmarkTools
using LinearAlgebra, Printf, StaticArrays
using KerrGeodesics

const CalibrationExt = Base.get_extension(KerrGeodesics, :CalibrationExt)


#function to get the state to normalized 0-ray state.
function prep_state(start_state, integrator)
    x0, x1, x2, x3, v0, v1, v2, v3 = start_state
    T = typeof(x0)
    local_metric = KerrGeodesics.metric(integrator)
    metric_tpl = KerrGeodesics.yield_inverse_metric(x0, x1, x2, x3, local_metric)
    #Normalize raised four-veloc's temporal part to 1 (assumed by integrators)
    w0, w1, w2, w3 = KerrGeodesics.mult_by_metric(metric_tpl, (v0, v1, v2, v3))
    v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0

    #AFTER this, norm the ray to be null.
    v0, v1, v2, v3 = KerrGeodesics.normalize_fourveloc(metric_tpl, v0, v1, v2, v3, norm = T(0), null = true)

    return @SVector [x0, x1, x2, x3, v0, v1, v2, v3]
end

function solve_implicit_midpoint_precise(state, dt::T, metric_instance; tol=1e-14) where T
    function residual(u, p)
        midpoint = (state .+ u) .* T(0.5)
        dstate_mid = KerrGeodesics.calculate_differential(midpoint, metric_instance)
        return u .- state .- dt .* dstate_mid
    end
    
    dstate_initial = KerrGeodesics.calculate_differential(state, metric_instance)
    u_initial = state .+ dt .* dstate_initial
    
    prob = NonlinearProblem(residual, u_initial)
    sol = solve(prob, SimpleNewtonRaphson(), abstol=tol)
    
    return sol.u
end

function run_adam_moulton_convergence_test(di = CalibrationExt.test_states, metric = CalibrationExt.test_metric, dtc = CalibrationExt.test_dtc)
    N_values = 1:12
    local_keys = collect(keys(di))
    num_states = length(local_keys)

    all_N = []
    all_diffs = []
    all_labels = []

    for (i, k) in enumerate(local_keys)
        local_state = di[k]
        state_N = []
        state_diffs = []

        for n in N_values
            integrator = AdamMoultonHeuretic(metric, dtc, n)

            gstate = prep_state(local_state, integrator)
            metric_tpl = KerrGeodesics.yield_inverse_metric(gstate[1], gstate[2], gstate[3], gstate[4], metric)
            dtc_cache = KerrGeodesics.initialize_cache(gstate, metric_tpl, KerrGeodesics.scaler(integrator))

            dt_ref, _ = KerrGeodesics.get_dt(gstate, metric, dtc)
            
            truth_state = solve_implicit_midpoint_precise(gstate, dt_ref, metric, tol=Float32(1e-14))
            
            nextval = KerrGeodesics.geodesic_step(gstate, integrator, dtc_cache)
            predicted_state = KerrGeodesics.state(nextval)

            diff = norm(predicted_state .- truth_state)

            push!(state_N, n)
            push!(state_diffs, diff)
        end
        
        append!(all_N, state_N)
        append!(all_diffs, state_diffs)
        append!(all_labels, fill(string(k), length(state_N)))
    end

    fig = Figure()
    ax = Axis(fig[1, 1], 
              xlabel = "N (Order of Adam-Moulton Heuristic)", 
              ylabel = "Difference (Norm of Error)",
              title = "Adam-Moulton Error vs. Implicit Midpoint",
              yscale = log10)
    
    colors = Makie.wong_colors()
    color_map = Dict(zip(local_keys, colors[1:num_states]))

    for (i, k) in enumerate(local_keys)
        indices = findall(x -> x == string(k), all_labels)
        current_N = [all_N[j] for j in indices]
        current_diffs = [all_diffs[j] for j in indices]
        
        scatter!(ax, current_N, current_diffs, 
                 color = color_map[k], 
                 markersize = 10,
                 label = string(k))
    end

    Legend(fig[1, 2], ax, "Starting States", framevisible = false)

    return fig
end

fig = run_adam_moulton_convergence_test()
display(fig)
save("exhibits/adam_moulton_order.png", fig, dpi = 600)