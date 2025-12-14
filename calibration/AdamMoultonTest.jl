include("./example_states.jl")
using LinearAlgebra, GLMakie, Printf

#test how close an Adam-Moulton step is to the "true" implicit midstep is.

#function to get the state to normalized 0-ray state.
function prep_state(start_state, integrator)
    x0, x1, x2, x3, v0, v1, v2, v3 = start_state

    local_metric = metric(integrator)
    metric_tpl = yield_inverse_metric(x0, x1, x2, x3, local_metric)
    #Normalize raised four-veloc's temporal part to 1 (assumed by integrators)
    w0, w1, w2, w3 = mult_by_metric(metric_tpl, (v0, v1, v2, v3))
    v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0

    #AFTER this, norm the ray to be null.
    v0, v1, v2, v3 = normalize_fourveloc(metric_tpl, v0, v1, v2, v3, norm = norm, null = null)
end

function solve_implicit_midpoint_precise(state, dt::T, metric_instance; tol=1e-14, max_iter=200) where T
    dstate_initial = KerrGeodesics.calculate_differential(state, metric_instance)
    u_current = @. state + dt * dstate_initial
    
    for i in 1:max_iter
        midpoint = (state + u_current) * T(0.5)
        dstate_mid = KerrGeodesics.calculate_differential(midpoint, metric_instance)
        u_next = @. state + dt * dstate_mid
        diff = norm(u_next - u_current)
        if diff < tol
            return u_next
        end
        u_current = u_next
    end
    @warn "Ground truth solver did not converge to tolerance $tol"
    return u_current
end

function run_adam_moulton_convergence_test(di = test_states, metric = test_metric, dtc = test_dtc)
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

            gstate = local_state
            
            dt_ref, _ = KerrGeodesics.get_dt(local_state, metric, dtc)
            
            truth_state = solve_implicit_midpoint_precise(local_state, dt_ref, metric, tol=Float32(1e-14))
            
            nextval = KerrGeodesics.geodesic_step(gstate, integrator)
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