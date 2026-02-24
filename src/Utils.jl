#CPU-bound function that integrates a single geodesic.
"""
    integrate_single_geodesic!(output_buffer::AbstractArray{T}, state::AbstractVector{T}, 
    integrator::AbstractStateLessCustomIntegrator; norm = T(-1), null = false) where {T}

    Debug-oriented function that integrates a single geodesic, and stores its state at every timestep into output_buffer of shape
    [N_state, N_timesteps], where N_state is 8 for standard integrators and 16 for SplitHamiltonianHeuretic.

    The function can normalize the initial input using the norm and null keyword arguments. 
    Returns the last index written and a bool indicating whether the ray was redshifted.
"""
function integrate_single_geodesic!(output_buffer::AbstractArray{T}, start_state::AbstractVector{T}, 
    integrator::AbstractStateLessCustomIntegrator; norm::T = T(-1), null::Bool = false, do_norm::Bool = false) where {T}

    N = max_timesteps(integrator)
    @assert N == size(output_buffer, 2)
    @assert 8 == size(output_buffer, 1)

    x0, x1, x2, x3, v0, v1, v2, v3 = start_state[1], start_state[2], start_state[3], start_state[4],
                                       start_state[5], start_state[6], start_state[7], start_state[8]

    local_metric = metric(integrator)
    metric_tpl = yield_inverse_metric(x0, x1, x2, x3, local_metric)

    if do_norm
        w0, w1, w2, w3 = mult_by_metric(metric_tpl, (v0, v1, v2, v3))
        v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0
    end

    v0, v1, v2, v3 = normalize_fourveloc(metric_tpl, v0, v1, v2, v3; norm = norm, null = null)

    base_state = @SVector [x0, x1, x2, x3, v0, v1, v2, v3]
    gstate = initialize_state(base_state, integrator, metric_tpl)
    dtc_cache = initialize_cache(gstate, metric_tpl, scaler(integrator))

    phys_indices = SVector(1,2,3,4,5,6,7,8)

    last_index = N
    isr = false
    @fastmath for t in 1:N
        
        output_buffer[:, t] .= gstate[phys_indices]
        

        nextval = geodesic_step(gstate, integrator, dtc_cache)

        if isterminated(nextval)
            output_buffer[:, t:end] .= gstate[phys_indices]
            last_index = t
            isr = isredshifted(nextval)
            break
        end

        gstate = full_state(nextval)
    end

    return last_index, isr
end