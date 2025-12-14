#CPU-bound function that integrates a single geodesic.
"""
    integrate_single_geodesic!(output_buffer::AbstractArray{T}, state::AbstractVector{T}, 
    integrator::AbstractStateLessCustomIntegrator; norm = T(-1), null = false) where {T}

    Debug-oriented function that integrates a single geodesic, and stores its state at every timestep into output_buffer of shape
    [8, N_timesteps].

    The function can normalize the initial input using the norm and null keyword arguments. 
    Returns the last index written by the integrator.
"""
function integrate_single_geodesic!(output_buffer::AbstractArray{T}, start_state::AbstractVector{T}, 
    integrator::AbstractStateLessCustomIntegrator; norm = T(-1), null = false) where {T}

    N = max_timesteps(integrator)
    @assert N == size(output_buffer, 2)

    x0, x1, x2, x3, v0, v1, v2, v3 = start_state

    local_metric = metric(integrator)
    metric_tpl = yield_inverse_metric(x0, x1, x2, x3, local_metric)
    #Normalize raised four-veloc's temporal part to 1 (assumed by integrators)
    w0, w1, w2, w3 = mult_by_metric(metric_tpl, (v0, v1, v2, v3))
    v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0

    #AFTER this, norm the ray to be null.
    v0, v1, v2, v3 = normalize_fourveloc(metric_tpl, v0, v1, v2, v3, norm = norm, null = null)
    

    gstate = @SVector [x0, x1, x2, x3, v0, v1, v2, v3]

    last_index = N
    isr = false
    @fastmath for t in 1:N
        # Store current state
        output_buffer[1, t] = gstate[1]
        output_buffer[2, t] = gstate[2]
        output_buffer[3, t] = gstate[3]
        output_buffer[4, t] = gstate[4]
        output_buffer[5, t] = gstate[5]
        output_buffer[6, t] = gstate[6]
        output_buffer[7, t] = gstate[7]
        output_buffer[8, t] = gstate[8]

        # Take a step using the integrator
        nextval = geodesic_step(gstate, integrator)

        # Check termination
        if isterminated(nextval)
            # Fill remaining buffer with final state
            output_buffer[1, t:end] .= gstate[1]
            output_buffer[2, t:end] .= gstate[2]
            output_buffer[3, t:end] .= gstate[3]
            output_buffer[4, t:end] .= gstate[4]
            output_buffer[5, t:end] .= gstate[5]
            output_buffer[6, t:end] .= gstate[6]
            output_buffer[7, t:end] .= gstate[7]
            output_buffer[8, t:end] .= gstate[8]
            last_index = t
            isr = isredshifted(nextval)
            break
        end

        # Update state
        gstate = state(nextval)
    end

    return last_index, isr
end