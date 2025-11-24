#CPU-bound function that integrates a single geodesic.

"""
    integrate_single_geodesic!(output_buffer::AbstractArray{T}, state::AbstractVector{T}, dtcontrol::TimeStepScaler{T},
    metric::KerrMetric{T}; norm = T(-1), null = false) where {T}

    Debug-orineted function that integrates a single geodesic, and stores its state at every timestep into output_buffer of shape
    [8, N_timesteps].


    The function can normalize the initial input using the norm and null keyword arguments. 
"""
function integrate_single_geodesic!(output_buffer::AbstractArray{T}, state::AbstractVector{T}, dtcontrol::TimeStepScaler{T},
    metric::KerrMetric{T}; norm = T(-1), null = false) where {T}


    N = dtcontrol.maxtimesteps
    @assert N == size(output_buffer, 2)

    x0, x1, x2, x3, v0, v1, v2, v3 = state

    metric_tpl = yield_inverse_metric(x0, x1, x2, x3, metric)
    v0, v1, v2, v3 = normalize_fourveloc(metric_tpl, v0, v1, v2, v3, norm = norm, null = null)
    metric_tpl = yield_inverse_metric(x0, x1, x2, x3, metric)
    w0, w1, w2, w3 = mult_by_metric(metric_tpl, (v0, v1, v2, v3))

    v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0

    @fastmath for t in 1:N

        output_buffer[1, t] = x0
        output_buffer[2, t] = x1
        output_buffer[3, t] = x2
        output_buffer[4, t] = x3

        output_buffer[5, t] = v0
        output_buffer[6, t] = v1
        output_buffer[7, t] = v2
        output_buffer[8, t] = v3
        r = sqrt(yield_r2(x0, x1, x2, x3, metric))
        
        #use RK4, we move backwards.
        dt = -get_dt(r, dtcontrol)

        dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3 = RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt)

        if r > dtcontrol.r_stop || dx0 > dtcontrol.redshift_stop
            output_buffer[1, t:end] .= x0
            output_buffer[2, t:end] .= x1
            output_buffer[3, t:end] .= x2
            output_buffer[4, t:end] .= x3

            output_buffer[5, t:end] .= v0
            output_buffer[6, t:end] .= v1
            output_buffer[7, t:end] .= v2
            output_buffer[8, t:end] .= v3
            break
        end

        x0, x1, x2, x3, v0, v1, v2, v3 = x0 + dt*dx0, x1 + dt*dx1, x2 + dt*dx2, x3 + dt*dx3,
                                            v0 + dt*dv0, v1 + dt*dv1, v2 + dt*dv2, v3 + dt*dv3
        
        

    end

    return nothing
end