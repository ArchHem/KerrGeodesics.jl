@kernel unsafe_indices=true function ensemble_ODE_RK4!(
    state::AbstractArray{T},
    @Const(metric::KerrMetric{T}), 
    @Const(batch::BatchInfo{V}), 
    @Const(dtcontrol::TimeStepScaler{T})) where {T, V}

    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V) .+ Int32(1)

    x0, x1, x2, x3, v0, v1, v2, v3 = @views state[lane, :, chunk]


    N = dtcontrol.maxtimesteps

    for t in 1:N
        r2 = yield_r2(x0, x1, x2, x3, metric)
        #use RK4
        dt = get_dt(r2, dtcontrol)

        x0, x1, x2, x3, v0, v1, v2, v3 = ifelse(r2 > dtcontrol.r_stop || v0 > dtcontrol.redshift_stop, 
            (x0, x1, x2, x3, v0, v1, v2, v3),  
            RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt))

    end
    
    state[lane, 1, chunk] = x0
    state[lane, 2, chunk] = x1
    state[lane, 3, chunk] = x2
    state[lane, 4, chunk] = x3
    state[lane, 5, chunk] = v0
    state[lane, 6, chunk] = v1
    state[lane, 7, chunk] = v2
    state[lane, 8, chunk] = v3
    
    

end