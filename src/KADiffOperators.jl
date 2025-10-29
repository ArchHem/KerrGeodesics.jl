#fallback method for arbitrary raybundles, with no spatial clustering...
@kernel unsafe_indices=true function ensemble_ODE_RK4!(output::AbstractArray{T},
    @Const(state::AbstractArray{T}),
    @Const(metric::KerrMetric{T}), 
    @Const(batch::SubStruct{V, H, NWarps, MWarps}), 
    @Const(dtcontrol::TimeStepScaler{T})) where {T, V, H, NWarps, MWarps}

    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    x0, x1, x2, x3, v0, v1, v2, v3 = @views state[lane, :, chunk]


    N = dtcontrol.maxtimesteps

    @fastmath for t in 1:N
        r2 = yield_r2(x0, x1, x2, x3, metric)
        #use RK4, we move backwards.
        dt = -get_dt(r2, dtcontrol)

        x0, x1, x2, x3, v0, v1, v2, v3 = ifelse(r2 > dtcontrol.r_stop || v0 > dtcontrol.redshift_stop, 
            (x0, x1, x2, x3, v0, v1, v2, v3),  
            RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt))

    end
    
    output[lane, :, chunk] .= (x0, x1, x2, x3, v0, v1, v2, v3)
    
    

end