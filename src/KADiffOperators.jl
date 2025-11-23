#fallback method for arbitrary raybundles, with no spatial clustering...


"""
    ensemble_ODE_RK4!(output::AbstractArray{T},
    @Const(state::AbstractArray{T}),
    @Const(metric::KerrMetric{T}), 
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}), 
    @Const(dtcontrol::TimeStepScaler{T})) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    KA Kernel that propegates an initial states using an uncodintional RK4 integrator, storing only the final valid value. 

"""
@kernel unsafe_indices=true function ensemble_ODE_RK4!(output::AbstractArray{T},
    @Const(state::AbstractArray{T}),
    @Const(metric::KerrMetric{T}), 
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}), 
    @Const(dtcontrol::TimeStepScaler{T})) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    x0, x1, x2, x3, v0, v1, v2, v3 = @views state[lane, :, chunk]
    #could add normlization here...

    N = dtcontrol.maxtimesteps

    @fastmath for t in 1:N
        r2 = yield_r2(x0, x1, x2, x3, metric)
        
        #use RK4, we move backwards.
        dt = -get_dt(r2, dtcontrol)

        dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3 = RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt)

        if r2 > dtcontrol.r_stop || dx0 > dtcontrol.redshift_stop
            break
        end

        x0, x1, x2, x3, v0, v1, v2, v3 = x0 + dt*dx0, x1 + dt*dx1, x2 + dt*dx2, x3 + dt*dx3,
                                            v0 + dt*dv0, v1 + dt*dv1, v2 + dt*dv2, v3 + dt*dv3

    end
    
    output[lane, :, chunk] .= (x0, x1, x2, x3, v0, v1, v2, v3)
    
    

end