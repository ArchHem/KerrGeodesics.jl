abstract type AbstractIntegratorBackend end

abstract type AbstractDEIntegrator <: AbstractIntegratorBackend end

abstract type AbstractCustomIntegrator <: AbstractIntegratorBackend end #for all of our custom needs

abstract type AbstractHeureticIntegrator <: AbstractCustomIntegrator end #uses some dt scaling heuretic

#concrete implementations: we need to check if abstract scaler will work on the GPU or not...

struct RK4HorizonHeuretic{T} <: AbstractHeureticIntegrator
    metric::KerrMetric{T} 
    stepscaler::HorizonHeureticScaler{T}
end
struct RK2HorizonHeuretic{T} <: AbstractHeureticIntegrator
    metric::KerrMetric{T} 
    stepscaler::HorizonHeureticScaler{T}
end

scaler(x::AbstractHeureticIntegrator) = x.stepscaler
metric(x::AbstractHeureticIntegrator) = x.metric

maxtimesteps(x::AbstractHeureticIntegrator) = maxtimesteps(scaler(x))
redshift_limit(x::AbstractHeureticIntegrator) = redshift_limit(scaler(x))

function geodesic_step(state, integrator::RK4HorizonHeuretic{T}) where T
    dtcontrol = scaler(integrator)
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    @fastmath begin
        #we can reuse vars here
        r = sqrt(yield_r2(x0, x1, x2, x3, metric(integrator)))
        dt = -get_dt(r, dtcontrol)
        dx0_1, dx1_1, dx2_1, dx3_1, dv0_1, dv1_1, dv2_1, dv3_1 = 
            calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric(integrator))
        dt_half = dt * T(0.5)
        dx0_2, dx1_2, dx2_2, dx3_2, dv0_2, dv1_2, dv2_2, dv3_2 = 
            calculate_differential(
                x0 + dt_half * dx0_1,
                x1 + dt_half * dx1_1,
                x2 + dt_half * dx2_1,
                x3 + dt_half * dx3_1,
                v0 + dt_half * dv0_1,
                v1 + dt_half * dv1_1,
                v2 + dt_half * dv2_1,
                v3 + dt_half * dv3_1,
                metric(integrator)
            )

        dx0_3, dx1_3, dx2_3, dx3_3, dv0_3, dv1_3, dv2_3, dv3_3 = 
            calculate_differential(
                x0 + dt_half * dx0_2,
                x1 + dt_half * dx1_2,
                x2 + dt_half * dx2_2,
                x3 + dt_half * dx3_2,
                v0 + dt_half * dv0_2,
                v1 + dt_half * dv1_2,
                v2 + dt_half * dv2_2,
                v3 + dt_half * dv3_2,
                metric(integrator)
            )

        dx0_4, dx1_4, dx2_4, dx3_4, dv0_4, dv1_4, dv2_4, dv3_4 = 
            calculate_differential(
                x0 + dt * dx0_3,
                x1 + dt * dx1_3,
                x2 + dt * dx2_3,
                x3 + dt * dx3_3,
                v0 + dt * dv0_3,
                v1 + dt * dv1_3,
                v2 + dt * dv2_3,
                v3 + dt * dv3_3,
                metric(integrator)
            )
        
        renorm_6 = 1 / T(6)
        renorm_3 = 1 / T(3)
        dx0 = renorm_6 * (dx0_1 + dx0_4) + renorm_3 * (dx0_2 + dx0_3)
        dx1 = renorm_6 * (dx1_1 + dx1_4) + renorm_3 * (dx1_2 + dx1_3)
        dx2 = renorm_6 * (dx2_1 + dx2_4) + renorm_3 * (dx2_2 + dx2_3)
        dx3 = renorm_6 * (dx3_1 + dx3_4) + renorm_3 * (dx3_2 + dx3_3)
        dv0 = renorm_6 * (dv0_1 + dv0_4) + renorm_3 * (dv0_2 + dv0_3)
        dv1 = renorm_6 * (dv1_1 + dv1_4) + renorm_3 * (dv1_2 + dv1_3)
        dv2 = renorm_6 * (dv2_1 + dv2_4) + renorm_3 * (dv2_2 + dv2_3)
        dv3 = renorm_6 * (dv3_1 + dv3_4) + renorm_3 * (dv3_2 + dv3_3)
        dstate = @SVector [dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3]
        newstate = @. state + dt * dstate
        escap = r > dtcontrol.r_stop 
        redshift = dx0 > dtcontrol.redshift_stop
    end
    
    res = StepResult(newstate, escap, redshift)
    return res
end

function geodesic_step(state, integrator::RK2HorizonHeuretic{T}) where T
    dtcontrol = scaler(integrator)
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    @fastmath begin
        #we can reuse vars here
        r = sqrt(yield_r2(x0, x1, x2, x3, metric(integrator)))
        dt = -get_dt(r, dtcontrol)
        dx0_1, dx1_1, dx2_1, dx3_1, dv0_1, dv1_1, dv2_1, dv3_1 = 
            calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric(integrator))
        dt_half = dt * T(0.5)
        dx0_2, dx1_2, dx2_2, dx3_2, dv0_2, dv1_2, dv2_2, dv3_2 = 
            calculate_differential(
                x0 + dt_half * dx0_1,
                x1 + dt_half * dx1_1,
                x2 + dt_half * dx2_1,
                x3 + dt_half * dx3_1,
                v0 + dt_half * dv0_1,
                v1 + dt_half * dv1_1,
                v2 + dt_half * dv2_1,
                v3 + dt_half * dv3_1,
                metric(integrator)
            )
        
        
        dx0 = dx0_2
        dx1 = dx1_2
        dx2 = dx2_2
        dx3 = dx3_2
        dv0 = dv0_2
        dv1 = dv1_2
        dv2 = dv2_2
        dv3 = dv3_2
        dstate = @SVector [dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3]
        newstate = @. state + dt * dstate
        escap = r > dtcontrol.r_stop 
        redshift = dx0 > dtcontrol.redshift_stop
    end
    
    res = StepResult(newstate, escap, redshift)
    return res
end
