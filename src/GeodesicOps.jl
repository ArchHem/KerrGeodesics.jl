@inline function calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric::KerrMetric{T}) where T

    #kerr metric depends not on the x0 coord, so no need to create a non-primal for it.
    x1_augmented = ForwardDiff.Dual(x1, (T(1), T(0), T(0)))
    x2_augmented = ForwardDiff.Dual(x2, (T(0), T(1), T(0)))
    x3_augmented = ForwardDiff.Dual(x3, (T(0), T(0), T(1)))

    inverse_metric_tpl = yield_inverse_metric(x0, x1_augmented, x2_augmented, x3_augmented, metric)

    #calculate the primal, 

    primal_inverse_metric = ForwardDiff.value.(inverse_metric_tpl)
    v_tpl = (v0, v1, v2, v3)
    w0, w1, w2, w3 = mult_by_metric(primal_inverse_metric, v_tpl)

    #dx0_inverse_metric = ForwardDiff.partials.(inverse_metric_tpl, 1)
    dx1_inverse_metric = ForwardDiff.partials.(inverse_metric_tpl, 1)
    dx2_inverse_metric = ForwardDiff.partials.(inverse_metric_tpl, 2)
    dx3_inverse_metric = ForwardDiff.partials.(inverse_metric_tpl, 3)

    dv0 = T(0)
    dv1 = T(-0.5) * yield_innerprod(dx1_inverse_metric, v0, v1, v2, v3)
    dv2 = T(-0.5) * yield_innerprod(dx2_inverse_metric, v0, v1, v2, v3)
    dv3 = T(-0.5) * yield_innerprod(dx3_inverse_metric, v0, v1, v2, v3)

    return (w0, w1, w2, w3, dv0, dv1, dv2, dv3)
    
end

@inline function calculate_differential(state::NTuple{N, T}, metric::KerrMetric{T}) where {N,T}
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    dstate = calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric)
    return dstate
end

@inline function yield_r2(x0::T, x1::T, x2::T, x3::T, metric::KerrMetric{T}) where T

    a = metric.a
    M = metric.M
    @fastmath begin
        x1_2 = x1*x1
        x2_2 = x2*x2
        x3_2 = x3*x3
        R2 = x1_2 + x2_2 + x3_2
        a2 = a*a
        sub1 = R2 - a2
        r2 = T(0.5) * (sub1 + sqrt(sub1^2 + T(4) * a2 * x3_2))
    end

    return r2

end

@inline function RK4step(x0::T, x1::T, x2::T, x3::T, v0::T, v1::T, v2::T, v3::T, 
                         metric::KerrMetric{T}, dt::T) where T

                        #for some reason, the original, tuble-base logic allocated here.
    @fastmath begin 
        dx0_1, dx1_1, dx2_1, dx3_1, dv0_1, dv1_1, dv2_1, dv3_1 = 
            calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric)
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
                metric
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
                metric
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
                metric
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
    end
    
    return (dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3)
end