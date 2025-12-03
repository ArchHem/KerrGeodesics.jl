
"""
    yield_inverse_metric(x0, x1, x2, x3, metric::KerrMetric{T}) where T

    Calculates the inverse Kerr metric, whcih is stored as a tuple 
        (u00, u10, u20, u30, u11, u21, u31, u22, u32, u33).
"""
@inline function yield_inverse_metric(x0, x1, x2, x3, metric::KerrMetric{T}) where T
    a = metric.a
    M = metric.M
    x1_2 = x1 * x1
    x2_2 = x2 * x2
    x3_2 = x3 * x3

    R2 = x1_2 + x2_2 + x3_2
    a2 = a*a
    sub1 = R2 - a2
    r2 = T(0.5) * (sub1 + sqrt(sub1^2 + T(4) * a2 * x3_2))
    r4 = r2 * r2
    r = sqrt(r2)
    f = 2 * M * r2 * r / (r4 + a2 * x3_2)
    common_subdiv = r2 + a2
    l0 = -T(1)
    l1 = (r * x1 + a * x2) / common_subdiv
    l2 = (r * x2 - a * x1) / common_subdiv
    l3 = x3 / r

    fl0 = f * l0
    fl1 = f * l1
    fl2 = f * l2

    u00 =  -1 - fl0 * l0
    u10 = -fl0 * l1
    u20 =  -fl0 * l2
    u30 =  -fl0 * l3
    u11 =  1 - fl1 * l1
    u21 = -l2 * fl1
    u31 = -l3 * fl1
    u22 = 1 - fl2 * l2
    u32 = -fl2 * l3
    u33 =  1 - f * l3 * l3

    return u00, u10, u20, u30, u11, u21, u31, u22, u32, u33
end

"""
    mult_by_metric(metric_tuple, v) where {N, T}

    Given a 10-length represnetation (metric_tuple) of a symmetric 4x4 matrix A,
        calculates the matrix product A * v
"""
@inline function mult_by_metric(metric_tuple, v)
    u00, u10, u20, u30, u11, u21, u31, u22, u32, u33 = metric_tuple
    v0, v1, v2, v3 = v

    w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
    w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
    w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
    w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3

    return w0, w1, w2, w3
end

#normalizes the four-velocity to be null, such that v0 is left unmodified.
#to do this, we use the fact the the v0 component only contributes to certain elements.


"""
    normalize_fourveloc(metric_tuple, v0::T, v1::T, v2::T, v3::T; norm = T(0), null = true) where {N, T}

    Given a four-velocity and the metric tuple, 

    This function does not assume that the metric tuple is the inevrse or metric, neither that v0, v1, v2, v3 is raised or lwered, just that
        we can take their innerproduct in a valid manner (ie. inverse metric and lowered velocity, or metric and raised velocity)

    The function has two distinct execution paths:
        if null = FALSE, the ray is assumed to be non-null, and it gets normalized by a simple scalar multiplication 
            so thats its innerproduct with the metric becomes norm
        if null = TRUE, the ray is assumed to be null. Since we can not employ scalar multiplication here, we instead assume that the tay can be made null,
            by scalar multiplying its v1, v2, v3 components. This is done via an EXACT fomrula for an arbitary metric, assuming that v0 is the timelike component.
            Please note that for such, one should set norm = 0
"""
@inline function normalize_fourveloc(metric_tuple, v0::T, v1::T, v2::T, v3::T; norm = T(0), null = true) where T
    u00, u10, u20, u30, u11, u21, u31, u22, u32, u33 = metric_tuple

    if null
        @fastmath begin
            zero_contrib = u00 * v0 * v0
            mixed_contrib = v0 * 2 * (u10  * v1 + u20 * v2 + u30 * v3)
            non_zero_contrib = v1 * v1 * u11 + v2 * v2 * u22 + v3 * v3 * u33 + 2 * (
                v1 * v2 * u21 + v1 * v3 * u31 + u32 * v3 * v2
            )
            
            #we chose a scaling quantity "a" such that: zero_contrib constributes zero (duh) 
            #mixed_contrib contributes as a * , non_zero_contrib contributes as a^2.
            #thus, zero_contrib + a * mixed_contrib + a^2 * non_zero_contrib = norm

            #for reasons of convinience, we choose a to be the positive root - we have a negative zero contrib
            #This is technicaly not stable, but should be fine for fp32 maybe even 16
            scaler_1 = (-mixed_contrib - sqrt(mixed_contrib^2 - 4 * (zero_contrib-norm) * non_zero_contrib)) / (2 * (non_zero_contrib))
            scaler_2 = (-mixed_contrib + sqrt(mixed_contrib^2 - 4 * (zero_contrib-norm) * non_zero_contrib)) / (2 * (non_zero_contrib))
            scaler = max(scaler_1, scaler_2)
            w1 = scaler * v1
            w2 = scaler * v2
            w3 = scaler * v3

        end

        return v0, w1, w2, w3
    else
        #just use standard linalg norming
        @fastmath begin
            u = yield_innerprod(metric_tuple, v0, v1, v2, v3)
            w0 = sqrt(norm/u) * v0
            w1 = sqrt(norm/u) * v1
            w2 = sqrt(norm/u) * v2
            w3 = sqrt(norm/u) * v3
        end
        return w0, w1, w2, w3
    end
end

@inline function yield_innerprod(metric_tuple, v0::T, v1::T, v2::T, v3::T) where {T}
    u00, u10, u20, u30, u11, u21, u31, u22, u32, u33 = metric_tuple

    innerprod = @fastmath u00 * v0 * v0 + v1 * v1 * u11 + v2 * v2 * u22 + v3 * v3 * u33 + 
        2 * (u10  * v1 * v0 + u20 * v2 * v0 + u30 * v3 * v0 + v1 * v2 * u21 + v1 * v3 * u31 + u32 * v3 * v2)
    return innerprod
end


"""
    yield_innerprod(metric_tuple, v, w)

    Calculates the generalized inner product m^ij v_i w_j where m^ij is assumed to be a symmetric 4x4 matrix, 
    stored in a trinagular format as a 10-tuple

"""
@inline function yield_innerprod(metric_tuple, v, w)
    @fastmath begin
        v0, v1, v2, v3 = v
        w0, w1, w2, w3 = mult_by_metric(metric_tuple, w)

        innerprod = w0 * v0 + v1 * w1 + v2 * w2 + v3 * w3
    end
    return innerprod
end



"""
    yield_determinant(metric_tuple)
    Yields the the determinant of a 4x4 symmetric matrix, 
    stored as 10 tuple in tringular format.

    For the Kerr Metric, this is uniquely +/- 1 everywhere.

"""
@inline function yield_determinant(metric_tuple)
    u00, u10, u20, u30, u11, u21, u31, u22, u32, u33 = metric_tuple

    @fastmath begin
        #double-check this
        C11 = u11 * (u22*u33 - u32*u32) - 
              u21 * (u21*u33 - u31*u32) + 
              u31 * (u21*u32 - u31*u22)

        C12 = u10 * (u22*u33 - u32*u32) - 
              u20 * (u21*u33 - u31*u32) + 
              u30 * (u21*u32 - u31*u22)

        C13 = u10 * (u21*u33 - u31*u31) - 
              u20 * (u11*u33 - u31*u31) + 
              u30 * (u11*u31 - u21*u31)

        C14 = u10 * (u21*u32 - u31*u22) - 
              u20 * (u11*u32 - u21*u31) + 
              u30 * (u11*u22 - u21*u21)
        
        det_val = u00*C11 - u10*C12 + u20*C13 - u30*C14
    end

    return det_val
end