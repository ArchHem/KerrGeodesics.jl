@inline function yield_inverse_metric(x0::T, x1::T, x2::T, x3::T, metric::KerrMetric{T}) where T
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
    f = -2 * M * r2 * r / (r4 + a2 * x3_2)
    common_subdiv = r2 + a2
    l0 = -T(1)
    l1 = (r * x1 + a * x2) / common_subdiv
    l2 = (r * x2 - a * x1) / common_subdiv
    l3 = x3 / r

    fl0 = f * l0
    fl1 = f * l1
    fl2 = f * l2

    u00 =  -1 + fl0 * l0
    u10 = fl0 * l1
    u20 =  fl0 * l2
    u30 =  fl0 * l3
    u11 =  1 + fl1 * l1
    u21 = l2 * fl1
    u31 =  l3 * fl1
    u22 = 1 + fl2 * l2
    u32 = fl2 * l3
    u33 =  1 + f * l3 * l3

    return u00, u10, u20, u30, u11, u21, u31, u22, u32, u33
end

@inline function mult_by_metric(metric_tuple::NTuple{N, T}, v) where {N, T}
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
@inline function normalize_fourveloc(metric_tuple::NTuple{N, T}, v0::T, v1::T, v2::T, v3::T; norm = T(0), null = true) where {N, T}
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
            scaler = (-mixed_contrib - sqrt(mixed_contrib^2 - 4 * (zero_contrib-norm) * non_zero_contrib)) / (2 * (zero_contrib-norm))
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

@inline function yield_innerprod(metric_tuple::NTuple{N, T}, v0::T, v1::T, v2::T, v3::T) where {N, T}
    u00, u10, u20, u30, u11, u21, u31, u22, u32, u33 = metric_tuple

    innerprod = @fastmath u00 * v0 * v0 + v1 * v1 * u11 + v2 * v2 * u22 + v3 * v3 * u33 + 
        2 * (u10  * v1 * v0 + u20 * v2 * v0 + u30 * v3 * v0 + v1 * v2 * u21 + v1 * v3 * u31 + u32 * v3 * v2)
    return innerprod
end


@inline function yield_innerprod(metric_tuple::NTuple{N, T}, v, w) where {N,T}
    @fastmath begin
        v0, v1, v2, v3 = v
        w0, w1, w2, w3 = mult_by_metric(metric_tuple, w)

        innerprod = w0 * v0 + v1 * w1 + v2 * w2 + v3 * w3
    end
    return innerprod
end

@inline function yield_determinant(metric_tuple::NTuple{N, T}) where {N, T}
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

struct PinHoleCamera{T}
    position::SVector{4, T}
    lowered_velocity::SVector{4, T}
    lowered_pointing::SVector{4, T}
    lowered_upward::SVector{4, T}
    lowered_rightward::SVector{4, T}
    horizontal_angle::T
    vertical_angle::T
    horizontal_px::Int
    vertical_px::Int
end

function PinHoleCamera(position::AbstractArray{T}, lowered_velocity::AbstractArray{T}, 
    lowered_pointing::AbstractArray{T}, lowered_upward::AbstractArray{T}, 
    metric::KerrMetric{T},
    horizontal_angle::T, vertical_angle::T, horizontal_px::Int, vertical_px::Int) where T

    #Use the modified version found in:
    #https://arxiv.org/pdf/1410.7775

    inv_metric_value = yield_inverse_metric(position..., metric)
    
    metric_det = 1/yield_determinant(inv_metric_value)
    
    #I am not sure if this -1 is correct, see:
    #Pg 202 of Charles W. Misner, Kip S. Thorne, and John Archibald
    # Wheeler. Gravitation.
    levis_norm = -1/sqrt(-metric_det)

    #I think its enough if we ensure timelike behaviour, and spacelike for the later...

    @assert yield_innerprod(inv_metric_value, lowered_velocity...) < 0
    @assert lowered_pointing[1] == T(0)
    @assert lowered_upward[1] == T(0)

    #now, do a round of GS

    lowered_velocity_tpl = normalize_fourveloc(inv_metric_value, lowered_velocity...; norm = T(-1), null = false)
    
    #since lowered_velocity is timelike, there is a sign flip here.
    lowered_pointing_tpl_interim = tuple(lowered_pointing...) .+ 
        yield_innerprod(inv_metric_value, lowered_velocity_tpl, tuple(lowered_pointing...)) .* lowered_velocity_tpl
    
    lowered_pointing_tpl = normalize_fourveloc(inv_metric_value, lowered_pointing_tpl_interim...; norm = T(1), null = false)
    
    lowered_upward_tpl_interim = tuple(lowered_upward...) .+ 
        yield_innerprod(inv_metric_value, lowered_velocity_tpl, tuple(lowered_upward...)) .* lowered_velocity_tpl .-
        yield_innerprod(inv_metric_value, lowered_pointing_tpl, tuple(lowered_upward...)) .* lowered_pointing_tpl

    lowered_upward_tpl = normalize_fourveloc(inv_metric_value, lowered_upward_tpl_interim...; norm = T(1), null = false)
    
    e0 = SVector{4, T}(lowered_velocity_tpl)
    e1 = SVector{4, T}(lowered_pointing_tpl)
    e2 = SVector{4, T}(lowered_upward_tpl)

    #do the levis-ciita generation

    e3_up = zeros(T, 4)

    for a in 1:4
        for i in 1:4
            for j in 1:4
                for k in 1:4
                    e3_up[a] += e0[i]*e1[j]*e2[k]*levis_norm * levicivita4(i, j, k, a)
                end
            end
        end
    end

    e3_up = SVector{4, T}(e3_up)
    
    e3_interim = mult_by_metric(inv_metric_value, e3_up)
    e3 = SVector{4, T}(normalize_fourveloc(inv_metric_value, e3_interim...; norm = T(1), null = false))

    pos = SVector{4, T}(position)

    return PinHoleCamera{T}(pos, e0, e1, e2, e3, horizontal_angle, vertical_angle, horizontal_px, vertical_px)

end



