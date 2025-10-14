#Lets try moving the vector product machinery in here.
@generated function calculate_innerprod!(
    innerprod_buffer::AbstractArray{T},
    dpos_buffer::AbstractArray{T},
    pos_buffer::AbstractArray{T},
    veloc_buffer::AbstractArray{T},
    batch::BatchInfo{V},
    metric::KerrMetric{T}
    ) where {T, V}
    
    res = quote
        a = metric.a
        M = metric.M
        @turbo thread = false for idx in 1:size(pos_buffer, 3)
            for lane in 1:$V  #hpefuly this unrolls
                
                x0 = pos_buffer[lane, 1, idx]
                x1 = pos_buffer[lane, 2, idx]
                x2 = pos_buffer[lane, 3, idx]
                x3 = pos_buffer[lane, 4, idx]

                
                R2 = x1 * x1 + x2 * x2 + x3 * x3
                sub1 = R2 - a * a
                r2 = T(0.5) * (sub1 + sqrt(sub1 * sub1 + T(4) * a * a * x3 * x3))

                r = sqrt(r2)
                f = -2 * M * r2 * r / (r2 * r2 + a * a * x3 * x3)
                common_subdiv = r2 + a * a
                l0 = -T(1)
                l1 = (r * x1 + a * x2) / common_subdiv
                l2 = (r * x2 - a * x1) / common_subdiv
                l3 = x3 / r
                u00 =  -1 + f * l0 * l0
                u10 = f * l1 * l0
                u20 =  f * l2 * l0
                u30 =  f * l3 * l0
                u11 =  1 + f * l1 * l1
                u21 = l2 * f * l1
                u31 =  l3 * f * l1
                u22 = 1 + f * l2 * l2
                u32 = f * l3 * l2
                u33 =  1 + f * l3 * l3

                
                v0 = veloc_buffer[lane, 1, idx]
                v1 = veloc_buffer[lane, 2, idx]
                v2 = veloc_buffer[lane, 3, idx]
                v3 = veloc_buffer[lane, 4, idx]

                w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
                w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
                w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
                w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3
                
                dpos_buffer[lane, 1, idx] = w0
                dpos_buffer[lane, 2, idx] = w1
                dpos_buffer[lane, 3, idx] = w2
                dpos_buffer[lane, 4, idx] = w3
                
                
                innerprod_buffer[lane, idx] = T(-0.5) * (w0*v0 + w1*v1 + w2*v2 + w3*v3)
            end
        end

        return nothing
    end

    return res
end

function calculate_differentials_backward!(innerprod_buffer::AbstractArray{T},
    d_innerprod_buffer::AbstractArray{T},
    dpos_buffer::AbstractArray{T},
    pos_buffer::AbstractArray{T},
    dveloc_buffer::AbstractArray{T},
    veloc_buffer::AbstractArray{T},
    batch::BatchInfo{V},
    metric::KerrMetric{T}) where {T, V}

    #use Enzyme to calculate backwards pass, calculting dveloc_u = d/dx_u (v_i g^ij v_j)

    dveloc_buffer .= T(0)
    d_innerprod_buffer .= T(1)
    Enzyme.autodiff(Reverse, calculate_innerprod!, DuplicatedNoNeed(innerprod_buffer, d_innerprod_buffer),
        #The Enzyme binder has a limitation on this: we would ideally use views from V * 8 * N arrays
        Const(dpos_buffer), DuplicatedNoNeed(pos_buffer, dveloc_buffer), Const(veloc_buffer), Const(batch), Const(metric))

    return nothing

    

end

