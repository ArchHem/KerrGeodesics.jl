
"""
    calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric::KerrMetric{T}) where T

    Calculates the RHS of the EOM ds_i / dγ = f(s) where γ is an arbitarry affine pareameter along the geodesic.

    This function was gnerated via a combination of Symbolics.jl, FastDifferentation.jl and via manual CSE and cost analysis.
"""
@inline function calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric::KerrMetric{T}) where T

    t, x, y, z = x0, x1, x2, x3
    #just so I dont get confused...
    M = metric.M
    a = metric.a
    
    @fastmath begin 
        a2 = a^2
        x2 = x^2
        y2 = y^2
        z2 = z^2

        R2 = x2 + y2 + z2

        R2_x = T(2)*x
        R2_y = T(2)*y
        R2_z = T(2)*z

        S1 = R2 - a2
        S2 = S1^2 + T(4) * a2 * z2

        S1_x = R2_x
        S1_y = R2_y
        S1_z = R2_z

        S2_x = T(2) * S1 * S1_x
        S2_y = T(2) * S1 * S1_y
        S2_z = T(2) * S1 * S1_z + T(8) * a2 * z

        S2_sqr = sqrt(S2)
        r2 = T(0.5) * (S1 + S2_sqr)
        r = sqrt(r2)

        const_prefac_1 = T(0.25)/S2_sqr

        r2_x = T(0.5) * S1_x + const_prefac_1 * S2_x
        r2_y = T(0.5) * S1_y + const_prefac_1 * S2_y
        r2_z = T(0.5) * S1_z + const_prefac_1 * S2_z

        r_m1 = T(1)/r
        const_prefac_2 = T(0.5) * r_m1

        r_x = const_prefac_2 * r2_x
        r_y = const_prefac_2 * r2_y
        r_z = const_prefac_2 * r2_z

        S4 = T(2) * M * r^2 * r 

        S3 = r2*r2 + a2 * z2
        S3_m1 = 1/S3

        S5 = r2 + a2
        S5_m1 = 1/S5
        f = S4 * S3_m1

        S6 = r*x + a*y
        S7 = r*y - a*x


        l0 = T(-1)
        l1 = S6 * S5_m1
        l2 = S7 * S5_m1
        l3 = z * r_m1

        fl0 = -f #use the fact that this is just a sign invertion
        fl1 = f * l1
        fl2 = f * l2
        fl3 = f * l3

        u00 = -(T(1) + f)
        u10 = -(fl0 * l1)
        u20 = -(fl0 * l2)
        u30 = -(fl0 * l3)

        u11 = T(1) -(fl1 * l1)
        u21 = -(fl1 * l2)
        u31 = -(fl1 * l3)

        u22 = T(1) -(fl2 * l2)
        u32 = -(fl2 * l3)

        u33 = T(1) -(fl3 * l3)

        w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
        w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
        w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
        w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3

        l0_x = T(0)
        l0_y = T(0)
        l0_z = T(0)

        cons_prefac_4 = T(6) * M * r2

        S4_x = cons_prefac_4 * r_x
        S4_y = cons_prefac_4 * r_y
        S4_z = cons_prefac_4 * r_z

        const_prefac_3 = T(4) * r2 * r #4 r^3

        S3_x = const_prefac_3 * r_x
        S3_y = const_prefac_3 * r_y
        S3_z = const_prefac_3 * r_z + T(2) * a2 * z

        S3_m2 = S3_m1 * S3_m1

        f_x = (S4_x * S3 - S3_x * S4) * S3_m2
        f_y = (S4_y * S3 - S3_y * S4) * S3_m2
        f_z = (S4_z * S3 - S3_z * S4) * S3_m2

        S5_m2 = S5_m1 * S5_m1

        
        S5_x = T(2) * r * r_x
        S5_y = T(2) * r * r_y
        S5_z = T(2) * r * r_z


        S6_x = r + x * r_x
        S6_y = a + x * r_y
        S6_z = x * r_z

        S7_x = r_x * y - a
        S7_y = r_y * y + r
        S7_z = r_z * y

        l1_x = (S6_x * S5 - S5_x * S6) * S5_m2
        l1_y = (S6_y * S5 - S5_y * S6) * S5_m2
        l1_z = (S6_z * S5 - S5_z * S6) * S5_m2

        l2_x = (S7_x * S5 - S5_x * S7) * S5_m2
        l2_y = (S7_y * S5 - S5_y * S7) * S5_m2
        l2_z = (S7_z * S5 - S5_z * S7) * S5_m2

        r_m2 = r_m1 * r_m1

        l3_x = -z * r_x * r_m2
        l3_y = -z * r_y * r_m2
        l3_z = (r - r_z * z) * r_m2

        #derivative wrt to x0 is 0, i.e. those can be skipped
        dv0 = T(0)

        
        u00_x = f_x + T(2) * fl0 * l0_x
        u10_x = f_x * l1 * l0 + fl0 * l1_x #derivative of ls is 0 wrt to 0
        u20_x = f_x * l2 * l0 + fl0 * l2_x 
        u30_x = f_x * l3 * l0 + fl0 * l3_x 

        u11_x = f_x * l1 * l1 + T(2) * fl1 * l1_x
        u21_x = f_x * l1 * l2 + fl1 * l2_x + fl2 * l1_x
        u31_x = f_x * l1 * l3 + fl1 * l3_x + fl3 * l1_x

        u22_x = f_x * l2 * l2 + T(2) * fl2 * l2_x
        u32_x = f_x * l3 * l2 + fl3 * l2_x + fl2 * l3_x

        u33_x = f_x * l3 * l3 + T(2) * fl3 * l3_x

        dv1 = T(0.5) * (u00_x * v0 * v0 + v1 * v1 * u11_x + v2 * v2 * u22_x + v3 * v3 * u33_x + 
            2 * (u10_x  * v1 * v0 + u20_x * v2 * v0 + u30_x * v3 * v0 + 
                v1 * v2 * u21_x + v1 * v3 * u31_x + u32_x * v3 * v2))


        u00_y = f_y + T(2) * fl0 * l0_y
        u10_y = f_y * l1 * l0 + fl0 * l1_y #derivative of ls is 0 wrt to 0
        u20_y = f_y * l2 * l0 + fl0 * l2_y 
        u30_y = f_y * l3 * l0 + fl0 * l3_y 

        u11_y = f_y * l1 * l1 + T(2) * fl1 * l1_y
        u21_y = f_y * l1 * l2 + fl1 * l2_y + fl2 * l1_y
        u31_y = f_y * l1 * l3 + fl1 * l3_y + fl3 * l1_y

        u22_y = f_y * l2 * l2 + T(2) * fl2 * l2_y
        u32_y = f_y * l3 * l2 + fl3 * l2_y + fl2 * l3_y

        u33_y = f_y * l3 * l3 + T(2) * fl3 * l3_y

        dv2 = T(0.5) * (u00_y * v0 * v0 + v1 * v1 * u11_y + v2 * v2 * u22_y + v3 * v3 * u33_y + 
            2 * (u10_y  * v1 * v0 + u20_y * v2 * v0 + u30_y * v3 * v0 + 
                v1 * v2 * u21_y + v1 * v3 * u31_y + u32_y * v3 * v2))

        u00_z = f_z + T(2) * fl0 * l0_z
        u10_z = f_z * l1 * l0 + fl0 * l1_z #derivative of ls is 0 wrt to 0
        u20_z = f_z * l2 * l0 + fl0 * l2_z 
        u30_z = f_z * l3 * l0 + fl0 * l3_z 

        u11_z = f_z * l1 * l1 + T(2) * fl1 * l1_z
        u21_z = f_z * l1 * l2 + fl1 * l2_z + fl2 * l1_z
        u31_z = f_z * l1 * l3 + fl1 * l3_z + fl3 * l1_z

        u22_z = f_z * l2 * l2 + T(2) * fl2 * l2_z
        u32_z = f_z * l3 * l2 + fl3 * l2_z + fl2 * l3_z

        u33_z = f_z * l3 * l3 + T(2) * fl3 * l3_z

        dv3 = T(0.5) * (u00_z * v0 * v0 + v1 * v1 * u11_z + v2 * v2 * u22_z + v3 * v3 * u33_z + 
            2 * (u10_z  * v1 * v0 + u20_z * v2 * v0 + u30_z * v3 * v0 + 
                v1 * v2 * u21_z + v1 * v3 * u31_z + u32_z * v3 * v2))



        


    end

    return (w0, w1, w2, w3, dv0, dv1, dv2, dv3)
    
end

"""
    calculate_differential_and_geom(x0, x1, x2, x3, v0, v1, v2, v3, metric::KerrMetric{T}) where T -> (dstate, geom_cache)

    Calculates the RHS of the EOM ds_i / dγ = f(s) where γ is an arbitarry affine pareameter along the geodesic and useful geometric properties of the Kerr metric (r)

    This function was gnerated via a combination of Symbolics.jl, FastDifferentation.jl and via manual CSE and cost analysis.
"""
@inline function calculate_differential_and_geom(x0, x1, x2, x3, v0, v1, v2, v3, metric::KerrMetric{T}) where T

    t, x, y, z = x0, x1, x2, x3
    #just so I dont get confused...
    M = metric.M
    a = metric.a
    
    @fastmath begin 
        a2 = a^2
        x2 = x^2
        y2 = y^2
        z2 = z^2

        R2 = x2 + y2 + z2

        R2_x = T(2)*x
        R2_y = T(2)*y
        R2_z = T(2)*z

        S1 = R2 - a2
        S2 = S1^2 + T(4) * a2 * z2

        S1_x = R2_x
        S1_y = R2_y
        S1_z = R2_z

        S2_x = T(2) * S1 * S1_x
        S2_y = T(2) * S1 * S1_y
        S2_z = T(2) * S1 * S1_z + T(8) * a2 * z

        S2_sqr = sqrt(S2)
        r2 = T(0.5) * (S1 + S2_sqr)
        r = sqrt(r2)

        const_prefac_1 = T(0.25)/S2_sqr

        r2_x = T(0.5) * S1_x + const_prefac_1 * S2_x
        r2_y = T(0.5) * S1_y + const_prefac_1 * S2_y
        r2_z = T(0.5) * S1_z + const_prefac_1 * S2_z

        r_m1 = T(1)/r
        const_prefac_2 = T(0.5) * r_m1

        r_x = const_prefac_2 * r2_x
        r_y = const_prefac_2 * r2_y
        r_z = const_prefac_2 * r2_z

        S4 = T(2) * M * r^2 * r 

        S3 = r2*r2 + a2 * z2
        S3_m1 = 1/S3

        S5 = r2 + a2
        S5_m1 = 1/S5
        f = S4 * S3_m1

        S6 = r*x + a*y
        S7 = r*y - a*x


        l0 = T(-1)
        l1 = S6 * S5_m1
        l2 = S7 * S5_m1
        l3 = z * r_m1

        fl0 = -f #use the fact that this is just a sign invertion
        fl1 = f * l1
        fl2 = f * l2
        fl3 = f * l3

        u00 = -(T(1) + f)
        u10 = -(fl0 * l1)
        u20 = -(fl0 * l2)
        u30 = -(fl0 * l3)

        u11 = T(1) -(fl1 * l1)
        u21 = -(fl1 * l2)
        u31 = -(fl1 * l3)

        u22 = T(1) -(fl2 * l2)
        u32 = -(fl2 * l3)

        u33 = T(1) -(fl3 * l3)

        w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
        w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
        w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
        w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3

        l0_x = T(0)
        l0_y = T(0)
        l0_z = T(0)

        cons_prefac_4 = T(6) * M * r2

        S4_x = cons_prefac_4 * r_x
        S4_y = cons_prefac_4 * r_y
        S4_z = cons_prefac_4 * r_z

        const_prefac_3 = T(4) * r2 * r #4 r^3

        S3_x = const_prefac_3 * r_x
        S3_y = const_prefac_3 * r_y
        S3_z = const_prefac_3 * r_z + T(2) * a2 * z

        S3_m2 = S3_m1 * S3_m1

        f_x = (S4_x * S3 - S3_x * S4) * S3_m2
        f_y = (S4_y * S3 - S3_y * S4) * S3_m2
        f_z = (S4_z * S3 - S3_z * S4) * S3_m2

        S5_m2 = S5_m1 * S5_m1

        
        S5_x = T(2) * r * r_x
        S5_y = T(2) * r * r_y
        S5_z = T(2) * r * r_z


        S6_x = r + x * r_x
        S6_y = a + x * r_y
        S6_z = x * r_z

        S7_x = r_x * y - a
        S7_y = r_y * y + r
        S7_z = r_z * y

        l1_x = (S6_x * S5 - S5_x * S6) * S5_m2
        l1_y = (S6_y * S5 - S5_y * S6) * S5_m2
        l1_z = (S6_z * S5 - S5_z * S6) * S5_m2

        l2_x = (S7_x * S5 - S5_x * S7) * S5_m2
        l2_y = (S7_y * S5 - S5_y * S7) * S5_m2
        l2_z = (S7_z * S5 - S5_z * S7) * S5_m2

        r_m2 = r_m1 * r_m1

        l3_x = -z * r_x * r_m2
        l3_y = -z * r_y * r_m2
        l3_z = (r - r_z * z) * r_m2

        #derivative wrt to x0 is 0, i.e. those can be skipped
        dv0 = T(0)

        
        u00_x = f_x + T(2) * fl0 * l0_x
        u10_x = f_x * l1 * l0 + fl0 * l1_x #derivative of ls is 0 wrt to 0
        u20_x = f_x * l2 * l0 + fl0 * l2_x 
        u30_x = f_x * l3 * l0 + fl0 * l3_x 

        u11_x = f_x * l1 * l1 + T(2) * fl1 * l1_x
        u21_x = f_x * l1 * l2 + fl1 * l2_x + fl2 * l1_x
        u31_x = f_x * l1 * l3 + fl1 * l3_x + fl3 * l1_x

        u22_x = f_x * l2 * l2 + T(2) * fl2 * l2_x
        u32_x = f_x * l3 * l2 + fl3 * l2_x + fl2 * l3_x

        u33_x = f_x * l3 * l3 + T(2) * fl3 * l3_x

        dv1 = T(0.5) * (u00_x * v0 * v0 + v1 * v1 * u11_x + v2 * v2 * u22_x + v3 * v3 * u33_x + 
            2 * (u10_x  * v1 * v0 + u20_x * v2 * v0 + u30_x * v3 * v0 + 
                v1 * v2 * u21_x + v1 * v3 * u31_x + u32_x * v3 * v2))


        u00_y = f_y + T(2) * fl0 * l0_y
        u10_y = f_y * l1 * l0 + fl0 * l1_y #derivative of ls is 0 wrt to 0
        u20_y = f_y * l2 * l0 + fl0 * l2_y 
        u30_y = f_y * l3 * l0 + fl0 * l3_y 

        u11_y = f_y * l1 * l1 + T(2) * fl1 * l1_y
        u21_y = f_y * l1 * l2 + fl1 * l2_y + fl2 * l1_y
        u31_y = f_y * l1 * l3 + fl1 * l3_y + fl3 * l1_y

        u22_y = f_y * l2 * l2 + T(2) * fl2 * l2_y
        u32_y = f_y * l3 * l2 + fl3 * l2_y + fl2 * l3_y

        u33_y = f_y * l3 * l3 + T(2) * fl3 * l3_y

        dv2 = T(0.5) * (u00_y * v0 * v0 + v1 * v1 * u11_y + v2 * v2 * u22_y + v3 * v3 * u33_y + 
            2 * (u10_y  * v1 * v0 + u20_y * v2 * v0 + u30_y * v3 * v0 + 
                v1 * v2 * u21_y + v1 * v3 * u31_y + u32_y * v3 * v2))

        u00_z = f_z + T(2) * fl0 * l0_z
        u10_z = f_z * l1 * l0 + fl0 * l1_z #derivative of ls is 0 wrt to 0
        u20_z = f_z * l2 * l0 + fl0 * l2_z 
        u30_z = f_z * l3 * l0 + fl0 * l3_z 

        u11_z = f_z * l1 * l1 + T(2) * fl1 * l1_z
        u21_z = f_z * l1 * l2 + fl1 * l2_z + fl2 * l1_z
        u31_z = f_z * l1 * l3 + fl1 * l3_z + fl3 * l1_z

        u22_z = f_z * l2 * l2 + T(2) * fl2 * l2_z
        u32_z = f_z * l3 * l2 + fl3 * l2_z + fl2 * l3_z

        u33_z = f_z * l3 * l3 + T(2) * fl3 * l3_z

        dv3 = T(0.5) * (u00_z * v0 * v0 + v1 * v1 * u11_z + v2 * v2 * u22_z + v3 * v3 * u33_z + 
            2 * (u10_z  * v1 * v0 + u20_z * v2 * v0 + u30_z * v3 * v0 + 
                v1 * v2 * u21_z + v1 * v3 * u31_z + u32_z * v3 * v2))



        


    end

    return ((w0, w1, w2, w3, dv0, dv1, dv2, dv3), (r))
    
end

"""
    calculate_differential(state, metric::KerrMetric{T}) where {T}
    Convinence wrapper for the EoM RHS for tuples.
"""
@inline function calculate_differential(state, metric::KerrMetric{T}) where {T}
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    dstate = calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric)
    return dstate
end

"""
    calculate_differential_and_geom(state, metric::KerrMetric{T}) where {T}
    Convinence wrapper for the EoM RHS for tuples.
"""
@inline function calculate_differential_and_geom(state, metric::KerrMetric{T}) where {T}
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    dstate, cache = calculate_differential(x0, x1, x2, x3, v0, v1, v2, v3, metric)
    return (dstate, cache)
end

"""
    yield_r2(x0::T, x1::T, x2::T, x3::T, metric::KerrMetric{T}) where T

    Calculates the square of the "radial" in the Kerr metric,
"""
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

@inline function yield_r2(state, metric::KerrMetric{T}) where T
    x0, x1, x2, x3, _, _, _, _ = state
    r2 = yield_r2(x0, x1, x2, x3, metric)

    return r2
end