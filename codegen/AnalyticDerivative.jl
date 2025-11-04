include("../src/KerrGeodesics.jl")
using .KerrGeodesics
using FastDifferentiation

function build_diffop_fastdiff(metric::KerrMetric{T}) where T

    #build expressions for metric comps
    @variables x0 x1 x2 x3
    @variables v0 v1 v2 v3
    @variables a M

    #keep power calls to minimumum

    x1_2 = x1 * x1
    x2_2 = x2 * x2
    x3_2 = x3 * x3

    R2 = x1_2 + x2_2 + x3_2
    a2 = a*a
    sub1 = R2 - a2
    #r2 is defined via

    #
    # (x^2 + y^2)/(r^2 + a^2) + z^2 / r^2 = 1) ->
    # (x^2 + y^2) * r^2 + (r^2 + a^2) * z^2 = r^2 (r^2 + a^2)
    # Let U = r^2 
    # (x^2 + y^2) * U + (U + a^2) * z^2 = U^2 + U * a^2
    # hence, 
    # (x^2 + y^2 + z^2 - a^2) * U - U^2 + a^2 * z^2

    r2 = T(0.5) * (sub1 + sqrt(sub1^2 + T(4) * a2 * x3_2))
    r4 = r2 * r2
    r = sqrt(r2)
    f = 2 * M * r2 * r / (r4 + a2 * x3_2)
    common_subdiv = r2 + a2
    #calculate INVERSE metric in signature -1, 1, 1, 1
    l0 = -T(1)
    l1 = (r * x1 + a * x2) / common_subdiv
    l2 = (r * x2 - a * x1) / common_subdiv
    l3 = x3 / r

    u00 =  -(T(1) + f * l0^2)
    u10 = -(f * l0 * l1)
    u20 =  -(f * l0 * l2)
    u30 =  -(f * l0 * l3)
    u11 =  T(1) - (f * l1^2)
    u21 =  -(f * l1 * l2)
    u31 = -(f * l1 * l3)
    u22 = (T(1) - f * l2^2)
    u32 = -(f * l2 * l3)
    u33 =  (T(1) - f * l3^2)

    w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
    w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
    w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
    w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3

    innerprod_result = T(-0.5) * (w0*v0 + w1*v1 + w2*v2 + w3*v3)

    jac = vec(FastDifferentiation.jacobian([innerprod_result],[x0, x1, x2, x3]))
    
    combined_ = append!([w0, w1, w2, w3], jac)
    
    jacexpr = make_Expr(combined_,[x0, x1, x2, x3, v0, v1, v2, v3, a, M])

    return jacexpr


end

test = KerrMetric{Float32}(1f0, 0.9f0)
build_diffop_fastdiff(test)