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

    w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
    w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
    w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
    w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3

    innerprod_result = T(-0.5) * (w0*v0 + w1*v1 + w2*v2 + w3*v3)

    jac = vec(FastDifferentiation.jacobian([innerprod_result],[x0, x1, x2, x3]))
    
    combined_ = append!([w0, w1, w2, w3], jac)
    println(size(combined_))
    jacexpr = make_Expr(combined_,[x0, x1, x2, x3, v0, v1, v2, v3, a, M])

    return jacexpr


end

test = KerrMetric{Float32}(1f0, 0.9f0)
build_diffop_fastdiff(test)