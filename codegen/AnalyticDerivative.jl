include("../src/KerrGeodesics.jl")
using .KerrGeodesics
using FastDifferentiation

function build_diffop_fastdiff(metric::KerrMetric{T}) where T

    #build expressions for metric comps
    @variables x0 x1 x2 x3
    @variables v0 v1 v2 v3
    @variables a M

    R2 = x1^2 + x2^2 + x3^2
    
    sub1 = R2 - a^2
    r2 = T(0.5) * (sub1 + sqrt(sub1^2 + T(4) * a^2 * x3^2))
    
    r = sqrt(r2)
    f = -2 * M * r2 * r / (r2^2 + a^2 * x3^2)
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

    w0 = u00 * v0 + u10 * v1 + u20 * v2 + u30 * v3
    w1 = u10 * v0 + u11 * v1 + u21 * v2 + u31 * v3
    w2 = u20 * v0 + u21 * v1 + u22 * v2 + u32 * v3
    w3 = u30 * v0 + u31 * v1 + u32 * v2 + u33 * v3

    innerprod_result = T(-0.5) * (w0*v0 + w1*v1 + w2*v2 + w3*v3)

    jac = FastDifferentiation.jacobian([innerprod_result],[x0, x1, x2, x3])

    jacexpr = make_Expr(jac,[x0, x1, x2, x3, v0, v1, v2, v3, a, M])

    return jacexpr


end

test = KerrMetric{Float32}(1f0, 0.9f0)
build_diffop_fastdiff(test)