using GLMakie, NonlinearSolve, BenchmarkTools #load everything for ext
using LinearAlgebra, Printf, StaticArrays
using KerrGeodesics

const CalibrationExt = Base.get_extension(KerrGeodesics, :CalibrationExt)

const test_states = CalibrationExt.test_states
const test_metric = CalibrationExt.test_metric
const test_dtc = CalibrationExt.test_dtc

const constant_dtc = HorizonHeureticScaler(
    test_dtc.max,
    test_metric,
    test_dtc.a0,
    0.0f0,
    0.0f0,
    test_dtc.redshift_stop,
    test_dtc.r_stop,
    1
)

rk2 = RK2Heuretic(test_metric, constant_dtc)
rk4 = RK4Heuretic(test_metric, constant_dtc)
split_ham = SplitHamiltonianHeuretic(test_metric, constant_dtc, 2.0f0)

function prep_state(start_state, integrator)
    x0, x1, x2, x3, v0, v1, v2, v3 = start_state
    T = typeof(x0)
    local_metric = KerrGeodesics.metric(integrator)
    metric_tpl = KerrGeodesics.yield_inverse_metric(x0, x1, x2, x3, local_metric)
    w0, w1, w2, w3 = KerrGeodesics.mult_by_metric(metric_tpl, (v0, v1, v2, v3))
    v0, v1, v2, v3 = v0 / w0, v1 / w0, v2 / w0, v3 / w0
    v0, v1, v2, v3 = KerrGeodesics.normalize_fourveloc(
        metric_tpl, v0, v1, v2, v3; norm=T(0), null=true)
    return @SVector [x0, x1, x2, x3, v0, v1, v2, v3]
end

function single_step(base_state, integrator)
    metric_tpl = KerrGeodesics.yield_inverse_metric(
        base_state[1], base_state[2], base_state[3], base_state[4], test_metric)
    gstate = KerrGeodesics.initialize_state(base_state, integrator, metric_tpl)
    cache = KerrGeodesics.initialize_cache(gstate, metric_tpl, KerrGeodesics.scaler(integrator))
    return KerrGeodesics.state(KerrGeodesics.geodesic_step(gstate, integrator, cache))
end

labels = ["x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3"]

for (name, raw_state) in test_states
    base_state = prep_state(raw_state, rk4)

    rk2_state = single_step(base_state, rk2)
    rk4_state = single_step(base_state, rk4)
    sh_state = single_step(base_state, split_ham)

    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("  $name")
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("  RK2 vs RK4:")
    for i in 1:8
        @printf("    %s  rk2=% .6f  rk4=% .6f  diff=% .2e\n",
            labels[i], rk2_state[i], rk4_state[i], rk2_state[i] - rk4_state[i])
    end
    @printf("    ||Δstate|| = %.2e\n\n", norm(rk2_state .- rk4_state))

    println("  RK4 vs SplitHamiltonian (ω=10):")
    for i in 1:8
        @printf("    %s  rk4=% .6f  split=% .6f  diff=% .2e\n",
            labels[i], rk4_state[i], sh_state[i], rk4_state[i] - sh_state[i])
    end
    @printf("    ||Δstate|| = %.2e\n\n", norm(rk4_state .- sh_state))
end