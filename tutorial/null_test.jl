include("../src/KerrGeodesics.jl")
using .KerrGeodesics, Plots
plotly()

star_state = [0.0, 5.0, 0.0, 5.0, -1.0, 0.0, 1.0, 0.0]

N = 100
dtc = TimeStepScaler(0.2, 0.01, 200.0^2, 500., 1600., N)

metric = KerrMetric{Float64}(1., 0.2)

buffer = zeros(Float64, 8, N)

integrate_single_geodesic!(buffer, star_state, dtc, metric, norm = 0.0, null = true)

m1 = KerrGeodesics.yield_inverse_metric(buffer[1, 1], buffer[2, 1], buffer[3, 1], buffer[4, 1], metric)
m2 = KerrGeodesics.yield_inverse_metric(buffer[1, end], buffer[2, end], buffer[3, end], buffer[4, end], metric)

w1 = KerrGeodesics.yield_innerprod(m1, buffer[5, 1], buffer[6, 1], buffer[7, 1], buffer[8, 1])
w2 = KerrGeodesics.yield_innerprod(m2, buffer[5, end], buffer[6, end], buffer[7, end], buffer[8, end])

#why are these not null?
trajectory = plot(buffer[2, :], buffer[3, :], buffer[4, :])