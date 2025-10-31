include("../src/KerrGeodesics.jl")
using .KerrGeodesics, Plots

N = 1000
threshold = 15.0^2
stc = TimeStepScaler(0.2, 0.02, 20.0^2, 500., 1600., N)
r = LinRange(1.0, 500.0, 10000)
dt = KerrGeodesics.get_dt.(r .^2, Ref(stc))

plot(r, dt)