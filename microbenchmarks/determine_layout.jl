include("../src/KerrGeodesics.jl")
using .KerrGeodesics
using BenchmarkTools

#Make this benchmark so that a batch fits into ~L1 cache
function find_optimal_layout(batchsize = 1600, T = Float32)
    VS = [2, 4, 8, 16]
    inner_times = []
    d_times = []
    for v in VS
        N = div(batchsize, v)
        metric = KerrMetric{T}(T(1), T(0.9))
        
        rbs = BatchInfo{v}()
        pos_buffer = zeros(T, v, 4, N)
        pos_buffer[:, 2, :] .= T(10)
        veloc_buffer = zeros(T, v, 4, N)
        veloc_buffer[:, 1:2,:] .= T(1)
        dpos_buffer = zeros(T, v, 4, N)
        dveloc_buffer = zeros(T, v, 4, N)

        innerprod_buffer = zeros(T, v, N)
        d_innerprod_buffer = ones(T, v, N)

        b1 = @benchmark calculate_innerprod!($innerprod_buffer, $dpos_buffer, $pos_buffer, $veloc_buffer, $rbs, $metric)
        t1 = median(b1).time / (v * N) #median time per ray

        b2 = @benchmark calculate_differentials_backward!($innerprod_buffer, $d_innerprod_buffer, $dpos_buffer,
            $pos_buffer, $dveloc_buffer, $veloc_buffer,  $rbs, $metric)
        t2 = median(b2).time / (v * N) #median time per ray
        push!(inner_times, t1)
        push!(d_times, t2)
    end
    return VS, inner_times, d_times
end

res = find_optimal_layout()