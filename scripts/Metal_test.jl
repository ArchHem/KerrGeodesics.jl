using KernelAbstractions, Metal
include("../src/KerrGeodesics.jl")
using .KerrGeodesics

function test_metal_kernel()
    T = Float32
    V = 32
    n_lanes = V
    n_chunks = 256
    n_vars = 8
    metric = KerrMetric{T}(1.0f0, 0.05f0)
    batch = BatchInfo{V}()
    
    state = zeros(T, n_lanes, n_vars, n_chunks)
    
    for chunk in 1:n_chunks
        for lane in 1:n_lanes
            state[lane, 1, chunk] = 0.0f0
            state[lane, 2, chunk] = 10.0f0
            state[lane, 3, chunk] = 0.0f0
            state[lane, 4, chunk] = 0.0f0
            state[lane, 5, chunk] = 1.0f0
            state[lane, 6, chunk] = 1.0f0
            state[lane, 7, chunk] = 0.0f0
            state[lane, 8, chunk] = 0.0f0
        end
    end
    dstate = zeros(T, n_lanes, n_vars, n_chunks)
    state_gpu = MtlArray(state)
    dstate_gpu = MtlArray(dstate)
    
    backend = MetalBackend()
    kernel! = calculate_differential!(backend, 256)
    
    total_work_items = n_lanes * n_chunks

    kernel!(
        dstate_gpu, 
        state_gpu, 
        metric, 
        batch,
        ndrange=total_work_items
    )
    KernelAbstractions.synchronize(backend)

    dstate_result = Array(dstate_gpu)
    return dstate_result
end

test_metal_kernel()