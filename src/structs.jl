#store metric information
struct KerrMetric{T}
    M::T
    a::T
end

#used for codegen for diffops
struct BatchInfo{V}

end

struct TimeStepScaler{T}
    min::T
    flat::T
    outer_scaling::T
    inner_scaling::T
    threshold::T
end