#store metric information
struct KerrMetric{T}
    M::T
    a::T
end

#used for codegen for diffops
struct BatchInfo{V}

end