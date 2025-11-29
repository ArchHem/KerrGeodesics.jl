abstract type AbstractKerrRender{N, T} end

length(AbstractKerrRender{N, T}) where {N, T} = N
eltype(AbstractKerrRender{N, T}) where {N, T} = T

struct PureColorRender{T} <: AbstractKerrRender{8, T}
end