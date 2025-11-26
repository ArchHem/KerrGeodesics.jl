abstract type AbstractInterpolant end

struct NearestInterpolant <: AbstractInterpolant
end

struct BiLinearInterpolant <: AbstractInterpolant
end