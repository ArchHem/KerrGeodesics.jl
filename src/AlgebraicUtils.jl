#direct way to get permutation

"""
    levicivita4(i::T, j::T, k::T, l::T) where T
    Generates a scalar corresponding to the 4-dimensional Levis-civita symbol ϵ_{ijkl}, 
    which is 1 if the permutation [ijkl] is even, -1 if odd, and 0 otherwise.
    
"""
@inline function levicivita4(i::T, j::T, k::T, l::T) where T
    (i == j || i == k || i == l || j == k || j == l || k == l) && return 0

    inversions = 0
    inversions += (i > j) + (i > k) + (i > l)
    inversions += (j > k) + (j > l)
    inversions += (k > l)
    
    return iseven(inversions) ? T(1) : T(-1)
end