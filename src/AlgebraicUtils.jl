#direct way to get permutation
@inline function levicivita4(i::T, j::T, k::T, l::T) where T
    (i == j || i == k || i == l || j == k || j == l || k == l) && return 0

    inversions = 0
    inversions += (i > j) + (i > k) + (i > l)
    inversions += (j > k) + (j > l)
    inversions += (k > l)
    
    return iseven(inversions) ? T(1) : T(-1)
end