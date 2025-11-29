"""
    PinHoleCamera{T}
    This is a struct that stores all the information needed to generate camera rays of a pinhole camera.
        Uses four local, lowered, four-velocities that form an orthonromal basis to generate camera rays (1 timelike, 3 spacelike)

    positon Stores 4 position as SVector
    lowered_velocity:  4 lowered velocity SVector. Must be timelike
    lowered_pointing: 4 vector, lowered, SVCector, must be spacelike. Can be interpreted as the "direction" the camera plane is facing
    lowered_upward: 4 vector, lowered, SVCector, must be spacelike. Can be interpreted as the direction towards the "top" of the picture
    lowered_righward: 4 vector, lowered, SVector, must be spacelike. 
    horizontal_angle: Angle of opening along the "upward" image direction. Note that pixel spacing scales as tan(α/2)
    vertical_angle: Angle of opening along the "rightward" image direction. Note that pixel spacing scales as tan(α/2)
    horizontal_px: Number of pixels in the horizontal image direction
    vertical_px: Number of pixels in the vertical direction
    inverse_metric_tpl: Stores the inverse metric's trinagular form using a 10-tuple.


    
"""
struct PinHoleCamera{T}
    position::SVector{4, T}
    lowered_velocity::SVector{4, T}
    lowered_pointing::SVector{4, T}
    lowered_upward::SVector{4, T}
    lowered_rightward::SVector{4, T}
    horizontal_angle::T
    vertical_angle::T
    horizontal_px::Int
    vertical_px::Int
    inverse_metric_tpl::NTuple{10, T}
end

"""
    PinHoleCamera(position::AbstractArray{T}, lowered_velocity::AbstractArray{T}, 
    lowered_pointing::AbstractArray{T}, lowered_upward::AbstractArray{T}, 
    metric::KerrMetric{T},
    horizontal_angle::T, vertical_angle::T, horizontal_px::Int, vertical_px::Int) where T


    Convinence constructor for the pinhole camera in case the exact vectors are not know.
        Provided the exact position, approximate lowered velocity of the camera, its pointing and upward vectors, 
        this will generate exact vectors, via running a Gram-schmidt procedure.
"""
function PinHoleCamera(position::AbstractArray{T}, lowered_velocity::AbstractArray{T}, 
    lowered_pointing::AbstractArray{T}, lowered_upward::AbstractArray{T}, 
    metric::KerrMetric{T},
    horizontal_angle::T, vertical_angle::T, horizontal_px::Int, vertical_px::Int) where T

    #Use the modified version found in:
    #https://arxiv.org/pdf/1410.7775

    inv_metric_value = yield_inverse_metric(position..., metric)
    
    metric_det = 1/yield_determinant(inv_metric_value)
    
    #I am not sure if this -1 is correct, see:
    #Pg 202 of Charles W. Misner, Kip S. Thorne, and John Archibald
    # Wheeler. Gravitation.
    levis_norm = -1/sqrt(-metric_det)

    #I think its enough if we ensure timelike behaviour, and spacelike for the later...

    @assert yield_innerprod(inv_metric_value, lowered_velocity...) < 0
    #@assert lowered_pointing[1] == T(0)
    #@assert lowered_upward[1] == T(0)

    #now, do a round of GS

    lowered_velocity_tpl = normalize_fourveloc(inv_metric_value, lowered_velocity...; norm = T(-1), null = false)
    
    #since lowered_velocity is timelike, there is a sign flip here.
    lowered_pointing_tpl_interim = tuple(lowered_pointing...) .+ 
        yield_innerprod(inv_metric_value, lowered_velocity_tpl, tuple(lowered_pointing...)) .* lowered_velocity_tpl
        
    lowered_pointing_tpl = normalize_fourveloc(inv_metric_value, lowered_pointing_tpl_interim...; norm = T(1), null = false)
    
    lowered_upward_tpl_interim = tuple(lowered_upward...) .+ 
        yield_innerprod(inv_metric_value, lowered_velocity_tpl, tuple(lowered_upward...)) .* lowered_velocity_tpl .-
        yield_innerprod(inv_metric_value, lowered_pointing_tpl, tuple(lowered_upward...)) .* lowered_pointing_tpl
    
    lowered_upward_tpl = normalize_fourveloc(inv_metric_value, lowered_upward_tpl_interim...; norm = T(1), null = false)
    
    e0 = SVector{4, T}(lowered_velocity_tpl)
    e1 = SVector{4, T}(lowered_pointing_tpl)
    e2 = SVector{4, T}(lowered_upward_tpl)

    #do the levis-ciita generation

    e3_up = zeros(T, 4)

    for a in 1:4
        for i in 1:4
            for j in 1:4
                for k in 1:4
                    e3_up[a] += e0[i]*e1[j]*e2[k]*levis_norm * levicivita4(i, j, k, a)
                end
            end
        end
    end

    e3_up = SVector{4, T}(e3_up)
    
    u00, u10, u20, u30, u11, u21, u31, u22, u32, u33 = inv_metric_value

    inv_metric = @SMatrix [u00 u10 u20 u30; 
                        u10 u11 u21 u31;
                        u20 u21 u22 u32;
                        u30 u31 u32 u33]

    #since v^u = g^uv v_v
    e3_interim = inv_metric \ e3_up


    e3 = SVector{4, T}(normalize_fourveloc(inv_metric_value, e3_interim...; norm = T(1), null = false))

    pos = SVector{4, T}(position)

    return PinHoleCamera{T}(pos, e0, e1, e2, e3, horizontal_angle, vertical_angle, horizontal_px, vertical_px, inv_metric_value)

end

"""
    PinHoleCamera(position::AbstractArray{T}, lowered_velocity::AbstractArray{T}, 
    lowered_pointing::AbstractArray{T}, lowered_upward::AbstractArray{T}, 
    metric::KerrMetric{T},
    horizontal_angle::T, vertical_angle::T, st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}) where 
    {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Convinence wrapper that uses a SubStruct to generate cameras, automatically aligning the number of pixels as needed.
"""
function PinHoleCamera(position::AbstractArray{T}, lowered_velocity::AbstractArray{T}, 
    lowered_pointing::AbstractArray{T}, lowered_upward::AbstractArray{T}, 
    metric::KerrMetric{T},
    horizontal_angle::T, vertical_angle::T, st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}) where 
    {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Ny = V * MicroNWarps * NBlocks
    Nx = H * MicroMWarps * MBlocks
    result = PinHoleCamera(position, lowered_velocity, 
        lowered_pointing, lowered_upward, 
        metric,
        horizontal_angle, vertical_angle, Nx, Ny)

    return result
end

#Rendering utils

"""
    cast_to_sphere(x0, x1, x2, x3, v0, v1, v2, v3)

    Casts a )sufficently far away) ray to spacelike infinity based on its current heading. Specific to the Kerr metric
"""
@inline function cast_to_sphere(x0, x1, x2, x3, v0, v1, v2, v3)
    #uses approximate flatness
    @fastmath r = sqrt(v1^2 + v2^2 + v3^2)
    @fastmath θ = acos(v3 / r)
    @fastmath ϕ = atan(v2, v1)

    return ϕ, θ
end

@inline function cast_to_sphere(state)
    x0, x1, x2, x3, v0, v1, v2, v3 = state
    ϕ, θ = cast_to_sphere(x0, x1, x2, x3, v0, v1, v2, v3)
    return ϕ, θ
end

