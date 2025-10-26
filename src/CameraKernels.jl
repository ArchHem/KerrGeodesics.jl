@inline function frame_pixel_coord_to_array_index(i, j, st::SubStruct{V, H}, array_warp_height) where {V, H}
    #warps of site V*H will get mapped to image pixels and vica versa.
    #array_warp_height is div(size(array)[1], V)
    macro_i, macro_j = div(i-1, V) + 1, div(j-1, H) + 1
    warp_index = macro_i + array_warp_height * (macro_j-1)
    inner_i, inner_j = mod1(i, V), mod1(j, H)
    inner_warp_index = inner_i + V * (inner_j-1)
    return warp_index, inner_warp_index
end

@inline function frame_array_index_to_pixel_coord(warp_index, inner_warp_index, st::SubStruct{V, H}, array_warp_height) where {V, H}
    macro_i = mod1(warp_index, array_warp_height)
    macro_j = div(warp_index - 1, array_warp_height) + 1

    inner_i = mod1(inner_warp_index, V)
    inner_j = div(inner_warp_index - 1, V) + 1
    i = (macro_i - 1) * V + inner_i
    j = (macro_j - 1) * H + inner_j
    return i, j
end

@inline function generate_camera_ray(vertical_scale, horizontal_scale, camera::PinHoleCamera{T}) where T

end

#This happens once per call, so needs not to be fully aligned...
@kernel unsafe_indices=true function institate_camera_rays!(device_array, @Const(camera::PinHoleCamera{T}), @Const(st::SubStruct{V, H})) where {T, V, H}
    N, M = camera.vertical_px, camera.horizontal_px
    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    i, j = frame_array_index_to_pixel_coord(chunk, lane, st, div(N, V))
    #use https://arxiv.org/pdf/1410.7775 

    @fastmath begin
        vertical_position = i / N
        vertical_scale = (2 * vertical_position - 1) * tan(T(0.5) * camera.vertical_angle)

        horizontal_position = j / M
        horizontal_scale = (2 * horizontal_position - 1) * tan(T(0.5) * camera.horizontal_angle)

        #since any vector maybe be represent as a linear combination of these 4 vectors of the camera, which are linearly independent
        #for the result to be null, we enforce a null condition, -C^2 + 1 + vertcial_scale^2 + horizontal_scale^2

        C = sqrt(1+ horizontal_scale^2 + vertical_scale^2)
        local_vector = @. C * camera.lowered_velocity - camera.lowered_pointing - 
            vertical_scale * camera.lowered_upward - 
            horizontal_scale * camera.lowered_rightward
    end

    device_array[lane, 1:4, chunk] .= camera.position
    device_array[lane, 5:8, chunk] .= local_vector


end

function create_camera_rays(backend, camera::PinHoleCamera{T}, st::SubStruct{V, H}; blocksize = ) where {T, V, H}
    N, M = camera.vertical_px, camera.horizontal_px
    W = V * H

end