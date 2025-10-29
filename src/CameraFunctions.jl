@inline function array_index_to_video_index(warp_index, lane_index,  st::SubStruct{V, H, NWarp, MWarp}) where {V, H, NWarp, MWarp}
    warps_per_frame = NWarp * MWarp
    #compute fram
    k = div(warp_index - 1, warps_per_frame) + 1
    local_warp_index = mod1(warp_index, warps_per_frame)
    macro_i = mod1(local_warp_index, NWarp)
    macro_j = div(local_warp_index - 1, NWarp) + 1

    inner_i = mod1(lane_index, V)
    inner_j = div(lane_index - 1, V) + 1
    i = (macro_i - 1) * V + inner_i
    j = (macro_j - 1) * H + inner_j
    
    return i, j, k
end

@inline function video_index_to_array_index(i, j, k, st::SubStruct{V, H, NWarp, MWarp}) where {V, H, NWarp, MWarp}

    macro_i, macro_j = div(i-1, V) + 1, div(j-1, H) + 1
    warp_index = macro_i + NWarp * (macro_j-1) + (k-1) * (NWarp * MWarp)
    inner_i, inner_j = mod1(i, V), mod1(j, H)
    lane_index = inner_i + V * (inner_j-1)

    return warp_index, lane_index
end

@inline function generate_camera_ray(vertical_position, horizontal_position, camera::PinHoleCamera{T}) where T
    vertical_scale = (2 * vertical_position - 1) * tan(T(0.5) * camera.vertical_angle)
    horizontal_scale = (2 * horizontal_position - 1) * tan(T(0.5) * camera.horizontal_angle)
    C = sqrt(1+ horizontal_scale^2 + vertical_scale^2)
    local_vector = @. C * camera.lowered_velocity - camera.lowered_pointing - 
        vertical_scale * camera.lowered_upward - 
        horizontal_scale * camera.lowered_rightward
    #scale vector so that u0 is 1

    local_vector = local_vector ./ local_vector[1]

    return local_vector
    
end