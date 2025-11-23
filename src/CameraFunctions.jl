
"""
    array_index_to_video_index(warp_index, lane_index,  
    st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}) where {V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Converts the warp_index (index of warp, V * H threads per warp) and the lane index (thread index within said warp) to
    the video buffer's (which is assumed to be an array of shape
    [V*MicroNwarps*NBlocks, H*MicroMwarps*MBlocks, k_frames].)

    The substruct argument is used to create a hierarchial memory layout of tiles of shape [V, H] being subset into tiles of shape 
    [V * MicroNWarps, H * MicroMWarps], which form the frame of shape [V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks].
    

"""
@inline function array_index_to_video_index(warp_index, lane_index,  
    st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}) where {V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}
    
    warps_per_block = MicroMWarps * MicroNWarps
    warps_per_frame = warps_per_block * NBlocks * MBlocks
    
    #compute frame index
    k = div(warp_index - 1, warps_per_frame) + 1
    warp_in_frame = mod1(warp_index, warps_per_frame)

    linear_block_index = div(warp_in_frame - 1, warps_per_block) + 1
    block_index_i = mod1(linear_block_index, NBlocks)
    block_index_j = div(linear_block_index - 1, NBlocks) + 1

    local_warp_linear_index = mod1(warp_in_frame, warps_per_block)

    macro_i = mod1(local_warp_linear_index, MicroNWarps)
    macro_j = div(local_warp_linear_index - 1, MicroNWarps) + 1

    inner_i = mod1(lane_index, V)
    inner_j = div(lane_index - 1, V) + 1

    global_i = (block_index_i - 1) * MicroNWarps * V + (macro_i - 1) * V + inner_i
    global_j = (block_index_j - 1) * MicroMWarps * H + (macro_j - 1) * H + inner_j
    
    return global_i, global_j, k
end

"""
    video_index_to_array_index(i, j, k, 
    st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}) where {V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Transforms the vide coordinates, that is, the k-th frame's i-j index, into the coordinates used by the integration buffer, which has a shape of

    (V * H, N_variables, number_of_total_warps_for_video), ensuring that each warp is continious in memory.

    The returned indeces may be used in the integration buffer as:

    state = buffer[lane_index, :, warp_index]

"""
@inline function video_index_to_array_index(i, j, k, 
    st::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}) where {V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    warps_per_block = MicroMWarps * MicroNWarps
    warps_per_frame = warps_per_block * NBlocks * MBlocks

    block_i = div(i - 1, MicroNWarps * V) + 1
    block_j = div(j - 1, MicroMWarps * H) + 1

    local_i = mod1(i, MicroNWarps * V)
    local_j = mod1(j, MicroMWarps * H)
    
    local_warp_i = div(local_i - 1, V) + 1
    local_warp_j = div(local_j - 1, H) + 1

    lane_i = mod1(i, V)
    lane_j = mod1(j, H)

    frame_offset = (k - 1) * warps_per_frame

    linear_block_index = (block_j - 1) * NBlocks + block_i

    block_offset = (linear_block_index - 1) * warps_per_block
    
    local_warp_linear = (local_warp_j - 1) * MicroNWarps + local_warp_i
    
    warp_index = frame_offset + block_offset + local_warp_linear
    lane_index = (lane_j - 1) * V + lane_i

    return warp_index, lane_index
end



"""
    generate_camera_ray(vertical_position, horizontal_position, camera::PinHoleCamera{T}) where T

    Generates a _null_ ray for a pinhole camera, that lays at roughly at pixel 
        (vertical_position * vertical pixel width, horizontal_position * horizontal pixel width) 
    
    Returns a lowered velocity static vector, but not the position.

"""
@inline function generate_camera_ray(vertical_position, horizontal_position, camera::PinHoleCamera{T}) where T
    vertical_scale = (2 * vertical_position - 1) * tan(T(0.5) * camera.vertical_angle)
    horizontal_scale = (2 * horizontal_position - 1) * tan(T(0.5) * camera.horizontal_angle)
    C = sqrt(1+ horizontal_scale^2 + vertical_scale^2)
    local_vector = @. C * camera.lowered_velocity - camera.lowered_pointing - 
        vertical_scale * camera.lowered_upward - 
        horizontal_scale * camera.lowered_rightward

    return local_vector
    
end