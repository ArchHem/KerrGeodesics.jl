"""
    render_kernel!(output::AbstractArray{T},
        @Const(metric::KerrMetric{T}), 
        @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}), 
        @Const(dtcontrol::HorizonHeureticScaler{T}), 
        @Const(camerachain::AbstractVector{PinHoleCamera{T}})) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Given an array of initical conditions in the (V*H, 8, N_warps) shaped array, propegate eah ray to its end state, 
    and cast them to spacelike infinity.
        This method will take variables stored in teh input array (positions and lowered velocities) and linearly scale the later such that
            the _raised_ four-velocities timelike component is 1 at the start of the integration.
        This function writes into the 'output' array 3 variables, the "cast angles" of ϕ and θ into [:, 2, :] and [:, 3, :]
        To [:, 1, :] it writes a flag, which is 0 if the tracked geodesic's timelike component has increased beyond a certain limit, 
        indicating an event horizon.
"""
@kernel unsafe_indices = true function render_kernel!(output::AbstractArray{T},
    @Const(integrator::AbstractHeureticIntegrator),
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}), 
    @Const(camerachain::AbstractVector{PinHoleCamera{T}})) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    i, j, k = array_index_to_video_index(chunk, lane, batch)

    local_camera = camerachain[k]

    x0, x1, x2, x3 = local_camera.position

    v0, v1, v2, v3 = @fastmath generate_camera_ray(T(i - T(0.5)) / (V * MicroNWarps * NBlocks), 
        T(j - T(0.5)) / (H * MicroMWarps * MBlocks), 
        local_camera)
    metric_tpl = local_camera.inverse_metric_tpl
    initial_state = @SVector [x0, x1, x2, x3, v0, v1, v2, v3]
    gstate = initialize_state(initial_state, integrator, metric_tpl)
    dt_scaler = scaler(integrator)
    dtc_cache = initialize_cache(gstate, metric_tpl, dt_scaler)

    N = max_timesteps(integrator)

    redshift_status = false
    @fastmath for t in 1:N

        nextval = geodesic_step(gstate, integrator, dtc_cache)

        if isterminated(nextval)
            redshift_status = isredshifted(nextval)
            break
        end
        gstate = full_state(nextval)
    end
    
    ϕ, θ = cast_to_sphere(gstate)

    output[lane, 1, chunk] = redshift_status ? T(0) : T(1)
    output[lane, 2, chunk] = ϕ
    output[lane, 3, chunk] = θ

end

"""
    propegate_camera_chain(
    camerachain::AbstractVector{PinHoleCamera{T}}, 
    batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, 
    dtcontrol::HorizonHeureticScaler{T},
    metric::KerrMetric{T}, backend
    ) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Given an array of cameras represneting frames of an array, render the entire video on the specified backed, 
    allocating and synchroninzing all needed backend arrays.

    The substruct argument is used to create a hierarchial memory layout of tiles of shape [V, H] being subset into tiles of shape 
    [V * MicroNWarps, H * MicroMWarps], which form the frame of shape [V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks]. Note that 
    this will launch the kernel with blocks of size [V * MicroNWarps * H * MicroMWarps], i.e. a single block for each microtile.
"""
function propegate_camera_chain(
    camerachain::AbstractVector{PinHoleCamera{T}}, 
    batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, 
    integrator::AbstractStateLessCustomIntegrator,
    backend
    ) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    N_frames = length(camerachain)
    N_total_warps = N_frames * MicroNWarps * MicroMWarps * NBlocks * MBlocks

    output = KernelAbstractions.zeros(backend, T, V * H, 3, N_total_warps)
    camerachain_device = adapt(backend, camerachain)

    kernel! = render_kernel!(backend, V * H * MicroNWarps * MicroMWarps)

    kernel!(
        output,
        integrator,
        batch,
        camerachain_device;
        ndrange = N_total_warps * V * H
    )

    KernelAbstractions.synchronize(backend)
    return output
end

"""
    render_video!(
    frame_buffer::AbstractArray{RGB{T}, 3},
    @Const(texture::AbstractArray{RGB{T}, 2}),
    @Const(output::AbstractArray{T, 3}),
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}),
    @Const(tex_height::Int), 
    @Const(tex_width::Int); interpolant::AbstractInterpolant = NearestInterpolant()

    ) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Nearest-pixel interpolant that turns an array of returned cast angles and flag into an RGB array, writing it into an an array of shape
    (V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks, k_frames)

    The substruct argument is used to interpret a hierarchial memory layout of tiles of shape [V, H] being subset into tiles of shape 
    [V * MicroNWarps, H * MicroMWarps], which form the frame of shape [V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks]. Note that 
    this kernel will be launched with the kernel with blocks of size [V * MicroNWarps * H * MicroMWarps], i.e. a single block for each microtile.
"""

@kernel unsafe_indices = true function render_video!(
    frame_buffer::AbstractArray{RGB{T}, 3},
    @Const(texture::AbstractArray{RGB{T}, 2}),
    @Const(output::AbstractArray{T, 3}),
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}),
    @Const(tex_height::Int), 
    @Const(tex_width::Int), interpolant::NearestInterpolant
    ) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}
    
    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1
    
    i, j, k = array_index_to_video_index(chunk, lane, batch)
    
    flag = output[lane, 1, chunk]
    ϕ = output[lane, 2, chunk]
    θ = output[lane, 3, chunk]
    
    #we _could_ rewrite this to use texture memory instead, by dispatching on backend...
    if flag == T(0)
        frame_buffer[i, j, k] = RGB{T}(T(0), T(0), T(0))
    else
        
        inv_2pi = T(1) / (T(2) * T(π))
        inv_pi = T(1) / T(π)
        
        u = (ϕ + T(π)) * inv_2pi
        v = θ * inv_pi
        
        v = clamp(v, T(0), T(1))
        tex_x = unsafe_trunc(Int, u * T(tex_width)) + 1
        tex_y = unsafe_trunc(Int, v * T(tex_height)) + 1
        tex_x = mod1(tex_x, tex_width)
        tex_y = clamp(tex_y, 1, tex_height)
        
        frame_buffer[i, j, k] = texture[tex_y, tex_x]
    end
end

"""
    render_output(output::AbstractArray{T}, batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, 
    texture::AbstractArray{RGB{T}}, backend,  framerate::Int;
    filename = nothing, interpolant::AbstractInterpolant = NearestInterpolant()) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}


    Wrapper function around the nearest-pixel interpolant kernel that turns an array of returned cast angles and flag into an RGB array, writing it into an an array of shape
    (V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks, k_frames)

    The substruct argument is used to interpret a hierarchial memory layout of tiles of shape [V, H] being subset into tiles of shape 
    [V * MicroNWarps, H * MicroMWarps], which form the frame of shape [V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks]. Note that 
    this kernel will be launched with the kernel with blocks of size [V * MicroNWarps * H * MicroMWarps], i.e. a single block for each microtile.
"""
function render_frames(output::AbstractArray{T}, batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, 
    texture::AbstractArray{RGB{T}}, backend, interpolant::AbstractInterpolant = NearestInterpolant()) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    output_backend = KernelAbstractions.get_backend(output)
    blocksize = V * H * MicroMWarps * MicroNWarps
    if output_backend != backend
        error("Backend mismatch: output is on $output_backend but backend=$backend")
    end

    I = V * MicroNWarps * NBlocks
    J = H * MicroMWarps * MBlocks
    N_frames = div(size(output, 3), MicroNWarps * NBlocks * MicroMWarps * MBlocks)
    K = N_frames

    frame_buffer = KernelAbstractions.zeros(backend, RGB{T}, I, J, K)
    texture_device = adapt(backend, texture)
    total_threads = V * H * MicroNWarps * NBlocks * MicroMWarps * MBlocks * N_frames
    tex_height, tex_width = size(texture)
    kernel! = render_video!(backend, blocksize)
    kernel!(
        frame_buffer,
        texture_device,
        output,
        batch, tex_height, tex_width, interpolant;
        ndrange = total_threads
    )

    KernelAbstractions.synchronize(backend)
    
    return frame_buffer

end

function write_video(frame_buffer; framerate = 30, filename = nothing, codec = "libx264", file_path = pwd())
    I, J, K = size(frame_buffer)
    backend = get_backend(frame_buffer)
    #transfer to CPU
    frame_buffer_cpu = Array(frame_buffer)
    KernelAbstractions.synchronize(backend)
    if isnothing(filename)
        filename = "output_$(I)x$(J)_$(K)frames_$(framerate)fps.mp4"
    end

    full_path = joinpath(file_path, filename)
    
    writer = open_video_out(
        full_path,
        RGB{N0f8},
        (I, J);
        framerate = framerate,
        codec_name = codec,
        encoder_options = (crf=23, preset="medium")
    )
    
    for k in 1:K
        frame = @views frame_buffer_cpu[:, :, k]
        write(writer, RGB{N0f8}.(clamp01.(frame)))
    end
    
    close_video_out!(writer)
    
    return nothing
end