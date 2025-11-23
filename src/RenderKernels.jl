"""
    render_kernel!(output::AbstractArray{T},
        @Const(metric::KerrMetric{T}), 
        @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}), 
        @Const(dtcontrol::TimeStepScaler{T}), 
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
    @Const(metric::KerrMetric{T}), 
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}), 
    @Const(dtcontrol::TimeStepScaler{T}), 
    @Const(camerachain::AbstractVector{PinHoleCamera{T}})) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    #initalize the ray.
    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    i, j, k = array_index_to_video_index(chunk, lane, batch)

    local_camera = camerachain[k]

    x0, x1, x2, x3 = local_camera.position

    #shift so that they hit "center" of the pixel
    v0, v1, v2, v3 = @fastmath generate_camera_ray(T(i - T(0.5)) / (V * MicroNWarps * NBlocks), 
        T(j - T(0.5)) / (H * MicroMWarps * MBlocks), 
        local_camera)

    #normalization steps (this could be wrapped into the came constructor, TODO)
    #renorm such that the raised velocity u0 = 1 for ALL rays
    metric_tpl = local_camera.inverse_metric_tpl
    w0, w1, w2, w3 = mult_by_metric(metric_tpl, (v0, v1, v2, v3))

    v0, v1, v2, v3 = v0/w0, v1/w0, v2/w0, v3/w0

    
    N = dtcontrol.maxtimesteps

    flag = T(1)
    @fastmath for t in 1:N
        #This causes warp divergence, however, it _does_ terminate early warps. Our spatial structure means that nearby pixels are in nearby warps...


        r2 = yield_r2(x0, x1, x2, x3, metric)
        
        #use RK4, we move backwards.
        dt = -get_dt(r2, dtcontrol)

        dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3 = RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt)

        if r2 > dtcontrol.r_stop || dx0 > dtcontrol.redshift_stop
            break
        end

        x0, x1, x2, x3, v0, v1, v2, v3 = x0 + dt*dx0, x1 + dt*dx1, x2 + dt*dx2, x3 + dt*dx3,
                                            v0 + dt*dv0, v1 + dt*dv1, v2 + dt*dv2, v3 + dt*dv3
        

    end
    ϕ, θ = cast_to_sphere(x0, x1, x2, x3, v0, v1, v2, v3)

    r2 = yield_r2(x0, x1, x2, x3, metric)
    #use RK4, we move backwards.
    dt = -get_dt(r2, dtcontrol)

    dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3 = RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt)

    if dx0 > dtcontrol.redshift_stop
        flag = T(0)
    end
    #output stores a flag, and the casted ray positions.
    output[lane, 1, chunk] = flag
    output[lane, 2, chunk] = ϕ
    output[lane, 3, chunk] = θ


end

"""
    propegate_camera_chain(
    camerachain::AbstractVector{PinHoleCamera{T}}, 
    batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, 
    dtcontrol::TimeStepScaler{T},
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
    dtcontrol::TimeStepScaler{T},
    metric::KerrMetric{T}, backend
    ) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    N_frames = length(camerachain)
    N_total_warps = N_frames * MicroNWarps * MicroMWarps * NBlocks * MBlocks

    output = KernelAbstractions.zeros(backend, T, V * H, 3, N_total_warps)
    camerachain_device = adapt(backend, camerachain)

    kernel! = render_kernel!(backend, V * H * MicroNWarps * MicroMWarps)

    kernel!(
        output,
        metric,
        batch,
        dtcontrol,
        camerachain_device;
        ndrange = N_total_warps * V * H
    )

    KernelAbstractions.synchronize(backend)
    return output
end

"""
    nearest_render!(
    frame_buffer::AbstractArray{RGB{T}, 3},
    @Const(texture::AbstractArray{RGB{T}, 2}),
    @Const(output::AbstractArray{T, 3}),
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}),
    @Const(tex_height::Int), 
    @Const(tex_width::Int)
    ) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

    Nearest-pixel interpolant that turns an array of returned cast angles and flag into an RGB array, writing it into an an array of shape
    (V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks, k_frames)

    The substruct argument is used to interpret a hierarchial memory layout of tiles of shape [V, H] being subset into tiles of shape 
    [V * MicroNWarps, H * MicroMWarps], which form the frame of shape [V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks]. Note that 
    this kernel will be launched with the kernel with blocks of size [V * MicroNWarps * H * MicroMWarps], i.e. a single block for each microtile.
"""

@kernel unsafe_indices = true function nearest_render!(
    frame_buffer::AbstractArray{RGB{T}, 3},
    @Const(texture::AbstractArray{RGB{T}, 2}),
    @Const(output::AbstractArray{T, 3}),
    @Const(batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}),
    @Const(tex_height::Int), 
    @Const(tex_width::Int)
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
    filename = nothing) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}


    Wrapper function around the nearest-pixel interpolant kernel that turns an array of returned cast angles and flag into an RGB array, writing it into an an array of shape
    (V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks, k_frames)

    The substruct argument is used to interpret a hierarchial memory layout of tiles of shape [V, H] being subset into tiles of shape 
    [V * MicroNWarps, H * MicroMWarps], which form the frame of shape [V * MicroNWarps * NBlocks, H * MicroMWarps * MBlocks]. Note that 
    this kernel will be launched with the kernel with blocks of size [V * MicroNWarps * H * MicroMWarps], i.e. a single block for each microtile.
"""
function render_output(output::AbstractArray{T}, batch::SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}, 
    texture::AbstractArray{RGB{T}}, backend,  framerate::Int;
    filename = nothing) where {T, V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

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
    kernel! = nearest_render!(backend, blocksize)
    kernel!(
        frame_buffer,
        texture_device,
        output,
        batch, tex_height, tex_width;
        ndrange = total_threads
    )

    KernelAbstractions.synchronize(backend)
    frame_buffer_cpu = Array(frame_buffer)

    if isnothing(filename)
        filename = "output_$(I)x$(J)_$(K)frames_$(framerate)fps.mp4"
    end
    
    writer = open_video_out(
        filename,
        RGB{N0f8},
        (I, J);
        framerate = framerate,
        codec_name = "libx264",
        encoder_options = (crf=23, preset="medium")
    )
    
    for k in 1:K
        frame = @views frame_buffer_cpu[:, :, k]
        write(writer, RGB{N0f8}.(clamp01.(frame)))
    end
    
    close_video_out!(writer)
    
    return frame_buffer_cpu

end