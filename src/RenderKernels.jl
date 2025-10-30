
@kernel unsafe_indices = true function render_kernel!(output::AbstractArray{T},
    @Const(metric::KerrMetric{T}), 
    @Const(batch::SubStruct{V, H, NWarps, MWarps}), 
    @Const(dtcontrol::TimeStepScaler{T}), 
    @Const(camerachain::AbstractVector{PinHoleCamera{T}})) where {T, V, H, NWarps, MWarps}

    #initalize the ray.
    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    i, j, k = array_index_to_video_index(chunk, lane, batch)

    local_camera = camerachain[k]

    x0, x1, x2, x3 = local_camera.position

    v0, v1, v2, v3 = @fastmath generate_camera_ray(T(i) / (V * NWarps), T(j) / (H * MWarps), local_camera)

    #normalization steps
    metric_tpl = yield_inverse_metric(x0, x1, x2, x3, metric)
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

function propegate_camera_chain(
    camerachain::AbstractVector{PinHoleCamera{T}}, 
    batch::SubStruct{V, H, NWarps, MWarps}, 
    dtcontrol::TimeStepScaler{T},
    metric::KerrMetric{T}, backend; 
    blockdim = 256
    ) where {T, V, H, NWarps, MWarps}

    N_frames = length(camerachain)
    N_total_warps = N_frames * NWarps * MWarps

    output = KernelAbstractions.zeros(backend, T, V * H, 3, N_total_warps)
    camerachain_device = adapt(backend, camerachain)

    kernel! = render_kernel!(backend, blockdim)

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

@kernel unsafe_indices = true function nearest_render!(
    frame_buffer::AbstractArray{RGB{T}, 3},
    @Const(texture::AbstractArray{RGB{T}, 2}),
    @Const(output::AbstractArray{T, 3}),
    @Const(batch::SubStruct{V, H, NWarps, MWarps}),
    @Const(tex_height::Int), 
    @Const(tex_width::Int)
    ) where {T, V, H, NWarps, MWarps}
    
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

function render_output(output::AbstractArray{T}, batch::SubStruct{V, H, NWarps, MWarps}, 
    texture::AbstractArray{RGB{T}}, backend,  framerate::Int; blocksize = 256) where {T, V, H, NWarps, MWarps}

    output_backend = KernelAbstractions.get_backend(output)
    if output_backend != backend
        error("Backend mismatch: output is on $output_backend but backend=$backend")
    end

    I = V * NWarps
    J = H * MWarps
    N_frames = div(size(output, 3), NWarps * MWarps)
    K = N_frames

    frame_buffer = KernelAbstractions.zeros(backend, RGB{T}, I, J, K)
    texture_device = adapt(backend, texture)
    total_threads = V * H * NWarps * MWarps * N_frames
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

    filename = "output_$(I)x$(J)_$(K)frames_$(framerate)fps.mp4"
    
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