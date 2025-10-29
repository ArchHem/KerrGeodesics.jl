
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

    
    N = dtcontrol.maxtimesteps

    flag = T(1)
    @fastmath for t in 1:N
        #This causes warp divergence, however, it _does_ terminate early warps. Our spatial structure means that nearby pixels are in nearby warps...


        r2 = yield_r2(x0, x1, x2, x3, metric)
        if r2 > dtcontrol.r_stop || v0 > dtcontrol.redshift_stop
            break
        end
        #use RK4, we move backwards.
        dt = -get_dt(r2, dtcontrol)

        dx0, dx1, dx2, dx3, dv0, dv1, dv2, dv3 = RK4step(x0, x1, x2, x3, v0, v1, v2, v3, metric, dt)
        

    end
    ϕ, θ = cast_to_sphere(x0, x1, x2, x3, v0, v1, v2, v3)

    if v0 > dtcontrol.redshift_stop
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

@kernel unsafe_indices = true function linear_background_render(
    frame_buffer::AbstractArray{RGB{T}},
    @Const(texture::AbstractArray{RGB{T}}),
    @Const(output::AbstractArray{T}),
    @Const(batch::SubStruct{V, H, NWarps, MWarps})
    ) where {T, V, H, NWarps, MWarps}

    g_index = @index(Global, Linear)
    chunk, lane = divrem(g_index - 1, V*H) .+ 1

    #i,j are pixel coordinates to color: k is the frame index.
    i, j, k = array_index_to_video_index(chunk, lane, batch)
    flag, ϕ, θ = @views output[lane, :, chunk]
    #a flag of T(0) means that that pixel should be black.

    if flag == T(0)
        frame_buffer[i, j, k] = RGB{T}(T(0), T(0), T(0))
        
    end

    tex_height, tex_width = size(texture)

    @fastmath begin
        u = (ϕ + T(π)) / T(2π)
        v = θ / T(π)
        u = clamp(u, T(0), T(1))
        v = clamp(v, T(0), T(1))

        x_cont = u * T(tex_width - 1)
        y_cont = v * T(tex_height - 1)

        x0 = floor(Int, x_cont)
        y0 = floor(Int, y_cont)
        x1 = min(x0 + 1, tex_width - 1)
        y1 = min(y0 + 1, tex_height - 1)
        
        fx = x_cont - T(x0)
        fy = y_cont - T(y0)

        x0 += 1
        y0 += 1
        x1 += 1
        y1 += 1
        
        x0 = mod1(x0, tex_width)
        x1 = mod1(x1, tex_width)
        
        c00 = texture[y0, x0]
        c10 = texture[y0, x1]
        c01 = texture[y1, x0]
        c11 = texture[y1, x1]

        r = (c00.r * (T(1) - fx) + c10.r * fx) * (T(1) - fy) +
            (c01.r * (T(1) - fx) + c11.r * fx) * fy
            
        g = (c00.g * (T(1) - fx) + c10.g * fx) * (T(1) - fy) +
            (c01.g * (T(1) - fx) + c11.g * fx) * fy
            
        b = (c00.b * (T(1) - fx) + c10.b * fx) * (T(1) - fy) +
            (c01.b * (T(1) - fx) + c11.b * fx) * fy
        
        frame_buffer[i, j, k] = RGB{T}(
            clamp(r, T(0), T(1)),
            clamp(g, T(0), T(1)),
            clamp(b, T(0), T(1))
        )
    end


end

function render_output(output::AbstractArray{T}, batch::SubStruct{V, H, NWarps, MWarps}, 
    backend, texture::AbstractArray{RGB{T}},  framerate::Int; blocksize = 256) where {T, V, H, NWarps, MWarps}

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
    
    kernel! = linear_background_render!(backend, blocksize)
    kernel!(
        frame_buffer,
        texture_device,
        output,
        batch;
        ndrange = total_threads
    )

    KernelAbstractions.synchronize(backend)
    frame_buffer_cpu = Array(frame_buffer)

    filename = "output_$(I)x$(J)_$(K)frames_$(framerate)fps.mp4"
    
    writer = open_video_out(
        filename,
        RGB{N0f8},
        (J, I);
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