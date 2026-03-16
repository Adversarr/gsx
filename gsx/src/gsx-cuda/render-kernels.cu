#include "internal.h"

#include "fastgs/backward.h"
#include "fastgs/forward.h"
#include <cuda_runtime.h>

#include <exception>
#include <stdio.h>
#include <tuple>

namespace {

__global__ void gsx_cuda_render_compose_background_to_chw_f32_kernel(
    const float *__restrict__ src_chw,
    const float *__restrict__ alpha,
    float *__restrict__ dst_chw,
    int width,
    int height,
    int channel_stride,
    float3 background_color
)
{
    int x = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int y = (int)blockIdx.y * blockDim.y + (int)threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    int index = y * width + x;
    float transmittance = 1.0f - alpha[index];

    dst_chw[index] = src_chw[index] + transmittance * background_color.x;
    dst_chw[channel_stride + index] = src_chw[channel_stride + index] + transmittance * background_color.y;
    dst_chw[2 * channel_stride + index] = src_chw[2 * channel_stride + index] + transmittance * background_color.z;
}

__global__ void gsx_cuda_render_compose_background_chw_inplace_f32_kernel(
    float *__restrict__ image_chw,
    const float *__restrict__ alpha,
    int width,
    int height,
    int channel_stride,
    float3 background_color
)
{
    int x = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int y = (int)blockIdx.y * blockDim.y + (int)threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    int index = y * width + x;
    float transmittance = 1.0f - alpha[index];

    image_chw[index] += transmittance * background_color.x;
    image_chw[channel_stride + index] += transmittance * background_color.y;
    image_chw[2 * channel_stride + index] += transmittance * background_color.z;
}

static cudaError_t gsx_cuda_wrap_fastgs_exception(const std::exception &exception)
{
    fprintf(stderr, "gsx cuda fastgs exception: %s\n", exception.what());
    return cudaErrorUnknown;
}

} /* namespace */

extern "C" {

cudaError_t gsx_cuda_render_compose_background_to_chw_f32_kernel_launch(
    const float *src_chw,
    const float *alpha_chw,
    float *dst_chw,
    gsx_index_t width,
    gsx_index_t height,
    gsx_vec3 background_color,
    cudaStream_t stream
)
{
    int channel_stride = (int)width * (int)height;
    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)(((int)width + 15) / 16), (unsigned int)(((int)height + 15) / 16), 1);

    gsx_cuda_render_compose_background_to_chw_f32_kernel<<<grid, block, 0, stream>>>(
        src_chw,
        alpha_chw,
        dst_chw,
        (int)width,
        (int)height,
        channel_stride,
        make_float3(background_color.x, background_color.y, background_color.z)
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_compose_background_chw_inplace_f32_kernel_launch(
    float *image_chw,
    const float *alpha_chw,
    gsx_index_t width,
    gsx_index_t height,
    gsx_vec3 background_color,
    cudaStream_t stream
)
{
    int channel_stride = (int)width * (int)height;
    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)(((int)width + 15) / 16), (unsigned int)(((int)height + 15) / 16), 1);

    gsx_cuda_render_compose_background_chw_inplace_f32_kernel<<<grid, block, 0, stream>>>(
        image_chw,
        alpha_chw,
        (int)width,
        (int)height,
        channel_stride,
        make_float3(background_color.x, background_color.y, background_color.z)
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_fastgs_forward_launch(
    gsx_cuda_resize_buffer_fn per_primitive_buffers_func,
    void *per_primitive_user_data,
    gsx_cuda_resize_buffer_fn per_tile_buffers_func,
    void *per_tile_user_data,
    gsx_cuda_resize_buffer_fn per_instance_buffers_func,
    void *per_instance_user_data,
    gsx_cuda_resize_buffer_fn per_bucket_buffers_func,
    void *per_bucket_user_data,
    const float3 *means,
    const float3 *scales_raw,
    const float4 *rotations_raw,
    const float *opacities_raw,
    const float *sh0,
    const float *sh1,
    const float *sh2,
    const float *sh3,
    const float4 *w2c,
    const float3 *cam_position,
    float *image,
    float *alpha,
    int n_primitives,
    int active_sh_bases,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    float near_plane,
    float far_plane,
    cudaStream_t major_stream,
    cudaStream_t helper_stream,
    char *zero_copy,
    cudaEvent_t memset_per_tile_done,
    cudaEvent_t copy_n_instances_done,
    cudaEvent_t preprocess_done,
    int *out_n_visible_primitives,
    int *out_n_instances,
    int *out_n_buckets,
    int *out_primitive_selector,
    int *out_instance_selector
)
{
    try {
        auto primitive_resize = [per_primitive_buffers_func, per_primitive_user_data](size_t size) -> char * {
            char *ptr = per_primitive_buffers_func(per_primitive_user_data, (gsx_size_t)size);

            if(ptr == nullptr) {
                throw std::bad_alloc();
            }
            return ptr;
        };
        auto tile_resize = [per_tile_buffers_func, per_tile_user_data](size_t size) -> char * {
            char *ptr = per_tile_buffers_func(per_tile_user_data, (gsx_size_t)size);

            if(ptr == nullptr) {
                throw std::bad_alloc();
            }
            return ptr;
        };
        auto instance_resize = [per_instance_buffers_func, per_instance_user_data](size_t size) -> char * {
            char *ptr = per_instance_buffers_func(per_instance_user_data, (gsx_size_t)size);

            if(ptr == nullptr) {
                throw std::bad_alloc();
            }
            return ptr;
        };
        auto bucket_resize = [per_bucket_buffers_func, per_bucket_user_data](size_t size) -> char * {
            char *ptr = per_bucket_buffers_func(per_bucket_user_data, (gsx_size_t)size);

            if(ptr == nullptr) {
                throw std::bad_alloc();
            }
            return ptr;
        };
        auto result = fast_gs::rasterization::forward(
            primitive_resize,
            tile_resize,
            instance_resize,
            bucket_resize,
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh0,
            sh1,
            sh2,
            sh3,
            w2c,
            cam_position,
            nullptr,
            image,
            alpha,
            n_primitives,
            active_sh_bases,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane,
            major_stream,
            helper_stream,
            zero_copy,
            memset_per_tile_done,
            copy_n_instances_done,
            preprocess_done,
            false,
            nullptr,
            nullptr
        );

        *out_n_visible_primitives = std::get<0>(result);
        *out_n_instances = std::get<1>(result);
        *out_n_buckets = std::get<2>(result);
        *out_primitive_selector = std::get<3>(result);
        *out_instance_selector = std::get<4>(result);
        return cudaGetLastError();
    } catch(const std::exception &exception) {
        return gsx_cuda_wrap_fastgs_exception(exception);
    }
}

cudaError_t gsx_cuda_fastgs_backward_launch(
    const float *grad_image,
    const float *image,
    const float3 *means,
    const float3 *scales_raw,
    const float4 *rotations_raw,
    const float *sh1,
    const float *sh2,
    const float *sh3,
    const float4 *w2c,
    const float3 *cam_position,
    char *per_primitive_buffers_blob,
    char *per_tile_buffers_blob,
    char *per_instance_buffers_blob,
    char *per_bucket_buffers_blob,
    float3 *grad_means,
    float3 *grad_scales_raw,
    float4 *grad_rotations_raw,
    float *grad_opacities_raw,
    float *grad_sh0,
    float *grad_sh1,
    float *grad_sh2,
    float *grad_sh3,
    float2 *grad_mean2d_helper,
    float *grad_conic_helper,
    float3 *grad_color,
    float4 *grad_w2c,
    float2 *absgrad_mean2d_helper,
    int n_primitives,
    int n_visible_primitives,
    int n_instances,
    int n_buckets,
    int primitive_selector,
    int instance_selector,
    int active_sh_bases,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    cudaStream_t stream
)
{
    try {
        fast_gs::rasterization::backward(
            grad_image,
            image,
            means,
            scales_raw,
            rotations_raw,
            sh1,
            sh2,
            sh3,
            w2c,
            cam_position,
            per_primitive_buffers_blob,
            per_tile_buffers_blob,
            per_instance_buffers_blob,
            per_bucket_buffers_blob,
            grad_means,
            grad_scales_raw,
            grad_rotations_raw,
            grad_opacities_raw,
            grad_sh0,
            grad_sh1,
            grad_sh2,
            grad_sh3,
            grad_mean2d_helper,
            grad_conic_helper,
            grad_color,
            grad_w2c,
            nullptr,
            nullptr,
            absgrad_mean2d_helper,
            n_primitives,
            n_visible_primitives,
            n_instances,
            n_buckets,
            primitive_selector,
            instance_selector,
            active_sh_bases,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            stream
        );
        return cudaGetLastError();
    } catch(const std::exception &exception) {
        return gsx_cuda_wrap_fastgs_exception(exception);
    }
}

} /* extern "C" */
