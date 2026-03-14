#include "internal.h"

#include "fastgs/backward.h"
#include "fastgs/forward.h"
#include "fastgs/tinygs/common.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <functional>
#include <stdio.h>
#include <tuple>

namespace {

__global__ void gsx_cuda_render_tiled_to_chw_f32_kernel(
    const float *__restrict__ src_tiled,
    const float *__restrict__ alpha_tiled,
    float *__restrict__ dst_chw,
    int width,
    int height,
    int width_in_tile,
    int channel_stride,
    float3 background_color
)
{
    int x = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int y = (int)blockIdx.y * blockDim.y + (int)threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    unsigned int tiled_index = tinygs::get_linear_index_tiled((unsigned int)y, (unsigned int)x, (unsigned int)width_in_tile);
    int chw_index = y * width + x;

    {
        float transmittance = 1.0f - alpha_tiled[tiled_index];

        dst_chw[chw_index] = src_tiled[tiled_index] + transmittance * background_color.x;
        dst_chw[width * height + chw_index] = src_tiled[channel_stride + tiled_index] + transmittance * background_color.y;
        dst_chw[2 * width * height + chw_index] = src_tiled[2 * channel_stride + tiled_index] + transmittance * background_color.z;
    }
}

__global__ void gsx_cuda_render_chw_to_tiled_f32_kernel(
    const float *__restrict__ src_chw,
    float *__restrict__ dst_tiled,
    int width,
    int height,
    int width_in_tile,
    int channel_stride
)
{
    int x = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int y = (int)blockIdx.y * blockDim.y + (int)threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    unsigned int tiled_index = tinygs::get_linear_index_tiled((unsigned int)y, (unsigned int)x, (unsigned int)width_in_tile);
    int chw_index = y * width + x;

    dst_tiled[tiled_index] = src_chw[chw_index];
    dst_tiled[channel_stride + tiled_index] = src_chw[width * height + chw_index];
    dst_tiled[2 * channel_stride + tiled_index] = src_chw[2 * width * height + chw_index];
}

__global__ void gsx_cuda_render_compose_background_tiled_f32_kernel(
    float *__restrict__ image_tiled,
    const float *__restrict__ alpha_tiled,
    int width,
    int height,
    int width_in_tile,
    int channel_stride,
    float3 background_color
)
{
    int x = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int y = (int)blockIdx.y * blockDim.y + (int)threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    unsigned int tiled_index = tinygs::get_linear_index_tiled((unsigned int)y, (unsigned int)x, (unsigned int)width_in_tile);
    float transmittance = 1.0f - alpha_tiled[tiled_index];

    image_tiled[tiled_index] += transmittance * background_color.x;
    image_tiled[channel_stride + tiled_index] += transmittance * background_color.y;
    image_tiled[2 * channel_stride + tiled_index] += transmittance * background_color.z;
}

__global__ void gsx_cuda_render_clear_tiled_f32_kernel(float *__restrict__ dst_tiled, int total_elements)
{
    int index = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;

    if(index >= total_elements) {
        return;
    }
    dst_tiled[index] = 0.0f;
}

__global__ void gsx_cuda_render_rotation_xyzw_to_wxyz_kernel(
    const float *__restrict__ src_xyzw,
    float *__restrict__ dst_wxyz,
    int gaussian_count
)
{
    int index = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;

    if(index >= gaussian_count) {
        return;
    }

    int src_base = index * 4;
    int dst_base = src_base;

    dst_wxyz[dst_base + 0] = src_xyzw[src_base + 3];
    dst_wxyz[dst_base + 1] = src_xyzw[src_base + 0];
    dst_wxyz[dst_base + 2] = src_xyzw[src_base + 1];
    dst_wxyz[dst_base + 3] = src_xyzw[src_base + 2];
}

__global__ void gsx_cuda_render_rotation_wxyz_to_xyzw_kernel(
    const float *__restrict__ src_wxyz,
    float *__restrict__ dst_xyzw,
    int gaussian_count
)
{
    int index = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;

    if(index >= gaussian_count) {
        return;
    }

    int src_base = index * 4;
    int dst_base = src_base;

    dst_xyzw[dst_base + 0] = src_wxyz[src_base + 1];
    dst_xyzw[dst_base + 1] = src_wxyz[src_base + 2];
    dst_xyzw[dst_base + 2] = src_wxyz[src_base + 3];
    dst_xyzw[dst_base + 3] = src_wxyz[src_base + 0];
}

__global__ void gsx_cuda_render_sh_aos_to_soa_kernel(
    const float *__restrict__ src_aos,
    float *__restrict__ dst_soa,
    int gaussian_count,
    int coeff_count
)
{
    int linear_index = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int total_values = gaussian_count * coeff_count * 3;

    if(linear_index >= total_values) {
        return;
    }

    int channel = linear_index % 3;
    int coeff_index = (linear_index / 3) % coeff_count;
    int gaussian_index = linear_index / (coeff_count * 3);
    int dst_index = coeff_index * (3 * gaussian_count) + channel * gaussian_count + gaussian_index;

    dst_soa[dst_index] = src_aos[linear_index];
}

__global__ void gsx_cuda_render_sh_soa_to_aos_kernel(
    const float *__restrict__ src_soa,
    float *__restrict__ dst_aos,
    int gaussian_count,
    int coeff_count
)
{
    int linear_index = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    int total_values = gaussian_count * coeff_count * 3;

    if(linear_index >= total_values) {
        return;
    }

    int channel = linear_index % 3;
    int coeff_index = (linear_index / 3) % coeff_count;
    int gaussian_index = linear_index / (coeff_count * 3);
    int src_index = coeff_index * (3 * gaussian_count) + channel * gaussian_count + gaussian_index;

    dst_aos[linear_index] = src_soa[src_index];
}

static cudaError_t gsx_cuda_wrap_fastgs_exception(const std::exception &exception)
{
    fprintf(stderr, "gsx cuda fastgs exception: %s\n", exception.what());
    return cudaErrorUnknown;
}

} /* namespace */

extern "C" {

cudaError_t gsx_cuda_render_tiled_to_chw_f32_kernel_launch(
    const float *src_tiled,
    const float *alpha_tiled,
    float *dst_chw,
    gsx_index_t width,
    gsx_index_t height,
    gsx_vec3 background_color,
    cudaStream_t stream
)
{
    int width_in_tile = ((int)width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int height_in_tile = ((int)height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)(((int)width + 15) / 16), (unsigned int)(((int)height + 15) / 16), 1);

    gsx_cuda_render_tiled_to_chw_f32_kernel<<<grid, block, 0, stream>>>(
        src_tiled,
        alpha_tiled,
        dst_chw,
        (int)width,
        (int)height,
        width_in_tile,
        channel_stride,
        make_float3(background_color.x, background_color.y, background_color.z)
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_chw_to_tiled_f32_kernel_launch(
    const float *src_chw,
    float *dst_tiled,
    gsx_index_t width,
    gsx_index_t height,
    cudaStream_t stream
)
{
    int width_in_tile = ((int)width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int height_in_tile = ((int)height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)(((int)width + 15) / 16), (unsigned int)(((int)height + 15) / 16), 1);

    gsx_cuda_render_chw_to_tiled_f32_kernel<<<grid, block, 0, stream>>>(
        src_chw,
        dst_tiled,
        (int)width,
        (int)height,
        width_in_tile,
        channel_stride
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_clear_tiled_f32_kernel_launch(
    float *dst_tiled,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    cudaStream_t stream
)
{
    int width_in_tile = ((int)width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int height_in_tile = ((int)height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    int total_elements = channel_stride * (int)channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    gsx_cuda_render_clear_tiled_f32_kernel<<<grid_size, block_size, 0, stream>>>(dst_tiled, total_elements);
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_compose_background_tiled_f32_kernel_launch(
    float *image_tiled,
    const float *alpha_tiled,
    gsx_index_t width,
    gsx_index_t height,
    gsx_vec3 background_color,
    cudaStream_t stream
)
{
    int width_in_tile = ((int)width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int height_in_tile = ((int)height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    int channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)(((int)width + 15) / 16), (unsigned int)(((int)height + 15) / 16), 1);

    gsx_cuda_render_compose_background_tiled_f32_kernel<<<grid, block, 0, stream>>>(
        image_tiled,
        alpha_tiled,
        (int)width,
        (int)height,
        width_in_tile,
        channel_stride,
        make_float3(background_color.x, background_color.y, background_color.z)
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_rotation_xyzw_to_wxyz_kernel_launch(
    const float *src_xyzw,
    float *dst_wxyz,
    gsx_size_t gaussian_count,
    cudaStream_t stream
)
{
    int block_size = 256;
    int grid_size = ((int)gaussian_count + block_size - 1) / block_size;

    if(gaussian_count == 0) {
        return cudaSuccess;
    }
    gsx_cuda_render_rotation_xyzw_to_wxyz_kernel<<<grid_size, block_size, 0, stream>>>(
        src_xyzw,
        dst_wxyz,
        (int)gaussian_count
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_rotation_wxyz_to_xyzw_kernel_launch(
    const float *src_wxyz,
    float *dst_xyzw,
    gsx_size_t gaussian_count,
    cudaStream_t stream
)
{
    int block_size = 256;
    int grid_size = ((int)gaussian_count + block_size - 1) / block_size;

    if(gaussian_count == 0) {
        return cudaSuccess;
    }
    gsx_cuda_render_rotation_wxyz_to_xyzw_kernel<<<grid_size, block_size, 0, stream>>>(
        src_wxyz,
        dst_xyzw,
        (int)gaussian_count
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_sh_aos_to_soa_kernel_launch(
    const float *src_aos,
    float *dst_soa,
    gsx_size_t gaussian_count,
    gsx_index_t coeff_count,
    cudaStream_t stream
)
{
    int total_values = (int)(gaussian_count * (gsx_size_t)coeff_count * 3u);
    int block_size = 256;
    int grid_size = (total_values + block_size - 1) / block_size;

    if(total_values == 0) {
        return cudaSuccess;
    }
    gsx_cuda_render_sh_aos_to_soa_kernel<<<grid_size, block_size, 0, stream>>>(
        src_aos,
        dst_soa,
        (int)gaussian_count,
        (int)coeff_count
    );
    return cudaGetLastError();
}

cudaError_t gsx_cuda_render_sh_soa_to_aos_kernel_launch(
    const float *src_soa,
    float *dst_aos,
    gsx_size_t gaussian_count,
    gsx_index_t coeff_count,
    cudaStream_t stream
)
{
    int total_values = (int)(gaussian_count * (gsx_size_t)coeff_count * 3u);
    int block_size = 256;
    int grid_size = (total_values + block_size - 1) / block_size;

    if(total_values == 0) {
        return cudaSuccess;
    }
    gsx_cuda_render_sh_soa_to_aos_kernel<<<grid_size, block_size, 0, stream>>>(
        src_soa,
        dst_aos,
        (int)gaussian_count,
        (int)coeff_count
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
