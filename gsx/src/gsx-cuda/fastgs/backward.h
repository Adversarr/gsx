#pragma once

#include "helper_math.h"

#include <functional>

#include "tinygs/core/gaussian.hpp"

namespace fast_gs::rasterization {

void backward(
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
    float4 *grad_w2c_per_gs,
    tinygs::DensificationInfo *densification_info,
    float2 *absgrad_mean2d_helper,
    int n_primitives,
    int n_visible_primitives,
    int n_instances,
    int n_buckets,
    int primitive_primitive_indices_selector,
    int instance_primitive_indices_selector,
    int active_sh_bases,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    cudaStream_t stream
);

} /* namespace fast_gs::rasterization */
