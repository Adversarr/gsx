#include <metal_stdlib>
using namespace metal;

#include "simd_utils.metal"
#include "render_common.metal"
#include "render_backward.metal"

kernel void gsx_metal_render_preprocess_kernel(
    device const float *mean3d [[buffer(0)]],
    device const float *rotation [[buffer(1)]],
    device const float *logscale [[buffer(2)]],
    device const float *sh0 [[buffer(3)]],
    device const float *opacity_raw [[buffer(4)]],
    device float *depth [[buffer(5)]],
    device int *visible [[buffer(6)]],
    device int *touched_tiles [[buffer(7)]],
    device float *bounds [[buffer(8)]],
    device float *mean2d [[buffer(9)]],
    device float *conic_opacity [[buffer(10)]],
    device float *color [[buffer(11)]],
    constant gsx_metal_render_preprocess_params &params [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    uint mean_base = gid * 3u;
    uint rot_base = gid * 4u;
    uint sh_base = gid * 3u;
    float3 mean = float3(mean3d[mean_base], mean3d[mean_base + 1u], mean3d[mean_base + 2u]);

    float pose_qx = params.pose_qx;
    float pose_qy = params.pose_qy;
    float pose_qz = params.pose_qz;
    float pose_qw = params.pose_qw;
    float pose_qrr_raw = pose_qw * pose_qw;
    float pose_qxx_raw = pose_qx * pose_qx;
    float pose_qyy_raw = pose_qy * pose_qy;
    float pose_qzz_raw = pose_qz * pose_qz;
    float pose_q_norm_sq = pose_qrr_raw + pose_qxx_raw + pose_qyy_raw + pose_qzz_raw;
    float w2c_r11;
    float w2c_r12;
    float w2c_r13;
    float w2c_r21;
    float w2c_r22;
    float w2c_r23;
    float w2c_r31;
    float w2c_r32;
    float w2c_r33;
    float x_cam;
    float y_cam;
    float z;
    float x;
    float y;

    if(pose_q_norm_sq < 1.0e-8f) {
        w2c_r11 = 1.0f;
        w2c_r12 = 0.0f;
        w2c_r13 = 0.0f;
        w2c_r21 = 0.0f;
        w2c_r22 = 1.0f;
        w2c_r23 = 0.0f;
        w2c_r31 = 0.0f;
        w2c_r32 = 0.0f;
        w2c_r33 = 1.0f;
    } else {
        float qxx = 2.0f * pose_qxx_raw / pose_q_norm_sq;
        float qyy = 2.0f * pose_qyy_raw / pose_q_norm_sq;
        float qzz = 2.0f * pose_qzz_raw / pose_q_norm_sq;
        float qxy = 2.0f * pose_qx * pose_qy / pose_q_norm_sq;
        float qxz = 2.0f * pose_qx * pose_qz / pose_q_norm_sq;
        float qyz = 2.0f * pose_qy * pose_qz / pose_q_norm_sq;
        float qrx = 2.0f * pose_qw * pose_qx / pose_q_norm_sq;
        float qry = 2.0f * pose_qw * pose_qy / pose_q_norm_sq;
        float qrz = 2.0f * pose_qw * pose_qz / pose_q_norm_sq;

        w2c_r11 = 1.0f - (qyy + qzz);
        w2c_r12 = qxy - qrz;
        w2c_r13 = qry + qxz;
        w2c_r21 = qrz + qxy;
        w2c_r22 = 1.0f - (qxx + qzz);
        w2c_r23 = qyz - qrx;
        w2c_r31 = qxz - qry;
        w2c_r32 = qrx + qyz;
        w2c_r33 = 1.0f - (qxx + qyy);
    }

    x_cam = w2c_r11 * mean.x + w2c_r12 * mean.y + w2c_r13 * mean.z + params.pose_tx;
    y_cam = w2c_r21 * mean.x + w2c_r22 * mean.y + w2c_r23 * mean.z + params.pose_ty;
    z = w2c_r31 * mean.x + w2c_r32 * mean.y + w2c_r33 * mean.z + params.pose_tz;
    float op = 1.0f / (1.0f + exp(-opacity_raw[gid]));

    depth[gid] = z;
    visible[gid] = 0;
    touched_tiles[gid] = 0;

    if(z <= params.near_plane || z >= params.far_plane || op < gsx_metal_render_min_alpha) {
        return;
    }

    float sx_act = exp(logscale[mean_base]);
    float sy_act = exp(logscale[mean_base + 1u]);
    float sz_act = exp(logscale[mean_base + 2u]);
    float var_x = sx_act * sx_act;
    float var_y = sy_act * sy_act;
    float var_z = sz_act * sz_act;
    float raw_qx = rotation[rot_base];
    float raw_qy = rotation[rot_base + 1u];
    float raw_qz = rotation[rot_base + 2u];
    float raw_qw = rotation[rot_base + 3u];
    float qrr_raw = raw_qw * raw_qw;
    float qxx_raw = raw_qx * raw_qx;
    float qyy_raw = raw_qy * raw_qy;
    float qzz_raw = raw_qz * raw_qz;
    float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
    float r11;
    float r12;
    float r13;
    float r21;
    float r22;
    float r23;
    float r31;
    float r32;
    float r33;
    float rs11;
    float rs12;
    float rs13;
    float rs21;
    float rs22;
    float rs23;
    float rs31;
    float rs32;
    float rs33;
    float cov11;
    float cov12;
    float cov13;
    float cov22;
    float cov23;
    float cov33;

    if(q_norm_sq < 1.0e-8f) {
        return;
    }

    {
        float qxx = 2.0f * qxx_raw / q_norm_sq;
        float qyy = 2.0f * qyy_raw / q_norm_sq;
        float qzz = 2.0f * qzz_raw / q_norm_sq;
        float qxy = 2.0f * raw_qx * raw_qy / q_norm_sq;
        float qxz = 2.0f * raw_qx * raw_qz / q_norm_sq;
        float qyz = 2.0f * raw_qy * raw_qz / q_norm_sq;
        float qrx = 2.0f * raw_qw * raw_qx / q_norm_sq;
        float qry = 2.0f * raw_qw * raw_qy / q_norm_sq;
        float qrz = 2.0f * raw_qw * raw_qz / q_norm_sq;

        r11 = 1.0f - (qyy + qzz);
        r12 = qxy - qrz;
        r13 = qry + qxz;
        r21 = qrz + qxy;
        r22 = 1.0f - (qxx + qzz);
        r23 = qyz - qrx;
        r31 = qxz - qry;
        r32 = qrx + qyz;
        r33 = 1.0f - (qxx + qyy);
    }

    rs11 = r11 * var_x;
    rs12 = r12 * var_y;
    rs13 = r13 * var_z;
    rs21 = r21 * var_x;
    rs22 = r22 * var_y;
    rs23 = r23 * var_z;
    rs31 = r31 * var_x;
    rs32 = r32 * var_y;
    rs33 = r33 * var_z;

    cov11 = rs11 * r11 + rs12 * r12 + rs13 * r13;
    cov12 = rs11 * r21 + rs12 * r22 + rs13 * r23;
    cov13 = rs11 * r31 + rs12 * r32 + rs13 * r33;
    cov22 = rs21 * r21 + rs22 * r22 + rs23 * r23;
    cov23 = rs21 * r31 + rs22 * r32 + rs23 * r33;
    cov33 = rs31 * r31 + rs32 * r32 + rs33 * r33;

    float inv_z = 1.0f / z;
    x = x_cam * inv_z;
    y = y_cam * inv_z;
    float clip_left = (-0.15f * float(params.width) - params.cx) / params.fx;
    float clip_right = (1.15f * float(params.width) - params.cx) / params.fx;
    float clip_top = (-0.15f * float(params.height) - params.cy) / params.fy;
    float clip_bottom = (1.15f * float(params.height) - params.cy) / params.fy;
    float tx = clamp(x, clip_left, clip_right);
    float ty = clamp(y, clip_top, clip_bottom);
    float j11 = params.fx / z;
    float j13 = -j11 * tx;
    float j22 = params.fy / z;
    float j23 = -j22 * ty;

    float3 jw1 = float3(j11 * w2c_r11 + j13 * w2c_r31, j11 * w2c_r12 + j13 * w2c_r32, j11 * w2c_r13 + j13 * w2c_r33);
    float3 jw2 = float3(j22 * w2c_r21 + j23 * w2c_r31, j22 * w2c_r22 + j23 * w2c_r32, j22 * w2c_r23 + j23 * w2c_r33);
    float3 jwc1 = float3(
        jw1.x * cov11 + jw1.y * cov12 + jw1.z * cov13,
        jw1.x * cov12 + jw1.y * cov22 + jw1.z * cov23,
        jw1.x * cov13 + jw1.y * cov23 + jw1.z * cov33);
    float3 jwc2 = float3(
        jw2.x * cov11 + jw2.y * cov12 + jw2.z * cov13,
        jw2.x * cov12 + jw2.y * cov22 + jw2.z * cov23,
        jw2.x * cov13 + jw2.y * cov23 + jw2.z * cov33);
    float cov2d_x = dot(jwc1, jw1) + 0.3f;
    float cov2d_y = dot(jwc1, jw2);
    float cov2d_z = dot(jwc2, jw2) + 0.3f;
    float det = cov2d_x * cov2d_z - cov2d_y * cov2d_y;
    if(det <= 1.0e-8f) {
        return;
    }

    float conic_x = cov2d_z / det;
    float conic_y = -cov2d_y / det;
    float conic_z = cov2d_x / det;
    float px = x * params.fx + params.cx;
    float py = y * params.fy + params.cy;

    mean2d[gid * 2u] = px;
    mean2d[gid * 2u + 1u] = py;
    conic_opacity[gid * 4u] = conic_x;
    conic_opacity[gid * 4u + 1u] = conic_y;
    conic_opacity[gid * 4u + 2u] = conic_z;
    conic_opacity[gid * 4u + 3u] = op;

    float power_threshold = log(op * gsx_metal_render_min_alpha_rcp);
    float power_threshold_factor = sqrt(2.0f * power_threshold);
    float extent_x = max(power_threshold_factor * sqrt(cov2d_x) - 0.5f, 0.0f);
    float extent_y = max(power_threshold_factor * sqrt(cov2d_z) - 0.5f, 0.0f);
    float x0f = floor((px - extent_x) / float(gsx_metal_render_tile_width));
    float x1f = ceil((px + extent_x) / float(gsx_metal_render_tile_width));
    float y0f = floor((py - extent_y) / float(gsx_metal_render_tile_height));
    float y1f = ceil((py + extent_y) / float(gsx_metal_render_tile_height));

    int x0 = clamp(int(x0f), 0, int(params.grid_width));
    int x1 = clamp(int(x1f), 0, int(params.grid_width));
    int y0 = clamp(int(y0f), 0, int(params.grid_height));
    int y1 = clamp(int(y1f), 0, int(params.grid_height));
    int tile_count = 0;

    bounds[gid * 4u] = float(x0);
    bounds[gid * 4u + 1u] = float(x1);
    bounds[gid * 4u + 2u] = float(y0);
    bounds[gid * 4u + 3u] = float(y1);
    if((x1 - x0) > 0 && (y1 - y0) > 0) {
        float2 mean_shifted = float2(px - 0.5f, py - 0.5f);
        float3 conic_vec = float3(conic_x, conic_y, conic_z);

        for(int yy = y0; yy < y1; ++yy) {
            for(int xx = x0; xx < x1; ++xx) {
                if(gsx_metal_render_will_primitive_contribute(mean_shifted, conic_vec, uint(xx), uint(yy), power_threshold)) {
                    tile_count += 1;
                }
            }
        }
    }

    touched_tiles[gid] = tile_count;
    if(tile_count == 0) {
        return;
    }

    visible[gid] = 1;
    color[sh_base] = 0.5f + 0.28209479177387814f * sh0[sh_base];
    color[sh_base + 1u] = 0.5f + 0.28209479177387814f * sh0[sh_base + 1u];
    color[sh_base + 2u] = 0.5f + 0.28209479177387814f * sh0[sh_base + 2u];

    (void)rotation;
    (void)rot_base;
}

kernel void gsx_metal_render_create_instances_kernel(
    device const int *sorted_primitive_ids [[buffer(0)]],
    device const int *primitive_offsets [[buffer(1)]],
    device const float *bounds [[buffer(2)]],
    device const float *mean2d [[buffer(3)]],
    device const float *conic_opacity [[buffer(4)]],
    device int *instance_keys [[buffer(5)]],
    device int *instance_primitive_ids [[buffer(6)]],
    constant gsx_metal_render_create_instances_params &params [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int primitive_id = sorted_primitive_ids[gid];
    int offset = primitive_offsets[gid];
    uint b = uint(primitive_id) * 4u;
    int x0 = int(bounds[b]);
    int x1 = int(bounds[b + 1u]);
    int y0 = int(bounds[b + 2u]);
    int y1 = int(bounds[b + 3u]);
    int local = 0;

    for(int y = y0; y < y1; ++y) {
        for(int x = x0; x < x1; ++x) {
            float2 mean_shifted = float2(mean2d[uint(primitive_id) * 2u] - 0.5f, mean2d[uint(primitive_id) * 2u + 1u] - 0.5f);
            float3 conic = float3(conic_opacity[uint(primitive_id) * 4u], conic_opacity[uint(primitive_id) * 4u + 1u], conic_opacity[uint(primitive_id) * 4u + 2u]);
            float opacity = conic_opacity[uint(primitive_id) * 4u + 3u];
            float power_threshold = log(opacity * gsx_metal_render_min_alpha_rcp);

            if(gsx_metal_render_will_primitive_contribute(mean_shifted, conic, uint(x), uint(y), power_threshold)) {
                int idx = offset + local;

                instance_keys[idx] = y * int(params.grid_width) + x;
                instance_primitive_ids[idx] = primitive_id;
                local += 1;
            }
        }
    }
}

kernel void gsx_metal_render_blend_kernel(
    device const int *tile_ranges [[buffer(0)]],
    device const int *instance_primitive_ids [[buffer(1)]],
    device const float *mean2d [[buffer(2)]],
    device const float *conic_opacity [[buffer(3)]],
    device const float *color [[buffer(4)]],
    device float *image_chw [[buffer(5)]],
    device float *alpha_hw [[buffer(6)]],
    constant gsx_metal_render_blend_params &params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    if(gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    uint tile_id = (gid.y / gsx_metal_render_tile_height) * params.grid_width + (gid.x / gsx_metal_render_tile_width);
    int start = tile_ranges[tile_id * 2u];
    int end = tile_ranges[tile_id * 2u + 1u];
    if(start < 0 || end <= start) {
        uint pixel_index_empty = gid.y * params.width + gid.x;
        image_chw[pixel_index_empty] = 0.0f;
        image_chw[params.channel_stride + pixel_index_empty] = 0.0f;
        image_chw[2u * params.channel_stride + pixel_index_empty] = 0.0f;
        alpha_hw[pixel_index_empty] = 0.0f;
        return;
    }

    {
        float2 pixel = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);
        float transmittance = 1.0f;
        float3 accum = float3(0.0f);

        for(int idx = start; idx < end; ++idx) {
            int primitive_id = instance_primitive_ids[idx];
            uint p2 = uint(primitive_id) * 2u;
            uint p3 = uint(primitive_id) * 3u;
            uint p4 = uint(primitive_id) * 4u;
            float2 mu = float2(mean2d[p2], mean2d[p2 + 1u]);
            float a = conic_opacity[p4];
            float b = conic_opacity[p4 + 1u];
            float c = conic_opacity[p4 + 2u];
            float opacity = conic_opacity[p4 + 3u];
            float2 d = mu - pixel;
            float sigma_over_2 = 0.5f * (a * d.x * d.x + c * d.y * d.y) + b * d.x * d.y;

            if(sigma_over_2 < 0.0f) {
                continue;
            }

            {
                float alpha = min(opacity * exp(-sigma_over_2), gsx_metal_render_max_alpha);

                if(alpha < gsx_metal_render_min_alpha) {
                    continue;
                }

                accum += transmittance * alpha * max(float3(color[p3], color[p3 + 1u], color[p3 + 2u]), float3(0.0f));
                transmittance *= (1.0f - alpha);
                if(transmittance < gsx_metal_render_min_transmittance) {
                    break;
                }
            }
        }

        {
            uint pixel_index = gid.y * params.width + gid.x;

            image_chw[pixel_index] = accum.x;
            image_chw[params.channel_stride + pixel_index] = accum.y;
            image_chw[2u * params.channel_stride + pixel_index] = accum.z;
            alpha_hw[pixel_index] = 1.0f - transmittance;
        }
    }
}

kernel void gsx_metal_render_compose_chw_f32_kernel(
    device const float *image_chw [[buffer(0)]],
    device const float *alpha_hw [[buffer(1)]],
    device float *out_chw [[buffer(2)]],
    constant gsx_metal_render_compose_params &params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if(gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    uint index = gid.y * params.width + gid.x;
    float transmittance = 1.0f - alpha_hw[index];

    out_chw[index] = image_chw[index] + transmittance * params.background_r;
    out_chw[params.channel_stride + index] = image_chw[params.channel_stride + index] + transmittance * params.background_g;
    out_chw[2u * params.channel_stride + index] = image_chw[2u * params.channel_stride + index] + transmittance * params.background_b;
}
