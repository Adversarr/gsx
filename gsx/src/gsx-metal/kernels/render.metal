#include <metal_stdlib>
using namespace metal;

#include "simd_utils.metal"
#include "render_common.metal"
#include "render_backward.metal"

static inline uint gsx_metal_first_set_lane_u64(ulong mask)
{
    uint low = (uint)(mask & 0xFFFFFFFFul);
    if(low != 0u) {
        return (uint)ctz(low);
    }
    return 32u + (uint)ctz((uint)(mask >> 32));
}

static inline int gsx_metal_render_count_touched_tiles_simd(
    float2 mean_shifted,
    float3 conic,
    float power_threshold,
    int x0,
    int x1,
    int y0,
    int y1,
    bool active,
    uint simd_lane_id)
{
    int tile_count = 0;
    int screen_bounds_width = x1 - x0;
    int screen_bounds_height = y1 - y0;
    int tile_count_max = screen_bounds_width * screen_bounds_height;

    if(active && tile_count_max > 0) {
        int sequential_count = min(tile_count_max, int(gsx_metal_render_sequential_tile_threshold));
        for(int local_idx = 0; local_idx < sequential_count; ++local_idx) {
            int yy = y0 + local_idx / screen_bounds_width;
            int xx = x0 + local_idx % screen_bounds_width;
            if(gsx_metal_render_will_primitive_contribute(mean_shifted, conic, uint(xx), uint(yy), power_threshold)) {
                tile_count += 1;
            }
        }
    }

    {
        bool cooperative = active && tile_count_max > int(gsx_metal_render_sequential_tile_threshold);
        ulong active_mask = gsx_metal_simd_active_threads_mask();
        ulong cooperative_mask = gsx_metal_simd_ballot(cooperative) & active_mask;
        uint active_lane_count = gsx_metal_simd_sum(1u);
        uint lane_rank = gsx_metal_simd_prefix_exclusive_sum(1u);

        if(cooperative_mask != 0ul) {
            ulong remaining_cooperative_mask = cooperative_mask;
            while(remaining_cooperative_mask != 0ul) {
                uint source_lane = gsx_metal_first_set_lane_u64(remaining_cooperative_mask);
                remaining_cooperative_mask &= (remaining_cooperative_mask - 1ul);

                int x0_coop = int(gsx_metal_simd_shuffle(x0, (ushort)source_lane));
                int x1_coop = int(gsx_metal_simd_shuffle(x1, (ushort)source_lane));
                int y0_coop = int(gsx_metal_simd_shuffle(y0, (ushort)source_lane));
                int y1_coop = int(gsx_metal_simd_shuffle(y1, (ushort)source_lane));
                int screen_bounds_width_coop = x1_coop - x0_coop;
                int tile_count_max_coop = screen_bounds_width_coop * (y1_coop - y0_coop);
                float2 mean_shifted_coop = float2(
                    gsx_metal_simd_shuffle(mean_shifted.x, (ushort)source_lane),
                    gsx_metal_simd_shuffle(mean_shifted.y, (ushort)source_lane));
                float3 conic_coop = float3(
                    gsx_metal_simd_shuffle(conic.x, (ushort)source_lane),
                    gsx_metal_simd_shuffle(conic.y, (ushort)source_lane),
                    gsx_metal_simd_shuffle(conic.z, (ushort)source_lane));
                float power_threshold_coop = gsx_metal_simd_shuffle(power_threshold, (ushort)source_lane);

                for(int local_idx_base = int(gsx_metal_render_sequential_tile_threshold);
                    local_idx_base < tile_count_max_coop;
                    local_idx_base += int(active_lane_count)) {
                    int local_idx = local_idx_base + int(lane_rank);
                    bool valid = local_idx < tile_count_max_coop;
                    int yy = y0_coop;
                    int xx = x0_coop;
                    if (valid) {
                        yy += local_idx / screen_bounds_width_coop;
                        xx += local_idx % screen_bounds_width_coop;
                    }
                    bool contributes = valid && gsx_metal_render_will_primitive_contribute(
                        mean_shifted_coop,
                        conic_coop,
                        uint(xx),
                        uint(yy),
                        power_threshold_coop);
                    uint contributes_count = gsx_metal_simd_sum(contributes ? 1u : 0u);

                    if(simd_lane_id == source_lane) {
                        tile_count += int(contributes_count);
                    }
                }
            }
        }
    }

    return tile_count;
}

static inline uint gsx_metal_forward_sh_degree_to_active_bases(uint sh_degree)
{
    if(sh_degree == 0u) {
        return 1u;
    }
    if(sh_degree == 1u) {
        return 4u;
    }
    if(sh_degree == 2u) {
        return 9u;
    }
    return 16u;
}

static inline float3 gsx_metal_forward_read_sh0_aos(device const float *sh0, uint primitive_idx)
{
    uint base = primitive_idx * 3u;
    return float3(sh0[base], sh0[base + 1u], sh0[base + 2u]);
}

static inline float3 gsx_metal_forward_read_sh_aos(device const float *sh, uint coeff_idx, uint coeff_count, uint primitive_idx)
{
    uint base = primitive_idx * (coeff_count * 3u) + coeff_idx * 3u;
    return float3(sh[base], sh[base + 1u], sh[base + 2u]);
}

static inline float3 gsx_metal_forward_convert_sh_to_color(
    device const float *sh0,
    device const float *sh1,
    device const float *sh2,
    device const float *sh3,
    float3 position,
    float3 cam_position,
    uint primitive_idx,
    uint active_sh_bases)
{
    float3 result = 0.5f + 0.28209479177387814f * gsx_metal_forward_read_sh0_aos(sh0, primitive_idx);
    if(active_sh_bases > 1u) {
        float3 direction = normalize(position - cam_position);
        float x = direction.x;
        float y = direction.y;
        float z = direction.z;
        float3 c0 = gsx_metal_forward_read_sh_aos(sh1, 0u, 3u, primitive_idx);
        float3 c1 = gsx_metal_forward_read_sh_aos(sh1, 1u, 3u, primitive_idx);
        float3 c2 = gsx_metal_forward_read_sh_aos(sh1, 2u, 3u, primitive_idx);
        result = result + (-0.48860251190291987f * y) * c0 + (0.48860251190291987f * z) * c1 + (-0.48860251190291987f * x) * c2;
        if(active_sh_bases > 4u) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float xz = x * z;
            float yz = y * z;
            float3 c3 = gsx_metal_forward_read_sh_aos(sh2, 0u, 5u, primitive_idx);
            float3 c4 = gsx_metal_forward_read_sh_aos(sh2, 1u, 5u, primitive_idx);
            float3 c5 = gsx_metal_forward_read_sh_aos(sh2, 2u, 5u, primitive_idx);
            float3 c6 = gsx_metal_forward_read_sh_aos(sh2, 3u, 5u, primitive_idx);
            float3 c7 = gsx_metal_forward_read_sh_aos(sh2, 4u, 5u, primitive_idx);
            result = result
                + (1.0925484305920792f * xy) * c3
                + (-1.0925484305920792f * yz) * c4
                + (0.94617469575755997f * zz - 0.31539156525251999f) * c5
                + (-1.0925484305920792f * xz) * c6
                + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * c7;
            if(active_sh_bases > 9u) {
                float3 c8 = gsx_metal_forward_read_sh_aos(sh3, 0u, 7u, primitive_idx);
                float3 c9 = gsx_metal_forward_read_sh_aos(sh3, 1u, 7u, primitive_idx);
                float3 c10 = gsx_metal_forward_read_sh_aos(sh3, 2u, 7u, primitive_idx);
                float3 c11 = gsx_metal_forward_read_sh_aos(sh3, 3u, 7u, primitive_idx);
                float3 c12 = gsx_metal_forward_read_sh_aos(sh3, 4u, 7u, primitive_idx);
                float3 c13 = gsx_metal_forward_read_sh_aos(sh3, 5u, 7u, primitive_idx);
                float3 c14 = gsx_metal_forward_read_sh_aos(sh3, 6u, 7u, primitive_idx);
                result = result
                    + (0.59004358992664352f * y * (-3.0f * xx + yy)) * c8
                    + (2.8906114426405538f * xy * z) * c9
                    + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * c10
                    + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * c11
                    + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * c12
                    + (1.4453057213202769f * z * (xx - yy)) * c13
                    + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * c14;
            }
        }
    }
    return result;
}

kernel void gsx_metal_render_preprocess_kernel(
    device const float *  __restrict__ mean3d [[buffer(0)]],
    device const float4 * __restrict__ rotation [[buffer(1)]],
    device const float *  __restrict__ logscale [[buffer(2)]],
    device const float *  __restrict__ sh0 [[buffer(3)]],
    device const float *  __restrict__ sh1 [[buffer(4)]],
    device const float *  __restrict__ sh2 [[buffer(5)]],
    device const float *  __restrict__ sh3 [[buffer(6)]],
    device const float *  __restrict__ opacity_raw [[buffer(7)]],
    device uint *         __restrict__ depth_keys [[buffer(8)]],
    device int *          __restrict__ visible_primitive_ids [[buffer(9)]],
    device int *          __restrict__ touched_tiles [[buffer(10)]],
    device int4 *         __restrict__ bounds [[buffer(11)]],
    device float2 *       __restrict__ mean2d [[buffer(12)]],
    device float4 *       __restrict__ conic_opacity [[buffer(13)]],
    device float *        __restrict__ color [[buffer(14)]],
    device atomic_uint *  __restrict__ visible_count [[buffer(15)]],
    device atomic_uint *  __restrict__ instance_count [[buffer(16)]],
    device float *        __restrict__ visible_counter [[buffer(17)]],
    device float *        __restrict__ max_screen_radius [[buffer(18)]],
    constant gsx_metal_render_preprocess_params &params [[buffer(19)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    if(params.gaussian_count == 0u) {
        return;
    }

    bool active = gid < params.gaussian_count;
    uint safe_gid = active ? gid : (params.gaussian_count - 1u);
    uint mean_base = safe_gid * 3u;
    float3 mean = float3(mean3d[mean_base], mean3d[mean_base + 1u], mean3d[mean_base + 2u]);
    float4 raw_rotation = rotation[safe_gid];
    float pose_qx = params.pose_qx;
    float pose_qy = params.pose_qy;
    float pose_qz = params.pose_qz;
    float pose_qw = params.pose_qw;
    float pose_qrr_raw = pose_qw * pose_qw;
    float pose_qxx_raw = pose_qx * pose_qx;
    float pose_qyy_raw = pose_qy * pose_qy;
    float pose_qzz_raw = pose_qz * pose_qz;
    float pose_q_norm_sq = pose_qrr_raw + pose_qxx_raw + pose_qyy_raw + pose_qzz_raw;
    float w2c_r11, w2c_r12, w2c_r13, w2c_r21, w2c_r22, w2c_r23, w2c_r31, w2c_r32, w2c_r33;
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
    float x_cam = w2c_r11 * mean.x + w2c_r12 * mean.y + w2c_r13 * mean.z + params.pose_tx;
    float y_cam = w2c_r21 * mean.x + w2c_r22 * mean.y + w2c_r23 * mean.z + params.pose_ty;
    float z = w2c_r31 * mean.x + w2c_r32 * mean.y + w2c_r33 * mean.z + params.pose_tz;
    float op = 1.0f / (1.0f + exp(-opacity_raw[safe_gid]));

    if(active) {
        touched_tiles[gid] = 0;
    }

    active = active && z > params.near_plane && z < params.far_plane && op >= gsx_metal_render_min_alpha;
    if(!gsx_metal_simd_any(active)) {
        return;
    }

    float sx_act = exp(logscale[mean_base]);
    float sy_act = exp(logscale[mean_base + 1u]);
    float sz_act = exp(logscale[mean_base + 2u]);
    float var_x = sx_act * sx_act;
    float var_y = sy_act * sy_act;
    float var_z = sz_act * sz_act;
    float raw_qx = raw_rotation.x;
    float raw_qy = raw_rotation.y;
    float raw_qz = raw_rotation.z;
    float raw_qw = raw_rotation.w;
    float qrr_raw = raw_qw * raw_qw;
    float qxx_raw = raw_qx * raw_qx;
    float qyy_raw = raw_qy * raw_qy;
    float qzz_raw = raw_qz * raw_qz;
    float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
    float r11, r12, r13, r21, r22, r23, r31, r32, r33;

    active = active && q_norm_sq >= 1.0e-8f;
    if(!gsx_metal_simd_any(active)) {
        return;
    }

    {
        float q_norm_sq_safe = max(q_norm_sq, 1.0e-8f);
        float qxx = 2.0f * qxx_raw / q_norm_sq_safe;
        float qyy = 2.0f * qyy_raw / q_norm_sq_safe;
        float qzz = 2.0f * qzz_raw / q_norm_sq_safe;
        float qxy = 2.0f * raw_qx * raw_qy / q_norm_sq_safe;
        float qxz = 2.0f * raw_qx * raw_qz / q_norm_sq_safe;
        float qyz = 2.0f * raw_qy * raw_qz / q_norm_sq_safe;
        float qrx = 2.0f * raw_qw * raw_qx / q_norm_sq_safe;
        float qry = 2.0f * raw_qw * raw_qy / q_norm_sq_safe;
        float qrz = 2.0f * raw_qw * raw_qz / q_norm_sq_safe;

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

    float rs11 = r11 * var_x;
    float rs12 = r12 * var_y;
    float rs13 = r13 * var_z;
    float rs21 = r21 * var_x;
    float rs22 = r22 * var_y;
    float rs23 = r23 * var_z;
    float rs31 = r31 * var_x;
    float rs32 = r32 * var_y;
    float rs33 = r33 * var_z;

    float cov11 = rs11 * r11 + rs12 * r12 + rs13 * r13;
    float cov12 = rs11 * r21 + rs12 * r22 + rs13 * r23;
    float cov13 = rs11 * r31 + rs12 * r32 + rs13 * r33;
    float cov22 = rs21 * r21 + rs22 * r22 + rs23 * r23;
    float cov23 = rs21 * r31 + rs22 * r32 + rs23 * r33;
    float cov33 = rs31 * r31 + rs32 * r32 + rs33 * r33;

    float z_safe = fabs(z) > 1.0e-8f ? z : 1.0f;
    float inv_z = 1.0f / z_safe;
    float x = x_cam * inv_z;
    float y = y_cam * inv_z;
    float clip_left = (-0.15f * float(params.width) - params.cx) / params.fx;
    float clip_right = (1.15f * float(params.width) - params.cx) / params.fx;
    float clip_top = (-0.15f * float(params.height) - params.cy) / params.fy;
    float clip_bottom = (1.15f * float(params.height) - params.cy) / params.fy;
    float tx = clamp(x, clip_left, clip_right);
    float ty = clamp(y, clip_top, clip_bottom);
    float j11 = params.fx / z_safe;
    float j13 = -j11 * tx;
    float j22 = params.fy / z_safe;
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
    active = active && det > 1.0e-8f;
    if(!gsx_metal_simd_any(active)) {
        return;
    }

    float det_safe = max(det, 1.0e-8f);
    float conic_x = cov2d_z / det_safe;
    float conic_y = -cov2d_y / det_safe;
    float conic_z = cov2d_x / det_safe;
    float px = x * params.fx + params.cx;
    float py = y * params.fy + params.cy;

    if(active) {
        mean2d[gid] = float2(px, py);
        conic_opacity[gid] = float4(conic_x, conic_y, conic_z, op);
    }

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

    bool contributes_region = (x1 - x0) > 0 && (y1 - y0) > 0;
    active = active && contributes_region;
    if(active) {
        bounds[gid] = int4(x0, x1, y0, y1);
    }
    {
        float2 mean_shifted = float2(px - 0.5f, py - 0.5f);
        float3 conic_vec = float3(conic_x, conic_y, conic_z);

        tile_count = gsx_metal_render_count_touched_tiles_simd(
            mean_shifted,
            conic_vec,
            power_threshold,
            x0,
            x1,
            y0,
            y1,
            active,
            simd_lane_id);
    }

    if(active) {
        touched_tiles[gid] = tile_count;
    }

    if(active && tile_count > 0) {
        uint visible_offset = atomic_fetch_add_explicit(visible_count, 1u, memory_order_relaxed);
        depth_keys[visible_offset] = as_type<uint>(z);
        visible_primitive_ids[visible_offset] = int(gid);
        atomic_fetch_add_explicit(instance_count, uint(tile_count), memory_order_relaxed);
        if(params.has_visible_counter != 0u) {
            gsx_metal_atomic_add_f32(visible_counter, gid, 1.0f);
        }
        if(params.has_max_screen_radius != 0u) {
            float mid = 0.5f * (cov2d_x + cov2d_z);
            float lambda_disc = sqrt(max(0.1f, mid * mid - det));
            float lambda1 = mid + lambda_disc;
            float lambda2 = mid - lambda_disc;
            float screen_radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));
            gsx_metal_atomic_max_f32_nonnegative(max_screen_radius, gid, screen_radius);
        }
        uint active_sh_bases = gsx_metal_forward_sh_degree_to_active_bases(params.sh_degree);
        float3 cam_position = -float3(
            w2c_r11 * params.pose_tx + w2c_r21 * params.pose_ty + w2c_r31 * params.pose_tz,
            w2c_r12 * params.pose_tx + w2c_r22 * params.pose_ty + w2c_r32 * params.pose_tz,
            w2c_r13 * params.pose_tx + w2c_r23 * params.pose_ty + w2c_r33 * params.pose_tz);
        float3 evaluated_color = gsx_metal_forward_convert_sh_to_color(sh0, sh1, sh2, sh3, mean, cam_position, gid, active_sh_bases);
        uint color_base = gid * 3u;
        color[color_base] = evaluated_color.x;
        color[color_base + 1u] = evaluated_color.y;
        color[color_base + 2u] = evaluated_color.z;
    }

    (void)rotation;
}

kernel void gsx_metal_render_apply_depth_ordering_kernel(
    device const int *     __restrict__ sorted_primitive_ids [[buffer(0)]],
    device const int *     __restrict__ touched_tiles [[buffer(1)]],
    device int *           __restrict__ primitive_offsets [[buffer(2)]],
    constant uint &visible_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= visible_count) {
        return;
    }
    int primitive_id = sorted_primitive_ids[gid];
    primitive_offsets[gid] = touched_tiles[(uint)primitive_id];
}

kernel void gsx_metal_render_create_instances_kernel(
    device const int *     __restrict__ sorted_primitive_ids [[buffer(0)]],
    device const int *     __restrict__ primitive_offsets [[buffer(1)]],
    device const int4 *    __restrict__ bounds [[buffer(2)]],
    device const float2 *  __restrict__ mean2d [[buffer(3)]],
    device const float4 *  __restrict__ conic_opacity [[buffer(4)]],
    device int *           __restrict__ instance_keys [[buffer(5)]],
    device int *           __restrict__ instance_primitive_ids [[buffer(6)]],
    constant gsx_metal_render_create_instances_params &params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    if(params.visible_count == 0u) {
        return;
    }
    bool active = gid < params.visible_count;
    if(!gsx_metal_simd_any(active)) {
        return;
    }

    uint safe_gid = active ? gid : (params.visible_count - 1u);
    int primitive_id = sorted_primitive_ids[safe_gid];
    int offset = primitive_offsets[safe_gid];
    int4 screen_bounds = bounds[uint(primitive_id)];
    int x0 = screen_bounds.x;
    int x1 = screen_bounds.y;
    int y0 = screen_bounds.z;
    int y1 = screen_bounds.w;
    int local = 0;

    float2 mean_shifted = mean2d[uint(primitive_id)] - 0.5f;
    float4 conic_opacity_value = conic_opacity[uint(primitive_id)];
    float3 conic = conic_opacity_value.xyz;
    float opacity = conic_opacity_value.w;
    float power_threshold = log(opacity * gsx_metal_render_min_alpha_rcp);
    int screen_bounds_width = x1 - x0;
    int tile_count = screen_bounds_width * (y1 - y0);
    int current_write_offset = offset;

    if(active) {
        for(int local_instance_idx = 0; local_instance_idx < tile_count && local_instance_idx < int(gsx_metal_render_sequential_tile_threshold); ++local_instance_idx) {
            int y = y0 + (local_instance_idx / screen_bounds_width);
            int x = x0 + (local_instance_idx % screen_bounds_width);

            if(gsx_metal_render_will_primitive_contribute(mean_shifted, conic, uint(x), uint(y), power_threshold)) {
                int idx = current_write_offset + local;

                instance_keys[idx] = y * int(params.grid_width) + x;
                instance_primitive_ids[idx] = primitive_id;
                local += 1;
            }
        }
        current_write_offset += local;
    }

    {
        bool cooperative = active && tile_count > int(gsx_metal_render_sequential_tile_threshold);
        ulong active_mask = gsx_metal_simd_active_threads_mask();
        ulong cooperative_mask = gsx_metal_simd_ballot(cooperative) & active_mask;
        uint active_lane_count = gsx_metal_simd_sum(1u);
        uint lane_rank = gsx_metal_simd_prefix_exclusive_sum(1u);

        if(cooperative_mask != 0ul) {
            ulong remaining_cooperative_mask = cooperative_mask;
            while(remaining_cooperative_mask != 0ul) {
                uint source_lane = gsx_metal_first_set_lane_u64(remaining_cooperative_mask);
                remaining_cooperative_mask &= (remaining_cooperative_mask - 1ul);

                int primitive_id_coop = gsx_metal_simd_shuffle(primitive_id, (ushort)source_lane);
                int write_offset_coop = gsx_metal_simd_shuffle(current_write_offset, (ushort)source_lane);
                int x0_coop = gsx_metal_simd_shuffle(x0, (ushort)source_lane);
                int x1_coop = gsx_metal_simd_shuffle(x1, (ushort)source_lane);
                int y0_coop = gsx_metal_simd_shuffle(y0, (ushort)source_lane);
                int y1_coop = gsx_metal_simd_shuffle(y1, (ushort)source_lane);
                int screen_bounds_width_coop = x1_coop - x0_coop;
                int tile_count_coop = screen_bounds_width_coop * (y1_coop - y0_coop);
                float2 mean_shifted_coop = float2(
                    gsx_metal_simd_shuffle(mean_shifted.x, (ushort)source_lane),
                    gsx_metal_simd_shuffle(mean_shifted.y, (ushort)source_lane));
                float3 conic_coop = float3(
                    gsx_metal_simd_shuffle(conic.x, (ushort)source_lane),
                    gsx_metal_simd_shuffle(conic.y, (ushort)source_lane),
                    gsx_metal_simd_shuffle(conic.z, (ushort)source_lane));
                float power_threshold_coop = gsx_metal_simd_shuffle(power_threshold, (ushort)source_lane);

                for(int local_instance_idx_base = int(gsx_metal_render_sequential_tile_threshold);
                    local_instance_idx_base < tile_count_coop;
                    local_instance_idx_base += int(active_lane_count)) {
                    int local_instance_idx = local_instance_idx_base + int(lane_rank);
                    bool valid = local_instance_idx < tile_count_coop;
                    int y = y0_coop;
                    int x = x0_coop;
                    if (valid) {
                        y += (local_instance_idx / screen_bounds_width_coop);
                        x += (local_instance_idx % screen_bounds_width_coop);
                    }
                    bool write_instance = valid && gsx_metal_render_will_primitive_contribute(
                        mean_shifted_coop,
                        conic_coop,
                        uint(x),
                        uint(y),
                        power_threshold_coop);
                    uint write_prefix = gsx_metal_simd_prefix_exclusive_sum(write_instance ? 1u : 0u);
                    uint write_count = gsx_metal_simd_sum(write_instance ? 1u : 0u);

                    if(write_instance) {
                        int write_idx = write_offset_coop + int(write_prefix);
                        instance_keys[write_idx] = y * int(params.grid_width) + x;
                        instance_primitive_ids[write_idx] = primitive_id_coop;
                    }
                    write_offset_coop += int(write_count);
                }
            }
        }
    }
}

kernel void gsx_metal_render_extract_instance_ranges_kernel(
    device const int * __restrict__ instance_keys [[buffer(0)]],
    device int2 *      __restrict__ tile_ranges [[buffer(1)]],
    constant uint &instance_count [[buffer(2)]],
    constant uint &tile_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= instance_count) {
        return;
    }

    device int *tile_ranges_scalar = (device int *)tile_ranges;
    uint tile_idx = uint(instance_keys[gid]);
    if(gid == 0u) {
        tile_ranges_scalar[tile_idx * 2u] = 0;
    } else {
        uint previous_tile_idx = uint(instance_keys[gid - 1u]);
        if(tile_idx != previous_tile_idx) {
            int gap_value = int(gid);

            tile_ranges_scalar[previous_tile_idx * 2u + 1u] = gap_value;
            for(uint fill_tile = previous_tile_idx + 1u; fill_tile < tile_idx; ++fill_tile) {
                tile_ranges[fill_tile] = int2(gap_value, gap_value);
            }
            tile_ranges_scalar[tile_idx * 2u] = int(gid);
        }
    }
    if(gid + 1u == instance_count) {
        int end_value = int(instance_count);

        tile_ranges_scalar[tile_idx * 2u + 1u] = end_value;
        for(uint fill_tile = tile_idx + 1u; fill_tile < tile_count; ++fill_tile) {
            tile_ranges[fill_tile] = int2(end_value, end_value);
        }
    }
}

kernel void gsx_metal_render_extract_bucket_counts_kernel(
    device const int2 * __restrict__ tile_ranges [[buffer(0)]],
    device int  *       __restrict__ tile_bucket_counts [[buffer(1)]],
    constant uint &tile_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= tile_count) {
        return;
    }

    int2 tile_range = tile_ranges[gid];
    int start = tile_range.x;
    int end = tile_range.y;
    uint instance_count = end > start ? uint(end - start) : 0u;
    tile_bucket_counts[gid] = int((instance_count + 31u) / 32u);
}

kernel void gsx_metal_render_finalize_bucket_offsets_kernel(
    device const int * __restrict__ tile_bucket_counts [[buffer(0)]],
    device int *       __restrict__ tile_bucket_offsets [[buffer(1)]],
    constant uint &    __restrict__ tile_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= tile_count) {
        return;
    }
    tile_bucket_offsets[gid] += tile_bucket_counts[gid];
}

kernel void gsx_metal_render_blend_kernel(
    device const int2 *   __restrict__ tile_ranges [[buffer(0)]],
    device const int *    __restrict__ tile_bucket_offsets [[buffer(1)]],
    device const int *    __restrict__ instance_primitive_ids [[buffer(2)]],
    device const float2 * __restrict__ mean2d [[buffer(3)]],
    device const float4 * __restrict__ conic_opacity [[buffer(4)]],
    device const float *  __restrict__ color [[buffer(5)]],
    device float *        __restrict__ image_chw [[buffer(6)]],
    device float *        __restrict__ alpha_hw [[buffer(7)]],
    device int *          __restrict__ tile_max_n_contributions [[buffer(8)]],
    device int *          __restrict__ tile_n_contributions [[buffer(9)]],
    device int *          __restrict__ bucket_tile_index [[buffer(10)]],
    device float4 *       __restrict__ bucket_color_transmittance [[buffer(11)]],
    constant gsx_metal_render_blend_params &params [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tile_coord [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tile_id = tile_coord.y * params.grid_width + tile_coord.x;
    if(tile_id >= params.tile_count) {
        return;
    }

    bool inside = gid.x < params.width && gid.y < params.height;
    int2 tile_range = tile_ranges[tile_id];
    int start = tile_range.x;
    int end = tile_range.y;
    if(start < 0 || end <= start) {
        if(tid == 0u) {
            tile_max_n_contributions[tile_id] = 0;
        }
        if(inside) {
            uint pixel_index_empty = gid.y * params.width + gid.x;
            image_chw[pixel_index_empty] = 0.0f;
            image_chw[params.channel_stride + pixel_index_empty] = 0.0f;
            image_chw[2u * params.channel_stride + pixel_index_empty] = 0.0f;
            alpha_hw[pixel_index_empty] = 0.0f;
            tile_n_contributions[pixel_index_empty] = 0;
        }
        return;
    }

    threadgroup float2 collected_mean2d[gsx_metal_render_tile_size];
    threadgroup float4 collected_conic_opacity[gsx_metal_render_tile_size];
    threadgroup float3 collected_color[gsx_metal_render_tile_size];
    threadgroup uint done_count_per_simd[gsx_metal_render_tile_size / gsx_metal_render_simd_width];

    float2 pixel = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);
    float transmittance = 1.0f;
    float3 accum = float3(0.0f);    // Accumulated color
    bool done = !inside;
    int n_possible_contributions = 0;
    int n_contributions = 0;
    uint simdgroup_id = tid / gsx_metal_render_simd_width;
    uint simdgroup_count = gsx_metal_render_tile_size / gsx_metal_render_simd_width;
    int tile_bucket_base = tile_id == 0u ? 0 : tile_bucket_offsets[tile_id - 1u];
    uint lane_off = tid;

    for(int batch_start = start; batch_start < end; batch_start += int(gsx_metal_render_tile_size)) {
        // Seems that Metal does not support __syncthreads_count, so we manually do this
        bool done_in_simd = gsx_metal_simd_all(done);
        if(simd_lane_id == 0u) {
            done_count_per_simd[simdgroup_id] = done_in_simd ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint done_total = 0u;
        for(uint simd_idx = 0u; simd_idx < simdgroup_count; ++simd_idx) {
            done_total += done_count_per_simd[simd_idx];
        }
        if(done_total == simdgroup_count) {
            break;
        }
        // <<< __syncthread_count end

        int fetch_idx = batch_start + int(tid);
        if(fetch_idx < end) {
            int primitive_id = instance_primitive_ids[fetch_idx];
            uint p3 = uint(primitive_id) * 3u;
            collected_mean2d[tid] = mean2d[uint(primitive_id)];
            collected_conic_opacity[tid] = conic_opacity[uint(primitive_id)];
            collected_color[tid] = max(float3(color[p3], color[p3 + 1u], color[p3 + 2u]), float3(0.0f));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int batch_size = min(int(gsx_metal_render_tile_size), end - batch_start);
        int j;
        for(j = 0; !done && j < batch_size; ++j) {
            int primitive_local_idx = (batch_start - start) + j;
            if((primitive_local_idx & 31) == 0) {
                int bucket_idx = tile_bucket_base + (primitive_local_idx >> 5);
                bucket_tile_index[(uint)bucket_idx] = int(tile_id);
                bucket_color_transmittance[(uint)bucket_idx * gsx_metal_render_tile_size + lane_off] = float4(accum, transmittance);
            }
            n_possible_contributions += 1;
            float2 mu = collected_mean2d[j];
            float4 conic_op = collected_conic_opacity[j];
            float2 d = mu - pixel;
            float sigma_over_2 = 0.5f * (conic_op.x * d.x * d.x + conic_op.z * d.y * d.y) + conic_op.y * d.x * d.y;
            if(sigma_over_2 < 0.0f) {
                continue;
            }

            float alpha = min(conic_op.w * exp(-sigma_over_2), gsx_metal_render_max_alpha);
            if(alpha < gsx_metal_render_min_alpha) {
                continue;
            }

            accum += transmittance * alpha * collected_color[j];
            transmittance *= (1.0f - alpha);
            n_contributions = n_possible_contributions;
            done = done || (transmittance < gsx_metal_render_min_transmittance);
        }

        // Cleanup: bucket remaining items in the batch if we exited early
        // Find the next j where primitive_local_idx is aligned to 32
        int base = batch_start - start;
        int aligned_j = j;
        int remainder = (base + aligned_j) & 31;
        if(remainder != 0) {
            aligned_j += 32 - remainder;
        }
        for(; aligned_j < batch_size; aligned_j += 32) {
            int primitive_local_idx = base + aligned_j;
            int bucket_idx = tile_bucket_base + (primitive_local_idx >> 5);
            bucket_color_transmittance[(uint)bucket_idx * gsx_metal_render_tile_size + lane_off] = float4(accum, transmittance);
        }

        // threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(inside) {
        uint pixel_index = gid.y * params.width + gid.x;
        image_chw[pixel_index] = accum.x;
        image_chw[params.channel_stride + pixel_index] = accum.y;
        image_chw[2u * params.channel_stride + pixel_index] = accum.z;
        alpha_hw[pixel_index] = 1.0f - transmittance;
        tile_n_contributions[pixel_index] = n_contributions;
    }

    uint simd_max_contributions = gsx_metal_simd_max(inside ? uint(n_contributions) : 0u);
    if(simd_lane_id == 0u) {
        // Reuse the done_count_per_simd to store the max contributions of each simd group
        done_count_per_simd[simdgroup_id] = simd_max_contributions;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(tid == 0u) {
        uint tile_max_contributions = 0u;

        for(uint i = 0u; i < simdgroup_count; ++i) {
            tile_max_contributions = max(tile_max_contributions, done_count_per_simd[i]);
        }
        tile_max_n_contributions[tile_id] = int(tile_max_contributions);
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
