#ifndef GSX_METAL_RENDER_COMMON_METAL
#define GSX_METAL_RENDER_COMMON_METAL

constant uint gsx_metal_render_tile_width = 16u;
constant uint gsx_metal_render_tile_height = 16u;
constant uint gsx_metal_render_tile_size = gsx_metal_render_tile_width * gsx_metal_render_tile_height;
constant uint gsx_metal_render_simd_width = 32u;
constant uint gsx_metal_render_sequential_tile_threshold = 8u;
constant float gsx_metal_render_min_alpha = 1.0f / 255.0f;
constant float gsx_metal_render_min_alpha_rcp = 255.0f;
constant float gsx_metal_render_max_alpha = 0.999f;
constant float gsx_metal_render_min_transmittance = 1.0e-4f;

struct gsx_metal_render_preprocess_params {
    uint gaussian_count;
    uint width;
    uint height;
    uint grid_width;
    uint grid_height;
    float fx;
    float fy;
    float cx;
    float cy;
    float near_plane;
    float far_plane;
    float pose_qx;
    float pose_qy;
    float pose_qz;
    float pose_qw;
    float pose_tx;
    float pose_ty;
    float pose_tz;
};

struct gsx_metal_render_create_instances_params {
    uint visible_count;
    uint grid_width;
    uint grid_height;
};

struct gsx_metal_render_blend_params {
    uint width;
    uint height;
    uint grid_width;
    uint grid_height;
    uint tile_count;
    uint channel_stride;
};

struct gsx_metal_render_preprocess_backward_params {
    uint gaussian_count;
    uint width;
    uint height;
    uint sh_degree;
    float fx;
    float fy;
    float cx;
    float cy;
    float near_plane;
    float far_plane;
    float pose_qx;
    float pose_qy;
    float pose_qz;
    float pose_qw;
    float pose_tx;
    float pose_ty;
    float pose_tz;
};

struct gsx_metal_render_blend_backward_params {
    uint gaussian_count;
    uint width;
    uint height;
    uint grid_width;
    uint grid_height;
    uint tile_count;
    uint total_bucket_count;
    uint channel_stride;
    float background_r;
    float background_g;
    float background_b;
};

struct gsx_metal_render_compose_params {
    uint width;
    uint height;
    uint channel_stride;
    float background_r;
    float background_g;
    float background_b;
};

static inline bool gsx_metal_render_will_primitive_contribute(float2 mean_shifted, float3 conic, uint tile_x, uint tile_y, float power_threshold)
{
    float2 rect_min = float2(float(tile_x * gsx_metal_render_tile_width), float(tile_y * gsx_metal_render_tile_height));
    float2 rect_max = float2(float((tile_x + 1u) * gsx_metal_render_tile_width - 1u), float((tile_y + 1u) * gsx_metal_render_tile_height - 1u));
    float x_min_diff = rect_min.x - mean_shifted.x;
    float x_left = x_min_diff > 0.0f ? 1.0f : 0.0f;
    float not_in_x_range = x_left + (mean_shifted.x > rect_max.x ? 1.0f : 0.0f);
    float y_min_diff = rect_min.y - mean_shifted.y;
    float y_above = y_min_diff > 0.0f ? 1.0f : 0.0f;
    float not_in_y_range = y_above + (mean_shifted.y > rect_max.y ? 1.0f : 0.0f);

    if(not_in_x_range + not_in_y_range == 0.0f) {
        return true;
    }

    float2 closest_corner = float2(x_left > 0.0f ? rect_min.x : rect_max.x, y_above > 0.0f ? rect_min.y : rect_max.y);
    float2 diff = mean_shifted - closest_corner;
    float2 d = float2(copysign(float(gsx_metal_render_tile_width - 1u), x_min_diff), copysign(float(gsx_metal_render_tile_height - 1u), y_min_diff));
    float tx = not_in_y_range * saturate((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x));
    float ty = not_in_x_range * saturate((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y));
    float2 max_point = closest_corner + float2(tx * d.x, ty * d.y);
    float2 delta = mean_shifted - max_point;
    float max_power = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
    return max_power <= power_threshold;
}

static inline bool gsx_metal_render_eval_contribution(
    float2 mean2d,
    float4 conic_opacity,
    float3 color_rgb,
    float2 pixel,
    thread float2 &delta,
    thread float3 &conic,
    thread float &opacity,
    thread float &alpha_raw,
    thread float &alpha,
    thread float3 &color_unclamped,
    thread float3 &color_clamped)
{
    conic = conic_opacity.xyz;
    opacity = conic_opacity.w;
    delta = mean2d - pixel;
    color_unclamped = color_rgb;
    color_clamped = max(color_rgb, float3(0.0f));

    float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
    if(sigma_over_2 < 0.0f) {
        alpha_raw = 0.0f;
        alpha = 0.0f;
        return false;
    }

    alpha_raw = opacity * exp(-sigma_over_2);
    alpha = min(alpha_raw, gsx_metal_render_max_alpha);
    return alpha >= gsx_metal_render_min_alpha;
}

#endif
