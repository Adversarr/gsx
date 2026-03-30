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
    uint sh_degree;
    uint has_visible_counter;
    uint has_max_screen_radius;
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
    uint has_grad_acc;
    uint has_absgrad_acc;
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
    float2 min_diff = rect_min - mean_shifted;
    bool2 below_min = min_diff > 0.0f;
    bool2 above_max = mean_shifted > rect_max;
    float2 outside = float2(float(below_min.x || above_max.x), float(below_min.y || above_max.y));

    if(all(outside == float2(0.0f))) {
        return true;
    }

    float2 closest_corner = float2(below_min.x ? rect_min.x : rect_max.x, below_min.y ? rect_min.y : rect_max.y);
    float2 diff = mean_shifted - closest_corner;
    float2 d = float2(
        copysign(float(gsx_metal_render_tile_width - 1u), min_diff.x),
        copysign(float(gsx_metal_render_tile_height - 1u), min_diff.y));
    float tx_numer = d.x * (conic.x * diff.x + conic.y * diff.y);
    float ty_numer = d.y * (conic.y * diff.x + conic.z * diff.y);
    float tx = outside.y * saturate(tx_numer * (1.0f / (d.x * conic.x * d.x)));
    float ty = outside.x * saturate(ty_numer * (1.0f / (d.y * conic.z * d.y)));
    float2 max_point = closest_corner + float2(tx * d.x, ty * d.y);
    float2 delta = mean_shifted - max_point;
    float max_power = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
    return max_power <= power_threshold;
}

#endif
