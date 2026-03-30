#include <metal_stdlib>
using namespace metal;

constant uint GSX_METAL_SSIM_KERNEL_RADIUS = 5;
constant float GSX_METAL_SSIM_C1 = 0.0001f;
constant float GSX_METAL_SSIM_C2 = 0.0009f;
constant uint GSX_METAL_SSIM_FUSED_BLOCK_X = 16;
constant uint GSX_METAL_SSIM_FUSED_BLOCK_Y = 16;
constant uint GSX_METAL_SSIM_FUSED_HALO = GSX_METAL_SSIM_KERNEL_RADIUS;
constant uint GSX_METAL_SSIM_FUSED_SHARED_X = 28;
constant uint GSX_METAL_SSIM_FUSED_SHARED_Y = 26;
constant uint GSX_METAL_SSIM_FUSED_CONV_X = 16;
constant uint GSX_METAL_SSIM_FUSED_CONV_Y = 26;
constant float gsx_metal_ssim_gauss_1d[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};

struct gsx_metal_loss_pointwise_params {
    uint element_count;
    float scale;
};

struct gsx_metal_loss_ssim_params {
    uint outer_count;
    uint channels;
    uint height;
    uint width;
    uint element_count;
    uint has_scratch;
    float scale;
};

static inline uint gsx_metal_loss_ssim_index_chw(
    uint outer,
    uint channel,
    uint y,
    uint x,
    constant gsx_metal_loss_ssim_params &params)
{
    return (((outer * params.channels + channel) * params.height + y) * params.width + x);
}

static inline uint gsx_metal_loss_ssim_index_hwc(
    uint outer,
    uint channel,
    uint y,
    uint x,
    constant gsx_metal_loss_ssim_params &params)
{
    return ((((outer * params.height + y) * params.width + x) * params.channels) + channel);
}

static inline uint gsx_metal_loss_ssim_index(
    bool is_hwc,
    uint outer,
    uint channel,
    uint y,
    uint x,
    constant gsx_metal_loss_ssim_params &params)
{
    if(is_hwc) {
        return gsx_metal_loss_ssim_index_hwc(outer, channel, y, x, params);
    }
    return gsx_metal_loss_ssim_index_chw(outer, channel, y, x, params);
}

static inline float gsx_metal_loss_ssim_sample_or_zero(
    device const float *values,
    bool is_hwc,
    uint outer,
    uint channel,
    int y,
    int x,
    constant gsx_metal_loss_ssim_params &params)
{
    if(y < 0 || x < 0 || y >= (int)params.height || x >= (int)params.width) {
        return 0.0f;
    }
    return values[gsx_metal_loss_ssim_index(is_hwc, outer, channel, (uint)y, (uint)x, params)];
}

kernel void gsx_metal_loss_mse_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *loss_map [[buffer(2)]],
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }

    float diff = prediction[gid] - target[gid];
    loss_map[gid] += params.scale * diff * diff;
}

kernel void gsx_metal_loss_l1_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *loss_map [[buffer(2)]],
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }

    float diff = prediction[gid] - target[gid];
    loss_map[gid] += params.scale * fabs(diff);
}

kernel void gsx_metal_loss_mse_backward_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *grad_prediction [[buffer(2)]],
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }

    float diff = prediction[gid] - target[gid];
    grad_prediction[gid] += 2.0f * diff * params.scale;
}

kernel void gsx_metal_loss_l1_backward_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *grad_prediction [[buffer(2)]],
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float sign = 0.0f;
    if(gid >= params.element_count) {
        return;
    }

    if(prediction[gid] > target[gid]) {
        sign = 1.0f;
    } else if(prediction[gid] < target[gid]) {
        sign = -1.0f;
    }
    grad_prediction[gid] += sign * params.scale;
}

kernel void gsx_metal_loss_ssim_chw_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *loss_map [[buffer(2)]],
    device float *scratch_a [[buffer(3)]],
    device float *scratch_b [[buffer(4)]],
    constant gsx_metal_loss_ssim_params &params [[buffer(5)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]])
{
    const uint pix_y = group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y + local_id.y;
    const uint pix_x = group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    const uint outer = group_id.z;
    const uint local_index = local_id.y * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    threadgroup float tile[GSX_METAL_SSIM_FUSED_SHARED_Y][GSX_METAL_SSIM_FUSED_SHARED_X][2];
    threadgroup float xconv[GSX_METAL_SSIM_FUSED_CONV_Y][GSX_METAL_SSIM_FUSED_CONV_X][5];

    if(outer >= params.outer_count) {
        return;
    }

    for(uint channel = 0; channel < params.channels; ++channel) {
        const uint tile_size = GSX_METAL_SSIM_FUSED_SHARED_Y * GSX_METAL_SSIM_FUSED_SHARED_X;
        const uint threads = GSX_METAL_SSIM_FUSED_BLOCK_X * GSX_METAL_SSIM_FUSED_BLOCK_Y;
        const uint steps = (tile_size + threads - 1) / threads;
        const int tile_start_y = (int)(group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y);
        const int tile_start_x = (int)(group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X);

        for(uint step = 0; step < steps; ++step) {
            uint load_index = step * threads + local_index;
            if(load_index < tile_size) {
                uint local_y = load_index / GSX_METAL_SSIM_FUSED_SHARED_X;
                uint local_x = load_index % GSX_METAL_SSIM_FUSED_SHARED_X;
                int gy = tile_start_y + (int)local_y - (int)GSX_METAL_SSIM_FUSED_HALO;
                int gx = tile_start_x + (int)local_x - (int)GSX_METAL_SSIM_FUSED_HALO;

                tile[local_y][local_x][0] =
                    gsx_metal_loss_ssim_sample_or_zero(prediction, false, outer, channel, gy, gx, params);
                tile[local_y][local_x][1] =
                    gsx_metal_loss_ssim_sample_or_zero(target, false, outer, channel, gy, gx, params);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint ly = local_id.y;
            uint lx = local_id.x + GSX_METAL_SSIM_FUSED_HALO;
            float sum_x = 0.0f;
            float sum_x2 = 0.0f;
            float sum_y = 0.0f;
            float sum_y2 = 0.0f;
            float sum_xy = 0.0f;

            for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                float x_left = tile[ly][lx - d][0];
                float y_left = tile[ly][lx - d][1];
                float x_right = tile[ly][lx + d][0];
                float y_right = tile[ly][lx + d][1];

                sum_x += (x_left + x_right) * w;
                sum_x2 += (x_left * x_left + x_right * x_right) * w;
                sum_y += (y_left + y_right) * w;
                sum_y2 += (y_left * y_left + y_right * y_right) * w;
                sum_xy += (x_left * y_left + x_right * y_right) * w;
            }
            {
                float center_x = tile[ly][lx][0];
                float center_y = tile[ly][lx][1];
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                sum_x += center_x * w;
                sum_x2 += center_x * center_x * w;
                sum_y += center_y * w;
                sum_y2 += center_y * center_y * w;
                sum_xy += center_x * center_y * w;
            }

            xconv[ly][local_id.x][0] = sum_x;
            xconv[ly][local_id.x][1] = sum_x2;
            xconv[ly][local_id.x][2] = sum_y;
            xconv[ly][local_id.x][3] = sum_y2;
            xconv[ly][local_id.x][4] = sum_xy;

            ly += GSX_METAL_SSIM_FUSED_BLOCK_Y;
            if(ly < GSX_METAL_SSIM_FUSED_CONV_Y) {
                sum_x = 0.0f;
                sum_x2 = 0.0f;
                sum_y = 0.0f;
                sum_y2 = 0.0f;
                sum_xy = 0.0f;

                for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                    float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                    float x_left = tile[ly][lx - d][0];
                    float y_left = tile[ly][lx - d][1];
                    float x_right = tile[ly][lx + d][0];
                    float y_right = tile[ly][lx + d][1];

                    sum_x += (x_left + x_right) * w;
                    sum_x2 += (x_left * x_left + x_right * x_right) * w;
                    sum_y += (y_left + y_right) * w;
                    sum_y2 += (y_left * y_left + y_right * y_right) * w;
                    sum_xy += (x_left * y_left + x_right * y_right) * w;
                }
                {
                    float center_x = tile[ly][lx][0];
                    float center_y = tile[ly][lx][1];
                    float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                    sum_x += center_x * w;
                    sum_x2 += center_x * center_x * w;
                    sum_y += center_y * w;
                    sum_y2 += center_y * center_y * w;
                    sum_xy += center_x * center_y * w;
                }

                xconv[ly][local_id.x][0] = sum_x;
                xconv[ly][local_id.x][1] = sum_x2;
                xconv[ly][local_id.x][2] = sum_y;
                xconv[ly][local_id.x][3] = sum_y2;
                xconv[ly][local_id.x][4] = sum_xy;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(pix_x < params.width && pix_y < params.height) {
            uint ly = local_id.y + GSX_METAL_SSIM_FUSED_HALO;
            uint lx = local_id.x;
            float out0 = 0.0f;
            float out1 = 0.0f;
            float out2 = 0.0f;
            float out3 = 0.0f;
            float out4 = 0.0f;

            for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                out0 += (xconv[ly - d][lx][0] + xconv[ly + d][lx][0]) * w;
                out1 += (xconv[ly - d][lx][1] + xconv[ly + d][lx][1]) * w;
                out2 += (xconv[ly - d][lx][2] + xconv[ly + d][lx][2]) * w;
                out3 += (xconv[ly - d][lx][3] + xconv[ly + d][lx][3]) * w;
                out4 += (xconv[ly - d][lx][4] + xconv[ly + d][lx][4]) * w;
            }
            {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                out0 += xconv[ly][lx][0] * w;
                out1 += xconv[ly][lx][1] * w;
                out2 += xconv[ly][lx][2] * w;
                out3 += xconv[ly][lx][3] * w;
                out4 += xconv[ly][lx][4] * w;
            }

            {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12 = out4 - mu1 * mu2;
                float a = mu1_sq + mu2_sq + GSX_METAL_SSIM_C1;
                float b = sigma1_sq + sigma2_sq + GSX_METAL_SSIM_C2;
                float c_term = 2.0f * mu1 * mu2 + GSX_METAL_SSIM_C1;
                float d_term = 2.0f * sigma12 + GSX_METAL_SSIM_C2;
                float denominator = a * b;
                float ssim = denominator == 0.0f ? 1.0f : (c_term * d_term) / denominator;
                uint idx = gsx_metal_loss_ssim_index(false, outer, channel, pix_y, pix_x, params);
                bool is_boundary = (pix_y < GSX_METAL_SSIM_KERNEL_RADIUS || 
                    pix_y >= params.height - GSX_METAL_SSIM_KERNEL_RADIUS ||
                    pix_x < GSX_METAL_SSIM_KERNEL_RADIUS || 
                    pix_x >= params.width - GSX_METAL_SSIM_KERNEL_RADIUS);

                float boundary_mask = is_boundary ? 0.0f : 1.0f;
                loss_map[idx] = boundary_mask * (loss_map[idx] + params.scale * (1.0f - ssim));

                if(params.has_scratch != 0u && scratch_a != nullptr && scratch_b != nullptr) {
                    float dm_dmu1 = 0.0f;
                    float dm_dsigma1_sq = 0.0f;
                    float dm_dsigma12 = 0.0f;

                    if(!is_boundary && denominator != 0.0f) {
                        dm_dmu1 = ((mu2 * 2.0f * d_term) / denominator - (mu2 * 2.0f * c_term) / denominator
                            - (mu1 * 2.0f * c_term * d_term) / (a * denominator)
                            + (mu1 * 2.0f * c_term * d_term) / (denominator * b));
                        dm_dsigma1_sq = -(c_term * d_term) / (denominator * b);
                        dm_dsigma12 = (2.0f * c_term) / denominator;
                    }

                    scratch_a[idx] = dm_dmu1;
                    scratch_a[idx + params.element_count] = dm_dsigma1_sq;
                    scratch_b[idx] = dm_dsigma12;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gsx_metal_loss_ssim_hwc_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *loss_map [[buffer(2)]],
    device float *scratch_a [[buffer(3)]],
    device float *scratch_b [[buffer(4)]],
    constant gsx_metal_loss_ssim_params &params [[buffer(5)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]])
{
    const uint pix_y = group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y + local_id.y;
    const uint pix_x = group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    const uint outer = group_id.z;
    const uint local_index = local_id.y * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    threadgroup float tile[GSX_METAL_SSIM_FUSED_SHARED_Y][GSX_METAL_SSIM_FUSED_SHARED_X][2];
    threadgroup float xconv[GSX_METAL_SSIM_FUSED_CONV_Y][GSX_METAL_SSIM_FUSED_CONV_X][5];

    if(outer >= params.outer_count) {
        return;
    }

    for(uint channel = 0; channel < params.channels; ++channel) {
        const uint tile_size = GSX_METAL_SSIM_FUSED_SHARED_Y * GSX_METAL_SSIM_FUSED_SHARED_X;
        const uint threads = GSX_METAL_SSIM_FUSED_BLOCK_X * GSX_METAL_SSIM_FUSED_BLOCK_Y;
        const uint steps = (tile_size + threads - 1) / threads;
        const int tile_start_y = (int)(group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y);
        const int tile_start_x = (int)(group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X);

        for(uint step = 0; step < steps; ++step) {
            uint load_index = step * threads + local_index;
            if(load_index < tile_size) {
                uint local_y = load_index / GSX_METAL_SSIM_FUSED_SHARED_X;
                uint local_x = load_index % GSX_METAL_SSIM_FUSED_SHARED_X;
                int gy = tile_start_y + (int)local_y - (int)GSX_METAL_SSIM_FUSED_HALO;
                int gx = tile_start_x + (int)local_x - (int)GSX_METAL_SSIM_FUSED_HALO;

                tile[local_y][local_x][0] =
                    gsx_metal_loss_ssim_sample_or_zero(prediction, true, outer, channel, gy, gx, params);
                tile[local_y][local_x][1] =
                    gsx_metal_loss_ssim_sample_or_zero(target, true, outer, channel, gy, gx, params);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint ly = local_id.y;
            uint lx = local_id.x + GSX_METAL_SSIM_FUSED_HALO;
            float sum_x = 0.0f;
            float sum_x2 = 0.0f;
            float sum_y = 0.0f;
            float sum_y2 = 0.0f;
            float sum_xy = 0.0f;

            for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                float x_left = tile[ly][lx - d][0];
                float y_left = tile[ly][lx - d][1];
                float x_right = tile[ly][lx + d][0];
                float y_right = tile[ly][lx + d][1];

                sum_x += (x_left + x_right) * w;
                sum_x2 += (x_left * x_left + x_right * x_right) * w;
                sum_y += (y_left + y_right) * w;
                sum_y2 += (y_left * y_left + y_right * y_right) * w;
                sum_xy += (x_left * y_left + x_right * y_right) * w;
            }
            {
                float center_x = tile[ly][lx][0];
                float center_y = tile[ly][lx][1];
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                sum_x += center_x * w;
                sum_x2 += center_x * center_x * w;
                sum_y += center_y * w;
                sum_y2 += center_y * center_y * w;
                sum_xy += center_x * center_y * w;
            }

            xconv[ly][local_id.x][0] = sum_x;
            xconv[ly][local_id.x][1] = sum_x2;
            xconv[ly][local_id.x][2] = sum_y;
            xconv[ly][local_id.x][3] = sum_y2;
            xconv[ly][local_id.x][4] = sum_xy;

            ly += GSX_METAL_SSIM_FUSED_BLOCK_Y;
            if(ly < GSX_METAL_SSIM_FUSED_CONV_Y) {
                sum_x = 0.0f;
                sum_x2 = 0.0f;
                sum_y = 0.0f;
                sum_y2 = 0.0f;
                sum_xy = 0.0f;

                for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                    float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                    float x_left = tile[ly][lx - d][0];
                    float y_left = tile[ly][lx - d][1];
                    float x_right = tile[ly][lx + d][0];
                    float y_right = tile[ly][lx + d][1];

                    sum_x += (x_left + x_right) * w;
                    sum_x2 += (x_left * x_left + x_right * x_right) * w;
                    sum_y += (y_left + y_right) * w;
                    sum_y2 += (y_left * y_left + y_right * y_right) * w;
                    sum_xy += (x_left * y_left + x_right * y_right) * w;
                }
                {
                    float center_x = tile[ly][lx][0];
                    float center_y = tile[ly][lx][1];
                    float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                    sum_x += center_x * w;
                    sum_x2 += center_x * center_x * w;
                    sum_y += center_y * w;
                    sum_y2 += center_y * center_y * w;
                    sum_xy += center_x * center_y * w;
                }

                xconv[ly][local_id.x][0] = sum_x;
                xconv[ly][local_id.x][1] = sum_x2;
                xconv[ly][local_id.x][2] = sum_y;
                xconv[ly][local_id.x][3] = sum_y2;
                xconv[ly][local_id.x][4] = sum_xy;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(pix_x < params.width && pix_y < params.height) {
            uint ly = local_id.y + GSX_METAL_SSIM_FUSED_HALO;
            uint lx = local_id.x;
            float out0 = 0.0f;
            float out1 = 0.0f;
            float out2 = 0.0f;
            float out3 = 0.0f;
            float out4 = 0.0f;

            for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                out0 += (xconv[ly - d][lx][0] + xconv[ly + d][lx][0]) * w;
                out1 += (xconv[ly - d][lx][1] + xconv[ly + d][lx][1]) * w;
                out2 += (xconv[ly - d][lx][2] + xconv[ly + d][lx][2]) * w;
                out3 += (xconv[ly - d][lx][3] + xconv[ly + d][lx][3]) * w;
                out4 += (xconv[ly - d][lx][4] + xconv[ly + d][lx][4]) * w;
            }
            {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                out0 += xconv[ly][lx][0] * w;
                out1 += xconv[ly][lx][1] * w;
                out2 += xconv[ly][lx][2] * w;
                out3 += xconv[ly][lx][3] * w;
                out4 += xconv[ly][lx][4] * w;
            }

            {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12 = out4 - mu1 * mu2;
                float a = mu1_sq + mu2_sq + GSX_METAL_SSIM_C1;
                float b = sigma1_sq + sigma2_sq + GSX_METAL_SSIM_C2;
                float c_term = 2.0f * mu1 * mu2 + GSX_METAL_SSIM_C1;
                float d_term = 2.0f * sigma12 + GSX_METAL_SSIM_C2;
                float denominator = a * b;
                float ssim = denominator == 0.0f ? 1.0f : (c_term * d_term) / denominator;
                uint idx = gsx_metal_loss_ssim_index(true, outer, channel, pix_y, pix_x, params);
                bool is_boundary = (pix_y < GSX_METAL_SSIM_KERNEL_RADIUS || 
                    pix_y >= params.height - GSX_METAL_SSIM_KERNEL_RADIUS ||
                    pix_x < GSX_METAL_SSIM_KERNEL_RADIUS || 
                    pix_x >= params.width - GSX_METAL_SSIM_KERNEL_RADIUS);

                float boundary_mask = is_boundary ? 0.0f : 1.0f;
                loss_map[idx] = boundary_mask * (loss_map[idx] + params.scale * (1.0f - ssim));

                if(params.has_scratch != 0u && scratch_a != nullptr && scratch_b != nullptr) {
                    float dm_dmu1 = 0.0f;
                    float dm_dsigma1_sq = 0.0f;
                    float dm_dsigma12 = 0.0f;

                    if(!is_boundary && denominator != 0.0f) {
                        dm_dmu1 = ((mu2 * 2.0f * d_term) / denominator - (mu2 * 2.0f * c_term) / denominator
                            - (mu1 * 2.0f * c_term * d_term) / (a * denominator)
                            + (mu1 * 2.0f * c_term * d_term) / (denominator * b));
                        dm_dsigma1_sq = -(c_term * d_term) / (denominator * b);
                        dm_dsigma12 = (2.0f * c_term) / denominator;
                    }

                    scratch_a[idx] = dm_dmu1;
                    scratch_a[idx + params.element_count] = dm_dsigma1_sq;
                    scratch_b[idx] = dm_dsigma12;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gsx_metal_loss_ssim_backward_chw_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *grad_prediction [[buffer(2)]],
    device const float *scratch_a [[buffer(3)]],
    device const float *scratch_b [[buffer(4)]],
    constant gsx_metal_loss_ssim_params &params [[buffer(5)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]])
{
    const uint pix_y = group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y + local_id.y;
    const uint pix_x = group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    const uint outer = group_id.z;
    const float chain = -params.scale;
    const uint local_index = local_id.y * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    threadgroup float sdata[3][GSX_METAL_SSIM_FUSED_SHARED_Y][GSX_METAL_SSIM_FUSED_SHARED_X];
    threadgroup float scratch[GSX_METAL_SSIM_FUSED_CONV_Y][GSX_METAL_SSIM_FUSED_CONV_X][3];

    if(outer >= params.outer_count || params.has_scratch == 0u || scratch_a == nullptr || scratch_b == nullptr) {
        return;
    }

    for(uint channel = 0; channel < params.channels; ++channel) {
        const uint tile_size = GSX_METAL_SSIM_FUSED_SHARED_Y * GSX_METAL_SSIM_FUSED_SHARED_X;
        const uint threads = GSX_METAL_SSIM_FUSED_BLOCK_X * GSX_METAL_SSIM_FUSED_BLOCK_Y;
        const uint steps = (tile_size + threads - 1) / threads;
        const int tile_start_y = (int)(group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y);
        const int tile_start_x = (int)(group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X);
        float p1 = 0.0f;
        float p2 = 0.0f;

        if(pix_x < params.width && pix_y < params.height) {
            uint idx = gsx_metal_loss_ssim_index(false, outer, channel, pix_y, pix_x, params);
            p1 = prediction[idx];
            p2 = target[idx];
        }

        for(uint step = 0; step < steps; ++step) {
            uint load_index = step * threads + local_index;
            if(load_index < tile_size) {
                uint local_y = load_index / GSX_METAL_SSIM_FUSED_SHARED_X;
                uint local_x = load_index % GSX_METAL_SSIM_FUSED_SHARED_X;
                int gy = tile_start_y + (int)local_y - (int)GSX_METAL_SSIM_FUSED_HALO;
                int gx = tile_start_x + (int)local_x - (int)GSX_METAL_SSIM_FUSED_HALO;

                if(gy < 0 || gx < 0 || gy >= (int)params.height || gx >= (int)params.width) {
                    sdata[0][local_y][local_x] = 0.0f;
                    sdata[1][local_y][local_x] = 0.0f;
                    sdata[2][local_y][local_x] = 0.0f;
                } else {
                    uint idx = gsx_metal_loss_ssim_index(false, outer, channel, (uint)gy, (uint)gx, params);
                    sdata[0][local_y][local_x] = scratch_a[idx] * chain;
                    sdata[1][local_y][local_x] = scratch_a[idx + params.element_count] * chain;
                    sdata[2][local_y][local_x] = scratch_b[idx] * chain;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint lx = local_id.x + GSX_METAL_SSIM_FUSED_HALO;
            for(uint pass = 0; pass < 2; ++pass) {
                uint yy = local_id.y + pass * GSX_METAL_SSIM_FUSED_BLOCK_Y;
                if(yy < GSX_METAL_SSIM_FUSED_CONV_Y) {
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;

                    for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                        float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                        sum0 += (sdata[0][yy][lx - d] + sdata[0][yy][lx + d]) * w;
                        sum1 += (sdata[1][yy][lx - d] + sdata[1][yy][lx + d]) * w;
                        sum2 += (sdata[2][yy][lx - d] + sdata[2][yy][lx + d]) * w;
                    }
                    {
                        float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                        sum0 += sdata[0][yy][lx] * w;
                        sum1 += sdata[1][yy][lx] * w;
                        sum2 += sdata[2][yy][lx] * w;
                    }

                    scratch[yy][local_id.x][0] = sum0;
                    scratch[yy][local_id.x][1] = sum1;
                    scratch[yy][local_id.x][2] = sum2;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(pix_x < params.width && pix_y < params.height) {
            uint ly = local_id.y + GSX_METAL_SSIM_FUSED_HALO;
            uint lx = local_id.x;
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;

            for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                sum0 += (scratch[ly - d][lx][0] + scratch[ly + d][lx][0]) * w;
                sum1 += (scratch[ly - d][lx][1] + scratch[ly + d][lx][1]) * w;
                sum2 += (scratch[ly - d][lx][2] + scratch[ly + d][lx][2]) * w;
            }
            {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                sum0 += scratch[ly][lx][0] * w;
                sum1 += scratch[ly][lx][1] * w;
                sum2 += scratch[ly][lx][2] * w;
            }

            {
                uint idx = gsx_metal_loss_ssim_index(false, outer, channel, pix_y, pix_x, params);
                grad_prediction[idx] += sum0 + 2.0f * p1 * sum1 + p2 * sum2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gsx_metal_loss_ssim_backward_hwc_f32_kernel(
    device const float *prediction [[buffer(0)]],
    device const float *target [[buffer(1)]],
    device float *grad_prediction [[buffer(2)]],
    device const float *scratch_a [[buffer(3)]],
    device const float *scratch_b [[buffer(4)]],
    constant gsx_metal_loss_ssim_params &params [[buffer(5)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]])
{
    const uint pix_y = group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y + local_id.y;
    const uint pix_x = group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    const uint outer = group_id.z;
    const float chain = -params.scale;
    const uint local_index = local_id.y * GSX_METAL_SSIM_FUSED_BLOCK_X + local_id.x;
    threadgroup float sdata[3][GSX_METAL_SSIM_FUSED_SHARED_Y][GSX_METAL_SSIM_FUSED_SHARED_X];
    threadgroup float scratch[GSX_METAL_SSIM_FUSED_CONV_Y][GSX_METAL_SSIM_FUSED_CONV_X][3];

    if(outer >= params.outer_count || params.has_scratch == 0u || scratch_a == nullptr || scratch_b == nullptr) {
        return;
    }

    for(uint channel = 0; channel < params.channels; ++channel) {
        const uint tile_size = GSX_METAL_SSIM_FUSED_SHARED_Y * GSX_METAL_SSIM_FUSED_SHARED_X;
        const uint threads = GSX_METAL_SSIM_FUSED_BLOCK_X * GSX_METAL_SSIM_FUSED_BLOCK_Y;
        const uint steps = (tile_size + threads - 1) / threads;
        const int tile_start_y = (int)(group_id.y * GSX_METAL_SSIM_FUSED_BLOCK_Y);
        const int tile_start_x = (int)(group_id.x * GSX_METAL_SSIM_FUSED_BLOCK_X);
        float p1 = 0.0f;
        float p2 = 0.0f;

        if(pix_x < params.width && pix_y < params.height) {
            uint idx = gsx_metal_loss_ssim_index(true, outer, channel, pix_y, pix_x, params);
            p1 = prediction[idx];
            p2 = target[idx];
        }

        for(uint step = 0; step < steps; ++step) {
            uint load_index = step * threads + local_index;
            if(load_index < tile_size) {
                uint local_y = load_index / GSX_METAL_SSIM_FUSED_SHARED_X;
                uint local_x = load_index % GSX_METAL_SSIM_FUSED_SHARED_X;
                int gy = tile_start_y + (int)local_y - (int)GSX_METAL_SSIM_FUSED_HALO;
                int gx = tile_start_x + (int)local_x - (int)GSX_METAL_SSIM_FUSED_HALO;

                if(gy < 0 || gx < 0 || gy >= (int)params.height || gx >= (int)params.width) {
                    sdata[0][local_y][local_x] = 0.0f;
                    sdata[1][local_y][local_x] = 0.0f;
                    sdata[2][local_y][local_x] = 0.0f;
                } else {
                    uint idx = gsx_metal_loss_ssim_index(true, outer, channel, (uint)gy, (uint)gx, params);
                    sdata[0][local_y][local_x] = scratch_a[idx] * chain;
                    sdata[1][local_y][local_x] = scratch_a[idx + params.element_count] * chain;
                    sdata[2][local_y][local_x] = scratch_b[idx] * chain;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint lx = local_id.x + GSX_METAL_SSIM_FUSED_HALO;
            for(uint pass = 0; pass < 2; ++pass) {
                uint yy = local_id.y + pass * GSX_METAL_SSIM_FUSED_BLOCK_Y;
                if(yy < GSX_METAL_SSIM_FUSED_CONV_Y) {
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;

                    for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                        float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                        sum0 += (sdata[0][yy][lx - d] + sdata[0][yy][lx + d]) * w;
                        sum1 += (sdata[1][yy][lx - d] + sdata[1][yy][lx + d]) * w;
                        sum2 += (sdata[2][yy][lx - d] + sdata[2][yy][lx + d]) * w;
                    }
                    {
                        float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                        sum0 += sdata[0][yy][lx] * w;
                        sum1 += sdata[1][yy][lx] * w;
                        sum2 += sdata[2][yy][lx] * w;
                    }

                    scratch[yy][local_id.x][0] = sum0;
                    scratch[yy][local_id.x][1] = sum1;
                    scratch[yy][local_id.x][2] = sum2;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(pix_x < params.width && pix_y < params.height) {
            uint ly = local_id.y + GSX_METAL_SSIM_FUSED_HALO;
            uint lx = local_id.x;
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;

            for(uint d = 1; d <= GSX_METAL_SSIM_FUSED_HALO; ++d) {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO - d];
                sum0 += (scratch[ly - d][lx][0] + scratch[ly + d][lx][0]) * w;
                sum1 += (scratch[ly - d][lx][1] + scratch[ly + d][lx][1]) * w;
                sum2 += (scratch[ly - d][lx][2] + scratch[ly + d][lx][2]) * w;
            }
            {
                float w = gsx_metal_ssim_gauss_1d[GSX_METAL_SSIM_FUSED_HALO];
                sum0 += scratch[ly][lx][0] * w;
                sum1 += scratch[ly][lx][1] * w;
                sum2 += scratch[ly][lx][2] * w;
            }

            {
                uint idx = gsx_metal_loss_ssim_index(true, outer, channel, pix_y, pix_x, params);
                grad_prediction[idx] += sum0 + 2.0f * p1 * sum1 + p2 * sum2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
