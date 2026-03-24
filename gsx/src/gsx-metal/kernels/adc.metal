#include <metal_stdlib>
using namespace metal;

struct gsx_metal_adc_classify_growth_params {
    uint gaussian_count;
    uint has_visible_counter;
    float duplicate_grad_threshold;
    float duplicate_scale_threshold;
    float scene_scale;
};

struct gsx_metal_adc_apply_split_params {
    uint split_count;
    ulong rng_state;
    ulong rng_inc;
};

struct gsx_metal_adc_keep_mask_params {
    uint gaussian_count;
    uint has_max_screen_radius;
    uint count_before_growth;
    uint prune_large;
    float scene_scale;
    float pruning_opacity_threshold;
    float max_world_scale;
    float max_screen_scale;
};

struct gsx_metal_adc_mcmc_noise_params {
    uint gaussian_count;
    float noise_strength;
    float learning_rate;
    ulong rng_state;
    ulong rng_inc;
};

struct gsx_metal_adc_mcmc_dead_mask_params {
    uint gaussian_count;
    float pruning_opacity_threshold;
};

struct gsx_metal_adc_mcmc_relocation_params {
    uint gaussian_count;
    float min_opacity;
    ulong rng_state;
    ulong rng_inc;
};

struct gsx_metal_pcg32 {
    ulong state;
    ulong inc;
};

inline uint gsx_metal_pcg32_next_uint(thread gsx_metal_pcg32 *rng)
{
    ulong oldstate = rng->state;
    uint xorshifted = (uint)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint rot = (uint)(oldstate >> 59u);

    rng->state = oldstate * 0x5851f42d4c957f2dUL + rng->inc;
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

inline float gsx_metal_pcg32_next_float(thread gsx_metal_pcg32 *rng)
{
    return as_type<float>((gsx_metal_pcg32_next_uint(rng) >> 9u) | 0x3f800000u) - 1.0f;
}

inline void gsx_metal_pcg32_advance(thread gsx_metal_pcg32 *rng, ulong delta)
{
    ulong cur_mult = 0x5851f42d4c957f2dUL;
    ulong cur_plus = rng->inc;
    ulong acc_mult = 1UL;
    ulong acc_plus = 0UL;

    while(delta > 0UL) {
        if((delta & 1UL) != 0UL) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1UL) * cur_plus;
        cur_mult *= cur_mult;
        delta >>= 1u;
    }

    rng->state = acc_mult * rng->state + acc_plus;
}

inline float gsx_metal_adc_sigmoid(float value)
{
    return 1.0f / (1.0f + exp(-value));
}

inline float gsx_metal_adc_logit(float value)
{
    return precise::log(value / (1.0f - value));
}

inline float gsx_metal_adc_clamp_probability(float value)
{
    if(value < 0.0f) {
        return 0.0f;
    }
    if(value > 1.0f - 1e-6f) {
        return 1.0f - 1e-6f;
    }
    return value;
}

inline float gsx_metal_adc_sample_normal(thread gsx_metal_pcg32 *rng)
{
    float u1 = gsx_metal_pcg32_next_float(rng);
    float u2 = gsx_metal_pcg32_next_float(rng);
    float radius = 0.0f;

    if(u1 < 1e-7f) {
        u1 = 1e-7f;
    }
    radius = precise::sqrt(-2.0f * precise::log(u1));
    return radius * precise::cos(6.2831853071795864769f * u2);
}

inline void gsx_metal_adc_normalize_quaternion(thread float &qx, thread float &qy, thread float &qz, thread float &qw)
{
    float q_norm = precise::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);

    if(q_norm <= 1e-8f) {
        return;
    }

    float inv_q = 1.0f / q_norm;
    qx *= inv_q;
    qy *= inv_q;
    qz *= inv_q;
    qw *= inv_q;
}

inline float gsx_metal_adc_mcmc_noise_gate(float opacity)
{
    float shifted = (1.0f - opacity) - 0.995f;

    return 1.0f / (1.0f + exp(-100.0f * shifted));
}

inline uint gsx_metal_adc_mcmc_clamp_ratio(uint ratio)
{
    if(ratio < 1u) {
        return 1u;
    }
    if(ratio > 51u) {
        return 51u;
    }
    return ratio;
}

inline float gsx_metal_adc_mcmc_binom(uint n, uint k)
{
    uint index = 0u;
    float value = 1.0f;

    if(k > n) {
        return 0.0f;
    }
    if(k == 0u || k == n) {
        return 1.0f;
    }
    if(k > n - k) {
        k = n - k;
    }
    for(index = 1u; index <= k; ++index) {
        value = value * (float)(n - (k - index)) / (float)index;
    }
    return value;
}

inline float gsx_metal_adc_mcmc_relocated_opacity(float opacity, uint ratio)
{
    float clamped = gsx_metal_adc_clamp_probability(opacity);
    float root = 1.0f / (float)gsx_metal_adc_mcmc_clamp_ratio(ratio);

    return 1.0f - pow(1.0f - clamped, root);
}

inline float gsx_metal_adc_mcmc_scale_coeff(float opacity, float new_opacity, uint ratio)
{
    float denom_sum = 0.0f;
    uint clamped_ratio = gsx_metal_adc_mcmc_clamp_ratio(ratio);
    uint i = 0u;

    for(i = 1u; i <= clamped_ratio; ++i) {
        uint k = 0u;

        for(k = 0u; k < i; ++k) {
            float sign = (k & 1u) == 0u ? 1.0f : -1.0f;
            float coeff = gsx_metal_adc_mcmc_binom(i - 1u, k);
            float power = pow(new_opacity, (float)(k + 1u));

            denom_sum += coeff * sign * power / precise::sqrt((float)(k + 1u));
        }
    }
    if(fabs(denom_sum) <= 1e-8f) {
        return 1.0f;
    }
    return opacity / denom_sum;
}

kernel void gsx_metal_adc_classify_growth_kernel(
    device const float *grad_acc [[buffer(0)]],
    device const float *visible_counter [[buffer(1)]],
    device const float *logscale [[buffer(2)]],
    device uchar *out_modes [[buffer(3)]],
    constant gsx_metal_adc_classify_growth_params &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    float counter = 1.0f;
    float grad = 0.0f;
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    float max_scale = 0.0f;

    if(gid >= params.gaussian_count) {
        return;
    }

    if(params.has_visible_counter != 0u && visible_counter != nullptr) {
        counter = visible_counter[gid];
    }
    if(counter <= 0.0f) {
        out_modes[gid] = 0u;
        return;
    }

    grad = grad_acc[gid] / (counter > 1.0f ? counter : 1.0f);
    if(grad <= params.duplicate_grad_threshold) {
        out_modes[gid] = 0u;
        return;
    }

    sx = exp(logscale[gid * 3u + 0u]);
    sy = exp(logscale[gid * 3u + 1u]);
    sz = exp(logscale[gid * 3u + 2u]);
    max_scale = max(max(sx, sy), sz);
    out_modes[gid] = max_scale > (params.duplicate_scale_threshold * params.scene_scale) ? 2u : 1u;
}

kernel void gsx_metal_adc_apply_split_kernel(
    device float *mean3d [[buffer(0)]],
    device float *logscale [[buffer(1)]],
    device float *opacity [[buffer(2)]],
    device const float *rotation [[buffer(3)]],
    device const int *split_sources [[buffer(4)]],
    device const int *split_targets [[buffer(5)]],
    constant gsx_metal_adc_apply_split_params &params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    gsx_metal_pcg32 rng = { params.rng_state, params.rng_inc };
    uint src = 0u;
    uint target = 0u;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 0.0f;
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    float rnd1x = 0.0f;
    float rnd1y = 0.0f;
    float rnd1z = 0.0f;
    float rnd2x = 0.0f;
    float rnd2y = 0.0f;
    float rnd2z = 0.0f;
    float source_opacity = 0.0f;
    float split_opacity = 0.0f;
    float m00 = 1.0f;
    float m01 = 0.0f;
    float m02 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;
    float m12 = 0.0f;
    float m20 = 0.0f;
    float m21 = 0.0f;
    float m22 = 1.0f;
    float t1x = 0.0f;
    float t1y = 0.0f;
    float t1z = 0.0f;
    float t2x = 0.0f;
    float t2y = 0.0f;
    float t2z = 0.0f;
    float off1x = 0.0f;
    float off1y = 0.0f;
    float off1z = 0.0f;
    float off2x = 0.0f;
    float off2y = 0.0f;
    float off2z = 0.0f;
    float new_scale_x = 0.0f;
    float new_scale_y = 0.0f;
    float new_scale_z = 0.0f;

    if(gid >= params.split_count) {
        return;
    }

    gsx_metal_pcg32_advance(&rng, (ulong)gid * 12UL);
    src = (uint)split_sources[gid];
    target = (uint)split_targets[gid];

    qx = rotation[src * 4u + 0u];
    qy = rotation[src * 4u + 1u];
    qz = rotation[src * 4u + 2u];
    qw = rotation[src * 4u + 3u];
    sx = exp(logscale[src * 3u + 0u]);
    sy = exp(logscale[src * 3u + 1u]);
    sz = exp(logscale[src * 3u + 2u]);
    rnd1x = gsx_metal_adc_sample_normal(&rng);
    rnd1y = gsx_metal_adc_sample_normal(&rng);
    rnd1z = gsx_metal_adc_sample_normal(&rng);
    rnd2x = gsx_metal_adc_sample_normal(&rng);
    rnd2y = gsx_metal_adc_sample_normal(&rng);
    rnd2z = gsx_metal_adc_sample_normal(&rng);
    source_opacity = gsx_metal_adc_clamp_probability(gsx_metal_adc_sigmoid(opacity[src]));
    split_opacity = 1.0f - precise::sqrt(1.0f - source_opacity);
    split_opacity = clamp(split_opacity, 1e-6f, 1.0f - 1e-6f);
    new_scale_x = sx / 1.6f;
    new_scale_y = sy / 1.6f;
    new_scale_z = sz / 1.6f;

    gsx_metal_adc_normalize_quaternion(qx, qy, qz, qw);
    m00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    m01 = 2.0f * (qx * qy - qw * qz);
    m02 = 2.0f * (qx * qz + qw * qy);
    m10 = 2.0f * (qx * qy + qw * qz);
    m11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    m12 = 2.0f * (qy * qz - qw * qx);
    m20 = 2.0f * (qx * qz - qw * qy);
    m21 = 2.0f * (qy * qz + qw * qx);
    m22 = 1.0f - 2.0f * (qx * qx + qy * qy);

    t1x = rnd1x * (sx + 1e-5f);
    t1y = rnd1y * (sy + 1e-5f);
    t1z = rnd1z * (sz + 1e-5f);
    t2x = rnd2x * (sx + 1e-5f);
    t2y = rnd2y * (sy + 1e-5f);
    t2z = rnd2z * (sz + 1e-5f);
    off1x = m00 * t1x + m01 * t1y + m02 * t1z;
    off1y = m10 * t1x + m11 * t1y + m12 * t1z;
    off1z = m20 * t1x + m21 * t1y + m22 * t1z;
    off2x = m00 * t2x + m01 * t2y + m02 * t2z;
    off2y = m10 * t2x + m11 * t2y + m12 * t2z;
    off2z = m20 * t2x + m21 * t2y + m22 * t2z;

    mean3d[target * 3u + 0u] = mean3d[src * 3u + 0u] + off1x;
    mean3d[target * 3u + 1u] = mean3d[src * 3u + 1u] + off1y;
    mean3d[target * 3u + 2u] = mean3d[src * 3u + 2u] + off1z;
    logscale[target * 3u + 0u] = precise::log(new_scale_x);
    logscale[target * 3u + 1u] = precise::log(new_scale_y);
    logscale[target * 3u + 2u] = precise::log(new_scale_z);
    opacity[target] = gsx_metal_adc_logit(split_opacity);

    mean3d[src * 3u + 0u] += off2x;
    mean3d[src * 3u + 1u] += off2y;
    mean3d[src * 3u + 2u] += off2z;
    logscale[src * 3u + 0u] = precise::log(new_scale_x);
    logscale[src * 3u + 1u] = precise::log(new_scale_y);
    logscale[src * 3u + 2u] = precise::log(new_scale_z);
    opacity[src] = gsx_metal_adc_logit(split_opacity);
}

kernel void gsx_metal_adc_keep_mask_kernel(
    device const float *opacity [[buffer(0)]],
    device const float *logscale [[buffer(1)]],
    device const float *rotation [[buffer(2)]],
    device const float *max_screen_radius [[buffer(3)]],
    device uchar *out_keep_mask [[buffer(4)]],
    constant gsx_metal_adc_keep_mask_params &params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    float opacity_value = 0.0f;
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    float max_scale = 0.0f;
    float q0 = 0.0f;
    float q1 = 0.0f;
    float q2 = 0.0f;
    float q3 = 0.0f;
    float rotation_norm = 0.0f;
    bool not_large_ws = true;
    bool not_large_ss = true;
    bool not_transparent = false;
    bool not_degenerate = false;

    if(gid >= params.gaussian_count) {
        return;
    }

    opacity_value = gsx_metal_adc_sigmoid(opacity[gid]);
    not_transparent = opacity_value > params.pruning_opacity_threshold;
    sx = exp(logscale[gid * 3u + 0u]);
    sy = exp(logscale[gid * 3u + 1u]);
    sz = exp(logscale[gid * 3u + 2u]);
    max_scale = max(max(sx, sy), sz);
    if(params.max_world_scale > 0.0f) {
        not_large_ws = max_scale < (params.max_world_scale * params.scene_scale);
    }
    if(params.max_screen_scale > 0.0f && params.has_max_screen_radius != 0u && gid < params.count_before_growth) {
        not_large_ss = max_screen_radius[gid] < params.max_screen_scale;
    }
    q0 = rotation[gid * 4u + 0u];
    q1 = rotation[gid * 4u + 1u];
    q2 = rotation[gid * 4u + 2u];
    q3 = rotation[gid * 4u + 3u];
    rotation_norm = fabs(q0) + fabs(q1) + fabs(q2) + fabs(q3);
    not_degenerate = rotation_norm > FLT_EPSILON;

    if(params.prune_large == 0u) {
        out_keep_mask[gid] = (not_transparent && not_degenerate) ? 1u : 0u;
        return;
    }
    out_keep_mask[gid] = (not_transparent && not_large_ws && not_large_ss && not_degenerate) ? 1u : 0u;
}

kernel void gsx_metal_adc_mcmc_noise_kernel(
    device float *mean3d [[buffer(0)]],
    device const float *logscale [[buffer(1)]],
    device const float *opacity [[buffer(2)]],
    device const float *rotation [[buffer(3)]],
    constant gsx_metal_adc_mcmc_noise_params &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    gsx_metal_pcg32 rng = { params.rng_state, params.rng_inc };
    float opacity_value = 0.0f;
    float gate = 0.0f;
    float scale_x = 0.0f;
    float scale_y = 0.0f;
    float scale_z = 0.0f;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 0.0f;
    float m00 = 1.0f;
    float m01 = 0.0f;
    float m02 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;
    float m12 = 0.0f;
    float m20 = 0.0f;
    float m21 = 0.0f;
    float m22 = 1.0f;
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c02 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;
    float c12 = 0.0f;
    float c20 = 0.0f;
    float c21 = 0.0f;
    float c22 = 0.0f;
    float noise_x = 0.0f;
    float noise_y = 0.0f;
    float noise_z = 0.0f;
    float scaled_noise_x = 0.0f;
    float scaled_noise_y = 0.0f;
    float scaled_noise_z = 0.0f;

    if(gid >= params.gaussian_count) {
        return;
    }

    gsx_metal_pcg32_advance(&rng, (ulong)gid * 6UL);
    opacity_value = gsx_metal_adc_sigmoid(opacity[gid]);
    gate = gsx_metal_adc_mcmc_noise_gate(opacity_value);
    scale_x = exp(logscale[gid * 3u + 0u]);
    scale_y = exp(logscale[gid * 3u + 1u]);
    scale_z = exp(logscale[gid * 3u + 2u]);
    qx = rotation[gid * 4u + 0u];
    qy = rotation[gid * 4u + 1u];
    qz = rotation[gid * 4u + 2u];
    qw = rotation[gid * 4u + 3u];
    noise_x = gsx_metal_adc_sample_normal(&rng);
    noise_y = gsx_metal_adc_sample_normal(&rng);
    noise_z = gsx_metal_adc_sample_normal(&rng);
    gsx_metal_adc_normalize_quaternion(qx, qy, qz, qw);
    m00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    m01 = 2.0f * (qx * qy - qw * qz);
    m02 = 2.0f * (qx * qz + qw * qy);
    m10 = 2.0f * (qx * qy + qw * qz);
    m11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    m12 = 2.0f * (qy * qz - qw * qx);
    m20 = 2.0f * (qx * qz - qw * qy);
    m21 = 2.0f * (qy * qz + qw * qx);
    m22 = 1.0f - 2.0f * (qx * qx + qy * qy);
    c00 = m00 * m00 * scale_x * scale_x + m01 * m01 * scale_y * scale_y + m02 * m02 * scale_z * scale_z;
    c01 = m00 * m10 * scale_x * scale_x + m01 * m11 * scale_y * scale_y + m02 * m12 * scale_z * scale_z;
    c02 = m00 * m20 * scale_x * scale_x + m01 * m21 * scale_y * scale_y + m02 * m22 * scale_z * scale_z;
    c10 = c01;
    c11 = m10 * m10 * scale_x * scale_x + m11 * m11 * scale_y * scale_y + m12 * m12 * scale_z * scale_z;
    c12 = m10 * m20 * scale_x * scale_x + m11 * m21 * scale_y * scale_y + m12 * m22 * scale_z * scale_z;
    c20 = c02;
    c21 = c12;
    c22 = m20 * m20 * scale_x * scale_x + m21 * m21 * scale_y * scale_y + m22 * m22 * scale_z * scale_z;
    scaled_noise_x = noise_x * gate * (params.noise_strength * params.learning_rate);
    scaled_noise_y = noise_y * gate * (params.noise_strength * params.learning_rate);
    scaled_noise_z = noise_z * gate * (params.noise_strength * params.learning_rate);
    mean3d[gid * 3u + 0u] += c00 * scaled_noise_x + c01 * scaled_noise_y + c02 * scaled_noise_z;
    mean3d[gid * 3u + 1u] += c10 * scaled_noise_x + c11 * scaled_noise_y + c12 * scaled_noise_z;
    mean3d[gid * 3u + 2u] += c20 * scaled_noise_x + c21 * scaled_noise_y + c22 * scaled_noise_z;
}

kernel void gsx_metal_adc_mcmc_dead_mask_kernel(
    device const float *opacity [[buffer(0)]],
    device uchar *out_dead_mask [[buffer(1)]],
    constant gsx_metal_adc_mcmc_dead_mask_params &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.gaussian_count) {
        return;
    }
    out_dead_mask[gid] = gsx_metal_adc_sigmoid(opacity[gid]) <= params.pruning_opacity_threshold ? 1u : 0u;
}

kernel void gsx_metal_adc_mcmc_relocation_kernel(
    device float *logscale [[buffer(0)]],
    device float *opacity [[buffer(1)]],
    device const uint *sample_counts [[buffer(2)]],
    constant gsx_metal_adc_mcmc_relocation_params &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint sample_count = 0u;
    uint ratio = 0u;
    float source_opacity = 0.0f;
    float new_opacity = 0.0f;
    float scale_coeff = 1.0f;
    float scale_x = 0.0f;
    float scale_y = 0.0f;
    float scale_z = 0.0f;

    if(gid >= params.gaussian_count) {
        return;
    }
    sample_count = sample_counts[gid];
    if(sample_count == 0u) {
        return;
    }

    ratio = sample_count + 1u;
    source_opacity = gsx_metal_adc_sigmoid(opacity[gid]);
    new_opacity = gsx_metal_adc_mcmc_relocated_opacity(source_opacity, ratio);
    scale_coeff = gsx_metal_adc_mcmc_scale_coeff(source_opacity, new_opacity, ratio);
    new_opacity = clamp(new_opacity, params.min_opacity, 1.0f - 1e-6f);
    scale_x = exp(logscale[gid * 3u + 0u]);
    scale_y = exp(logscale[gid * 3u + 1u]);
    scale_z = exp(logscale[gid * 3u + 2u]);
    opacity[gid] = gsx_metal_adc_logit(new_opacity);
    logscale[gid * 3u + 0u] = precise::log(scale_x * scale_coeff);
    logscale[gid * 3u + 1u] = precise::log(scale_y * scale_coeff);
    logscale[gid * 3u + 2u] = precise::log(scale_z * scale_coeff);
}
