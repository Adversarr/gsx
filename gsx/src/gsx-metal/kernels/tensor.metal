#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

struct gsx_metal_tensor_gather_params {
    uint x_row_count;
    uint out_row_count;
    uint row_bytes;
};

struct gsx_metal_tensor_unary_f32_params {
    uint element_count;
};

struct gsx_metal_tensor_rand_f32_params {
    ulong rng_state;
    ulong rng_inc;
    uint element_count;
};

struct gsx_metal_tensor_randn_f32_params {
    ulong rng_state;
    ulong rng_inc;
    float sigma;
    uint element_count;
};

struct gsx_metal_tensor_randint_i32_params {
    ulong rng_state;
    ulong rng_inc;
    uint bound;
    uint element_count;
};

struct gsx_metal_tensor_unary_reduce_f32_params {
    uint outer_count;
    uint reduce_count;
};

struct gsx_metal_tensor_binary_reduce_f32_params {
    uint outer_count;
    uint reduce_count;
};

struct gsx_metal_tensor_clamp_f32_params {
    float min_value;
    float max_value;
    uint element_count;
};

struct gsx_metal_tensor_clamp_i32_params {
    int min_value;
    int max_value;
    uint element_count;
};

kernel void gsx_metal_tensor_gather_kernel(
    device const uchar *x_bytes [[buffer(0)]],
    device const int *index_data [[buffer(1)]],
    device uchar *out_bytes [[buffer(2)]],
    constant gsx_metal_tensor_gather_params &params [[buffer(3)]],
    device atomic_uint *out_status [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_bytes = params.out_row_count * params.row_bytes;
    if(gid >= total_bytes) {
        return;
    }

    uint row = gid / params.row_bytes;
    uint col = gid - row * params.row_bytes;
    int src_row = index_data[row];
    if(src_row < 0 || (uint)src_row >= params.x_row_count) {
        atomic_fetch_or_explicit(out_status, 1u, memory_order_relaxed);
        return;
    }

    out_bytes[gid] = x_bytes[(uint)src_row * params.row_bytes + col];
}

kernel void gsx_metal_tensor_exp_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_f32_params &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }

    out_values[gid] = exp(x_values[gid]);
}

kernel void gsx_metal_tensor_sigmoid_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_f32_params &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float x_value = 0.0f;

    if(gid >= params.element_count) {
        return;
    }

    x_value = x_values[gid];
    out_values[gid] = 1.0f / (1.0f + exp(-x_value));
}

kernel void gsx_metal_tensor_sigmoid_derivative_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_f32_params &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float x_value = 0.0f;
    float sigmoid_value = 0.0f;

    if(gid >= params.element_count) {
        return;
    }

    x_value = x_values[gid];
    sigmoid_value = 1.0f / (1.0f + exp(-x_value));
    out_values[gid] = sigmoid_value * (1.0f - sigmoid_value);
}

kernel void gsx_metal_tensor_abs_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_f32_params &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }

    out_values[gid] = fabs(x_values[gid]);
}

struct gsx_metal_pcg32 {
    ulong state;
    ulong inc;
};

constant uint GSX_METAL_RAND_VALUES_PER_THREAD = 4u;
constant uint GSX_METAL_RANDN_PAIRS_PER_THREAD = 2u;

inline uint gsx_metal_pcg32_next_uint(thread gsx_metal_pcg32 *rng)
{
    ulong oldstate = rng->state;
    uint xorshifted = (uint)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint rot = (uint)(oldstate >> 59u);

    rng->state = oldstate * 0x5851f42d4c957f2dUL + rng->inc;
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

inline uint gsx_metal_pcg32_next_uint_bound(thread gsx_metal_pcg32 *rng, uint bound)
{
    uint threshold = (~bound + 1u) % bound;

    for(;;) {
        uint value = gsx_metal_pcg32_next_uint(rng);

        if(value >= threshold) {
            return value % bound;
        }
    }
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

kernel void gsx_metal_tensor_rand_f32_kernel(
    device float *out_values [[buffer(0)]],
    constant gsx_metal_tensor_rand_f32_params &params [[buffer(1)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]])
{
    gsx_metal_pcg32 rng = { params.rng_state, params.rng_inc };
    uint gid = gid3.x;
    uint thread_count = threads_per_grid.x;

    if(gid >= thread_count) {
        return;
    }

    gsx_metal_pcg32_advance(&rng, (ulong)gid * (ulong)GSX_METAL_RAND_VALUES_PER_THREAD);
    for(uint j = 0; j < GSX_METAL_RAND_VALUES_PER_THREAD; ++j) {
        uint idx = gid + thread_count * j;

        if(idx >= params.element_count) {
            return;
        }
        out_values[idx] = gsx_metal_pcg32_next_float(&rng);
    }
}

kernel void gsx_metal_tensor_randn_f32_kernel(
    device float *out_values [[buffer(0)]],
    constant gsx_metal_tensor_randn_f32_params &params [[buffer(1)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]])
{
    gsx_metal_pcg32 rng = { params.rng_state, params.rng_inc };
    uint gid = gid3.x;
    uint thread_count = threads_per_grid.x;

    if(gid >= thread_count) {
        return;
    }

    gsx_metal_pcg32_advance(&rng, (ulong)gid * (ulong)(GSX_METAL_RANDN_PAIRS_PER_THREAD * 2u));
    for(uint p = 0; p < GSX_METAL_RANDN_PAIRS_PER_THREAD; ++p) {
        float u1 = gsx_metal_pcg32_next_float(&rng);
        float u2 = gsx_metal_pcg32_next_float(&rng);
        float radius = 0.0f;
        float theta = 0.0f;
        float z0 = 0.0f;
        float z1 = 0.0f;
        uint idx0 = gid + thread_count * (2u * p);
        uint idx1 = gid + thread_count * (2u * p + 1u);

        if(u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        radius = precise::sqrt(-2.0f * precise::log(u1));
        theta = 6.2831853071795864769f * u2;
        z0 = radius * precise::cos(theta);
        z1 = radius * precise::sin(theta);
        if(idx0 < params.element_count) {
            out_values[idx0] = z0 * params.sigma;
        }
        if(idx1 < params.element_count) {
            out_values[idx1] = z1 * params.sigma;
        }
    }
}

kernel void gsx_metal_tensor_randint_i32_kernel(
    device int *out_values [[buffer(0)]],
    constant gsx_metal_tensor_randint_i32_params &params [[buffer(1)]],
    uint3 gid3 [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]])
{
    gsx_metal_pcg32 rng = { params.rng_state, params.rng_inc };
    uint gid = gid3.x;
    uint thread_count = threads_per_grid.x;

    if(gid >= thread_count) {
        return;
    }

    gsx_metal_pcg32_advance(&rng, (ulong)gid * (ulong)GSX_METAL_RAND_VALUES_PER_THREAD);
    for(uint j = 0; j < GSX_METAL_RAND_VALUES_PER_THREAD; ++j) {
        uint idx = gid + thread_count * j;

        if(idx >= params.element_count) {
            return;
        }
        out_values[idx] = (int)gsx_metal_pcg32_next_uint_bound(&rng, params.bound);
    }
}

kernel void gsx_metal_tensor_sum_reduce_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_reduce_f32_params &params [[buffer(2)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint reduce_index = 0;
    uint stride = 0;
    uint base_index = 0;
    float accum = 0.0f;

    if(group_id >= params.outer_count) {
        return;
    }

    base_index = group_id * params.reduce_count;
    for(reduce_index = tid; reduce_index < params.reduce_count; reduce_index += threads_per_group) {
        accum += x_values[base_index + reduce_index];
    }

    scratch[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0) {
        out_values[group_id] = scratch[0];
    }
}

kernel void gsx_metal_tensor_mean_reduce_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_reduce_f32_params &params [[buffer(2)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint reduce_index = 0;
    uint stride = 0;
    uint base_index = 0;
    float accum = 0.0f;

    if(group_id >= params.outer_count) {
        return;
    }

    base_index = group_id * params.reduce_count;
    for(reduce_index = tid; reduce_index < params.reduce_count; reduce_index += threads_per_group) {
        accum += x_values[base_index + reduce_index];
    }

    scratch[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0) {
        out_values[group_id] = scratch[0] / (float)params.reduce_count;
    }
}

kernel void gsx_metal_tensor_max_reduce_f32_kernel(
    device const float *x_values [[buffer(0)]],
    device float *out_values [[buffer(1)]],
    constant gsx_metal_tensor_unary_reduce_f32_params &params [[buffer(2)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint reduce_index = 0;
    uint stride = 0;
    uint base_index = 0;
    float accum = -INFINITY;

    if(group_id >= params.outer_count) {
        return;
    }

    base_index = group_id * params.reduce_count;
    for(reduce_index = tid; reduce_index < params.reduce_count; reduce_index += threads_per_group) {
        accum = max(accum, x_values[base_index + reduce_index]);
    }

    scratch[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0) {
        out_values[group_id] = scratch[0];
    }
}

kernel void gsx_metal_tensor_mse_reduce_f32_kernel(
    device const float *lhs_values [[buffer(0)]],
    device const float *rhs_values [[buffer(1)]],
    device float *out_values [[buffer(2)]],
    constant gsx_metal_tensor_binary_reduce_f32_params &params [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint reduce_index = 0;
    uint stride = 0;
    uint base_index = 0;
    float accum = 0.0f;

    if(group_id >= params.outer_count) {
        return;
    }

    base_index = group_id * params.reduce_count;
    for(reduce_index = tid; reduce_index < params.reduce_count; reduce_index += threads_per_group) {
        float diff = lhs_values[base_index + reduce_index] - rhs_values[base_index + reduce_index];
        accum += diff * diff;
    }

    scratch[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0) {
        out_values[group_id] = scratch[0] / (float)params.reduce_count;
    }
}

kernel void gsx_metal_tensor_mae_reduce_f32_kernel(
    device const float *lhs_values [[buffer(0)]],
    device const float *rhs_values [[buffer(1)]],
    device float *out_values [[buffer(2)]],
    constant gsx_metal_tensor_binary_reduce_f32_params &params [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint reduce_index = 0;
    uint stride = 0;
    uint base_index = 0;
    float accum = 0.0f;

    if(group_id >= params.outer_count) {
        return;
    }

    base_index = group_id * params.reduce_count;
    for(reduce_index = tid; reduce_index < params.reduce_count; reduce_index += threads_per_group) {
        float diff = lhs_values[base_index + reduce_index] - rhs_values[base_index + reduce_index];
        accum += fabs(diff);
    }

    scratch[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0) {
        out_values[group_id] = scratch[0] / (float)params.reduce_count;
    }
}

kernel void gsx_metal_tensor_clamp_f32_inplace_kernel(
    device float *values [[buffer(0)]],
    constant gsx_metal_tensor_clamp_f32_params &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }
    values[gid] = clamp(values[gid], params.min_value, params.max_value);
}

kernel void gsx_metal_tensor_clamp_i32_inplace_kernel(
    device int *values [[buffer(0)]],
    constant gsx_metal_tensor_clamp_i32_params &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }
    values[gid] = clamp(values[gid], params.min_value, params.max_value);
}

struct gsx_metal_tensor_check_finite_params {
    uint element_count;
};

inline bool gsx_metal_f16_is_finite(uint16_t value)
{
    return ((value >> 10) & 0x1FU) != 0x1FU;
}

inline bool gsx_metal_bf16_is_finite(uint16_t value)
{
    return ((value >> 7) & 0xFFU) != 0xFFU;
}

kernel void gsx_metal_tensor_check_finite_f32_kernel(
    device const float *values [[buffer(0)]],
    constant gsx_metal_tensor_check_finite_params &params [[buffer(1)]],
    device atomic_uint *has_non_finite [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }
    if(!isfinite(values[gid])) {
        atomic_fetch_or_explicit(has_non_finite, 1u, memory_order_relaxed);
    }
}

kernel void gsx_metal_tensor_check_finite_f16_kernel(
    device const uint16_t *values [[buffer(0)]],
    constant gsx_metal_tensor_check_finite_params &params [[buffer(1)]],
    device atomic_uint *has_non_finite [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }
    if(!gsx_metal_f16_is_finite(values[gid])) {
        atomic_fetch_or_explicit(has_non_finite, 1u, memory_order_relaxed);
    }
}

kernel void gsx_metal_tensor_check_finite_bf16_kernel(
    device const uint16_t *values [[buffer(0)]],
    constant gsx_metal_tensor_check_finite_params &params [[buffer(1)]],
    device atomic_uint *has_non_finite [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }
    if(!gsx_metal_bf16_is_finite(values[gid])) {
        atomic_fetch_or_explicit(has_non_finite, 1u, memory_order_relaxed);
    }
}
