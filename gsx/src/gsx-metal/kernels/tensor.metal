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
