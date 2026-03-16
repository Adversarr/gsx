#include <metal_stdlib>
using namespace metal;

struct gsx_metal_loss_pointwise_params {
    uint element_count;
    float scale;
};

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
