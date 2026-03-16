#include <metal_stdlib>
using namespace metal;

struct gsx_metal_adam_step_params {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_grad;
    float inv_beta1_correction;
    float inv_beta2_correction;
    uint element_count;
};

kernel void gsx_metal_adam_step_f32_kernel(
    device float *parameter [[buffer(0)]],
    device const float *gradient [[buffer(1)]],
    device float *first_moment [[buffer(2)]],
    device float *second_moment [[buffer(3)]],
    constant gsx_metal_adam_step_params &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid >= params.element_count) {
        return;
    }

    float grad = gradient[gid];
    if(params.max_grad > 0.0f) {
        grad = clamp(grad, -params.max_grad, params.max_grad);
    }

    float m = params.beta1 * first_moment[gid] + (1.0f - params.beta1) * grad;
    float v = params.beta2 * second_moment[gid] + (1.0f - params.beta2) * grad * grad;
    float param = parameter[gid];

    first_moment[gid] = m;
    second_moment[gid] = v;

    if(params.weight_decay > 0.0f) {
        param -= params.learning_rate * params.weight_decay * param;
    }

    float m_hat = m * params.inv_beta1_correction;
    float v_hat = v * params.inv_beta2_correction;
    param -= params.learning_rate * (m_hat / (sqrt(v_hat) + params.epsilon));
    parameter[gid] = param;
}
