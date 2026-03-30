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
    uint base_idx = gid * 4u;

    if(base_idx >= params.element_count) {
        return;
    }

    uint remaining = min(4u, params.element_count - base_idx);
    bool use_vector_path = remaining == 4u
        && (((base_idx | uint(params.element_count)) & 3u) == 0u);

    if(use_vector_path) {
        float4 grad = *reinterpret_cast<device const float4 *>(gradient + base_idx);
        float4 m_prev = *reinterpret_cast<device const float4 *>(first_moment + base_idx);
        float4 v_prev = *reinterpret_cast<device const float4 *>(second_moment + base_idx);
        float4 param = *reinterpret_cast<device float4 *>(parameter + base_idx);

        if(params.max_grad > 0.0f) {
            grad = clamp(grad, float4(-params.max_grad), float4(params.max_grad));
        }

        float4 m = params.beta1 * m_prev + (1.0f - params.beta1) * grad;
        float4 v = params.beta2 * v_prev + (1.0f - params.beta2) * grad * grad;

        *reinterpret_cast<device float4 *>(first_moment + base_idx) = m;
        *reinterpret_cast<device float4 *>(second_moment + base_idx) = v;

        if(params.weight_decay > 0.0f) {
            param -= params.learning_rate * params.weight_decay * param;
        }

        float4 m_hat = m * params.inv_beta1_correction;
        float4 v_hat = v * params.inv_beta2_correction;
        param -= params.learning_rate * (m_hat / (sqrt(v_hat) + params.epsilon));
        *reinterpret_cast<device float4 *>(parameter + base_idx) = param;
        return;
    }

    for(uint lane = 0u; lane < remaining; ++lane) {
        uint idx = base_idx + lane;
        float grad = gradient[idx];
        if(params.max_grad > 0.0f) {
            grad = clamp(grad, -params.max_grad, params.max_grad);
        }

        float m = params.beta1 * first_moment[idx] + (1.0f - params.beta1) * grad;
        float v = params.beta2 * second_moment[idx] + (1.0f - params.beta2) * grad * grad;
        float param = parameter[idx];

        first_moment[idx] = m;
        second_moment[idx] = v;

        if(params.weight_decay > 0.0f) {
            param -= params.learning_rate * params.weight_decay * param;
        }

        float m_hat = m * params.inv_beta1_correction;
        float v_hat = v * params.inv_beta2_correction;
        param -= params.learning_rate * (m_hat / (sqrt(v_hat) + params.epsilon));
        parameter[idx] = param;
    }
}
