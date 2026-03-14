#include <cuda_runtime.h>

#include <cstddef>

extern "C" {

namespace {

constexpr int GSX_CUDA_SSIM_WINDOW_SIZE = 11;
constexpr int GSX_CUDA_SSIM_WINDOW_RADIUS = GSX_CUDA_SSIM_WINDOW_SIZE / 2;
constexpr float GSX_CUDA_SSIM_C1 = 0.0001f;
constexpr float GSX_CUDA_SSIM_C2 = 0.0009f;

__constant__ float gsx_cuda_ssim_window[GSX_CUDA_SSIM_WINDOW_SIZE * GSX_CUDA_SSIM_WINDOW_SIZE] = {
    0.000001057565598153f, 0.000007814411533054f, 0.000037022477082749f, 0.000112464355116679f, 0.000219050652866017f, 0.000273561160085806f, 0.000219050652866017f, 0.000112464355116679f, 0.000037022477082749f, 0.000007814411533054f, 0.000001057565598153f,
    0.000007814411533054f, 0.000057741125197864f, 0.000273561160085806f, 0.000831005429087199f, 0.001618577562534386f, 0.002021358758362567f, 0.001618577562534386f, 0.000831005429087199f, 0.000273561160085806f, 0.000057741125197864f, 0.000007814411533054f,
    0.000037022477082749f, 0.000273561160085806f, 0.001296055593843201f, 0.003937069262846785f, 0.007668363825236721f, 0.009576627490240294f, 0.007668363825236721f, 0.003937069262846785f, 0.001296055593843201f, 0.000273561160085806f, 0.000037022477082749f,
    0.000112464355116679f, 0.000831005429087199f, 0.003937069262846785f, 0.011959760410037009f, 0.023294432473487107f, 0.029091225648550437f, 0.023294432473487107f, 0.011959760410037009f, 0.003937069262846785f, 0.000831005429087199f, 0.000112464355116679f,
    0.000219050652866017f, 0.001618577562534386f, 0.007668363825236721f, 0.023294432473487107f, 0.045371359095660320f, 0.056661970491684574f, 0.045371359095660320f, 0.023294432473487107f, 0.007668363825236721f, 0.001618577562534386f, 0.000219050652866017f,
    0.000273561160085806f, 0.002021358758362567f, 0.009576627490240294f, 0.029091225648550437f, 0.056661970491684574f, 0.070762237763946967f, 0.056661970491684574f, 0.029091225648550437f, 0.009576627490240294f, 0.002021358758362567f, 0.000273561160085806f,
    0.000219050652866017f, 0.001618577562534386f, 0.007668363825236721f, 0.023294432473487107f, 0.045371359095660320f, 0.056661970491684574f, 0.045371359095660320f, 0.023294432473487107f, 0.007668363825236721f, 0.001618577562534386f, 0.000219050652866017f,
    0.000112464355116679f, 0.000831005429087199f, 0.003937069262846785f, 0.011959760410037009f, 0.023294432473487107f, 0.029091225648550437f, 0.023294432473487107f, 0.011959760410037009f, 0.003937069262846785f, 0.000831005429087199f, 0.000112464355116679f,
    0.000037022477082749f, 0.000273561160085806f, 0.001296055593843201f, 0.003937069262846785f, 0.007668363825236721f, 0.009576627490240294f, 0.007668363825236721f, 0.003937069262846785f, 0.001296055593843201f, 0.000273561160085806f, 0.000037022477082749f,
    0.000007814411533054f, 0.000057741125197864f, 0.000273561160085806f, 0.000831005429087199f, 0.001618577562534386f, 0.002021358758362567f, 0.001618577562534386f, 0.000831005429087199f, 0.000273561160085806f, 0.000057741125197864f, 0.000007814411533054f,
    0.000001057565598153f, 0.000007814411533054f, 0.000037022477082749f, 0.000112464355116679f, 0.000219050652866017f, 0.000273561160085806f, 0.000219050652866017f, 0.000112464355116679f, 0.000037022477082749f, 0.000007814411533054f, 0.000001057565598153f
};

struct gsx_cuda_ssim_stats {
    float mu_prediction;
    float mu_target;
    float sigma_prediction_sq;
    float sigma_target_sq;
    float sigma_prediction_target;
};

__device__ __forceinline__ int gsx_cuda_ssim_clamp_index(int value, int limit)
{
    if(value < 0) {
        return 0;
    }
    if(value >= limit) {
        return limit - 1;
    }
    return value;
}

__device__ __forceinline__ size_t gsx_cuda_ssim_chw_offset(int c, int h, int w, int height, int width)
{
    return (size_t)c * (size_t)height * (size_t)width + (size_t)h * (size_t)width + (size_t)w;
}

__device__ __forceinline__ size_t gsx_cuda_ssim_hwc_offset(int c, int h, int w, int channels, int width)
{
    return ((size_t)h * (size_t)width + (size_t)w) * (size_t)channels + (size_t)c;
}

__device__ gsx_cuda_ssim_stats gsx_cuda_ssim_compute_stats_chw(
    const float *prediction,
    const float *target,
    int channel,
    int center_h,
    int center_w,
    int height,
    int width
)
{
    gsx_cuda_ssim_stats stats = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int window_y = 0;

    for(window_y = 0; window_y < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_y) {
        int src_h = gsx_cuda_ssim_clamp_index(center_h + window_y - GSX_CUDA_SSIM_WINDOW_RADIUS, height);
        int window_x = 0;

        for(window_x = 0; window_x < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_x) {
            int src_w = gsx_cuda_ssim_clamp_index(center_w + window_x - GSX_CUDA_SSIM_WINDOW_RADIUS, width);
            size_t offset = gsx_cuda_ssim_chw_offset(channel, src_h, src_w, height, width);
            float weight = gsx_cuda_ssim_window[window_y * GSX_CUDA_SSIM_WINDOW_SIZE + window_x];
            float prediction_value = prediction[offset];
            float target_value = target[offset];

            stats.mu_prediction += weight * prediction_value;
            stats.mu_target += weight * target_value;
            stats.sigma_prediction_sq += weight * prediction_value * prediction_value;
            stats.sigma_target_sq += weight * target_value * target_value;
            stats.sigma_prediction_target += weight * prediction_value * target_value;
        }
    }

    stats.sigma_prediction_sq -= stats.mu_prediction * stats.mu_prediction;
    stats.sigma_target_sq -= stats.mu_target * stats.mu_target;
    stats.sigma_prediction_target -= stats.mu_prediction * stats.mu_target;
    return stats;
}

__device__ gsx_cuda_ssim_stats gsx_cuda_ssim_compute_stats_hwc(
    const float *prediction,
    const float *target,
    int channel,
    int center_h,
    int center_w,
    int channels,
    int height,
    int width
)
{
    gsx_cuda_ssim_stats stats = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    int window_y = 0;

    for(window_y = 0; window_y < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_y) {
        int src_h = gsx_cuda_ssim_clamp_index(center_h + window_y - GSX_CUDA_SSIM_WINDOW_RADIUS, height);
        int window_x = 0;

        for(window_x = 0; window_x < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_x) {
            int src_w = gsx_cuda_ssim_clamp_index(center_w + window_x - GSX_CUDA_SSIM_WINDOW_RADIUS, width);
            size_t offset = gsx_cuda_ssim_hwc_offset(channel, src_h, src_w, channels, width);
            float weight = gsx_cuda_ssim_window[window_y * GSX_CUDA_SSIM_WINDOW_SIZE + window_x];
            float prediction_value = prediction[offset];
            float target_value = target[offset];

            stats.mu_prediction += weight * prediction_value;
            stats.mu_target += weight * target_value;
            stats.sigma_prediction_sq += weight * prediction_value * prediction_value;
            stats.sigma_target_sq += weight * target_value * target_value;
            stats.sigma_prediction_target += weight * prediction_value * target_value;
        }
    }

    stats.sigma_prediction_sq -= stats.mu_prediction * stats.mu_prediction;
    stats.sigma_target_sq -= stats.mu_target * stats.mu_target;
    stats.sigma_prediction_target -= stats.mu_prediction * stats.mu_target;
    return stats;
}

__device__ __forceinline__ float gsx_cuda_ssim_value_clamped(const gsx_cuda_ssim_stats &stats, bool *out_clamped)
{
    float a1 = 2.0f * stats.mu_prediction * stats.mu_target + GSX_CUDA_SSIM_C1;
    float a2 = 2.0f * stats.sigma_prediction_target + GSX_CUDA_SSIM_C2;
    float b1 = stats.mu_prediction * stats.mu_prediction + stats.mu_target * stats.mu_target + GSX_CUDA_SSIM_C1;
    float b2 = stats.sigma_prediction_sq + stats.sigma_target_sq + GSX_CUDA_SSIM_C2;
    float value = (a1 * a2) / (b1 * b2);
    bool clamped = false;

    if(value < -1.0f) {
        value = -1.0f;
        clamped = true;
    } else if(value > 1.0f) {
        value = 1.0f;
        clamped = true;
    }
    if(out_clamped != nullptr) {
        *out_clamped = clamped;
    }

    return value;
}

__device__ float gsx_cuda_ssim_prediction_derivative(
    const gsx_cuda_ssim_stats &stats,
    float prediction_value,
    float target_value,
    float effective_weight
)
{
    float a1 = 2.0f * stats.mu_prediction * stats.mu_target + GSX_CUDA_SSIM_C1;
    float a2 = 2.0f * stats.sigma_prediction_target + GSX_CUDA_SSIM_C2;
    float b1 = stats.mu_prediction * stats.mu_prediction + stats.mu_target * stats.mu_target + GSX_CUDA_SSIM_C1;
    float b2 = stats.sigma_prediction_sq + stats.sigma_target_sq + GSX_CUDA_SSIM_C2;
    float numerator = a1 * a2;
    float denominator = b1 * b2;
    float d_mu_prediction = effective_weight;
    float d_sigma_prediction_sq = 2.0f * effective_weight * (prediction_value - stats.mu_prediction);
    float d_sigma_prediction_target = effective_weight * (target_value - stats.mu_target);
    float d_a1 = 2.0f * stats.mu_target * d_mu_prediction;
    float d_a2 = 2.0f * d_sigma_prediction_target;
    float d_b1 = 2.0f * stats.mu_prediction * d_mu_prediction;
    float d_b2 = d_sigma_prediction_sq;
    float d_numerator = d_a1 * a2 + a1 * d_a2;
    float d_denominator = d_b1 * b2 + b1 * d_b2;

    return (d_numerator * denominator - numerator * d_denominator) / (denominator * denominator);
}

__global__ void gsx_cuda_loss_ssim_forward_chw_f32_kernel(
    float *__restrict__ loss_map,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    int channels,
    int height,
    int width,
    float scale
)
{
    size_t total_elements = (size_t)channels * (size_t)height * (size_t)width;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t linear_index = idx; linear_index < total_elements; linear_index += stride) {
        int channel = (int)(linear_index / ((size_t)height * (size_t)width));
        size_t hw_index = linear_index % ((size_t)height * (size_t)width);
        int y = (int)(hw_index / (size_t)width);
        int x = (int)(hw_index % (size_t)width);
        bool clamped = false;
        gsx_cuda_ssim_stats stats = gsx_cuda_ssim_compute_stats_chw(prediction, target, channel, y, x, height, width);
        float ssim = gsx_cuda_ssim_value_clamped(stats, &clamped);

        (void)clamped;
        loss_map[linear_index] += scale * (1.0f - ssim);
    }
}

__global__ void gsx_cuda_loss_ssim_forward_hwc_f32_kernel(
    float *__restrict__ loss_map,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    int channels,
    int height,
    int width,
    float scale
)
{
    size_t total_elements = (size_t)channels * (size_t)height * (size_t)width;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t linear_index = idx; linear_index < total_elements; linear_index += stride) {
        size_t pixel_index = linear_index / (size_t)channels;
        int channel = (int)(linear_index % (size_t)channels);
        int y = (int)(pixel_index / (size_t)width);
        int x = (int)(pixel_index % (size_t)width);
        bool clamped = false;
        gsx_cuda_ssim_stats stats = gsx_cuda_ssim_compute_stats_hwc(prediction, target, channel, y, x, channels, height, width);
        float ssim = gsx_cuda_ssim_value_clamped(stats, &clamped);

        (void)clamped;
        loss_map[linear_index] += scale * (1.0f - ssim);
    }
}

__global__ void gsx_cuda_loss_ssim_backward_chw_f32_kernel(
    float *__restrict__ grad_prediction,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    int channels,
    int height,
    int width,
    float grad_scale
)
{
    size_t total_elements = (size_t)channels * (size_t)height * (size_t)width;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t linear_index = idx; linear_index < total_elements; linear_index += stride) {
        int channel = (int)(linear_index / ((size_t)height * (size_t)width));
        size_t hw_index = linear_index % ((size_t)height * (size_t)width);
        int input_h = (int)(hw_index / (size_t)width);
        int input_w = (int)(hw_index % (size_t)width);
        float prediction_value = prediction[linear_index];
        float target_value = target[linear_index];
        float grad_value = 0.0f;
        int output_h = 0;

        for(output_h = input_h - GSX_CUDA_SSIM_WINDOW_RADIUS; output_h <= input_h + GSX_CUDA_SSIM_WINDOW_RADIUS; ++output_h) {
            int output_w = 0;

            if(output_h < 0 || output_h >= height) {
                continue;
            }
            for(output_w = input_w - GSX_CUDA_SSIM_WINDOW_RADIUS; output_w <= input_w + GSX_CUDA_SSIM_WINDOW_RADIUS; ++output_w) {
                float effective_weight = 0.0f;
                bool clamped = false;
                gsx_cuda_ssim_stats stats;
                int window_y = 0;

                if(output_w < 0 || output_w >= width) {
                    continue;
                }
                stats = gsx_cuda_ssim_compute_stats_chw(prediction, target, channel, output_h, output_w, height, width);
                (void)gsx_cuda_ssim_value_clamped(stats, &clamped);
                if(clamped) {
                    continue;
                }
                for(window_y = 0; window_y < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_y) {
                    int src_h = gsx_cuda_ssim_clamp_index(output_h + window_y - GSX_CUDA_SSIM_WINDOW_RADIUS, height);
                    int window_x = 0;

                    for(window_x = 0; window_x < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_x) {
                        int src_w = gsx_cuda_ssim_clamp_index(output_w + window_x - GSX_CUDA_SSIM_WINDOW_RADIUS, width);

                        if(src_h == input_h && src_w == input_w) {
                            effective_weight += gsx_cuda_ssim_window[window_y * GSX_CUDA_SSIM_WINDOW_SIZE + window_x];
                        }
                    }
                }
                if(effective_weight != 0.0f) {
                    grad_value -= gsx_cuda_ssim_prediction_derivative(
                        stats, prediction_value, target_value, effective_weight);
                }
            }
        }

        grad_prediction[linear_index] += grad_scale * grad_value;
    }
}

__global__ void gsx_cuda_loss_ssim_backward_hwc_f32_kernel(
    float *__restrict__ grad_prediction,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    int channels,
    int height,
    int width,
    float grad_scale
)
{
    size_t total_elements = (size_t)channels * (size_t)height * (size_t)width;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t linear_index = idx; linear_index < total_elements; linear_index += stride) {
        size_t pixel_index = linear_index / (size_t)channels;
        int channel = (int)(linear_index % (size_t)channels);
        int input_h = (int)(pixel_index / (size_t)width);
        int input_w = (int)(pixel_index % (size_t)width);
        float prediction_value = prediction[linear_index];
        float target_value = target[linear_index];
        float grad_value = 0.0f;
        int output_h = 0;

        for(output_h = input_h - GSX_CUDA_SSIM_WINDOW_RADIUS; output_h <= input_h + GSX_CUDA_SSIM_WINDOW_RADIUS; ++output_h) {
            int output_w = 0;

            if(output_h < 0 || output_h >= height) {
                continue;
            }
            for(output_w = input_w - GSX_CUDA_SSIM_WINDOW_RADIUS; output_w <= input_w + GSX_CUDA_SSIM_WINDOW_RADIUS; ++output_w) {
                float effective_weight = 0.0f;
                bool clamped = false;
                gsx_cuda_ssim_stats stats;
                int window_y = 0;

                if(output_w < 0 || output_w >= width) {
                    continue;
                }
                stats = gsx_cuda_ssim_compute_stats_hwc(prediction, target, channel, output_h, output_w, channels, height, width);
                (void)gsx_cuda_ssim_value_clamped(stats, &clamped);
                if(clamped) {
                    continue;
                }
                for(window_y = 0; window_y < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_y) {
                    int src_h = gsx_cuda_ssim_clamp_index(output_h + window_y - GSX_CUDA_SSIM_WINDOW_RADIUS, height);
                    int window_x = 0;

                    for(window_x = 0; window_x < GSX_CUDA_SSIM_WINDOW_SIZE; ++window_x) {
                        int src_w = gsx_cuda_ssim_clamp_index(output_w + window_x - GSX_CUDA_SSIM_WINDOW_RADIUS, width);

                        if(src_h == input_h && src_w == input_w) {
                            effective_weight += gsx_cuda_ssim_window[window_y * GSX_CUDA_SSIM_WINDOW_SIZE + window_x];
                        }
                    }
                }
                if(effective_weight != 0.0f) {
                    grad_value -= gsx_cuda_ssim_prediction_derivative(
                        stats, prediction_value, target_value, effective_weight);
                }
            }
        }

        grad_prediction[linear_index] += grad_scale * grad_value;
    }
}

static int gsx_cuda_loss_ssim_grid_size(size_t total_elements, int block_size)
{
    int grid_size = 0;

    if(total_elements == 0) {
        return 0;
    }

    grid_size = (int)((total_elements + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    return grid_size;
}

}  // namespace

cudaError_t gsx_cuda_loss_ssim_chw_f32_kernel_launch(
    float *loss_map,
    float *grad_prediction,
    const float *prediction,
    const float *target,
    int channels,
    int height,
    int width,
    float scale,
    float grad_scale,
    cudaStream_t stream
)
{
    const int block_size = 128;
    /* Extreme shape multiplication overflow is not handled in this round. */
    size_t total_elements = (size_t)channels * (size_t)height * (size_t)width;
    int grid_size = gsx_cuda_loss_ssim_grid_size(total_elements, block_size);
    cudaError_t error = cudaSuccess;

    if(grid_size == 0) {
        return cudaSuccess;
    }

    gsx_cuda_loss_ssim_forward_chw_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        loss_map, prediction, target, channels, height, width, scale);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        return error;
    }
    if(grad_prediction != nullptr) {
        gsx_cuda_loss_ssim_backward_chw_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            grad_prediction, prediction, target, channels, height, width, grad_scale);
        return cudaGetLastError();
    }

    return cudaSuccess;
}

cudaError_t gsx_cuda_loss_ssim_hwc_f32_kernel_launch(
    float *loss_map,
    float *grad_prediction,
    const float *prediction,
    const float *target,
    int channels,
    int height,
    int width,
    float scale,
    float grad_scale,
    cudaStream_t stream
)
{
    const int block_size = 128;
    size_t total_elements = (size_t)channels * (size_t)height * (size_t)width;
    int grid_size = gsx_cuda_loss_ssim_grid_size(total_elements, block_size);
    cudaError_t error = cudaSuccess;

    if(grid_size == 0) {
        return cudaSuccess;
    }

    gsx_cuda_loss_ssim_forward_hwc_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        loss_map, prediction, target, channels, height, width, scale);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        return error;
    }
    if(grad_prediction != nullptr) {
        gsx_cuda_loss_ssim_backward_hwc_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            grad_prediction, prediction, target, channels, height, width, grad_scale);
        return cudaGetLastError();
    }

    return cudaSuccess;
}

}
