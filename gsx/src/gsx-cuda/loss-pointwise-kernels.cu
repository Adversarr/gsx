#include <cuda_runtime.h>

#include <cstddef>

extern "C" {

__global__ void gsx_cuda_loss_mse_f32_kernel(
    float *__restrict__ loss_map,
    float *__restrict__ grad_prediction,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    size_t total_elements,
    float scale,
    float grad_scale
)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < total_elements; i += stride) {
        float diff = prediction[i] - target[i];

        loss_map[i] += scale * diff * diff;
        if(grad_prediction != nullptr) {
            grad_prediction[i] += 2.0f * diff * grad_scale;
        }
    }
}

__global__ void gsx_cuda_loss_l1_f32_kernel(
    float *__restrict__ loss_map,
    float *__restrict__ grad_prediction,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    size_t total_elements,
    float scale,
    float grad_scale
)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < total_elements; i += stride) {
        float diff = prediction[i] - target[i];
        float sign = 0.0f;

        if(diff > 0.0f) {
            sign = 1.0f;
        } else if(diff < 0.0f) {
            sign = -1.0f;
        }
        loss_map[i] += scale * fabsf(diff);
        if(grad_prediction != nullptr) {
            grad_prediction[i] += sign * grad_scale;
        }
    }
}

static int gsx_cuda_loss_grid_size(size_t total_elements)
{
    int grid_size = 0;

    if(total_elements == 0) {
        return 0;
    }

    grid_size = (int)((total_elements + 255u) / 256u);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    return grid_size;
}

cudaError_t gsx_cuda_loss_mse_f32_kernel_launch(
    float *loss_map,
    float *grad_prediction,
    const float *prediction,
    const float *target,
    size_t total_elements,
    float scale,
    float grad_scale,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = gsx_cuda_loss_grid_size(total_elements);

    if(grid_size == 0) {
        return cudaSuccess;
    }

    gsx_cuda_loss_mse_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        loss_map, grad_prediction, prediction, target, total_elements, scale, grad_scale);
    return cudaGetLastError();
}

cudaError_t gsx_cuda_loss_l1_f32_kernel_launch(
    float *loss_map,
    float *grad_prediction,
    const float *prediction,
    const float *target,
    size_t total_elements,
    float scale,
    float grad_scale,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = gsx_cuda_loss_grid_size(total_elements);

    if(grid_size == 0) {
        return cudaSuccess;
    }

    gsx_cuda_loss_l1_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        loss_map, grad_prediction, prediction, target, total_elements, scale, grad_scale);
    return cudaGetLastError();
}

}
