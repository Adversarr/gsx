#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <cmath>

extern "C" {

__device__ __forceinline__ bool gsx_cuda_f16_is_non_finite(uint16_t value)
{
    return ((value >> 10U) & 0x1FU) == 0x1FU;
}

__device__ __forceinline__ bool gsx_cuda_bf16_is_non_finite(uint16_t value)
{
    return ((value >> 7U) & 0xFFU) == 0xFFU;
}

__device__ __forceinline__ bool gsx_cuda_f16_packed4_has_non_finite(uint64_t packed)
{
    uint16_t v0 = (uint16_t)(packed & 0xFFFFULL);
    uint16_t v1 = (uint16_t)((packed >> 16U) & 0xFFFFULL);
    uint16_t v2 = (uint16_t)((packed >> 32U) & 0xFFFFULL);
    uint16_t v3 = (uint16_t)((packed >> 48U) & 0xFFFFULL);
    return gsx_cuda_f16_is_non_finite(v0) || gsx_cuda_f16_is_non_finite(v1) || gsx_cuda_f16_is_non_finite(v2) || gsx_cuda_f16_is_non_finite(v3);
}

__device__ __forceinline__ bool gsx_cuda_bf16_packed4_has_non_finite(uint64_t packed)
{
    uint16_t v0 = (uint16_t)(packed & 0xFFFFULL);
    uint16_t v1 = (uint16_t)((packed >> 16U) & 0xFFFFULL);
    uint16_t v2 = (uint16_t)((packed >> 32U) & 0xFFFFULL);
    uint16_t v3 = (uint16_t)((packed >> 48U) & 0xFFFFULL);
    return gsx_cuda_bf16_is_non_finite(v0) || gsx_cuda_bf16_is_non_finite(v1) || gsx_cuda_bf16_is_non_finite(v2) || gsx_cuda_bf16_is_non_finite(v3);
}

__global__ void gsx_cuda_fill_tensor_bytes_kernel(uint8_t *__restrict__ dst, const uint8_t *__restrict__ value, size_t value_size, size_t total_bytes)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < total_bytes; i += stride) {
        dst[i] = value[i % value_size];
    }
}

__global__ void gsx_cuda_fill_tensor_u32_kernel(uint32_t *__restrict__ dst, const uint32_t *__restrict__ value, size_t total_words)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    uint32_t fill_value = value[0];

    for(size_t i = idx; i < total_words; i += stride) {
        dst[i] = fill_value;
    }
}

__global__ void gsx_cuda_fill_tensor_u64_kernel(uint64_t *__restrict__ dst, const uint64_t *__restrict__ value, size_t total_words)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    uint64_t fill_value = value[0];

    for(size_t i = idx; i < total_words; i += stride) {
        dst[i] = fill_value;
    }
}

__global__ void gsx_cuda_fill_tensor_u128_kernel(uint4 *__restrict__ dst, const uint4 *__restrict__ value, size_t total_words)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    uint4 fill_value = value[0];

    for(size_t i = idx; i < total_words; i += stride) {
        dst[i] = fill_value;
    }
}

void gsx_cuda_fill_tensor_kernel_launch(void *dst, const void *value, size_t value_size, size_t total_bytes, size_t alignment_bytes, cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (int)((total_bytes + block_size - 1) / block_size);
    size_t vec_count = 0;
    size_t tail_bytes = 0;
    if(total_bytes == 0) {
        return;
    }

    if(grid_size > 65535) {
        grid_size = 65535;
    }

    if(value_size == sizeof(uint4) && alignment_bytes >= sizeof(uint4)) {
        vec_count = total_bytes / sizeof(uint4);
        tail_bytes = total_bytes % sizeof(uint4);
        if(vec_count != 0) {
            gsx_cuda_fill_tensor_u128_kernel<<<grid_size, block_size, 0, stream>>>(
                (uint4*)dst,
                (const uint4*)value,
                vec_count
            );
        }
        if(tail_bytes != 0) {
            gsx_cuda_fill_tensor_bytes_kernel<<<grid_size, block_size, 0, stream>>>(
                (uint8_t*)dst + vec_count * sizeof(uint4),
                (const uint8_t*)value,
                value_size,
                tail_bytes
            );
        }
        return;
    }

    if(value_size == sizeof(uint64_t) && alignment_bytes >= sizeof(uint64_t)) {
        vec_count = total_bytes / sizeof(uint64_t);
        tail_bytes = total_bytes % sizeof(uint64_t);
        if(vec_count != 0) {
            gsx_cuda_fill_tensor_u64_kernel<<<grid_size, block_size, 0, stream>>>(
                (uint64_t*)dst,
                (const uint64_t*)value,
                vec_count
            );
        }
        if(tail_bytes != 0) {
            gsx_cuda_fill_tensor_bytes_kernel<<<grid_size, block_size, 0, stream>>>(
                (uint8_t*)dst + vec_count * sizeof(uint64_t),
                (const uint8_t*)value,
                value_size,
                tail_bytes
            );
        }
        return;
    }

    if(value_size == sizeof(uint32_t) && alignment_bytes >= sizeof(uint32_t)) {
        vec_count = total_bytes / sizeof(uint32_t);
        tail_bytes = total_bytes % sizeof(uint32_t);
        if(vec_count != 0) {
            gsx_cuda_fill_tensor_u32_kernel<<<grid_size, block_size, 0, stream>>>(
                (uint32_t*)dst,
                (const uint32_t*)value,
                vec_count
            );
        }
        if(tail_bytes != 0) {
            gsx_cuda_fill_tensor_bytes_kernel<<<grid_size, block_size, 0, stream>>>(
                (uint8_t*)dst + vec_count * sizeof(uint32_t),
                (const uint8_t*)value,
                value_size,
                tail_bytes
            );
        }
        return;
    }

    gsx_cuda_fill_tensor_bytes_kernel<<<grid_size, block_size, 0, stream>>>(
        (uint8_t*)dst,
        (const uint8_t*)value,
        value_size,
        total_bytes
    );
}

__global__ void gsx_cuda_check_finite_f32_scalar_kernel(const float *__restrict__ src, size_t count, int *__restrict__ has_non_finite)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride) {
        if(!isfinite(src[i])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

__global__ void gsx_cuda_check_finite_f32_float4_kernel(const float4 *__restrict__ src, size_t count, int *__restrict__ has_non_finite)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride) {
        float4 values = src[i];
        if(!isfinite(values.x) || !isfinite(values.y) || !isfinite(values.z) || !isfinite(values.w)) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

__global__ void gsx_cuda_check_finite_f16_scalar_kernel(const uint16_t *__restrict__ src, size_t count, int *__restrict__ has_non_finite)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride) {
        if(gsx_cuda_f16_is_non_finite(src[i])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

__global__ void gsx_cuda_check_finite_f16_packed4_kernel(const uint64_t *__restrict__ src, size_t count, int *__restrict__ has_non_finite)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride) {
        if(gsx_cuda_f16_packed4_has_non_finite(src[i])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

__global__ void gsx_cuda_check_finite_bf16_scalar_kernel(const uint16_t *__restrict__ src, size_t count, int *__restrict__ has_non_finite)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride) {
        if(gsx_cuda_bf16_is_non_finite(src[i])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

__global__ void gsx_cuda_check_finite_bf16_packed4_kernel(const uint64_t *__restrict__ src, size_t count, int *__restrict__ has_non_finite)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride) {
        if(gsx_cuda_bf16_packed4_has_non_finite(src[i])) {
            atomicExch(has_non_finite, 1);
            return;
        }
    }
}

void gsx_cuda_check_finite_tensor_f32_kernel_launch(const void *src, size_t total_elements, size_t alignment_bytes, int *out_has_non_finite, cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    size_t vector_count = 0;
    size_t tail_count = 0;
    if(total_elements == 0) {
        return;
    }
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    if(alignment_bytes >= sizeof(float4) && total_elements >= 4) {
        vector_count = total_elements / 4;
        tail_count = total_elements % 4;
        gsx_cuda_check_finite_f32_float4_kernel<<<grid_size, block_size, 0, stream>>>(
            (const float4*)src,
            vector_count,
            out_has_non_finite
        );
        if(tail_count != 0) {
            gsx_cuda_check_finite_f32_scalar_kernel<<<1, (int)tail_count, 0, stream>>>(
                (const float*)src + vector_count * 4,
                tail_count,
                out_has_non_finite
            );
        }
        return;
    }

    gsx_cuda_check_finite_f32_scalar_kernel<<<grid_size, block_size, 0, stream>>>((const float*)src, total_elements, out_has_non_finite);
}

void gsx_cuda_check_finite_tensor_f16_kernel_launch(const void *src, size_t total_elements, size_t alignment_bytes, int *out_has_non_finite, cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    size_t packed_count = 0;
    size_t tail_count = 0;
    if(total_elements == 0) {
        return;
    }
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    if(alignment_bytes >= sizeof(uint64_t) && total_elements >= 4) {
        packed_count = total_elements / 4;
        tail_count = total_elements % 4;
        gsx_cuda_check_finite_f16_packed4_kernel<<<grid_size, block_size, 0, stream>>>(
            (const uint64_t*)src,
            packed_count,
            out_has_non_finite
        );
        if(tail_count != 0) {
            gsx_cuda_check_finite_f16_scalar_kernel<<<1, (int)tail_count, 0, stream>>>(
                (const uint16_t*)src + packed_count * 4,
                tail_count,
                out_has_non_finite
            );
        }
        return;
    }

    gsx_cuda_check_finite_f16_scalar_kernel<<<grid_size, block_size, 0, stream>>>((const uint16_t*)src, total_elements, out_has_non_finite);
}

void gsx_cuda_check_finite_tensor_bf16_kernel_launch(const void *src, size_t total_elements, size_t alignment_bytes, int *out_has_non_finite, cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    size_t packed_count = 0;
    size_t tail_count = 0;
    if(total_elements == 0) {
        return;
    }
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    if(alignment_bytes >= sizeof(uint64_t) && total_elements >= 4) {
        packed_count = total_elements / 4;
        tail_count = total_elements % 4;
        gsx_cuda_check_finite_bf16_packed4_kernel<<<grid_size, block_size, 0, stream>>>(
            (const uint64_t*)src,
            packed_count,
            out_has_non_finite
        );
        if(tail_count != 0) {
            gsx_cuda_check_finite_bf16_scalar_kernel<<<1, (int)tail_count, 0, stream>>>(
                (const uint16_t*)src + packed_count * 4,
                tail_count,
                out_has_non_finite
            );
        }
        return;
    }

    gsx_cuda_check_finite_bf16_scalar_kernel<<<grid_size, block_size, 0, stream>>>((const uint16_t*)src, total_elements, out_has_non_finite);
}

__global__ void gsx_cuda_adam_step_f32_kernel(
    float *__restrict__ parameter,
    const float *__restrict__ gradient,
    float *__restrict__ first_moment,
    float *__restrict__ second_moment,
    size_t total_elements,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float max_grad,
    double inv_beta1_correction,
    double inv_beta2_correction
)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < total_elements; i += stride) {
        float gradient_value = gradient[i];
        if(max_grad > 0.0f) {
            if(gradient_value > max_grad) {
                gradient_value = max_grad;
            } else if(gradient_value < -max_grad) {
                gradient_value = -max_grad;
            }
        }
        float first_moment_value = beta1 * first_moment[i] + (1.0f - beta1) * gradient_value;
        float second_moment_value = beta2 * second_moment[i] + (1.0f - beta2) * gradient_value * gradient_value;
        float first_moment_hat = (float)((double)first_moment_value * inv_beta1_correction);
        float second_moment_hat = (float)((double)second_moment_value * inv_beta2_correction);
        float parameter_value = parameter[i];

        first_moment[i] = first_moment_value;
        second_moment[i] = second_moment_value;
        if(weight_decay > 0.0f) {
            parameter_value -= learning_rate * weight_decay * parameter_value;
        }
        parameter_value -= learning_rate * (first_moment_hat / (sqrtf(second_moment_hat) + epsilon));
        parameter[i] = parameter_value;
    }
}

// TODO: implement vectorized gather kernel to improve memory throughput.
__global__ void gsx_cuda_gather_rows_kernel(
    const uint8_t *__restrict__ src,
    uint8_t *__restrict__ dst,
    size_t row_bytes,
    size_t row_count,
    const int32_t *__restrict__ src_indices,
    size_t src_row_count,
    int *__restrict__ out_has_out_of_range
)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t row_index = idx; row_index < row_count; row_index += stride) {
        int32_t src_row_i32 = src_indices[row_index];
        size_t src_row = 0;

        if(src_row_i32 < 0 || (size_t)src_row_i32 >= src_row_count) {
            if(out_has_out_of_range != NULL) {
                atomicExch(out_has_out_of_range, 1);
            }
            continue;
        }
        src_row = (size_t)src_row_i32;
        for(size_t row_offset = 0; row_offset < row_bytes; ++row_offset) {
            dst[row_index * row_bytes + row_offset] = src[src_row * row_bytes + row_offset];
        }
    }
}

__global__ void gsx_cuda_exp_tensor_f32_kernel(const float *__restrict__ src, float *__restrict__ dst, size_t element_count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < element_count; i += stride) {
        dst[i] = expf(src[i]);
    }
}

__global__ void gsx_cuda_clamp_inplace_tensor_f32_kernel(float *__restrict__ values, size_t element_count, float min_value, float max_value)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < element_count; i += stride) {
        float value = values[i];
        if(value < min_value) {
            value = min_value;
        } else if(value > max_value) {
            value = max_value;
        }
        values[i] = value;
    }
}

__global__ void gsx_cuda_clamp_inplace_tensor_i32_kernel(int32_t *__restrict__ values, size_t element_count, int32_t min_value, int32_t max_value)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < element_count; i += stride) {
        int32_t value = values[i];
        if(value < min_value) {
            value = min_value;
        } else if(value > max_value) {
            value = max_value;
        }
        values[i] = value;
    }
}

__global__ void gsx_cuda_clamp_inplace_tensor_u8_kernel(uint8_t *__restrict__ values, size_t element_count, uint8_t min_value, uint8_t max_value)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for(size_t i = idx; i < element_count; i += stride) {
        uint8_t value = values[i];
        if(value < min_value) {
            value = min_value;
        } else if(value > max_value) {
            value = max_value;
        }
        values[i] = value;
    }
}

void gsx_cuda_adam_step_f32_kernel_launch(
    float *parameter,
    const float *gradient,
    float *first_moment,
    float *second_moment,
    size_t total_elements,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float max_grad,
    double inv_beta1_correction,
    double inv_beta2_correction,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = 0;

    if(total_elements == 0) {
        return;
    }

    grid_size = (int)((total_elements + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    gsx_cuda_adam_step_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        parameter,
        gradient,
        first_moment,
        second_moment,
        total_elements,
        beta1,
        beta2,
        learning_rate,
        weight_decay,
        epsilon,
        max_grad,
        inv_beta1_correction,
        inv_beta2_correction
    );
}

cudaError_t gsx_cuda_gather_rows_kernel_launch(
    const void *src,
    void *dst,
    size_t row_bytes,
    size_t row_count,
    const int32_t *src_indices,
    size_t src_row_count,
    int *out_has_out_of_range,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = 0;
    cudaError_t status = cudaSuccess;

    if(row_bytes != 0 && (row_count > ((size_t)-1) / row_bytes || src_row_count > ((size_t)-1) / row_bytes)) {
        return cudaErrorInvalidValue;
    }
    if(row_count == 0 || row_bytes == 0) {
        return cudaSuccess;
    }

    grid_size = (int)((row_count + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    gsx_cuda_gather_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        (const uint8_t *)src,
        (uint8_t *)dst,
        row_bytes,
        row_count,
        src_indices,
        src_row_count,
        out_has_out_of_range
    );
    status = cudaGetLastError();
    if(status != cudaSuccess) {
        return status;
    }
    return cudaSuccess;
}

cudaError_t gsx_cuda_exp_tensor_f32_kernel_launch(const float *src, float *dst, size_t element_count, cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = 0;
    cudaError_t status = cudaSuccess;

    if(element_count == 0) {
        return cudaSuccess;
    }
    if(src == NULL || dst == NULL) {
        return cudaErrorInvalidValue;
    }

    grid_size = (int)((element_count + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    gsx_cuda_exp_tensor_f32_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, element_count);
    status = cudaGetLastError();
    if(status != cudaSuccess) {
        return status;
    }
    return cudaSuccess;
}

cudaError_t gsx_cuda_clamp_inplace_tensor_f32_kernel_launch(
    float *values,
    size_t element_count,
    float min_value,
    float max_value,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = 0;
    cudaError_t status = cudaSuccess;

    if(element_count == 0) {
        return cudaSuccess;
    }
    if(values == NULL) {
        return cudaErrorInvalidValue;
    }

    grid_size = (int)((element_count + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    gsx_cuda_clamp_inplace_tensor_f32_kernel<<<grid_size, block_size, 0, stream>>>(values, element_count, min_value, max_value);
    status = cudaGetLastError();
    if(status != cudaSuccess) {
        return status;
    }
    return cudaSuccess;
}

cudaError_t gsx_cuda_clamp_inplace_tensor_i32_kernel_launch(
    int32_t *values,
    size_t element_count,
    int32_t min_value,
    int32_t max_value,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = 0;
    cudaError_t status = cudaSuccess;

    if(element_count == 0) {
        return cudaSuccess;
    }
    if(values == NULL) {
        return cudaErrorInvalidValue;
    }

    grid_size = (int)((element_count + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    gsx_cuda_clamp_inplace_tensor_i32_kernel<<<grid_size, block_size, 0, stream>>>(values, element_count, min_value, max_value);
    status = cudaGetLastError();
    if(status != cudaSuccess) {
        return status;
    }
    return cudaSuccess;
}

cudaError_t gsx_cuda_clamp_inplace_tensor_u8_kernel_launch(
    uint8_t *values,
    size_t element_count,
    uint8_t min_value,
    uint8_t max_value,
    cudaStream_t stream
)
{
    const int block_size = 256;
    int grid_size = 0;
    cudaError_t status = cudaSuccess;

    if(element_count == 0) {
        return cudaSuccess;
    }
    if(values == NULL) {
        return cudaErrorInvalidValue;
    }

    grid_size = (int)((element_count + (size_t)block_size - 1) / (size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    gsx_cuda_clamp_inplace_tensor_u8_kernel<<<grid_size, block_size, 0, stream>>>(values, element_count, min_value, max_value);
    status = cudaGetLastError();
    if(status != cudaSuccess) {
        return status;
    }
    return cudaSuccess;
}

}
