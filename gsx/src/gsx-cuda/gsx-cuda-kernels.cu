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

}
