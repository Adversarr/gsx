#include "internal.h"

#include <cub/cub.cuh>
#include <limits.h>
#include <math.h>

static int gsx_cuda_reduce_compute_grid_size(gsx_size_t element_count, int block_size)
{
    int device_id = 0;
    int sm_count = 0;
    int suggested_grid = 1;
    gsx_size_t blocks_by_work = 0;

    if(block_size <= 0 || element_count == 0) {
        return 1;
    }
    blocks_by_work = (element_count + (gsx_size_t)block_size - 1) / (gsx_size_t)block_size;
    if(blocks_by_work == 0) {
        return 1;
    }
    if(cudaGetDevice(&device_id) != cudaSuccess) {
        if(blocks_by_work > (gsx_size_t)INT_MAX) {
            return INT_MAX;
        }
        return (int)blocks_by_work;
    }
    if(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) != cudaSuccess || sm_count <= 0) {
        if(blocks_by_work > (gsx_size_t)INT_MAX) {
            return INT_MAX;
        }
        return (int)blocks_by_work;
    }
    suggested_grid = sm_count * 32;
    if(suggested_grid <= 0) {
        suggested_grid = 1;
    }
    if(blocks_by_work < (gsx_size_t)suggested_grid) {
        return (int)blocks_by_work;
    }
    return suggested_grid;
}

static gsx_size_t gsx_cuda_round_up(gsx_size_t value, gsx_size_t alignment)
{
    if(alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

__global__ static void gsx_cuda_scale_outputs_kernel(float *values, gsx_size_t count, float scale)
{
    gsx_size_t index = (gsx_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    gsx_size_t stride = (gsx_size_t)blockDim.x * gridDim.x;

    for(; index < count; index += stride) {
        values[index] *= scale;
    }
}

__global__ static void gsx_cuda_prepare_binary_terms_kernel(
    const float *lhs,
    const float *rhs,
    float *terms,
    gsx_size_t count,
    gsx_impl_binary_reduce_op op
)
{
    gsx_size_t index = (gsx_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    gsx_size_t stride = (gsx_size_t)blockDim.x * gridDim.x;

    for(; index < count; index += stride) {
        float diff = lhs[index] - rhs[index];
        if(op == GSX_IMPL_BINARY_REDUCE_OP_MSE) {
            terms[index] = diff * diff;
        } else {
            terms[index] = fabsf(diff);
        }
    }
}

extern "C" cudaError_t gsx_cuda_unary_reduce_workspace_size_query(
    gsx_size_t reduce_count,
    gsx_impl_unary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes
)
{
    size_t temp_storage_bytes = 0;
    const float *input = NULL;
    float *output = NULL;
    cudaError_t cuda_err = cudaSuccess;

    if(out_workspace_size_bytes == NULL || reduce_count == 0) {
        return cudaErrorInvalidValue;
    }
    switch(op) {
    case GSX_IMPL_UNARY_REDUCE_OP_SUM:
    case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
        cuda_err = cub::DeviceReduce::Sum(NULL, temp_storage_bytes, input, output, reduce_count);
        break;
    case GSX_IMPL_UNARY_REDUCE_OP_MAX:
        cuda_err = cub::DeviceReduce::Max(NULL, temp_storage_bytes, input, output, reduce_count);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    if(cuda_err != cudaSuccess) {
        return cuda_err;
    }
    *out_workspace_size_bytes = (gsx_size_t)temp_storage_bytes;
    return cudaSuccess;
}

extern "C" cudaError_t gsx_cuda_binary_reduce_workspace_size_query(
    gsx_size_t reduce_count,
    gsx_impl_binary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes
)
{
    size_t cub_workspace_size = 0;
    gsx_size_t cub_workspace_aligned = 0;
    gsx_size_t transform_bytes = 0;
    gsx_size_t total_bytes = 0;
    const float *input = NULL;
    float *output = NULL;
    cudaError_t cuda_err = cudaSuccess;

    if(out_workspace_size_bytes == NULL || reduce_count == 0) {
        return cudaErrorInvalidValue;
    }
    if(op != GSX_IMPL_BINARY_REDUCE_OP_MSE && op != GSX_IMPL_BINARY_REDUCE_OP_MAE) {
        return cudaErrorInvalidValue;
    }
    cuda_err = cub::DeviceReduce::Sum(NULL, cub_workspace_size, input, output, reduce_count);
    if(cuda_err != cudaSuccess) {
        return cuda_err;
    }
    cub_workspace_aligned = gsx_cuda_round_up((gsx_size_t)cub_workspace_size, sizeof(float));
    transform_bytes = reduce_count * sizeof(float);
    total_bytes = cub_workspace_aligned + transform_bytes;
    *out_workspace_size_bytes = total_bytes;
    return cudaSuccess;
}

extern "C" cudaError_t gsx_cuda_unary_reduce_f32_launch(
    const float *input,
    float *output,
    void *workspace,
    gsx_size_t workspace_size_bytes,
    gsx_size_t outer_count,
    gsx_size_t reduce_count,
    gsx_impl_unary_reduce_op op,
    cudaStream_t stream
)
{
    gsx_size_t outer_index = 0;
    gsx_size_t required_workspace_size = 0;
    cudaError_t cuda_err = cudaSuccess;

    if(input == NULL || output == NULL || reduce_count == 0) {
        return cudaErrorInvalidValue;
    }
    cuda_err = gsx_cuda_unary_reduce_workspace_size_query(reduce_count, op, &required_workspace_size);
    if(cuda_err != cudaSuccess) {
        return cuda_err;
    }
    if(workspace_size_bytes < required_workspace_size) {
        return cudaErrorInvalidValue;
    }
    if(required_workspace_size != 0 && workspace == NULL) {
        return cudaErrorInvalidValue;
    }
    for(outer_index = 0; outer_index < outer_count; ++outer_index) {
        const float *slice_input = input + outer_index * reduce_count;
        float *slice_output = output + outer_index;
        size_t temp_storage_bytes = (size_t)required_workspace_size;

        switch(op) {
        case GSX_IMPL_UNARY_REDUCE_OP_SUM:
        case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
            cuda_err = cub::DeviceReduce::Sum(workspace, temp_storage_bytes, slice_input, slice_output, reduce_count, stream);
            break;
        case GSX_IMPL_UNARY_REDUCE_OP_MAX:
            cuda_err = cub::DeviceReduce::Max(workspace, temp_storage_bytes, slice_input, slice_output, reduce_count, stream);
            break;
        default:
            return cudaErrorInvalidValue;
        }
        if(cuda_err != cudaSuccess) {
            return cuda_err;
        }
    }
    if(op == GSX_IMPL_UNARY_REDUCE_OP_MEAN) {
        const int block_size = 256;
        int grid_size = gsx_cuda_reduce_compute_grid_size(outer_count, block_size);
        gsx_cuda_scale_outputs_kernel<<<grid_size, block_size, 0, stream>>>(output, outer_count, 1.0f / (float)reduce_count);
        cuda_err = cudaGetLastError();
        if(cuda_err != cudaSuccess) {
            return cuda_err;
        }
    }
    return cudaSuccess;
}

extern "C" cudaError_t gsx_cuda_binary_reduce_f32_launch(
    const float *lhs,
    const float *rhs,
    float *output,
    void *workspace,
    gsx_size_t workspace_size_bytes,
    gsx_size_t outer_count,
    gsx_size_t reduce_count,
    gsx_impl_binary_reduce_op op,
    cudaStream_t stream
)
{
    gsx_size_t outer_index = 0;
    gsx_size_t required_workspace_size = 0;
    gsx_size_t cub_workspace_size = 0;
    gsx_size_t cub_workspace_aligned = 0;
    char *workspace_bytes = (char *)workspace;
    float *transform_terms = NULL;
    cudaError_t cuda_err = cudaSuccess;

    if(lhs == NULL || rhs == NULL || output == NULL || reduce_count == 0) {
        return cudaErrorInvalidValue;
    }
    if(op != GSX_IMPL_BINARY_REDUCE_OP_MSE && op != GSX_IMPL_BINARY_REDUCE_OP_MAE) {
        return cudaErrorInvalidValue;
    }
    cuda_err = gsx_cuda_binary_reduce_workspace_size_query(reduce_count, op, &required_workspace_size);
    if(cuda_err != cudaSuccess) {
        return cuda_err;
    }
    if(workspace_size_bytes < required_workspace_size) {
        return cudaErrorInvalidValue;
    }
    if(required_workspace_size != 0 && workspace == NULL) {
        return cudaErrorInvalidValue;
    }
    cuda_err = gsx_cuda_unary_reduce_workspace_size_query(reduce_count, GSX_IMPL_UNARY_REDUCE_OP_SUM, &cub_workspace_size);
    if(cuda_err != cudaSuccess) {
        return cuda_err;
    }
    cub_workspace_aligned = gsx_cuda_round_up(cub_workspace_size, sizeof(float));
    transform_terms = (float *)(workspace_bytes + cub_workspace_aligned);

    for(outer_index = 0; outer_index < outer_count; ++outer_index) {
        const float *slice_lhs = lhs + outer_index * reduce_count;
        const float *slice_rhs = rhs + outer_index * reduce_count;
        float *slice_output = output + outer_index;
        size_t temp_storage_bytes = (size_t)cub_workspace_size;
        const int block_size = 256;
        int grid_size = gsx_cuda_reduce_compute_grid_size(reduce_count, block_size);
        gsx_cuda_prepare_binary_terms_kernel<<<grid_size, block_size, 0, stream>>>(
            slice_lhs, slice_rhs, transform_terms, reduce_count, op);
        cuda_err = cudaGetLastError();
        if(cuda_err != cudaSuccess) {
            return cuda_err;
        }
        cuda_err = cub::DeviceReduce::Sum(
            workspace,
            temp_storage_bytes,
            transform_terms,
            slice_output,
            reduce_count,
            stream
        );
        if(cuda_err != cudaSuccess) {
            return cuda_err;
        }
    }

    {
        const int block_size = 256;
        int grid_size = gsx_cuda_reduce_compute_grid_size(outer_count, block_size);
        gsx_cuda_scale_outputs_kernel<<<grid_size, block_size, 0, stream>>>(output, outer_count, 1.0f / (float)reduce_count);
        cuda_err = cudaGetLastError();
        if(cuda_err != cudaSuccess) {
            return cuda_err;
        }
    }
    return cudaSuccess;
}
