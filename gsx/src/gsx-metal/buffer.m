#include "internal.h"
#include "../pcg32.h"

#import <Metal/Metal.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *gsx_metal_backend_buffer_bytes(gsx_metal_backend_buffer *buffer)
{
    return (unsigned char *)[(id<MTLBuffer>)buffer->mtl_buffer contents];
}

static unsigned char *gsx_metal_backend_tensor_data(gsx_metal_backend_buffer *metal_buffer, const gsx_backend_tensor_view *tensor_view, gsx_size_t offset_bytes)
{
    return gsx_metal_backend_buffer_bytes(metal_buffer) + (size_t)(tensor_view->offset_bytes + offset_bytes);
}

static bool gsx_metal_backend_buffer_is_cpu_visible(gsx_metal_backend_buffer *buffer)
{
    return buffer->type_class != GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static MTLResourceOptions gsx_metal_backend_buffer_type_resource_options(gsx_backend_buffer_type_class type_class)
{
    switch(type_class) {
    case GSX_BACKEND_BUFFER_TYPE_DEVICE:
        return MTLResourceStorageModePrivate | MTLResourceCPUCacheModeDefaultCache;
    case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
        return MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
        return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    case GSX_BACKEND_BUFFER_TYPE_HOST:
        break;
    }
    return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
}

static gsx_error gsx_metal_backend_commit_copy(
    gsx_metal_backend *metal_backend,
    id<MTLBuffer> src_buffer,
    gsx_size_t src_offset_bytes,
    id<MTLBuffer> dst_buffer,
    gsx_size_t dst_offset_bytes,
    gsx_size_t byte_count
)
{
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit_encoder = nil;

    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_buffer == nil || dst_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "source and destination Metal buffers must be non-null");
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
    }

    blit_encoder = [command_buffer blitCommandEncoder];
    if(blit_encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
    }

    [blit_encoder
        copyFromBuffer:src_buffer
        sourceOffset:(NSUInteger)src_offset_bytes
        toBuffer:dst_buffer
        destinationOffset:(NSUInteger)dst_offset_bytes
        size:(NSUInteger)byte_count];
    [blit_encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_commit_fill(
    gsx_metal_backend *metal_backend,
    id<MTLBuffer> buffer,
    gsx_size_t offset_bytes,
    gsx_size_t byte_count,
    uint8_t value
)
{
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit_encoder = nil;

    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "Metal buffer must be non-null");
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
    }

    blit_encoder = [command_buffer blitCommandEncoder];
    if(blit_encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
    }

    [blit_encoder fillBuffer:buffer range:NSMakeRange((NSUInteger)offset_bytes, (NSUInteger)byte_count) value:value];
    [blit_encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_random_fill_f32_bytes(void *dst_bytes, gsx_size_t byte_count, uint64_t rng_state, uint64_t rng_inc)
{
    float *values = (float *)dst_bytes;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_pcg32 rng = { 0 };

    if(dst_bytes == NULL && byte_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when byte_count is non-zero");
    }
    if(byte_count % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    element_count = byte_count / sizeof(float);
    rng.state = rng_state;
    rng.inc = rng_inc;
    for(element_index = 0; element_index < element_count; ++element_index) {
        values[element_index] = pcg32_next_float(&rng);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_random_fill_f32_normal_bytes(
    void *dst_bytes,
    gsx_size_t byte_count,
    uint64_t rng_state,
    uint64_t rng_inc,
    gsx_float_t sigma
)
{
    float *values = (float *)dst_bytes;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_pcg32 rng = { 0 };

    if(dst_bytes == NULL && byte_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when byte_count is non-zero");
    }
    if(byte_count % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    element_count = byte_count / sizeof(float);
    rng.state = rng_state;
    rng.inc = rng_inc;
    for(element_index = 0; element_index < element_count; element_index += 2) {
        float u1 = pcg32_next_float(&rng);
        float u2 = pcg32_next_float(&rng);
        float radius = 0.0f;
        float theta = 0.0f;
        float z0 = 0.0f;
        float z1 = 0.0f;

        if(u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        radius = sqrtf(-2.0f * logf(u1));
        theta = 6.2831853071795864769f * u2;
        z0 = radius * cosf(theta);
        z1 = radius * sinf(theta);
        values[element_index] = z0 * sigma;
        if(element_index + 1 < element_count) {
            values[element_index + 1] = z1 * sigma;
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_random_fill_i32_bytes(
    void *dst_bytes,
    gsx_size_t byte_count,
    uint64_t rng_state,
    uint64_t rng_inc,
    uint32_t bound
)
{
    int32_t *values = (int32_t *)dst_bytes;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_pcg32 rng = { 0 };

    if(dst_bytes == NULL && byte_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when byte_count is non-zero");
    }
    if(byte_count % sizeof(int32_t) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "int32 tensor byte size must be divisible by sizeof(int32_t)");
    }

    element_count = byte_count / sizeof(int32_t);
    rng.state = rng_state;
    rng.inc = rng_inc;
    for(element_index = 0; element_index < element_count; ++element_index) {
        values[element_index] = (int32_t)pcg32_next_uint_bound(&rng, bound);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_metal_backend_buffer_prefers_gpu_compute(gsx_metal_backend_buffer *buffer)
{
    return buffer->type_class != GSX_BACKEND_BUFFER_TYPE_HOST_PINNED;
}

static gsx_error gsx_metal_backend_dispatch_tensor_unary(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_impl_unary_op op,
    const gsx_metal_tensor_unary_f32_params *params
)
{
    switch(op) {
    case GSX_IMPL_UNARY_OP_EXP:
        return gsx_metal_backend_dispatch_tensor_exp(backend, x_view, out_view, params);
    case GSX_IMPL_UNARY_OP_SIGMOID:
        return gsx_metal_backend_dispatch_tensor_sigmoid(backend, x_view, out_view, params);
    case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE:
        return gsx_metal_backend_dispatch_tensor_sigmoid_derivative(backend, x_view, out_view, params);
    case GSX_IMPL_UNARY_OP_ABS:
        return gsx_metal_backend_dispatch_tensor_abs(backend, x_view, out_view, params);
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
    }
}

static gsx_error gsx_metal_backend_apply_unary_f32_scalar(float x_value, gsx_impl_unary_op op, float *out_value)
{
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }

    switch(op) {
    case GSX_IMPL_UNARY_OP_EXP:
        *out_value = expf(x_value);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_IMPL_UNARY_OP_SIGMOID:
        *out_value = 1.0f / (1.0f + expf(-x_value));
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE: {
        float sigmoid_value = 1.0f / (1.0f + expf(-x_value));

        *out_value = sigmoid_value * (1.0f - sigmoid_value);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_IMPL_UNARY_OP_ABS:
        *out_value = fabsf(x_value);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
    }
}

static gsx_error gsx_metal_backend_buffer_apply_unary_reduce_tensor_f32(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
)
{
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    gsx_metal_tensor_unary_reduce_f32_params params = { 0 };
    const float *x_values = NULL;
    float *out_values = NULL;
    gsx_size_t outer_count = 0;
    gsx_size_t reduce_count = 0;
    gsx_size_t outer_index = 0;
    gsx_size_t reduce_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || workspace_view == NULL || x_shape == NULL || out_shape == NULL
        || x_view->buffer == NULL || out_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce buffers, views, and shapes must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce tensors must belong to the same backend");
    }
    if(workspace_view->buffer != NULL && workspace_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "workspace tensor must belong to the same backend");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x_view and out_view data_type must match");
    }
    if(x_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal unary_reduce only supports float32 tensors");
    }

    error = gsx_metal_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(workspace_view->buffer != NULL) {
        error = gsx_metal_backend_buffer_check_range(
            workspace_view->buffer, workspace_view->offset_bytes, workspace_view->size_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(workspace_view->offset_bytes != 0 || workspace_view->size_bytes != 0) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "workspace view must have zero offset/size when workspace buffer is null");
    }
    error = gsx_metal_backend_reduce_validate_shape_contract(
        x_view, out_view, x_rank, x_shape, out_rank, out_shape, start_axis, &outer_count, &reduce_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(dst_buffer);
    if(!gsx_metal_backend_buffer_prefers_gpu_compute(x_buffer) && !gsx_metal_backend_buffer_prefers_gpu_compute(out_buffer)) {
        x_values = (const float *)gsx_metal_backend_tensor_data(x_buffer, x_view, 0);
        out_values = (float *)gsx_metal_backend_tensor_data(out_buffer, out_view, 0);
        for(outer_index = 0; outer_index < outer_count; ++outer_index) {
            gsx_size_t base_index = outer_index * reduce_count;
            float accum = 0.0f;

            if(op == GSX_IMPL_UNARY_REDUCE_OP_MAX) {
                accum = x_values[base_index];
            }
            for(reduce_index = 0; reduce_index < reduce_count; ++reduce_index) {
                float value = x_values[base_index + reduce_index];

                switch(op) {
                case GSX_IMPL_UNARY_REDUCE_OP_SUM:
                case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
                    accum += value;
                    break;
                case GSX_IMPL_UNARY_REDUCE_OP_MAX:
                    if(value > accum) {
                        accum = value;
                    }
                    break;
                default:
                    return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary_reduce op");
                }
            }
            if(op == GSX_IMPL_UNARY_REDUCE_OP_MEAN) {
                accum /= (float)reduce_count;
            }
            out_values[outer_index] = accum;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(outer_count > UINT32_MAX || reduce_count > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce launch parameters exceed Metal kernel limits");
    }
    switch(op) {
    case GSX_IMPL_UNARY_REDUCE_OP_SUM:
    case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
    case GSX_IMPL_UNARY_REDUCE_OP_MAX:
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary_reduce op");
    }

    params.outer_count = (uint32_t)outer_count;
    params.reduce_count = (uint32_t)reduce_count;
    return gsx_metal_backend_dispatch_tensor_unary_reduce_f32(dst_buffer->buffer_type->backend, x_view, out_view, &params, op);
}

static gsx_error gsx_metal_backend_buffer_apply_binary_reduce_tensor_f32(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
)
{
    gsx_metal_backend_buffer *lhs_buffer = NULL;
    gsx_metal_backend_buffer *rhs_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    gsx_metal_tensor_binary_reduce_f32_params params = { 0 };
    const float *lhs_values = NULL;
    const float *rhs_values = NULL;
    float *out_values = NULL;
    gsx_size_t outer_count_lhs = 0;
    gsx_size_t reduce_count_lhs = 0;
    gsx_size_t outer_count_rhs = 0;
    gsx_size_t reduce_count_rhs = 0;
    gsx_size_t outer_index = 0;
    gsx_size_t reduce_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || lhs_view == NULL || rhs_view == NULL || out_view == NULL || workspace_view == NULL || lhs_shape == NULL
        || rhs_shape == NULL || out_shape == NULL || lhs_view->buffer == NULL || rhs_view->buffer == NULL || out_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce buffers, views, and shapes must be non-null");
    }
    if(rhs_rank != lhs_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs_rank and rhs_rank must match");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(lhs_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || rhs_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce tensors must belong to the same backend");
    }
    if(workspace_view->buffer != NULL && workspace_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "workspace tensor must belong to the same backend");
    }
    if(lhs_view->data_type != rhs_view->data_type || lhs_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce tensor data_type must match");
    }
    if(lhs_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal binary_reduce only supports float32 tensors");
    }

    error = gsx_metal_backend_tensor_view_validate(lhs_view->buffer, lhs_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(rhs_view->buffer, rhs_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(workspace_view->buffer != NULL) {
        error = gsx_metal_backend_buffer_check_range(
            workspace_view->buffer, workspace_view->offset_bytes, workspace_view->size_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(workspace_view->offset_bytes != 0 || workspace_view->size_bytes != 0) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "workspace view must have zero offset/size when workspace buffer is null");
    }
    error = gsx_metal_backend_reduce_validate_shape_contract(
        lhs_view, out_view, lhs_rank, lhs_shape, out_rank, out_shape, start_axis, &outer_count_lhs, &reduce_count_lhs);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_reduce_validate_shape_contract(
        rhs_view, out_view, rhs_rank, rhs_shape, out_rank, out_shape, start_axis, &outer_count_rhs, &reduce_count_rhs);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(outer_count_lhs != outer_count_rhs || reduce_count_lhs != reduce_count_rhs) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs and rhs reduce metadata must match");
    }

    lhs_buffer = gsx_metal_backend_buffer_from_base(lhs_view->buffer);
    rhs_buffer = gsx_metal_backend_buffer_from_base(rhs_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(dst_buffer);
    if(!gsx_metal_backend_buffer_prefers_gpu_compute(lhs_buffer) && !gsx_metal_backend_buffer_prefers_gpu_compute(rhs_buffer)
        && !gsx_metal_backend_buffer_prefers_gpu_compute(out_buffer)) {
        lhs_values = (const float *)gsx_metal_backend_tensor_data(lhs_buffer, lhs_view, 0);
        rhs_values = (const float *)gsx_metal_backend_tensor_data(rhs_buffer, rhs_view, 0);
        out_values = (float *)gsx_metal_backend_tensor_data(out_buffer, out_view, 0);
        for(outer_index = 0; outer_index < outer_count_lhs; ++outer_index) {
            gsx_size_t base_index = outer_index * reduce_count_lhs;
            float accum = 0.0f;

            for(reduce_index = 0; reduce_index < reduce_count_lhs; ++reduce_index) {
                float diff = lhs_values[base_index + reduce_index] - rhs_values[base_index + reduce_index];

                switch(op) {
                case GSX_IMPL_BINARY_REDUCE_OP_MSE:
                    accum += diff * diff;
                    break;
                case GSX_IMPL_BINARY_REDUCE_OP_MAE:
                    accum += fabsf(diff);
                    break;
                default:
                    return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown binary_reduce op");
                }
            }
            out_values[outer_index] = accum / (float)reduce_count_lhs;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(outer_count_lhs > UINT32_MAX || reduce_count_lhs > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce launch parameters exceed Metal kernel limits");
    }
    switch(op) {
    case GSX_IMPL_BINARY_REDUCE_OP_MSE:
    case GSX_IMPL_BINARY_REDUCE_OP_MAE:
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown binary_reduce op");
    }

    params.outer_count = (uint32_t)outer_count_lhs;
    params.reduce_count = (uint32_t)reduce_count_lhs;
    return gsx_metal_backend_dispatch_tensor_binary_reduce_f32(
        dst_buffer->buffer_type->backend, lhs_view, rhs_view, out_view, &params, op);
}

static gsx_error gsx_metal_backend_buffer_apply_unary_tensor_f32(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
)
{
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    gsx_metal_tensor_unary_f32_params params = { 0 };
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    const float *x_values = NULL;
    float *out_values = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    (void)rank;
    (void)shape;

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || shape == NULL || x_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shape must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must belong to the same backend");
    }
    if(x_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op only supports float32 tensors on metal backend");
    }

    error = gsx_metal_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(x_view->size_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out view byte sizes must match");
    }
    if(x_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(dst_buffer);
    element_count = x_view->size_bytes / sizeof(float);
    if(element_count > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "unary element count exceeds Metal kernel limits");
    }

    if(!gsx_metal_backend_buffer_prefers_gpu_compute(x_buffer) && !gsx_metal_backend_buffer_prefers_gpu_compute(out_buffer)) {
        x_values = (const float *)gsx_metal_backend_tensor_data(x_buffer, x_view, 0);
        out_values = (float *)gsx_metal_backend_tensor_data(out_buffer, out_view, 0);
        for(element_index = 0; element_index < element_count; ++element_index) {
            error = gsx_metal_backend_apply_unary_f32_scalar(x_values[element_index], op, &out_values[element_index]);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    params.element_count = (uint32_t)element_count;
    return gsx_metal_backend_dispatch_tensor_unary(dst_buffer->buffer_type->backend, x_view, out_view, op, &params);
}

static gsx_error gsx_metal_backend_buffer_apply_unary_inplace_tensor_f32(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
)
{
    gsx_metal_backend_buffer *metal_buffer = NULL;
    gsx_metal_tensor_unary_f32_params params = { 0 };
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    float *values = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer == NULL || tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference buffer");
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op only supports float32 tensors on metal backend");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    element_count = tensor_view->size_bytes / sizeof(float);
    if(element_count > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "unary element count exceeds Metal kernel limits");
    }

    if(!gsx_metal_backend_buffer_prefers_gpu_compute(metal_buffer)) {
        values = (float *)gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
        for(element_index = 0; element_index < element_count; ++element_index) {
            error = gsx_metal_backend_apply_unary_f32_scalar(values[element_index], op, &values[element_index]);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    params.element_count = (uint32_t)element_count;
    return gsx_metal_backend_dispatch_tensor_unary(buffer->buffer_type->backend, tensor_view, tensor_view, op, &params);
}

gsx_error gsx_metal_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info)
{
    gsx_metal_backend_buffer_type *metal_buffer_type = gsx_metal_backend_buffer_type_from_base(buffer_type);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    *out_info = metal_buffer_type->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes)
{
    gsx_metal_backend_buffer_type *metal_buffer_type = gsx_metal_backend_buffer_type_from_base(buffer_type);

    if(out_alloc_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_alloc_size_bytes must be non-null");
    }

    if(gsx_round_up_overflows(requested_size_bytes, metal_buffer_type->info.alignment_bytes, out_alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer)
{
    gsx_metal_backend_buffer_type *metal_buffer_type = gsx_metal_backend_buffer_type_from_base(buffer_type);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer_type->backend);
    gsx_metal_backend_buffer *metal_buffer = NULL;
    gsx_size_t alloc_size_bytes = 0;
    gsx_size_t effective_alignment = 0;
    MTLResourceOptions resource_options = MTLResourceStorageModeShared;
    id<MTLBuffer> mtl_buffer = nil;

    if(out_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer must be non-null");
    }
    *out_buffer = NULL;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->size_bytes must be non-zero");
    }

    effective_alignment = metal_buffer_type->info.alignment_bytes;
    if(desc->alignment_bytes > effective_alignment) {
        effective_alignment = desc->alignment_bytes;
    }

    if(gsx_round_up_overflows(desc->size_bytes, effective_alignment, &alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }

    resource_options = gsx_metal_backend_buffer_type_resource_options(metal_buffer_type->info.type);

    metal_buffer = (gsx_metal_backend_buffer *)calloc(1, sizeof(*metal_buffer));
    if(metal_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate buffer struct");
    }

    mtl_buffer = [(id<MTLDevice>)metal_backend->mtl_device
        newBufferWithLength:(NSUInteger)alloc_size_bytes
        options:resource_options];
    if(mtl_buffer == nil) {
        free(metal_buffer);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal buffer");
    }

    metal_buffer->base.iface = &gsx_metal_backend_buffer_iface;
    metal_buffer->base.buffer_type = buffer_type;
    metal_buffer->base.size_bytes = desc->size_bytes;
    metal_buffer->base.alignment_bytes = effective_alignment;
    metal_buffer->mtl_buffer = mtl_buffer;
    metal_buffer->alloc_size_bytes = alloc_size_bytes;
    metal_buffer->type_class = metal_buffer_type->info.type;
    metal_buffer->resource_options = (uint32_t)resource_options;

    metal_backend->base.live_buffer_count += 1;
    *out_buffer = &metal_buffer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_free(gsx_backend_buffer_t buffer)
{
    gsx_metal_backend_buffer *metal_buffer = NULL;
    gsx_metal_backend *metal_backend = NULL;

    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }
    metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);

    if(metal_backend->base.live_buffer_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_buffer_count underflow in buffer free");
    }

    if(metal_buffer->mtl_buffer != NULL) {
        [(id<MTLBuffer>)metal_buffer->mtl_buffer release];
        metal_buffer->mtl_buffer = NULL;
    }

    metal_backend->base.live_buffer_count -= 1;
    free(metal_buffer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info)
{
    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->backend = buffer->buffer_type->backend;
    out_info->buffer_type = buffer->buffer_type;
    out_info->size_bytes = buffer->size_bytes;
    out_info->alignment_bytes = buffer->alignment_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_get_native_handle(gsx_backend_buffer_t buffer, void **out_handle)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);

    if(out_handle == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_handle must be non-null");
    }

    *out_handle = metal_buffer->mtl_buffer;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    unsigned char *dst_ptr = NULL;

    error = gsx_metal_backend_buffer_check_range(buffer, dst_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero byte_count");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        dst_ptr = gsx_metal_backend_buffer_bytes(metal_buffer);
        if(dst_ptr == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal buffer contents are unavailable");
        }

        memcpy(dst_ptr + (size_t)dst_offset_bytes, src_bytes, (size_t)byte_count);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithBytes:src_bytes
            length:(NSUInteger)byte_count
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal upload staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:staging_buffer
            sourceOffset:0
            toBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            destinationOffset:(NSUInteger)dst_offset_bytes
            size:(NSUInteger)byte_count];
        [blit_encoder endEncoding];
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
            (void)completed_buffer;
            [staging_buffer release];
        }];
        [command_buffer commit];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    const unsigned char *src_ptr = NULL;

    error = gsx_metal_backend_buffer_check_range(buffer, src_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero byte_count");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        src_ptr = gsx_metal_backend_buffer_bytes(metal_buffer);
        if(src_ptr == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal buffer contents are unavailable");
        }

        memcpy(dst_bytes, src_ptr + (size_t)src_offset_bytes, (size_t)byte_count);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithLength:(NSUInteger)byte_count
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal download staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            sourceOffset:(NSUInteger)src_offset_bytes
            toBuffer:staging_buffer
            destinationOffset:0
            size:(NSUInteger)byte_count];
        [blit_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        {
            const void *staging_ptr = [staging_buffer contents];

            if(staging_ptr != NULL) {
                memcpy(dst_bytes, staging_ptr, (size_t)byte_count);
            }
        }
        [staging_buffer release];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_set_zero(gsx_backend_buffer_t buffer)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);

    if(metal_buffer->alloc_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        unsigned char *bytes = gsx_metal_backend_buffer_bytes(metal_buffer);

        if(bytes == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal buffer contents are unavailable");
        }

        memset(bytes, 0, (size_t)metal_buffer->alloc_size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_metal_backend_commit_fill(
        metal_backend,
        (id<MTLBuffer>)metal_buffer->mtl_buffer,
        0,
        metal_buffer->alloc_size_bytes,
        0
    );
}

gsx_error gsx_metal_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = gsx_metal_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);

    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        memset(gsx_metal_backend_tensor_data(metal_buffer, tensor_view, offset_bytes), value, (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_metal_backend_commit_fill(
        metal_backend,
        (id<MTLBuffer>)metal_buffer->mtl_buffer,
        tensor_view->offset_bytes + offset_bytes,
        size_bytes,
        value
    );
}

gsx_error gsx_metal_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(size_bytes != 0 && src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null when size_bytes is non-zero");
    }

    error = gsx_metal_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);
    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        memcpy(gsx_metal_backend_tensor_data(metal_buffer, tensor_view, offset_bytes), src_bytes, (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithBytes:src_bytes
            length:(NSUInteger)size_bytes
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal tensor upload staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:staging_buffer
            sourceOffset:0
            toBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            destinationOffset:(NSUInteger)(tensor_view->offset_bytes + offset_bytes)
            size:(NSUInteger)size_bytes];
        [blit_encoder endEncoding];
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
            (void)completed_buffer;
            [staging_buffer release];
        }];
        [command_buffer commit];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(size_bytes != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when size_bytes is non-zero");
    }

    error = gsx_metal_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);
    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        memcpy(dst_bytes, gsx_metal_backend_tensor_data(metal_buffer, tensor_view, offset_bytes), (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithLength:(NSUInteger)size_bytes
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal tensor download staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            sourceOffset:(NSUInteger)(tensor_view->offset_bytes + offset_bytes)
            toBuffer:staging_buffer
            destinationOffset:0
            size:(NSUInteger)size_bytes];
        [blit_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        {
            const void *staging_ptr = [staging_buffer contents];

            if(staging_ptr != NULL) {
                memcpy(dst_bytes, staging_ptr, (size_t)size_bytes);
            }
        }
        [staging_buffer release];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_copy_tensor(gsx_backend_buffer_t dst_buffer, const gsx_backend_tensor_view *src_view, const gsx_backend_tensor_view *dst_view)
{
    gsx_metal_backend_buffer *src_metal_buffer = NULL;
    gsx_metal_backend_buffer *dst_metal_buffer = NULL;
    gsx_metal_backend *metal_backend = NULL;
    gsx_size_t src_begin_bytes = 0;
    gsx_size_t src_end_bytes = 0;
    gsx_size_t dst_begin_bytes = 0;
    gsx_size_t dst_end_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || src_view == NULL || dst_view == NULL || src_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and tensor views must be non-null");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend || dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy requires source and destination to belong to the same backend");
    }

    error = gsx_metal_backend_tensor_view_validate(src_view->buffer, src_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, dst_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(src_view->size_bytes != dst_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy requires equal source and destination sizes");
    }
    if(src_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    src_begin_bytes = src_view->offset_bytes;
    src_end_bytes = src_view->offset_bytes + src_view->size_bytes;
    dst_begin_bytes = dst_view->offset_bytes;
    dst_end_bytes = dst_view->offset_bytes + dst_view->size_bytes;
    if(src_view->buffer == dst_buffer) {
        if(src_begin_bytes == dst_begin_bytes && src_end_bytes == dst_end_bytes) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        if(!(dst_end_bytes <= src_begin_bytes || src_end_bytes <= dst_begin_bytes)) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy rejects overlapping source and destination ranges");
        }
    }

    src_metal_buffer = gsx_metal_backend_buffer_from_base(src_view->buffer);
    dst_metal_buffer = gsx_metal_backend_buffer_from_base(dst_buffer);
    metal_backend = gsx_metal_backend_from_base(dst_buffer->buffer_type->backend);
    if(gsx_metal_backend_buffer_is_cpu_visible(src_metal_buffer) && gsx_metal_backend_buffer_is_cpu_visible(dst_metal_buffer)) {
        error = gsx_metal_backend_major_stream_sync(&metal_backend->base);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        memcpy(
            gsx_metal_backend_buffer_bytes(dst_metal_buffer) + (size_t)dst_begin_bytes,
            gsx_metal_backend_buffer_bytes(src_metal_buffer) + (size_t)src_begin_bytes,
            (size_t)src_view->size_bytes
        );
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_metal_backend_commit_copy(
        gsx_metal_backend_from_base(dst_buffer->buffer_type->backend),
        (id<MTLBuffer>)src_metal_buffer->mtl_buffer,
        src_begin_bytes,
        (id<MTLBuffer>)dst_metal_buffer->mtl_buffer,
        dst_begin_bytes,
        src_view->size_bytes
    );
}

gsx_error gsx_metal_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    unsigned char *dst_bytes = NULL;
    gsx_size_t offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(value_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_size_bytes must be non-zero");
    }
    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->size_bytes != 0 && value_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_bytes must be non-null when tensor is non-empty");
    }
    if(tensor_view->size_bytes % value_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of value_size_bytes");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error) || tensor_view->size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        dst_bytes = gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
        for(offset_bytes = 0; offset_bytes < tensor_view->size_bytes; offset_bytes += value_size_bytes) {
            memcpy(dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(value_size_bytes == 1) {
        return gsx_metal_backend_commit_fill(
            metal_backend,
            (id<MTLBuffer>)metal_buffer->mtl_buffer,
            tensor_view->offset_bytes,
            tensor_view->size_bytes,
            *(const uint8_t *)value_bytes
        );
    }

    return gsx_make_error(
        GSX_ERROR_NOT_SUPPORTED,
        "fill_tensor on Metal device buffers currently supports value_size_bytes == 1 only without explicit synchronization APIs"
    );
}

gsx_error gsx_metal_backend_buffer_fill_rand_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_tensor_rand_f32_params params = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "rand fill only supports float32 tensors on metal backend");
    }
    if(tensor_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        return gsx_metal_backend_random_fill_f32_bytes(
            gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0),
            tensor_view->size_bytes,
            rng_state,
            rng_inc
        );
    }

    if(tensor_view->size_bytes / sizeof(float) > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "rand element count exceeds Metal kernel limits");
    }

    params.rng_state = rng_state;
    params.rng_inc = rng_inc;
    params.element_count = (uint32_t)(tensor_view->size_bytes / sizeof(float));
    return gsx_metal_backend_dispatch_tensor_rand_f32(buffer->buffer_type->backend, tensor_view, &params);
}

gsx_error gsx_metal_backend_buffer_fill_randn_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc,
    gsx_float_t sigma
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_tensor_randn_f32_params params = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "randn fill only supports float32 tensors on metal backend");
    }
    if(tensor_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        return gsx_metal_backend_random_fill_f32_normal_bytes(
            gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0),
            tensor_view->size_bytes,
            rng_state,
            rng_inc,
            sigma
        );
    }

    if(tensor_view->size_bytes / sizeof(float) > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "randn element count exceeds Metal kernel limits");
    }

    params.rng_state = rng_state;
    params.rng_inc = rng_inc;
    params.sigma = sigma;
    params.element_count = (uint32_t)(tensor_view->size_bytes / sizeof(float));
    return gsx_metal_backend_dispatch_tensor_randn_f32(buffer->buffer_type->backend, tensor_view, &params);
}

gsx_error gsx_metal_backend_buffer_fill_randint_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc,
    uint32_t bound
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_tensor_randint_i32_params params = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "randint fill only supports int32 tensors on metal backend");
    }
    if(tensor_view->size_bytes % sizeof(int32_t) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "int32 tensor byte size must be divisible by sizeof(int32_t)");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        return gsx_metal_backend_random_fill_i32_bytes(
            gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0),
            tensor_view->size_bytes,
            rng_state,
            rng_inc,
            bound
        );
    }

    if(tensor_view->size_bytes / sizeof(int32_t) > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "randint element count exceeds Metal kernel limits");
    }

    params.rng_state = rng_state;
    params.rng_inc = rng_inc;
    params.bound = bound;
    params.element_count = (uint32_t)(tensor_view->size_bytes / sizeof(int32_t));
    return gsx_metal_backend_dispatch_tensor_randint_i32(buffer->buffer_type->backend, tensor_view, &params);
}

gsx_error gsx_metal_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    const unsigned char *bytes = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_is_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_is_finite must be non-null");
    }
    *out_is_finite = true;

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        gsx_metal_tensor_check_finite_params params = { 0 };
        uint32_t has_non_finite = 0;
        gsx_size_t element_size = 0;

        switch(tensor_view->data_type) {
        case GSX_DATA_TYPE_F32:
            element_size = 4;
            break;
        case GSX_DATA_TYPE_F16:
        case GSX_DATA_TYPE_BF16:
            element_size = 2;
            break;
        default:
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "check_finite only supports floating point types");
        }
        if(tensor_view->size_bytes % element_size != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of the checked element size");
        }
        if(tensor_view->size_bytes > UINT32_MAX) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor size exceeds Metal kernel limits");
        }

        params.element_count = (uint32_t)(tensor_view->size_bytes / element_size);

        switch(tensor_view->data_type) {
        case GSX_DATA_TYPE_F32:
            error = gsx_metal_backend_dispatch_tensor_check_finite_f32(
                buffer->buffer_type->backend, tensor_view, &params, &has_non_finite);
            break;
        case GSX_DATA_TYPE_F16:
            error = gsx_metal_backend_dispatch_tensor_check_finite_f16(
                buffer->buffer_type->backend, tensor_view, &params, &has_non_finite);
            break;
        case GSX_DATA_TYPE_BF16:
            error = gsx_metal_backend_dispatch_tensor_check_finite_bf16(
                buffer->buffer_type->backend, tensor_view, &params, &has_non_finite);
            break;
        default:
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "check_finite only supports floating point types");
        }

        if(!gsx_error_is_success(error)) {
            return error;
        }
        *out_is_finite = (has_non_finite == 0);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    bytes = gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32: {
        const float *values = (const float *)bytes;

        if(tensor_view->size_bytes % sizeof(float) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensors must have a byte size divisible by sizeof(float)");
        }
        element_count = tensor_view->size_bytes / sizeof(float);
        for(element_index = 0; element_index < element_count; ++element_index) {
            if(!isfinite((double)values[element_index])) {
                *out_is_finite = false;
                return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_F16: {
        const uint16_t *values = (const uint16_t *)bytes;

        if(tensor_view->size_bytes % sizeof(uint16_t) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float16 tensors must have a byte size divisible by sizeof(uint16_t)");
        }
        element_count = tensor_view->size_bytes / sizeof(uint16_t);
        for(element_index = 0; element_index < element_count; ++element_index) {
            if(!gsx_metal_backend_f16_is_finite(values[element_index])) {
                *out_is_finite = false;
                return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_BF16: {
        const uint16_t *values = (const uint16_t *)bytes;

        if(tensor_view->size_bytes % sizeof(uint16_t) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "bfloat16 tensors must have a byte size divisible by sizeof(uint16_t)");
        }
        element_count = tensor_view->size_bytes / sizeof(uint16_t);
        for(element_index = 0; element_index < element_count; ++element_index) {
            if(!gsx_metal_backend_bf16_is_finite(values[element_index])) {
                *out_is_finite = false;
                return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "finite check supports f32/f16/bf16 only");
    }
}

gsx_error gsx_metal_backend_buffer_gather_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape
)
{
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_index_bytes = 0;
    gsx_size_t x_row_bytes = 0;
    gsx_size_t row_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_metal_tensor_gather_params params = { 0 };

    if(dst_buffer == NULL || x_view == NULL || index_view == NULL || out_view == NULL || x_shape == NULL || out_shape == NULL
        || x_view->buffer == NULL || index_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shapes must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || index_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "all gather tensors must belong to the same backend");
    }

    error = gsx_metal_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(index_view->buffer, index_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    row_count = (gsx_size_t)out_shape[0];
    if(row_count == 0 || x_rank < 1 || out_rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gather shape metadata is invalid");
    }
    if(out_view->size_bytes % row_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gather output view byte size must be row-aligned");
    }
    x_row_bytes = out_view->size_bytes / row_count;
    if(gsx_size_mul_overflows((gsx_size_t)x_shape[0], x_row_bytes, &expected_x_bytes) || expected_x_bytes != x_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gather x view byte size is inconsistent with row metadata");
    }
    if(gsx_size_mul_overflows(row_count, sizeof(int32_t), &expected_index_bytes) || expected_index_bytes != index_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index view byte size must match out leading dimension");
    }

    if((gsx_size_t)x_shape[0] > UINT32_MAX || row_count > UINT32_MAX || x_row_bytes > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather launch parameters exceed Metal kernel limits");
    }

    params.x_row_count = (uint32_t)x_shape[0];
    params.out_row_count = (uint32_t)row_count;
    params.row_bytes = (uint32_t)x_row_bytes;
    return gsx_metal_backend_dispatch_tensor_gather(
        dst_buffer->buffer_type->backend,
        x_view,
        index_view,
        out_view,
        &params
    );
}

gsx_error gsx_metal_backend_buffer_unary_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
)
{
    return gsx_metal_backend_buffer_apply_unary_tensor_f32(dst_buffer, x_view, out_view, rank, shape, op);
}

gsx_error gsx_metal_backend_buffer_unary_tensor_inplace(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
)
{
    return gsx_metal_backend_buffer_apply_unary_inplace_tensor_f32(buffer, tensor_view, op);
}

gsx_error gsx_metal_backend_buffer_unary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
)
{
    return gsx_metal_backend_buffer_apply_unary_reduce_tensor_f32(
        dst_buffer,
        x_view,
        out_view,
        workspace_view,
        x_rank,
        x_shape,
        out_rank,
        out_shape,
        start_axis,
        op
    );
}

gsx_error gsx_metal_backend_buffer_binary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
)
{
    return gsx_metal_backend_buffer_apply_binary_reduce_tensor_f32(
        dst_buffer,
        lhs_view,
        rhs_view,
        out_view,
        workspace_view,
        lhs_rank,
        lhs_shape,
        rhs_rank,
        rhs_shape,
        out_rank,
        out_shape,
        start_axis,
        op
    );
}

gsx_error gsx_metal_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
)
{
    gsx_metal_backend_buffer *metal_buffer = NULL;
    gsx_size_t element_size_bytes = 0;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer == NULL || tensor_view == NULL || min_value == NULL || max_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor_view, min_value, and max_value must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference buffer");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_data_type_get_size_bytes(tensor_view->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes % element_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor view byte size is not aligned to element size");
    }

    metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    element_count = tensor_view->size_bytes / element_size_bytes;

    if(!gsx_metal_backend_buffer_prefers_gpu_compute(metal_buffer)) {
        switch(tensor_view->data_type) {
        case GSX_DATA_TYPE_F32: {
            float *values = (float *)gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
            const float min_bound = *(const float *)min_value;
            const float max_bound = *(const float *)max_value;

            for(element_index = 0; element_index < element_count; ++element_index) {
                if(values[element_index] < min_bound) {
                    values[element_index] = min_bound;
                } else if(values[element_index] > max_bound) {
                    values[element_index] = max_bound;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        case GSX_DATA_TYPE_I32: {
            int32_t *values = (int32_t *)gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
            const int32_t min_bound = *(const int32_t *)min_value;
            const int32_t max_bound = *(const int32_t *)max_value;

            for(element_index = 0; element_index < element_count; ++element_index) {
                if(values[element_index] < min_bound) {
                    values[element_index] = min_bound;
                } else if(values[element_index] > max_bound) {
                    values[element_index] = max_bound;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        default:
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp_inplace only supports f32 and i32 tensors on metal backend");
        }
    }

    if(element_count > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "clamp_inplace element count exceeds Metal kernel limits");
    }

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32: {
        gsx_metal_tensor_clamp_f32_params params = {
            *(const float *)min_value,
            *(const float *)max_value,
            (uint32_t)element_count
        };

        return gsx_metal_backend_dispatch_tensor_clamp_f32_inplace(buffer->buffer_type->backend, tensor_view, &params);
    }
    case GSX_DATA_TYPE_I32: {
        gsx_metal_tensor_clamp_i32_params params = {
            *(const int32_t *)min_value,
            *(const int32_t *)max_value,
            (uint32_t)element_count
        };

        return gsx_metal_backend_dispatch_tensor_clamp_i32_inplace(buffer->buffer_type->backend, tensor_view, &params);
    }
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp_inplace only supports f32 and i32 tensors on metal backend");
    }
}
