#include "../objc-helpers.h"

// TODO: vectorized kernels for wider data types and better performance

extern const char gsx_metal_tensor_metallib_start[];
extern const char gsx_metal_tensor_metallib_end[];

static gsx_error gsx_metal_backend_ensure_tensor_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->tensor_library,
        gsx_metal_tensor_metallib_start,
        gsx_metal_tensor_metallib_end,
        "embedded Metal tensor metallib is empty",
        "failed to create dispatch data for embedded Metal tensor metallib",
        "failed to load embedded Metal tensor metallib",
        out_library);
}

static gsx_error gsx_metal_backend_ensure_tensor_gather_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_gather_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_gather_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_exp_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_exp_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_exp_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_sigmoid_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_sigmoid_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_sigmoid_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_sigmoid_derivative_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_sigmoid_derivative_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_sigmoid_derivative_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_abs_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_abs_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_abs_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_clamp_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_clamp_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_clamp_f32_inplace_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_clamp_i32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_clamp_i32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_clamp_i32_inplace_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

gsx_error gsx_metal_backend_dispatch_tensor_gather(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_gather_params *params
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *index_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLBuffer> status_buffer = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    NSUInteger total_bytes = 0;
    uint32_t *status_ptr = NULL;

    if(backend == NULL || x_view == NULL || index_view == NULL || out_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, and params must be non-null");
    }
    if(params->out_row_count == 0 || params->row_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    index_buffer = gsx_metal_backend_buffer_from_base(index_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_view->buffer);

    error = gsx_metal_backend_ensure_tensor_gather_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    status_buffer = [(id<MTLDevice>)metal_backend->mtl_device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
    if(status_buffer == nil) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal gather status buffer");
    }
    status_ptr = (uint32_t *)[status_buffer contents];
    if(status_ptr == NULL) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to access Metal gather status buffer contents");
    }
    *status_ptr = 0;

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        [status_buffer release];
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)x_buffer->mtl_buffer offset:(NSUInteger)x_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)index_buffer->mtl_buffer offset:(NSUInteger)index_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_view->offset_bytes atIndex:2];
    [encoder setBytes:params length:sizeof(*params) atIndex:3];
    [encoder setBuffer:status_buffer offset:0 atIndex:4];

    total_bytes = (NSUInteger)params->out_row_count * (NSUInteger)params->row_bytes;
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, total_bytes);

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if(*status_ptr != 0) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather index is out of range");
    }

    [status_buffer release];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_dispatch_tensor_unary_f32_with_pipeline(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || x_view == NULL || out_view == NULL || params == NULL || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, params, and pipeline helper must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_view->buffer);

    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)x_buffer->mtl_buffer offset:(NSUInteger)x_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_view->offset_bytes atIndex:1];
    [encoder setBytes:params length:sizeof(*params) atIndex:2];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_tensor_exp(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
)
{
    return gsx_metal_backend_dispatch_tensor_unary_f32_with_pipeline(
        backend,
        x_view,
        out_view,
        params,
        gsx_metal_backend_ensure_tensor_exp_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_sigmoid(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
)
{
    return gsx_metal_backend_dispatch_tensor_unary_f32_with_pipeline(
        backend,
        x_view,
        out_view,
        params,
        gsx_metal_backend_ensure_tensor_sigmoid_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_sigmoid_derivative(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
)
{
    return gsx_metal_backend_dispatch_tensor_unary_f32_with_pipeline(
        backend,
        x_view,
        out_view,
        params,
        gsx_metal_backend_ensure_tensor_sigmoid_derivative_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_abs(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
)
{
    return gsx_metal_backend_dispatch_tensor_unary_f32_with_pipeline(
        backend,
        x_view,
        out_view,
        params,
        gsx_metal_backend_ensure_tensor_abs_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_clamp_f32_inplace(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_clamp_f32_params *params
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *metal_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tensor_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor_view, and params must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    metal_buffer = gsx_metal_backend_buffer_from_base(tensor_view->buffer);

    error = gsx_metal_backend_ensure_tensor_clamp_f32_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer offset:(NSUInteger)tensor_view->offset_bytes atIndex:0];
    [encoder setBytes:params length:sizeof(*params) atIndex:1];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_tensor_clamp_i32_inplace(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_clamp_i32_params *params
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *metal_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tensor_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor_view, and params must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    metal_buffer = gsx_metal_backend_buffer_from_base(tensor_view->buffer);

    error = gsx_metal_backend_ensure_tensor_clamp_i32_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer offset:(NSUInteger)tensor_view->offset_bytes atIndex:0];
    [encoder setBytes:params length:sizeof(*params) atIndex:1];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
