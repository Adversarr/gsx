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

static gsx_error gsx_metal_backend_ensure_image_linear_to_srgb_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->image_linear_to_srgb_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_image_linear_to_srgb_f32_kernel",
        "failed to look up Metal image kernel function",
        "failed to create Metal image pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_image_srgb_to_linear_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->image_srgb_to_linear_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_image_srgb_to_linear_f32_kernel",
        "failed to look up Metal image kernel function",
        "failed to create Metal image pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_image_chw_to_hwc_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->image_chw_to_hwc_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_image_chw_to_hwc_kernel",
        "failed to look up Metal image kernel function",
        "failed to create Metal image pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_image_hwc_to_chw_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->image_hwc_to_chw_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_image_hwc_to_chw_kernel",
        "failed to look up Metal image kernel function",
        "failed to create Metal image pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_image_f32_to_u8_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->image_f32_to_u8_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_image_f32_to_u8_kernel",
        "failed to look up Metal image kernel function",
        "failed to create Metal image pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_image_u8_to_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->image_u8_to_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_image_u8_to_f32_kernel",
        "failed to look up Metal image kernel function",
        "failed to create Metal image pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_rand_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_rand_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_rand_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_randn_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_randn_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_randn_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_randint_i32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_randint_i32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_randint_i32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_sum_reduce_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_sum_reduce_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_sum_reduce_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_mean_reduce_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_mean_reduce_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_mean_reduce_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_max_reduce_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_max_reduce_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_max_reduce_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_mse_reduce_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_mse_reduce_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_mse_reduce_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_mae_reduce_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_mae_reduce_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_mae_reduce_f32_kernel",
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

static gsx_error gsx_metal_backend_ensure_tensor_check_finite_f32_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_check_finite_f32_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_check_finite_f32_kernel",
        "failed to look up Metal tensor kernel function",
        "failed to create Metal tensor pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_tensor_check_finite_f16_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->tensor_check_finite_f16_pipeline,
        gsx_metal_backend_ensure_tensor_library,
        "gsx_metal_tensor_check_finite_f16_kernel",
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

static gsx_error gsx_metal_backend_dispatch_image_tensor_with_pipeline(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *src_buffer = NULL;
    gsx_metal_backend_buffer *dst_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || src_view == NULL || dst_view == NULL || params == NULL || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, image tensor views, params, and pipeline helper must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    src_buffer = gsx_metal_backend_buffer_from_base(src_view->buffer);
    dst_buffer = gsx_metal_backend_buffer_from_base(dst_view->buffer);
    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    [encoder setBuffer:(id<MTLBuffer>)src_buffer->mtl_buffer offset:(NSUInteger)src_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)dst_buffer->mtl_buffer offset:(NSUInteger)dst_view->offset_bytes atIndex:1];
    [encoder setBytes:params length:sizeof(*params) atIndex:2];
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_dispatch_image_layout_with_pipeline(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_layout_params *params,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *src_buffer = NULL;
    gsx_metal_backend_buffer *dst_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    NSUInteger element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || src_view == NULL || dst_view == NULL || params == NULL || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, image layout views, params, and pipeline helper must be non-null");
    }
    element_count = (NSUInteger)params->channels * (NSUInteger)params->height * (NSUInteger)params->width;
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    src_buffer = gsx_metal_backend_buffer_from_base(src_view->buffer);
    dst_buffer = gsx_metal_backend_buffer_from_base(dst_view->buffer);
    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    [encoder setBuffer:(id<MTLBuffer>)src_buffer->mtl_buffer offset:(NSUInteger)src_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)dst_buffer->mtl_buffer offset:(NSUInteger)dst_view->offset_bytes atIndex:1];
    [encoder setBytes:params length:sizeof(*params) atIndex:2];
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, element_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_dispatch_tensor_fill_with_pipeline(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const void *params,
    size_t params_size,
    NSUInteger thread_count,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *metal_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tensor_view == NULL || params == NULL || params_size == 0 || thread_count == 0 || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor_view, params, thread_count, and pipeline helper must be non-null");
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    metal_buffer = gsx_metal_backend_buffer_from_base(tensor_view->buffer);

    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer offset:(NSUInteger)tensor_view->offset_bytes atIndex:0];
    [encoder setBytes:params length:params_size atIndex:1];
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, thread_count);

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static NSUInteger gsx_metal_backend_reduce_threadgroup_width(id<MTLComputePipelineState> pipeline, uint32_t reduce_count)
{
    NSUInteger width = gsx_metal_backend_compute_threadgroup_width(pipeline);
    NSUInteger power_of_two_width = 1;

    if(reduce_count != 0 && width > (NSUInteger)reduce_count) {
        width = (NSUInteger)reduce_count;
    }
    while((power_of_two_width << 1) <= width) {
        power_of_two_width <<= 1;
    }
    return power_of_two_width;
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

gsx_error gsx_metal_backend_dispatch_image_linear_to_srgb_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
)
{
    return gsx_metal_backend_dispatch_image_tensor_with_pipeline(
        backend,
        src_view,
        dst_view,
        params,
        gsx_metal_backend_ensure_image_linear_to_srgb_f32_pipeline);
}

gsx_error gsx_metal_backend_dispatch_image_srgb_to_linear_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
)
{
    return gsx_metal_backend_dispatch_image_tensor_with_pipeline(
        backend,
        src_view,
        dst_view,
        params,
        gsx_metal_backend_ensure_image_srgb_to_linear_f32_pipeline);
}

gsx_error gsx_metal_backend_dispatch_image_chw_to_hwc(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_layout_params *params
)
{
    return gsx_metal_backend_dispatch_image_layout_with_pipeline(
        backend,
        src_view,
        dst_view,
        params,
        gsx_metal_backend_ensure_image_chw_to_hwc_pipeline);
}

gsx_error gsx_metal_backend_dispatch_image_hwc_to_chw(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_layout_params *params
)
{
    return gsx_metal_backend_dispatch_image_layout_with_pipeline(
        backend,
        src_view,
        dst_view,
        params,
        gsx_metal_backend_ensure_image_hwc_to_chw_pipeline);
}

gsx_error gsx_metal_backend_dispatch_image_f32_to_u8(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
)
{
    return gsx_metal_backend_dispatch_image_tensor_with_pipeline(
        backend,
        src_view,
        dst_view,
        params,
        gsx_metal_backend_ensure_image_f32_to_u8_pipeline);
}

gsx_error gsx_metal_backend_dispatch_image_u8_to_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
)
{
    return gsx_metal_backend_dispatch_image_tensor_with_pipeline(
        backend,
        src_view,
        dst_view,
        params,
        gsx_metal_backend_ensure_image_u8_to_f32_pipeline);
}

gsx_error gsx_metal_backend_dispatch_tensor_rand_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_rand_f32_params *params
)
{
    NSUInteger thread_count = 0;

    if(params == NULL || params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    thread_count = ((NSUInteger)params->element_count + 3u) / 4u;
    return gsx_metal_backend_dispatch_tensor_fill_with_pipeline(
        backend,
        tensor_view,
        params,
        sizeof(*params),
        thread_count,
        gsx_metal_backend_ensure_tensor_rand_f32_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_randn_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_randn_f32_params *params
)
{
    NSUInteger thread_count = 0;

    if(params == NULL || params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    thread_count = ((NSUInteger)params->element_count + 3u) / 4u;
    return gsx_metal_backend_dispatch_tensor_fill_with_pipeline(
        backend,
        tensor_view,
        params,
        sizeof(*params),
        thread_count,
        gsx_metal_backend_ensure_tensor_randn_f32_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_randint_i32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_randint_i32_params *params
)
{
    NSUInteger thread_count = 0;

    if(params == NULL || params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    thread_count = ((NSUInteger)params->element_count + 3u) / 4u;
    return gsx_metal_backend_dispatch_tensor_fill_with_pipeline(
        backend,
        tensor_view,
        params,
        sizeof(*params),
        thread_count,
        gsx_metal_backend_ensure_tensor_randint_i32_pipeline
    );
}

gsx_error gsx_metal_backend_dispatch_tensor_unary_reduce_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_reduce_f32_params *params,
    gsx_impl_unary_reduce_op op
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    NSUInteger threadgroup_width = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || x_view == NULL || out_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, and params must be non-null");
    }
    if(params->outer_count == 0 || params->reduce_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_view->buffer);

    switch(op) {
    case GSX_IMPL_UNARY_REDUCE_OP_SUM:
        error = gsx_metal_backend_ensure_tensor_sum_reduce_f32_pipeline(metal_backend, &pipeline);
        break;
    case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
        error = gsx_metal_backend_ensure_tensor_mean_reduce_f32_pipeline(metal_backend, &pipeline);
        break;
    case GSX_IMPL_UNARY_REDUCE_OP_MAX:
        error = gsx_metal_backend_ensure_tensor_max_reduce_f32_pipeline(metal_backend, &pipeline);
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary_reduce op");
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    threadgroup_width = gsx_metal_backend_reduce_threadgroup_width(pipeline, params->reduce_count);
    [encoder setBuffer:(id<MTLBuffer>)x_buffer->mtl_buffer offset:(NSUInteger)x_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_view->offset_bytes atIndex:1];
    [encoder setBytes:params length:sizeof(*params) atIndex:2];
    [encoder setThreadgroupMemoryLength:threadgroup_width * sizeof(float) atIndex:0];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)params->outer_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_tensor_binary_reduce_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_binary_reduce_f32_params *params,
    gsx_impl_binary_reduce_op op
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *lhs_buffer = NULL;
    gsx_metal_backend_buffer *rhs_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    NSUInteger threadgroup_width = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || lhs_view == NULL || rhs_view == NULL || out_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, and params must be non-null");
    }
    if(params->outer_count == 0 || params->reduce_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    lhs_buffer = gsx_metal_backend_buffer_from_base(lhs_view->buffer);
    rhs_buffer = gsx_metal_backend_buffer_from_base(rhs_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_view->buffer);

    switch(op) {
    case GSX_IMPL_BINARY_REDUCE_OP_MSE:
        error = gsx_metal_backend_ensure_tensor_mse_reduce_f32_pipeline(metal_backend, &pipeline);
        break;
    case GSX_IMPL_BINARY_REDUCE_OP_MAE:
        error = gsx_metal_backend_ensure_tensor_mae_reduce_f32_pipeline(metal_backend, &pipeline);
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown binary_reduce op");
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    threadgroup_width = gsx_metal_backend_reduce_threadgroup_width(pipeline, params->reduce_count);
    [encoder setBuffer:(id<MTLBuffer>)lhs_buffer->mtl_buffer offset:(NSUInteger)lhs_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)rhs_buffer->mtl_buffer offset:(NSUInteger)rhs_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_view->offset_bytes atIndex:2];
    [encoder setBytes:params length:sizeof(*params) atIndex:3];
    [encoder setThreadgroupMemoryLength:threadgroup_width * sizeof(float) atIndex:0];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)params->outer_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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

gsx_error gsx_metal_backend_dispatch_tensor_check_finite_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_check_finite_params *params,
    uint32_t *out_has_non_finite
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *metal_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLBuffer> status_buffer = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    uint32_t *status_ptr = NULL;

    if(backend == NULL || tensor_view == NULL || params == NULL || out_has_non_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor_view, params, and out_has_non_finite must be non-null");
    }
    *out_has_non_finite = 0;
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    metal_buffer = gsx_metal_backend_buffer_from_base(tensor_view->buffer);

    status_buffer = [(id<MTLDevice>)metal_backend->mtl_device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
    if(status_buffer == nil) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal check_finite status buffer");
    }
    status_ptr = (uint32_t *)[status_buffer contents];
    if(status_ptr == NULL) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to access Metal check_finite status buffer contents");
    }
    *status_ptr = 0;

    error = gsx_metal_backend_ensure_tensor_check_finite_f32_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        [status_buffer release];
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        [status_buffer release];
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer offset:(NSUInteger)tensor_view->offset_bytes atIndex:0];
    [encoder setBytes:params length:sizeof(*params) atIndex:1];
    [encoder setBuffer:status_buffer offset:0 atIndex:2];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    *out_has_non_finite = *status_ptr;
    [status_buffer release];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_tensor_check_finite_f16(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_check_finite_params *params,
    uint32_t *out_has_non_finite
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *metal_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLBuffer> status_buffer = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    uint32_t *status_ptr = NULL;

    if(backend == NULL || tensor_view == NULL || params == NULL || out_has_non_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor_view, params, and out_has_non_finite must be non-null");
    }
    *out_has_non_finite = 0;
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    metal_buffer = gsx_metal_backend_buffer_from_base(tensor_view->buffer);

    status_buffer = [(id<MTLDevice>)metal_backend->mtl_device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
    if(status_buffer == nil) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal check_finite status buffer");
    }
    status_ptr = (uint32_t *)[status_buffer contents];
    if(status_ptr == NULL) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to access Metal check_finite status buffer contents");
    }
    *status_ptr = 0;

    error = gsx_metal_backend_ensure_tensor_check_finite_f16_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        [status_buffer release];
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        [status_buffer release];
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer offset:(NSUInteger)tensor_view->offset_bytes atIndex:0];
    [encoder setBytes:params length:sizeof(*params) atIndex:1];
    [encoder setBuffer:status_buffer offset:0 atIndex:2];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    *out_has_non_finite = *status_ptr;
    [status_buffer release];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

