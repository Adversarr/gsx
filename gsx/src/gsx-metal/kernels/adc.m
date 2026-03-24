#include "../objc-helpers.h"

extern const char gsx_metal_adc_metallib_start[];
extern const char gsx_metal_adc_metallib_end[];

static gsx_error gsx_metal_backend_ensure_adc_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->adc_library,
        gsx_metal_adc_metallib_start,
        gsx_metal_adc_metallib_end,
        "embedded Metal adc metallib is empty",
        "failed to create dispatch data for embedded Metal adc metallib",
        "failed to load embedded Metal adc metallib",
        out_library);
}

static gsx_error gsx_metal_backend_ensure_adc_classify_growth_pipeline(
    gsx_metal_backend *metal_backend,
    id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->adc_classify_growth_pipeline,
        gsx_metal_backend_ensure_adc_library,
        "gsx_metal_adc_classify_growth_kernel",
        "failed to look up Metal adc growth-classify kernel function",
        "failed to create Metal adc growth-classify pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_adc_apply_split_pipeline(
    gsx_metal_backend *metal_backend,
    id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->adc_apply_split_pipeline,
        gsx_metal_backend_ensure_adc_library,
        "gsx_metal_adc_apply_split_kernel",
        "failed to look up Metal adc split kernel function",
        "failed to create Metal adc split pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_adc_keep_mask_pipeline(
    gsx_metal_backend *metal_backend,
    id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->adc_keep_mask_pipeline,
        gsx_metal_backend_ensure_adc_library,
        "gsx_metal_adc_keep_mask_kernel",
        "failed to look up Metal adc keep-mask kernel function",
        "failed to create Metal adc keep-mask pipeline state",
        out_pipeline);
}

gsx_error gsx_metal_backend_dispatch_adc_classify_growth(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *grad_acc_view,
    const gsx_backend_tensor_view *visible_counter_view,
    const gsx_backend_tensor_view *logscale_view,
    gsx_backend_buffer_t out_mode_buffer,
    const gsx_metal_adc_classify_growth_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *grad_acc_buffer = NULL;
    gsx_metal_backend_buffer *visible_counter_buffer = NULL;
    gsx_metal_backend_buffer *logscale_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || grad_acc_view == NULL || logscale_view == NULL || out_mode_buffer == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, adc classify views, output buffer, and params must be non-null");
    }
    if(params->gaussian_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    grad_acc_buffer = gsx_metal_backend_buffer_from_base(grad_acc_view->buffer);
    logscale_buffer = gsx_metal_backend_buffer_from_base(logscale_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_mode_buffer);
    if(visible_counter_view != NULL) {
        visible_counter_buffer = gsx_metal_backend_buffer_from_base(visible_counter_view->buffer);
    }

    error = gsx_metal_backend_ensure_adc_classify_growth_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)grad_acc_buffer->mtl_buffer offset:(NSUInteger)grad_acc_view->offset_bytes atIndex:0];
    [encoder setBuffer:visible_counter_buffer != NULL ? (id<MTLBuffer>)visible_counter_buffer->mtl_buffer : nil
                offset:visible_counter_view != NULL ? (NSUInteger)visible_counter_view->offset_bytes : 0
               atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)logscale_buffer->mtl_buffer offset:(NSUInteger)logscale_view->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:0 atIndex:3];
    [encoder setBytes:params length:sizeof(*params) atIndex:4];
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_adc_apply_split(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *mean3d_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *rotation_view,
    gsx_backend_buffer_t split_source_buffer,
    gsx_backend_buffer_t split_target_buffer,
    const gsx_metal_adc_apply_split_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *mean3d_buffer = NULL;
    gsx_metal_backend_buffer *logscale_buffer = NULL;
    gsx_metal_backend_buffer *opacity_buffer = NULL;
    gsx_metal_backend_buffer *rotation_buffer = NULL;
    gsx_metal_backend_buffer *source_buffer = NULL;
    gsx_metal_backend_buffer *target_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || mean3d_view == NULL || logscale_view == NULL || opacity_view == NULL || rotation_view == NULL
        || split_source_buffer == NULL || split_target_buffer == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, adc split inputs, source/target buffers, and params must be non-null");
    }
    if(params->split_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    mean3d_buffer = gsx_metal_backend_buffer_from_base(mean3d_view->buffer);
    logscale_buffer = gsx_metal_backend_buffer_from_base(logscale_view->buffer);
    opacity_buffer = gsx_metal_backend_buffer_from_base(opacity_view->buffer);
    rotation_buffer = gsx_metal_backend_buffer_from_base(rotation_view->buffer);
    source_buffer = gsx_metal_backend_buffer_from_base(split_source_buffer);
    target_buffer = gsx_metal_backend_buffer_from_base(split_target_buffer);

    error = gsx_metal_backend_ensure_adc_apply_split_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)mean3d_buffer->mtl_buffer offset:(NSUInteger)mean3d_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)logscale_buffer->mtl_buffer offset:(NSUInteger)logscale_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)opacity_buffer->mtl_buffer offset:(NSUInteger)opacity_view->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)rotation_buffer->mtl_buffer offset:(NSUInteger)rotation_view->offset_bytes atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)source_buffer->mtl_buffer offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)target_buffer->mtl_buffer offset:0 atIndex:5];
    [encoder setBytes:params length:sizeof(*params) atIndex:6];
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->split_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_adc_keep_mask(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *rotation_view,
    const gsx_backend_tensor_view *max_screen_radius_view,
    gsx_backend_buffer_t out_keep_mask_buffer,
    const gsx_metal_adc_keep_mask_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *opacity_buffer = NULL;
    gsx_metal_backend_buffer *logscale_buffer = NULL;
    gsx_metal_backend_buffer *rotation_buffer = NULL;
    gsx_metal_backend_buffer *max_screen_radius_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || opacity_view == NULL || logscale_view == NULL || rotation_view == NULL
        || out_keep_mask_buffer == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, adc keep-mask inputs, output buffer, and params must be non-null");
    }
    if(params->gaussian_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    opacity_buffer = gsx_metal_backend_buffer_from_base(opacity_view->buffer);
    logscale_buffer = gsx_metal_backend_buffer_from_base(logscale_view->buffer);
    rotation_buffer = gsx_metal_backend_buffer_from_base(rotation_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_keep_mask_buffer);
    if(max_screen_radius_view != NULL) {
        max_screen_radius_buffer = gsx_metal_backend_buffer_from_base(max_screen_radius_view->buffer);
    }

    error = gsx_metal_backend_ensure_adc_keep_mask_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)opacity_buffer->mtl_buffer offset:(NSUInteger)opacity_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)logscale_buffer->mtl_buffer offset:(NSUInteger)logscale_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)rotation_buffer->mtl_buffer offset:(NSUInteger)rotation_view->offset_bytes atIndex:2];
    [encoder setBuffer:max_screen_radius_buffer != NULL ? (id<MTLBuffer>)max_screen_radius_buffer->mtl_buffer : nil
                offset:max_screen_radius_view != NULL ? (NSUInteger)max_screen_radius_view->offset_bytes : 0
               atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:0 atIndex:4];
    [encoder setBytes:params length:sizeof(*params) atIndex:5];
    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
