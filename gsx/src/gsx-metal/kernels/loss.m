#include "../objc-helpers.h"

extern const char gsx_metal_loss_metallib_start[];
extern const char gsx_metal_loss_metallib_end[];

static gsx_error gsx_metal_backend_ensure_loss_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->loss_library,
        gsx_metal_loss_metallib_start,
        gsx_metal_loss_metallib_end,
        "embedded Metal loss metallib is empty",
        "failed to create dispatch data for embedded Metal loss metallib",
        "failed to load embedded Metal loss metallib",
        out_library);
}

static gsx_error gsx_metal_backend_ensure_loss_mse_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_mse_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_mse_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_l1_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_l1_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_l1_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_mse_backward_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_mse_backward_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_mse_backward_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_l1_backward_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_l1_backward_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_l1_backward_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_ssim_chw_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_ssim_chw_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_ssim_chw_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_ssim_hwc_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_ssim_hwc_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_ssim_hwc_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_ssim_backward_chw_pipeline(
    gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_ssim_backward_chw_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_ssim_backward_chw_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_loss_ssim_backward_hwc_pipeline(
    gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->loss_ssim_backward_hwc_pipeline,
        gsx_metal_backend_ensure_loss_library,
        "gsx_metal_loss_ssim_backward_hwc_f32_kernel",
        "failed to look up Metal loss kernel function",
        "failed to create Metal loss pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_dispatch_loss_pointwise_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *, id<MTLComputePipelineState> *))
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *prediction_buffer = NULL;
    gsx_metal_backend_buffer *target_buffer = NULL;
    gsx_metal_backend_buffer *accumulator_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || prediction_view == NULL || target_view == NULL || accumulator_view == NULL || params == NULL
        || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, params, and ensure_pipeline must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    prediction_buffer = gsx_metal_backend_buffer_from_base(prediction_view->buffer);
    target_buffer = gsx_metal_backend_buffer_from_base(target_view->buffer);
    accumulator_buffer = gsx_metal_backend_buffer_from_base(accumulator_view->buffer);

    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)prediction_buffer->mtl_buffer offset:(NSUInteger)prediction_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)target_buffer->mtl_buffer offset:(NSUInteger)target_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)accumulator_buffer->mtl_buffer offset:(NSUInteger)accumulator_view->offset_bytes atIndex:2];
    [encoder setBytes:params length:sizeof(*params) atIndex:3];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_loss_mse_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        accumulator_view,
        params,
        gsx_metal_backend_ensure_loss_mse_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_l1_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        accumulator_view,
        params,
        gsx_metal_backend_ensure_loss_l1_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_mse_backward_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        grad_view,
        params,
        gsx_metal_backend_ensure_loss_mse_backward_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_l1_backward_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        grad_view,
        params,
        gsx_metal_backend_ensure_loss_l1_backward_pipeline);
}

static gsx_error gsx_metal_backend_dispatch_loss_ssim_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *, id<MTLComputePipelineState> *))
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *prediction_buffer = NULL;
    gsx_metal_backend_buffer *target_buffer = NULL;
    gsx_metal_backend_buffer *accumulator_buffer = NULL;
    gsx_metal_backend_buffer *scratch_a_buffer = NULL;
    gsx_metal_backend_buffer *scratch_b_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    const NSUInteger tg_w = 16u;
    const NSUInteger tg_h = 16u;

    if(backend == NULL || prediction_view == NULL || target_view == NULL || accumulator_view == NULL || params == NULL
        || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, params, and ensure_pipeline must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    prediction_buffer = gsx_metal_backend_buffer_from_base(prediction_view->buffer);
    target_buffer = gsx_metal_backend_buffer_from_base(target_view->buffer);
    accumulator_buffer = gsx_metal_backend_buffer_from_base(accumulator_view->buffer);
    if(scratch_a_view != NULL) {
        scratch_a_buffer = gsx_metal_backend_buffer_from_base(scratch_a_view->buffer);
    }
    if(scratch_b_view != NULL) {
        scratch_b_buffer = gsx_metal_backend_buffer_from_base(scratch_b_view->buffer);
    }

    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup < tg_w * tg_h) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM kernel requires 256-thread Metal threadgroups");
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)prediction_buffer->mtl_buffer offset:(NSUInteger)prediction_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)target_buffer->mtl_buffer offset:(NSUInteger)target_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)accumulator_buffer->mtl_buffer offset:(NSUInteger)accumulator_view->offset_bytes atIndex:2];
    if(scratch_a_view != NULL) {
        [encoder setBuffer:(id<MTLBuffer>)scratch_a_buffer->mtl_buffer offset:(NSUInteger)scratch_a_view->offset_bytes atIndex:3];
    } else {
        [encoder setBuffer:nil offset:0 atIndex:3];
    }
    if(scratch_b_view != NULL) {
        [encoder setBuffer:(id<MTLBuffer>)scratch_b_buffer->mtl_buffer offset:(NSUInteger)scratch_b_view->offset_bytes atIndex:4];
    } else {
        [encoder setBuffer:nil offset:0 atIndex:4];
    }
    [encoder setBytes:params length:sizeof(*params) atIndex:5];

    [encoder
        dispatchThreadgroups:MTLSizeMake(
            ((NSUInteger)params->width + tg_w - 1u) / tg_w,
            ((NSUInteger)params->height + tg_h - 1u) / tg_h,
            (NSUInteger)params->outer_count)
        threadsPerThreadgroup:MTLSizeMake(tg_w, tg_h, 1u)];

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_loss_ssim_chw_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *loss_map_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params)
{
    return gsx_metal_backend_dispatch_loss_ssim_f32(
        backend,
        prediction_view,
        target_view,
        loss_map_view,
        scratch_a_view,
        scratch_b_view,
        params,
        gsx_metal_backend_ensure_loss_ssim_chw_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_ssim_hwc_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *loss_map_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params)
{
    return gsx_metal_backend_dispatch_loss_ssim_f32(
        backend,
        prediction_view,
        target_view,
        loss_map_view,
        scratch_a_view,
        scratch_b_view,
        params,
        gsx_metal_backend_ensure_loss_ssim_hwc_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_ssim_backward_chw_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params)
{
    return gsx_metal_backend_dispatch_loss_ssim_f32(
        backend,
        prediction_view,
        target_view,
        grad_view,
        scratch_a_view,
        scratch_b_view,
        params,
        gsx_metal_backend_ensure_loss_ssim_backward_chw_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_ssim_backward_hwc_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params)
{
    return gsx_metal_backend_dispatch_loss_ssim_f32(
        backend,
        prediction_view,
        target_view,
        grad_view,
        scratch_a_view,
        scratch_b_view,
        params,
        gsx_metal_backend_ensure_loss_ssim_backward_hwc_pipeline);
}
