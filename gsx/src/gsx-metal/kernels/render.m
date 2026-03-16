#include "../objc-helpers.h"

extern const char gsx_metal_render_metallib_start[];
extern const char gsx_metal_render_metallib_end[];

static gsx_error gsx_metal_backend_ensure_render_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->render_library,
        gsx_metal_render_metallib_start,
        gsx_metal_render_metallib_end,
        "embedded Metal render metallib is empty",
        "failed to create dispatch data for embedded Metal render metallib",
        "failed to load embedded Metal render metallib",
        out_library);
}

static gsx_error gsx_metal_backend_ensure_render_compose_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_compose_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_compose_chw_f32_kernel",
        "failed to look up Metal render compose kernel function",
        "failed to create Metal render compose pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_render_preprocess_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_preprocess_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_preprocess_kernel",
        "failed to look up Metal render preprocess kernel function",
        "failed to create Metal render preprocess pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_render_create_instances_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_create_instances_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_create_instances_kernel",
        "failed to look up Metal render create-instances kernel function",
        "failed to create Metal render create-instances pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_render_blend_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_blend_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_blend_kernel",
        "failed to look up Metal render blend kernel function",
        "failed to create Metal render blend pipeline state",
        out_pipeline);
}

gsx_error gsx_metal_backend_dispatch_render_preprocess(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *mean3d_view,
    const gsx_backend_tensor_view *rotation_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *sh0_view,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *depth_view,
    const gsx_backend_tensor_view *visible_view,
    const gsx_backend_tensor_view *touched_view,
    const gsx_backend_tensor_view *bounds_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_metal_render_preprocess_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || mean3d_view == NULL || rotation_view == NULL || logscale_view == NULL || sh0_view == NULL || opacity_view == NULL
        || depth_view == NULL || visible_view == NULL || touched_view == NULL || bounds_view == NULL || mean2d_view == NULL
        || conic_opacity_view == NULL || color_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render preprocess dispatch arguments must be non-null");
    }
    if(params->gaussian_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_preprocess_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean3d_view->buffer)->mtl_buffer offset:(NSUInteger)mean3d_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(rotation_view->buffer)->mtl_buffer offset:(NSUInteger)rotation_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(logscale_view->buffer)->mtl_buffer offset:(NSUInteger)logscale_view->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh0_view->buffer)->mtl_buffer offset:(NSUInteger)sh0_view->offset_bytes atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(opacity_view->buffer)->mtl_buffer offset:(NSUInteger)opacity_view->offset_bytes atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(depth_view->buffer)->mtl_buffer offset:(NSUInteger)depth_view->offset_bytes atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(visible_view->buffer)->mtl_buffer offset:(NSUInteger)visible_view->offset_bytes atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(touched_view->buffer)->mtl_buffer offset:(NSUInteger)touched_view->offset_bytes atIndex:7];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bounds_view->buffer)->mtl_buffer offset:(NSUInteger)bounds_view->offset_bytes atIndex:8];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:9];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:10];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(color_view->buffer)->mtl_buffer offset:(NSUInteger)color_view->offset_bytes atIndex:11];
    [encoder setBytes:params length:sizeof(*params) atIndex:12];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_create_instances(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *sorted_primitive_ids_view,
    const gsx_backend_tensor_view *primitive_offsets_view,
    const gsx_backend_tensor_view *bounds_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *instance_keys_view,
    const gsx_backend_tensor_view *instance_primitive_ids_view,
    const gsx_metal_render_create_instances_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || sorted_primitive_ids_view == NULL || primitive_offsets_view == NULL || bounds_view == NULL
        || mean2d_view == NULL || conic_opacity_view == NULL || instance_keys_view == NULL || instance_primitive_ids_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render create-instances dispatch arguments must be non-null");
    }
    if(params->visible_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_create_instances_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sorted_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)sorted_primitive_ids_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(primitive_offsets_view->buffer)->mtl_buffer offset:(NSUInteger)primitive_offsets_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bounds_view->buffer)->mtl_buffer offset:(NSUInteger)bounds_view->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_keys_view->buffer)->mtl_buffer offset:(NSUInteger)instance_keys_view->offset_bytes atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)instance_primitive_ids_view->offset_bytes atIndex:6];
    [encoder setBytes:params length:sizeof(*params) atIndex:7];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->visible_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_blend(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_ranges_view,
    const gsx_backend_tensor_view *instance_primitive_ids_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_backend_tensor_view *image_view,
    const gsx_backend_tensor_view *alpha_view,
    const gsx_metal_render_blend_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    NSUInteger tg_w = 16;
    NSUInteger tg_h = 16;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tile_ranges_view == NULL || instance_primitive_ids_view == NULL || mean2d_view == NULL
        || conic_opacity_view == NULL || color_view == NULL || image_view == NULL || alpha_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render blend dispatch arguments must be non-null");
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_blend_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_ranges_view->buffer)->mtl_buffer offset:(NSUInteger)tile_ranges_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)instance_primitive_ids_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(color_view->buffer)->mtl_buffer offset:(NSUInteger)color_view->offset_bytes atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(image_view->buffer)->mtl_buffer offset:(NSUInteger)image_view->offset_bytes atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(alpha_view->buffer)->mtl_buffer offset:(NSUInteger)alpha_view->offset_bytes atIndex:6];
    [encoder setBytes:params length:sizeof(*params) atIndex:7];

    if((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup < tg_w * tg_h) {
        tg_w = (NSUInteger)pipeline.threadExecutionWidth;
        if(tg_w == 0) {
            tg_w = 8;
        }
        tg_h = (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup / tg_w;
        if(tg_h == 0) {
            tg_h = 1;
        }
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)params->width, (NSUInteger)params->height, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_w, tg_h, 1)];

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_compose_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *image_view,
    const gsx_backend_tensor_view *alpha_view,
    const gsx_backend_tensor_view *out_rgb_view,
    const gsx_metal_render_compose_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *image_buffer = NULL;
    gsx_metal_backend_buffer *alpha_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    NSUInteger tg_w = 16;
    NSUInteger tg_h = 16;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || image_view == NULL || alpha_view == NULL || out_rgb_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, and params must be non-null");
    }
    if(params->width == 0 || params->height == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    image_buffer = gsx_metal_backend_buffer_from_base(image_view->buffer);
    alpha_buffer = gsx_metal_backend_buffer_from_base(alpha_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_rgb_view->buffer);

    error = gsx_metal_backend_ensure_render_compose_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)image_buffer->mtl_buffer offset:(NSUInteger)image_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)alpha_buffer->mtl_buffer offset:(NSUInteger)alpha_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_rgb_view->offset_bytes atIndex:2];
    [encoder setBytes:params length:sizeof(*params) atIndex:3];

    if((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup < tg_w * tg_h) {
        tg_w = (NSUInteger)pipeline.threadExecutionWidth;
        if(tg_w == 0) {
            tg_w = 8;
        }
        tg_h = (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup / tg_w;
        if(tg_h == 0) {
            tg_h = 1;
        }
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)params->width, (NSUInteger)params->height, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_w, tg_h, 1)];

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
