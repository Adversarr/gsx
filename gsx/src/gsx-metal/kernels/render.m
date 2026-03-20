#include "../objc-helpers.h"

extern const char gsx_metal_render_metallib_start[];
extern const char gsx_metal_render_metallib_end[];
extern const char gsx_metal_sort_metallib_start[];
extern const char gsx_metal_sort_metallib_end[];
extern const char gsx_metal_scan_metallib_start[];
extern const char gsx_metal_scan_metallib_end[];

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

static gsx_error gsx_metal_backend_ensure_sort_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->sort_library,
        gsx_metal_sort_metallib_start,
        gsx_metal_sort_metallib_end,
        "embedded Metal sort metallib is empty",
        "failed to create dispatch data for embedded Metal sort metallib",
        "failed to load embedded Metal sort metallib",
        out_library);
}

static gsx_error gsx_metal_backend_ensure_scan_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->scan_library,
        gsx_metal_scan_metallib_start,
        gsx_metal_scan_metallib_end,
        "embedded Metal scan metallib is empty",
        "failed to create dispatch data for embedded Metal scan metallib",
        "failed to load embedded Metal scan metallib",
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

static gsx_error gsx_metal_backend_ensure_render_apply_depth_ordering_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_apply_depth_ordering_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_apply_depth_ordering_kernel",
        "failed to look up Metal render apply-depth-ordering kernel function",
        "failed to create Metal render apply-depth-ordering pipeline state",
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

static gsx_error gsx_metal_backend_ensure_render_extract_instance_ranges_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_extract_instance_ranges_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_extract_instance_ranges_kernel",
        "failed to look up Metal render extract-instance-ranges kernel function",
        "failed to create Metal render extract-instance-ranges pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_render_extract_bucket_counts_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_extract_bucket_counts_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_extract_bucket_counts_kernel",
        "failed to look up Metal render extract-bucket-counts kernel function",
        "failed to create Metal render extract-bucket-counts pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_render_finalize_bucket_offsets_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->render_finalize_bucket_offsets_pipeline,
        gsx_metal_backend_ensure_render_library,
        "gsx_metal_render_finalize_bucket_offsets_kernel",
        "failed to look up Metal render finalize-bucket-offsets kernel function",
        "failed to create Metal render finalize-bucket-offsets pipeline state",
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

static gsx_error gsx_metal_backend_ensure_sort_histogram_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->sort_histogram_pipeline,
        gsx_metal_backend_ensure_sort_library,
        "radix_histogram",
        "failed to look up Metal sort histogram kernel function",
        "failed to create Metal sort histogram pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_sort_prefix_offsets_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->sort_reduce_pipeline,
        gsx_metal_backend_ensure_sort_library,
        "radix_prefix_offsets",
        "failed to look up Metal sort prefix-offsets kernel function",
        "failed to create Metal sort prefix-offsets pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_sort_scatter_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->sort_scatter_pipeline,
        gsx_metal_backend_ensure_sort_library,
        "radix_scatter_simd",
        "failed to look up Metal sort scatter kernel function",
        "failed to create Metal sort scatter pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_scan_blocks_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->scan_blocks_pipeline,
        gsx_metal_backend_ensure_scan_library,
        "prefix_scan_blocks",
        "failed to look up Metal scan kernel function",
        "failed to create Metal scan pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_scan_block_sums_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->scan_block_sums_pipeline,
        gsx_metal_backend_ensure_scan_library,
        "prefix_scan_block_sums",
        "failed to look up Metal scan block-sums kernel function",
        "failed to create Metal scan block-sums pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_scan_add_offsets_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->scan_add_offsets_pipeline,
        gsx_metal_backend_ensure_scan_library,
        "prefix_scan_add_block_offsets",
        "failed to look up Metal scan add-offsets kernel function",
        "failed to create Metal scan add-offsets pipeline state",
        out_pipeline);
}

static gsx_error gsx_metal_render_require_view_alignment(
    const gsx_backend_tensor_view *tensor_view,
    gsx_size_t required_alignment_bytes,
    const char *message)
{
    if(tensor_view == NULL || message == NULL || required_alignment_bytes == 0u) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render view alignment validation requires non-null inputs");
    }
    if(tensor_view->effective_alignment_bytes < required_alignment_bytes
        || (tensor_view->offset_bytes % required_alignment_bytes) != 0u) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, message);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_preprocess(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *mean3d_view,
    const gsx_backend_tensor_view *rotation_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *sh0_view,
    const gsx_backend_tensor_view *sh1_view,
    const gsx_backend_tensor_view *sh2_view,
    const gsx_backend_tensor_view *sh3_view,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *depth_keys_view,
    const gsx_backend_tensor_view *visible_primitive_ids_view,
    const gsx_backend_tensor_view *touched_tiles_view,
    const gsx_backend_tensor_view *bounds_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_backend_tensor_view *visible_count_view,
    const gsx_backend_tensor_view *instance_count_view,
    const gsx_backend_tensor_view *visible_counter_view,
    const gsx_backend_tensor_view *max_screen_radius_view,
    const gsx_metal_render_preprocess_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || mean3d_view == NULL || rotation_view == NULL || logscale_view == NULL || sh0_view == NULL || opacity_view == NULL
        || depth_keys_view == NULL || visible_primitive_ids_view == NULL || touched_tiles_view == NULL || bounds_view == NULL || mean2d_view == NULL
        || conic_opacity_view == NULL || color_view == NULL || visible_count_view == NULL || instance_count_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render preprocess dispatch arguments must be non-null");
    }
    if(params->sh_degree > 3u) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render preprocess sh_degree must be in [0,3]");
    }
    if(sh1_view == NULL || sh2_view == NULL || sh3_view == NULL || visible_counter_view == NULL || max_screen_radius_view == NULL) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT,
            "render preprocess requires sh1, sh2, sh3, visible_counter, and max_screen_radius views; bind a dummy tensor when unused");
    }
    if(params->gaussian_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_metal_render_require_view_alignment(rotation_view, 16u, "gs_rotation alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(bounds_view, 16u, "internal bounds alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(mean2d_view, 8u, "internal mean2d alignment must be >= 8 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(
        conic_opacity_view,
        16u,
        "internal conic_opacity alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
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
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh1_view->buffer)->mtl_buffer
        offset:(NSUInteger)sh1_view->offset_bytes
        atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh2_view->buffer)->mtl_buffer
        offset:(NSUInteger)sh2_view->offset_bytes
        atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh3_view->buffer)->mtl_buffer
        offset:(NSUInteger)sh3_view->offset_bytes
        atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(opacity_view->buffer)->mtl_buffer offset:(NSUInteger)opacity_view->offset_bytes atIndex:7];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(depth_keys_view->buffer)->mtl_buffer offset:(NSUInteger)depth_keys_view->offset_bytes atIndex:8];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(visible_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)visible_primitive_ids_view->offset_bytes atIndex:9];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(touched_tiles_view->buffer)->mtl_buffer offset:(NSUInteger)touched_tiles_view->offset_bytes atIndex:10];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bounds_view->buffer)->mtl_buffer offset:(NSUInteger)bounds_view->offset_bytes atIndex:11];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:12];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:13];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(color_view->buffer)->mtl_buffer offset:(NSUInteger)color_view->offset_bytes atIndex:14];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(visible_count_view->buffer)->mtl_buffer offset:(NSUInteger)visible_count_view->offset_bytes atIndex:15];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_count_view->buffer)->mtl_buffer offset:(NSUInteger)instance_count_view->offset_bytes atIndex:16];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(visible_counter_view->buffer)->mtl_buffer
        offset:(NSUInteger)visible_counter_view->offset_bytes
        atIndex:17];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(max_screen_radius_view->buffer)->mtl_buffer
        offset:(NSUInteger)max_screen_radius_view->offset_bytes
        atIndex:18];
    [encoder setBytes:params length:sizeof(*params) atIndex:19];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_apply_depth_ordering(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *sorted_primitive_ids_view,
    const gsx_backend_tensor_view *touched_tiles_view,
    const gsx_backend_tensor_view *primitive_offsets_view,
    uint32_t visible_count)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || sorted_primitive_ids_view == NULL || touched_tiles_view == NULL || primitive_offsets_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render apply-depth-ordering dispatch arguments must be non-null");
    }
    if(visible_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_apply_depth_ordering_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sorted_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)sorted_primitive_ids_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(touched_tiles_view->buffer)->mtl_buffer offset:(NSUInteger)touched_tiles_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(primitive_offsets_view->buffer)->mtl_buffer offset:(NSUInteger)primitive_offsets_view->offset_bytes atIndex:2];
    [encoder setBytes:&visible_count length:sizeof(visible_count) atIndex:3];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)visible_count);
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
    error = gsx_metal_render_require_view_alignment(bounds_view, 16u, "internal bounds alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(mean2d_view, 8u, "internal mean2d alignment must be >= 8 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(
        conic_opacity_view,
        16u,
        "internal conic_opacity alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
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

gsx_error gsx_metal_backend_dispatch_render_extract_instance_ranges(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *instance_keys_view,
    const gsx_backend_tensor_view *tile_ranges_view,
    uint32_t instance_count,
    uint32_t tile_count)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || instance_keys_view == NULL || tile_ranges_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render extract-instance-ranges dispatch arguments must be non-null");
    }
    if(instance_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_metal_render_require_view_alignment(tile_ranges_view, 8u, "internal tile_ranges alignment must be >= 8 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_extract_instance_ranges_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_keys_view->buffer)->mtl_buffer offset:(NSUInteger)instance_keys_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_ranges_view->buffer)->mtl_buffer offset:(NSUInteger)tile_ranges_view->offset_bytes atIndex:1];
    [encoder setBytes:&instance_count length:sizeof(instance_count) atIndex:2];
    [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:3];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)instance_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_extract_bucket_counts(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_ranges_view,
    const gsx_backend_tensor_view *tile_bucket_counts_view,
    uint32_t tile_count)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tile_ranges_view == NULL || tile_bucket_counts_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render extract-bucket-counts dispatch arguments must be non-null");
    }
    if(tile_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_metal_render_require_view_alignment(tile_ranges_view, 8u, "internal tile_ranges alignment must be >= 8 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_extract_bucket_counts_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_ranges_view->buffer)->mtl_buffer offset:(NSUInteger)tile_ranges_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_bucket_counts_view->buffer)->mtl_buffer offset:(NSUInteger)tile_bucket_counts_view->offset_bytes atIndex:1];
    [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:2];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)tile_count);
    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_finalize_bucket_offsets(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_bucket_counts_view,
    const gsx_backend_tensor_view *tile_bucket_offsets_view,
    uint32_t tile_count)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tile_bucket_counts_view == NULL || tile_bucket_offsets_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render finalize-bucket-offsets dispatch arguments must be non-null");
    }
    if(tile_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    error = gsx_metal_backend_ensure_render_finalize_bucket_offsets_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_bucket_counts_view->buffer)->mtl_buffer offset:(NSUInteger)tile_bucket_counts_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_bucket_offsets_view->buffer)->mtl_buffer offset:(NSUInteger)tile_bucket_offsets_view->offset_bytes atIndex:1];
    [encoder setBytes:&tile_count length:sizeof(tile_count) atIndex:2];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)tile_count);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_blend(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_ranges_view,
    const gsx_backend_tensor_view *tile_bucket_offsets_view,
    const gsx_backend_tensor_view *instance_primitive_ids_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_backend_tensor_view *image_view,
    const gsx_backend_tensor_view *alpha_view,
    const gsx_backend_tensor_view *tile_max_n_contributions_view,
    const gsx_backend_tensor_view *tile_n_contributions_view,
    const gsx_backend_tensor_view *bucket_tile_index_view,
    const gsx_backend_tensor_view *bucket_color_transmittance_view,
    const gsx_metal_render_blend_params *params)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    NSUInteger tg_w = 16;
    NSUInteger tg_h = 16;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || tile_ranges_view == NULL || tile_bucket_offsets_view == NULL || instance_primitive_ids_view == NULL || mean2d_view == NULL
        || conic_opacity_view == NULL || color_view == NULL || image_view == NULL || alpha_view == NULL
        || tile_max_n_contributions_view == NULL || tile_n_contributions_view == NULL || bucket_tile_index_view == NULL
        || bucket_color_transmittance_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render blend dispatch arguments must be non-null");
    }
    error = gsx_metal_render_require_view_alignment(tile_ranges_view, 8u, "internal tile_ranges alignment must be >= 8 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(mean2d_view, 8u, "internal mean2d alignment must be >= 8 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(
        conic_opacity_view,
        16u,
        "internal conic_opacity alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_require_view_alignment(
        bucket_color_transmittance_view,
        16u,
        "internal bucket_color_transmittance alignment must be >= 16 bytes for Metal vector access");
    if(!gsx_error_is_success(error)) {
        return error;
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
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_bucket_offsets_view->buffer)->mtl_buffer offset:(NSUInteger)tile_bucket_offsets_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)instance_primitive_ids_view->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(color_view->buffer)->mtl_buffer offset:(NSUInteger)color_view->offset_bytes atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(image_view->buffer)->mtl_buffer offset:(NSUInteger)image_view->offset_bytes atIndex:6];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(alpha_view->buffer)->mtl_buffer offset:(NSUInteger)alpha_view->offset_bytes atIndex:7];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_max_n_contributions_view->buffer)->mtl_buffer offset:(NSUInteger)tile_max_n_contributions_view->offset_bytes atIndex:8];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_n_contributions_view->buffer)->mtl_buffer offset:(NSUInteger)tile_n_contributions_view->offset_bytes atIndex:9];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bucket_tile_index_view->buffer)->mtl_buffer offset:(NSUInteger)bucket_tile_index_view->offset_bytes atIndex:10];
    [encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bucket_color_transmittance_view->buffer)->mtl_buffer offset:(NSUInteger)bucket_color_transmittance_view->offset_bytes atIndex:11];
    [encoder setBytes:params length:sizeof(*params) atIndex:12];

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

gsx_error gsx_metal_backend_dispatch_scan_exclusive_u32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *data_view,
    const gsx_backend_tensor_view *block_sums_view,
    const gsx_backend_tensor_view *scanned_block_sums_view,
    uint32_t count)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> blocks_pipeline = nil;
    id<MTLComputePipelineState> block_sums_pipeline = nil;
    id<MTLComputePipelineState> add_offsets_pipeline = nil;
    gsx_metal_backend_buffer *data_buffer = NULL;
    gsx_metal_backend_buffer *block_sums_buffer = NULL;
    gsx_metal_backend_buffer *scanned_block_sums_buffer = NULL;
    id<MTLCommandBuffer> command_buffer = nil;
    uint32_t block_count = 0u;
    uint32_t second_level_block_count = 0u;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || data_view == NULL || block_sums_view == NULL || scanned_block_sums_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scan dispatch arguments must be non-null");
    }
    if(count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    data_buffer = gsx_metal_backend_buffer_from_base(data_view->buffer);
    block_sums_buffer = gsx_metal_backend_buffer_from_base(block_sums_view->buffer);
    scanned_block_sums_buffer = gsx_metal_backend_buffer_from_base(scanned_block_sums_view->buffer);
    block_count = (count + 255u) / 256u;
    second_level_block_count = (block_count + 255u) / 256u;
    if(second_level_block_count > 256u) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scan dispatch currently supports up to 16777216 elements");
    }

    error = gsx_metal_backend_ensure_scan_blocks_pipeline(metal_backend, &blocks_pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_ensure_scan_block_sums_pipeline(metal_backend, &block_sums_pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_ensure_scan_add_offsets_pipeline(metal_backend, &add_offsets_pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan command buffer");
    }

    {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan blocks encoder");
        }
        [encoder setComputePipelineState:blocks_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)data_buffer->mtl_buffer offset:(NSUInteger)data_view->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)block_sums_buffer->mtl_buffer offset:(NSUInteger)block_sums_view->offset_bytes atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)block_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }

    if(block_count <= 1u) {
        [command_buffer commit];
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(block_count <= 256u) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan block-sums encoder");
        }
        [encoder setComputePipelineState:block_sums_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)block_sums_buffer->mtl_buffer offset:(NSUInteger)block_sums_view->offset_bytes atIndex:0];
        [encoder setBytes:&block_count length:sizeof(block_count) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    } else {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan second-level blocks encoder");
        }
        [encoder setComputePipelineState:blocks_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)block_sums_buffer->mtl_buffer offset:(NSUInteger)block_sums_view->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)scanned_block_sums_buffer->mtl_buffer offset:(NSUInteger)scanned_block_sums_view->offset_bytes atIndex:1];
        [encoder setBytes:&block_count length:sizeof(block_count) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)second_level_block_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan second-level block-sums encoder");
        }
        [encoder setComputePipelineState:block_sums_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)scanned_block_sums_buffer->mtl_buffer offset:(NSUInteger)scanned_block_sums_view->offset_bytes atIndex:0];
        [encoder setBytes:&second_level_block_count length:sizeof(second_level_block_count) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan add-offsets encoder");
        }
        [encoder setComputePipelineState:add_offsets_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)block_sums_buffer->mtl_buffer offset:(NSUInteger)block_sums_view->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)scanned_block_sums_buffer->mtl_buffer offset:(NSUInteger)scanned_block_sums_view->offset_bytes atIndex:1];
        [encoder setBytes:&block_count length:sizeof(block_count) atIndex:2];
        gsx_metal_backend_dispatch_threads_1d(encoder, add_offsets_pipeline, (NSUInteger)block_count);
        [encoder endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal scan final add-offsets encoder");
        }
        [encoder setComputePipelineState:add_offsets_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)data_buffer->mtl_buffer offset:(NSUInteger)data_view->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)block_sums_buffer->mtl_buffer offset:(NSUInteger)block_sums_view->offset_bytes atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        gsx_metal_backend_dispatch_threads_1d(encoder, add_offsets_pipeline, (NSUInteger)count);
        [encoder endEncoding];
    }

    [command_buffer commit];

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_sort_pairs_u32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *keys_in_view,
    const gsx_backend_tensor_view *values_in_view,
    const gsx_backend_tensor_view *keys_out_view,
    const gsx_backend_tensor_view *values_out_view,
    const gsx_backend_tensor_view *histogram_view,
    const gsx_backend_tensor_view *global_histogram_view,
    const gsx_backend_tensor_view *scatter_offsets_view,
    uint32_t count,
    uint32_t significant_bits)
{
    gsx_metal_backend *metal_backend = NULL;
    id<MTLComputePipelineState> histogram_pipeline = nil;
    id<MTLComputePipelineState> prefix_offsets_pipeline = nil;
    id<MTLComputePipelineState> scatter_pipeline = nil;
    gsx_metal_backend_buffer *keys_in_buffer = NULL;
    gsx_metal_backend_buffer *values_in_buffer = NULL;
    gsx_metal_backend_buffer *keys_out_buffer = NULL;
    gsx_metal_backend_buffer *values_out_buffer = NULL;
    gsx_metal_backend_buffer *histogram_buffer = NULL;
    gsx_metal_backend_buffer *global_histogram_buffer = NULL;
    gsx_metal_backend_buffer *scatter_offsets_buffer = NULL;
    id<MTLCommandBuffer> command_buffer = nil;
    uint32_t num_threadgroups = 0u;
    uint32_t shift = 0u;
    bool use_ping = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || keys_in_view == NULL || values_in_view == NULL || keys_out_view == NULL || values_out_view == NULL
        || histogram_view == NULL || global_histogram_view == NULL || scatter_offsets_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sort dispatch arguments must be non-null");
    }
    if(count <= 1u || significant_bits == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    num_threadgroups = (count + 1023u) / 1024u;
    metal_backend = gsx_metal_backend_from_base(backend);
    keys_in_buffer = gsx_metal_backend_buffer_from_base(keys_in_view->buffer);
    values_in_buffer = gsx_metal_backend_buffer_from_base(values_in_view->buffer);
    keys_out_buffer = gsx_metal_backend_buffer_from_base(keys_out_view->buffer);
    values_out_buffer = gsx_metal_backend_buffer_from_base(values_out_view->buffer);
    histogram_buffer = gsx_metal_backend_buffer_from_base(histogram_view->buffer);
    global_histogram_buffer = gsx_metal_backend_buffer_from_base(global_histogram_view->buffer);
    scatter_offsets_buffer = gsx_metal_backend_buffer_from_base(scatter_offsets_view->buffer);

    error = gsx_metal_backend_ensure_sort_histogram_pipeline(metal_backend, &histogram_pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_ensure_sort_prefix_offsets_pipeline(metal_backend, &prefix_offsets_pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_ensure_sort_scatter_pipeline(metal_backend, &scatter_pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal sort command buffer");
    }

    if(significant_bits > 32u) {
        significant_bits = 32u;
    }

    for(shift = 0u; shift < significant_bits; shift += 8u) {
        const gsx_backend_tensor_view *src_keys = use_ping ? keys_out_view : keys_in_view;
        const gsx_backend_tensor_view *src_values = use_ping ? values_out_view : values_in_view;
        const gsx_backend_tensor_view *dst_keys = use_ping ? keys_in_view : keys_out_view;
        const gsx_backend_tensor_view *dst_values = use_ping ? values_in_view : values_out_view;
        gsx_metal_backend_buffer *src_keys_buffer = use_ping ? keys_out_buffer : keys_in_buffer;
        gsx_metal_backend_buffer *src_values_buffer = use_ping ? values_out_buffer : values_in_buffer;
        gsx_metal_backend_buffer *dst_keys_buffer = use_ping ? keys_in_buffer : keys_out_buffer;
        gsx_metal_backend_buffer *dst_values_buffer = use_ping ? values_in_buffer : values_out_buffer;
        id<MTLComputeCommandEncoder> encoder = nil;

        encoder = [command_buffer computeCommandEncoder];
        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal sort histogram encoder");
        }
        [encoder setComputePipelineState:histogram_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)src_keys_buffer->mtl_buffer offset:(NSUInteger)src_keys->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)histogram_buffer->mtl_buffer offset:(NSUInteger)histogram_view->offset_bytes atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&shift length:sizeof(shift) atIndex:3];
        [encoder setThreadgroupMemoryLength:sizeof(uint32_t) * 256u * 8u atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)num_threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal sort prefix-offsets encoder");
        }
        [encoder setComputePipelineState:prefix_offsets_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)histogram_buffer->mtl_buffer offset:(NSUInteger)histogram_view->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)global_histogram_buffer->mtl_buffer offset:(NSUInteger)global_histogram_view->offset_bytes atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)scatter_offsets_buffer->mtl_buffer offset:(NSUInteger)scatter_offsets_view->offset_bytes atIndex:2];
        [encoder setBytes:&num_threadgroups length:sizeof(num_threadgroups) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        if(encoder == nil) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal sort scatter encoder");
        }
        [encoder setComputePipelineState:scatter_pipeline];
        [encoder setBuffer:(id<MTLBuffer>)src_keys_buffer->mtl_buffer offset:(NSUInteger)src_keys->offset_bytes atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)src_values_buffer->mtl_buffer offset:(NSUInteger)src_values->offset_bytes atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)dst_keys_buffer->mtl_buffer offset:(NSUInteger)dst_keys->offset_bytes atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)dst_values_buffer->mtl_buffer offset:(NSUInteger)dst_values->offset_bytes atIndex:3];
        [encoder setBuffer:(id<MTLBuffer>)scatter_offsets_buffer->mtl_buffer offset:(NSUInteger)scatter_offsets_view->offset_bytes atIndex:4];
        [encoder setBytes:&count length:sizeof(count) atIndex:5];
        [encoder setBytes:&shift length:sizeof(shift) atIndex:6];
        [encoder setThreadgroupMemoryLength:sizeof(uint32_t) * 256u atIndex:0];
        [encoder setThreadgroupMemoryLength:sizeof(uint32_t) * 256u atIndex:1];
        [encoder setThreadgroupMemoryLength:sizeof(uint32_t) * 256u * 8u atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)num_threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        use_ping = !use_ping;
    }

    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
