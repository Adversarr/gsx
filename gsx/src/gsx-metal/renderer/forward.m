#include "../internal.h"

#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct gsx_metal_forward_scratch {
    gsx_tensor_t depth;
    gsx_tensor_t visible;
    gsx_tensor_t touched;
    gsx_tensor_t bounds;
    gsx_tensor_t mean2d;
    gsx_tensor_t conic_opacity;
    gsx_tensor_t color;
    gsx_tensor_t sorted_primitive_ids;
    gsx_tensor_t primitive_offsets;
    gsx_tensor_t instance_keys;
    gsx_tensor_t instance_primitive_ids;
    gsx_tensor_t tile_ranges;
} gsx_metal_forward_scratch;

static bool gsx_metal_render_tensor_is_device_f32(gsx_tensor_t tensor)
{
    return tensor != NULL
        && tensor->data_type == GSX_DATA_TYPE_F32
        && tensor->backing_buffer != NULL
        && gsx_metal_backend_buffer_get_type_class(tensor->backing_buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static bool gsx_metal_render_tensor_is_gpu_i32(gsx_tensor_t tensor)
{
    gsx_backend_buffer_type_class type_class = GSX_BACKEND_BUFFER_TYPE_HOST;

    if(tensor == NULL || tensor->data_type != GSX_DATA_TYPE_I32 || tensor->backing_buffer == NULL) {
        return false;
    }
    type_class = gsx_metal_backend_buffer_get_type_class(tensor->backing_buffer);
    return type_class != GSX_BACKEND_BUFFER_TYPE_HOST;
}

static gsx_error gsx_metal_render_make_tensor(
    gsx_arena_t arena,
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_tensor_t *out_tensor)
{
    gsx_tensor_desc desc = { 0 };

    if(arena == NULL || shape == NULL || out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena, shape, and out_tensor must be non-null");
    }

    desc.rank = rank;
    for(gsx_index_t i = 0; i < rank; ++i) {
        desc.shape[i] = shape[i];
    }
    desc.data_type = data_type;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    return gsx_tensor_init(out_tensor, &desc);
}

static void gsx_metal_render_release_tensor(gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        (void)gsx_tensor_free(*tensor);
        *tensor = NULL;
    }
}

static void gsx_metal_render_cleanup_forward_scratch(gsx_metal_forward_scratch *scratch)
{
    gsx_metal_render_release_tensor(&scratch->tile_ranges);
    gsx_metal_render_release_tensor(&scratch->instance_primitive_ids);
    gsx_metal_render_release_tensor(&scratch->instance_keys);
    gsx_metal_render_release_tensor(&scratch->primitive_offsets);
    gsx_metal_render_release_tensor(&scratch->sorted_primitive_ids);
    gsx_metal_render_release_tensor(&scratch->color);
    gsx_metal_render_release_tensor(&scratch->conic_opacity);
    gsx_metal_render_release_tensor(&scratch->mean2d);
    gsx_metal_render_release_tensor(&scratch->bounds);
    gsx_metal_render_release_tensor(&scratch->touched);
    gsx_metal_render_release_tensor(&scratch->visible);
    gsx_metal_render_release_tensor(&scratch->depth);
}

static void gsx_metal_render_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    out_view->buffer = tensor->backing_buffer;
    out_view->offset_bytes = tensor->offset_bytes;
    out_view->size_bytes = tensor->size_bytes;
    out_view->effective_alignment_bytes = tensor->effective_alignment_bytes;
    out_view->data_type = tensor->data_type;
}

static gsx_error gsx_metal_renderer_validate_forward_scope(const gsx_render_forward_request *request)
{
    if(request->precision != GSX_RENDER_PRECISION_FLOAT32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer currently supports only float32 precision");
    }
    if(request->forward_type != GSX_RENDER_FORWARD_TYPE_INFERENCE
        && request->forward_type != GSX_RENDER_FORWARD_TYPE_TRAIN) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer currently supports only inference/train forward");
    }
    if(request->sh_degree != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer currently supports only sh_degree=0");
    }
    if(request->gs_cov3d != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer does not support gs_cov3d input");
    }
    if(request->out_alpha != NULL || request->out_invdepth != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer alpha/invdepth outputs are not implemented");
    }
    if(request->metric_map != NULL || request->gs_metric_accumulator != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer metric path is not implemented");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_reserve_host_buffer(void **buffer, gsx_size_t *capacity, gsx_size_t required_count, gsx_size_t element_size, const char *error_label)
{
    void *new_buffer = NULL;
    gsx_size_t required_bytes = 0;

    if(buffer == NULL || capacity == NULL || error_label == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "host staging reserve arguments must be non-null");
    }
    if(required_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(*buffer != NULL && *capacity >= required_count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_mul_overflows(required_count, element_size, &required_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "host staging size overflow");
    }

    new_buffer = realloc(*buffer, (size_t)required_bytes);
    if(new_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, error_label);
    }
    *buffer = new_buffer;
    *capacity = required_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_tensor_map_host_bytes(gsx_tensor_t tensor, void **out_bytes, gsx_size_t *out_size_bytes)
{
    void *native_handle = NULL;
    gsx_size_t offset_bytes = 0;
    id<MTLBuffer> mtl_buffer = nil;
    unsigned char *base_bytes = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL || out_bytes == NULL || out_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and output pointers must be non-null");
    }

    *out_bytes = NULL;
    *out_size_bytes = 0;
    error = gsx_tensor_get_native_handle(tensor, &native_handle, &offset_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    mtl_buffer = (id<MTLBuffer>)native_handle;
    if(mtl_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "Metal tensor native handle is unavailable");
    }

    base_bytes = (unsigned char *)[mtl_buffer contents];
    if(base_bytes == NULL) {
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "Metal tensor is not CPU-visible; map_host_bytes requires unified/host-visible storage");
    }

    *out_bytes = (void *)(base_bytes + (size_t)offset_bytes);
    *out_size_bytes = tensor->size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static uint32_t gsx_metal_render_depth_sort_key(float depth)
{
    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = depth;
    return bits.u;
}

static gsx_error gsx_metal_render_gather_sort_and_build_offsets(
    const float *depth,
    const int32_t *visible,
    const int32_t *touched,
    gsx_size_t gaussian_count,
    gsx_metal_sort_pair_u32 *pairs,
    int32_t *out_sorted_primitive_ids,
    int32_t *out_primitive_offsets,
    gsx_size_t *out_visible_count,
    gsx_size_t *out_instance_count)
{
    gsx_size_t visible_count = 0;
    gsx_size_t instance_count = 0;

    if(depth == NULL || visible == NULL || touched == NULL || pairs == NULL
        || out_sorted_primitive_ids == NULL || out_primitive_offsets == NULL
        || out_visible_count == NULL || out_instance_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gather_sort_and_build_offsets: all pointers must be non-null");
    }

    for(gsx_size_t i = 0; i < gaussian_count; ++i) {
        if(visible[i] != 0 && touched[i] > 0) {
            pairs[visible_count].key = gsx_metal_render_depth_sort_key(depth[i]);
            pairs[visible_count].value = (uint32_t)i;
            pairs[visible_count].stable_index = (uint32_t)visible_count;
            visible_count += 1;
        }
    }
    gsx_metal_render_sort_pairs_u32(pairs, (uint32_t)visible_count);

    for(gsx_size_t i = 0; i < visible_count; ++i) {
        uint32_t primitive_index = pairs[i].value;

        out_sorted_primitive_ids[i] = (int32_t)primitive_index;
        out_primitive_offsets[i] = (int32_t)instance_count;
        instance_count += (gsx_size_t)touched[primitive_index];
    }

    *out_visible_count = visible_count;
    *out_instance_count = instance_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_render_sort_instances_inplace(
    int32_t *instance_keys,
    int32_t *instance_primitive_ids,
    gsx_size_t instance_count,
    gsx_metal_sort_pair_u32 *pairs)
{
    for(gsx_size_t i = 0; i < instance_count; ++i) {
        pairs[i].key = (uint32_t)instance_keys[i];
        pairs[i].value = (uint32_t)instance_primitive_ids[i];
        pairs[i].stable_index = (uint32_t)i;
    }
    gsx_metal_render_sort_pairs_u32(pairs, (uint32_t)instance_count);
    for(gsx_size_t i = 0; i < instance_count; ++i) {
        instance_keys[i] = (int32_t)pairs[i].key;
        instance_primitive_ids[i] = (int32_t)pairs[i].value;
    }
}

static void gsx_metal_render_fill_tile_ranges(
    const int32_t *instance_keys,
    gsx_size_t instance_count,
    int32_t *tile_ranges,
    gsx_size_t tile_count)
{
    for(gsx_size_t i = 0; i < tile_count; ++i) {
        tile_ranges[i * 2u] = 0;
        tile_ranges[i * 2u + 1u] = 0;
    }
    for(gsx_size_t i = 0; i < instance_count; ++i) {
        int32_t key = instance_keys[i];

        if(key < 0 || (gsx_size_t)key >= tile_count) {
            continue;
        }
        if(tile_ranges[(gsx_size_t)key * 2u] == 0 && tile_ranges[(gsx_size_t)key * 2u + 1u] == 0) {
            tile_ranges[(gsx_size_t)key * 2u] = (int32_t)i;
        }
        tile_ranges[(gsx_size_t)key * 2u + 1u] = (int32_t)(i + 1u);
    }
}

gsx_error gsx_metal_renderer_forward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request)
{
    gsx_metal_render_context *metal_context = (gsx_metal_render_context *)context;
    gsx_metal_forward_scratch scratch;
    int32_t *host_visible = NULL;
    int32_t *host_touched = NULL;
    int32_t *host_sorted_primitive_ids = NULL;
    int32_t *host_primitive_offsets = NULL;
    int32_t *host_instance_keys = NULL;
    int32_t *host_instance_primitive_ids = NULL;
    int32_t *host_tile_ranges = NULL;
    float *host_depth = NULL;
    gsx_size_t gaussian_count = 0;
    gsx_size_t visible_count = 0;
    gsx_size_t instance_count = 0;
    gsx_size_t tile_count = 0;
    gsx_size_t scratch_required_bytes = 0;
    gsx_size_t staging_required_bytes = 0;
    gsx_size_t mapped_size_bytes = 0;
    gsx_index_t grid_width = 0;
    gsx_index_t grid_height = 0;
    gsx_backend_tensor_view image_view = { 0 };
    gsx_backend_tensor_view alpha_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_backend_tensor_view mean3d_in_view = { 0 };
    gsx_backend_tensor_view rotation_in_view = { 0 };
    gsx_backend_tensor_view logscale_in_view = { 0 };
    gsx_backend_tensor_view sh0_in_view = { 0 };
    gsx_backend_tensor_view opacity_in_view = { 0 };
    gsx_backend_tensor_view depth_view = { 0 };
    gsx_backend_tensor_view visible_view = { 0 };
    gsx_backend_tensor_view touched_view = { 0 };
    gsx_backend_tensor_view bounds_view = { 0 };
    gsx_backend_tensor_view mean2d_view = { 0 };
    gsx_backend_tensor_view conic_opacity_view = { 0 };
    gsx_backend_tensor_view color_view = { 0 };
    gsx_backend_tensor_view sorted_primitive_ids_view = { 0 };
    gsx_backend_tensor_view primitive_offsets_view = { 0 };
    gsx_backend_tensor_view instance_keys_view = { 0 };
    gsx_backend_tensor_view instance_primitive_ids_view = { 0 };
    gsx_backend_tensor_view tile_ranges_view = { 0 };
    gsx_metal_render_compose_params compose_params = { 0 };
    gsx_metal_render_preprocess_params preprocess_params = { 0 };
    gsx_metal_render_create_instances_params create_params = { 0 };
    gsx_metal_render_blend_params blend_params = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    memset(&scratch, 0, sizeof(scratch));
    if(renderer == NULL || context == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer, context, and request must be non-null");
    }

    error = gsx_metal_renderer_validate_forward_scope(request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_render_context_clear_train_state(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_metal_render_tensor_is_device_f32(request->gs_mean3d)
        || !gsx_metal_render_tensor_is_device_f32(request->gs_rotation)
        || !gsx_metal_render_tensor_is_device_f32(request->gs_logscale)
        || !gsx_metal_render_tensor_is_device_f32(request->gs_sh0)
        || !gsx_metal_render_tensor_is_device_f32(request->gs_opacity)
        || !gsx_metal_render_tensor_is_device_f32(request->out_rgb)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer currently requires device-backed float32 render tensors");
    }

    error = gsx_arena_reset(metal_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(metal_context->staging_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_tensor_set_zero(metal_context->helper_image_chw);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(metal_context->helper_alpha_hw);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gaussian_count = (gsx_size_t)request->gs_mean3d->shape[0];
    grid_width = (renderer->info.width + 15) / 16;
    grid_height = (renderer->info.height + 15) / 16;
    tile_count = (gsx_size_t)grid_width * (gsx_size_t)grid_height;

    if(gsx_size_mul_overflows(gaussian_count, 256u, &scratch_required_bytes)
        || gsx_size_add_overflows(scratch_required_bytes, tile_count * 16u, &scratch_required_bytes)
        || gsx_size_add_overflows(scratch_required_bytes, 65536u, &scratch_required_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward scratch sizing overflow");
    }
    error = gsx_arena_reserve(metal_context->scratch_arena, scratch_required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(gsx_size_mul_overflows(gaussian_count, 96u, &staging_required_bytes)
        || gsx_size_add_overflows(staging_required_bytes, tile_count * 8u, &staging_required_bytes)
        || gsx_size_add_overflows(staging_required_bytes, 65536u, &staging_required_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward staging sizing overflow");
    }
    error = gsx_arena_reserve(metal_context->staging_arena, staging_required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gaussian_count > 0) {
        gsx_index_t shape_n[1] = { (gsx_index_t)gaussian_count };
        gsx_index_t shape_n2[2] = { (gsx_index_t)gaussian_count, 2 };
        gsx_index_t shape_n3[2] = { (gsx_index_t)gaussian_count, 3 };
        gsx_index_t shape_n4[2] = { (gsx_index_t)gaussian_count, 4 };

        error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_F32, 1, shape_n, &scratch.depth);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.visible);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.touched);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        /* sorted_primitive_ids and primitive_offsets in staging so the GPU reads them
         * zero-copy and the CPU writes them directly after sync 1, with no upload step. */
        error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.sorted_primitive_ids);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.primitive_offsets);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch.bounds);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n2, &scratch.mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch.conic_opacity);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &scratch.color);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        gsx_metal_render_make_tensor_view(request->gs_mean3d, &mean3d_in_view);
        gsx_metal_render_make_tensor_view(request->gs_rotation, &rotation_in_view);
        gsx_metal_render_make_tensor_view(request->gs_logscale, &logscale_in_view);
        gsx_metal_render_make_tensor_view(request->gs_sh0, &sh0_in_view);
        gsx_metal_render_make_tensor_view(request->gs_opacity, &opacity_in_view);
        gsx_metal_render_make_tensor_view(scratch.depth, &depth_view);
        gsx_metal_render_make_tensor_view(scratch.visible, &visible_view);
        gsx_metal_render_make_tensor_view(scratch.touched, &touched_view);
        gsx_metal_render_make_tensor_view(scratch.bounds, &bounds_view);
        gsx_metal_render_make_tensor_view(scratch.mean2d, &mean2d_view);
        gsx_metal_render_make_tensor_view(scratch.conic_opacity, &conic_opacity_view);
        gsx_metal_render_make_tensor_view(scratch.color, &color_view);

        preprocess_params.gaussian_count = (uint32_t)gaussian_count;
        preprocess_params.width = (uint32_t)renderer->info.width;
        preprocess_params.height = (uint32_t)renderer->info.height;
        preprocess_params.grid_width = (uint32_t)grid_width;
        preprocess_params.grid_height = (uint32_t)grid_height;
        preprocess_params.fx = request->intrinsics->fx;
        preprocess_params.fy = request->intrinsics->fy;
        preprocess_params.cx = request->intrinsics->cx;
        preprocess_params.cy = request->intrinsics->cy;
        preprocess_params.near_plane = request->near_plane;
        preprocess_params.far_plane = request->far_plane;
        preprocess_params.pose_qx = request->pose->rot.x;
        preprocess_params.pose_qy = request->pose->rot.y;
        preprocess_params.pose_qz = request->pose->rot.z;
        preprocess_params.pose_qw = request->pose->rot.w;
        preprocess_params.pose_tx = request->pose->transl.x;
        preprocess_params.pose_ty = request->pose->transl.y;
        preprocess_params.pose_tz = request->pose->transl.z;

        error = gsx_metal_backend_dispatch_render_preprocess(
            renderer->backend,
            &mean3d_in_view,
            &rotation_in_view,
            &logscale_in_view,
            &sh0_in_view,
            &opacity_in_view,
            &depth_view,
            &visible_view,
            &touched_view,
            &bounds_view,
            &mean2d_view,
            &conic_opacity_view,
            &color_view,
            &preprocess_params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        /* Sync 1 of 3: preprocess outputs (depth, visible, touched) are in staging
         * (unified memory). sorted_primitive_ids and primitive_offsets are also in
         * staging, so the CPU writes them directly here without an upload step. */
        error = gsx_backend_major_stream_sync(renderer->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.depth, (void **)&host_depth, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(float)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map preprocess depth tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.visible, (void **)&host_visible, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map preprocess visible tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.touched, (void **)&host_touched, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map preprocess touched tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.sorted_primitive_ids, (void **)&host_sorted_primitive_ids, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted-primitive-id tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.primitive_offsets, (void **)&host_primitive_offsets, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map primitive-offset tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_reserve_host_buffer(
            (void **)&metal_context->host_visible_pairs,
            &metal_context->host_gaussian_capacity,
            gaussian_count,
            sizeof(gsx_metal_sort_pair_u32),
            "failed to reserve host visible pair staging buffer");
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        /* Gather visible Gaussians, sort front-to-back by depth, and build
         * sorted_primitive_ids and primitive_offsets into staging memory. */
        error = gsx_metal_render_gather_sort_and_build_offsets(
            host_depth, host_visible, host_touched, gaussian_count,
            metal_context->host_visible_pairs,
            host_sorted_primitive_ids, host_primitive_offsets,
            &visible_count, &instance_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        /* CPU→GPU synchronization note: sorted_primitive_ids and primitive_offsets
         * are now populated in unified (MTLStorageModeShared) memory. Metal shared
         * storage is coherent, and GPU command encoding happens sequentially on this
         * main thread after CPU writes complete, so no explicit barrier is needed.
         * This assumption would NOT hold for discrete GPU memory or async encoding. */

        if(instance_count > 0) {
            gsx_index_t shape_instances[1] = { (gsx_index_t)instance_count };

            error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 1, shape_instances, &scratch.instance_keys);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 1, shape_instances, &scratch.instance_primitive_ids);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            gsx_metal_render_make_tensor_view(scratch.sorted_primitive_ids, &sorted_primitive_ids_view);
            gsx_metal_render_make_tensor_view(scratch.primitive_offsets, &primitive_offsets_view);
            gsx_metal_render_make_tensor_view(scratch.bounds, &bounds_view);
            gsx_metal_render_make_tensor_view(scratch.instance_keys, &instance_keys_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids, &instance_primitive_ids_view);

            create_params.visible_count = (uint32_t)visible_count;
            create_params.grid_width = (uint32_t)grid_width;
            create_params.grid_height = (uint32_t)grid_height;
            error = gsx_metal_backend_dispatch_render_create_instances(
                renderer->backend,
                &sorted_primitive_ids_view,
                &primitive_offsets_view,
                &bounds_view,
                &mean2d_view,
                &conic_opacity_view,
                &instance_keys_view,
                &instance_primitive_ids_view,
                &create_params);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            /* Sync 2 of 3: instance_keys and instance_primitive_ids are in staging
             * (unified memory). Sort them in-place immediately after this fence. */
            error = gsx_backend_major_stream_sync(renderer->backend);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_keys, (void **)&host_instance_keys, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)instance_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map instance key tensor to host-visible bytes");
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_primitive_ids, (void **)&host_instance_primitive_ids, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)instance_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map instance primitive-id tensor to host-visible bytes");
                goto cleanup;
            }
            error = gsx_metal_render_reserve_host_buffer(
                (void **)&metal_context->host_instance_pairs,
                &metal_context->host_instance_capacity,
                instance_count,
                sizeof(gsx_metal_sort_pair_u32),
                "failed to reserve host instance pair staging buffer");
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            /* Sort instances by tile key, writing sorted keys and primitive IDs
             * back in-place to the staging (unified) tensors. */
            gsx_metal_render_sort_instances_inplace(
                host_instance_keys, host_instance_primitive_ids,
                instance_count, metal_context->host_instance_pairs);

        }

        if(tile_count > 0) {
            gsx_index_t shape_tile_ranges[2] = { (gsx_index_t)tile_count, 2 };

            /* tile_ranges in staging (unified): CPU fills it zero-copy, GPU reads without upload. */
            error = gsx_metal_render_make_tensor(metal_context->staging_arena, GSX_DATA_TYPE_I32, 2, shape_tile_ranges, &scratch.tile_ranges);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_ranges, (void **)&host_tile_ranges, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)tile_count * 2u * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile-range tensor to host-visible bytes");
                goto cleanup;
            }
            gsx_metal_render_fill_tile_ranges(
                host_instance_keys, instance_count, host_tile_ranges, tile_count);
        }

        if(instance_count > 0 && tile_count > 0) {
            if(!gsx_metal_render_tensor_is_gpu_i32(scratch.tile_ranges)
                || !gsx_metal_render_tensor_is_gpu_i32(scratch.instance_primitive_ids)
                || !gsx_metal_render_tensor_is_device_f32(scratch.mean2d)
                || !gsx_metal_render_tensor_is_device_f32(scratch.conic_opacity)
                || !gsx_metal_render_tensor_is_device_f32(scratch.color)) {
                error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal blend staging tensors are not in required device formats");
                goto cleanup;
            }

            gsx_metal_render_make_tensor_view(scratch.tile_ranges, &tile_ranges_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids, &instance_primitive_ids_view);
            gsx_metal_render_make_tensor_view(scratch.mean2d, &mean2d_view);
            gsx_metal_render_make_tensor_view(scratch.conic_opacity, &conic_opacity_view);
            gsx_metal_render_make_tensor_view(scratch.color, &color_view);
            gsx_metal_render_make_tensor_view(metal_context->helper_image_chw, &image_view);
            gsx_metal_render_make_tensor_view(metal_context->helper_alpha_hw, &alpha_view);

            blend_params.width = (uint32_t)renderer->info.width;
            blend_params.height = (uint32_t)renderer->info.height;
            blend_params.grid_width = (uint32_t)grid_width;
            blend_params.grid_height = (uint32_t)grid_height;
            blend_params.tile_count = (uint32_t)tile_count;
            blend_params.channel_stride = (uint32_t)((gsx_size_t)renderer->info.width * (gsx_size_t)renderer->info.height);

            error = gsx_metal_backend_dispatch_render_blend(
                renderer->backend,
                &tile_ranges_view,
                &instance_primitive_ids_view,
                &mean2d_view,
                &conic_opacity_view,
                &color_view,
                &image_view,
                &alpha_view,
                &blend_params);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }
    }

    gsx_metal_render_make_tensor_view(metal_context->helper_image_chw, &image_view);
    gsx_metal_render_make_tensor_view(metal_context->helper_alpha_hw, &alpha_view);
    gsx_metal_render_make_tensor_view(request->out_rgb, &out_view);

    compose_params.width = (uint32_t)renderer->info.width;
    compose_params.height = (uint32_t)renderer->info.height;
    compose_params.channel_stride = (uint32_t)((gsx_size_t)renderer->info.width * (gsx_size_t)renderer->info.height);
    compose_params.background_r = request->background_color.x;
    compose_params.background_g = request->background_color.y;
    compose_params.background_b = request->background_color.z;
    error = gsx_metal_backend_dispatch_render_compose_f32(
        renderer->backend,
        &image_view,
        &alpha_view,
        &out_view,
        &compose_params);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    /* Sync 3 of 3: ensure compose_f32 completes before returning the output
     * tensor to the caller as valid. */
    error = gsx_backend_major_stream_sync(renderer->backend);

    if(gsx_error_is_success(error) && request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN) {
        error = gsx_metal_render_context_snapshot_train_state(
            metal_context,
            request,
            scratch.mean2d,
            scratch.conic_opacity,
            scratch.color,
            scratch.instance_primitive_ids,
            scratch.tile_ranges);
    }

cleanup:
    gsx_metal_render_cleanup_forward_scratch(&scratch);
    (void)gsx_arena_reset(metal_context->scratch_arena);
    (void)gsx_arena_reset(metal_context->staging_arena);
    return error;
}
