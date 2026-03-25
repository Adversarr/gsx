#include "../internal.h"

#import <Metal/Metal.h>

bool gsx_metal_render_tensor_is_device_f32(gsx_tensor_t tensor) { return gsx_metal_tensor_is_device_f32(tensor); }

bool gsx_metal_render_tensor_is_optional_device_f32(gsx_tensor_t tensor) { return gsx_metal_tensor_is_optional_device_f32(tensor); }

bool gsx_metal_render_tensor_is_backed_f32(gsx_tensor_t tensor) { return gsx_metal_tensor_is_backed_f32(tensor); }

bool gsx_metal_render_tensor_is_backed_i32(gsx_tensor_t tensor) { return gsx_metal_tensor_is_backed_i32(tensor); }

gsx_error gsx_metal_render_make_tensor(
    gsx_arena_t arena,
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_tensor_t *out_tensor)
{
    return gsx_metal_render_make_tensor_aligned(arena, data_type, rank, shape, sizeof(float), out_tensor);
}

gsx_error gsx_metal_render_make_tensor_aligned(
    gsx_arena_t arena,
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_size_t requested_alignment_bytes,
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
    desc.requested_alignment_bytes = requested_alignment_bytes;
    desc.arena = arena;
    return gsx_tensor_init(out_tensor, &desc);
}

gsx_error gsx_metal_render_validate_tensor_alignment(gsx_tensor_t tensor, gsx_size_t required_alignment_bytes, const char *tensor_name)
{
    if(tensor == NULL || tensor_name == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor alignment validation requires non-null inputs");
    }
    if(required_alignment_bytes == 0u) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "required_alignment_bytes must be non-zero");
    }
    if(tensor->effective_alignment_bytes < required_alignment_bytes) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, tensor_name);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_metal_render_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    gsx_tensor_fill_backend_view(tensor, out_view);
}

void gsx_metal_render_release_tensor(gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        (void)gsx_tensor_free(*tensor);
        *tensor = NULL;
    }
}

gsx_error gsx_metal_render_reserve_arena_with_dry_run(
    gsx_arena_t target_arena,
    gsx_metal_render_dry_run_plan_fn plan_fn,
    void *plan_user_data)
{
    gsx_arena_desc dry_run_desc = { 0 };
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_t dry_run_arena = NULL;
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(target_arena == NULL || plan_fn == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "target_arena and plan_fn must be non-null");
    }

    error = gsx_arena_get_buffer_type(target_arena, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    dry_run_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, buffer_type, &dry_run_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = plan_fn(dry_run_arena, plan_user_data);
    if(!gsx_error_is_success(error)) {
        (void)gsx_arena_free(dry_run_arena);
        return error;
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, &required_bytes);
    if(!gsx_error_is_success(error)) {
        (void)gsx_arena_free(dry_run_arena);
        return error;
    }

    error = gsx_arena_free(dry_run_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_arena_reserve(target_arena, required_bytes);
}

gsx_error gsx_metal_render_validate_train_state_for_backward(const gsx_metal_render_context *metal_context)
{
    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }
    if(!metal_context->has_train_state) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backward requires a retained TRAIN forward on the same context");
    }
    if(metal_context->saved_sh_degree < 0 || metal_context->saved_sh_degree > 3) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer backward supports sh_degree in range [0,3]");
    }
    if(metal_context->saved_mean3d == NULL || metal_context->saved_rotation == NULL || metal_context->saved_logscale == NULL
        || metal_context->saved_sh0 == NULL || metal_context->saved_opacity == NULL || metal_context->saved_mean2d == NULL
        || metal_context->saved_conic_opacity == NULL || metal_context->saved_color == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal renderer backward train state is incomplete");
    }
    if(metal_context->saved_sh_degree >= 1 && metal_context->saved_sh1 == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal renderer backward requires saved_sh1 for sh_degree >= 1");
    }
    if(metal_context->saved_sh_degree >= 2 && metal_context->saved_sh2 == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal renderer backward requires saved_sh2 for sh_degree >= 2");
    }
    if(metal_context->saved_sh_degree >= 3 && metal_context->saved_sh3 == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal renderer backward requires saved_sh3 for sh_degree >= 3");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_render_tensor_map_host_bytes(gsx_tensor_t tensor, void **out_bytes, gsx_size_t *out_size_bytes)
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

gsx_index_t gsx_metal_render_get_grid_width(gsx_index_t width)
{
    return (width + 15) / 16;
}

gsx_index_t gsx_metal_render_get_grid_height(gsx_index_t height)
{
    return (height + 15) / 16;
}

gsx_size_t gsx_metal_render_get_tile_count(gsx_index_t width, gsx_index_t height)
{
    return (gsx_size_t)gsx_metal_render_get_grid_width(width) * (gsx_size_t)gsx_metal_render_get_grid_height(height);
}

gsx_size_t gsx_metal_render_get_channel_stride(gsx_index_t width, gsx_index_t height)
{
    return (gsx_size_t)width * (gsx_size_t)height;
}
