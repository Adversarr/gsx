#include "../internal.h"

#include <stdlib.h>
#include <string.h>

static gsx_error gsx_metal_render_init_arena(gsx_backend_buffer_type_t buffer_type, gsx_arena_t *out_arena)
{
    gsx_arena_desc arena_desc = { 0 };

    if(buffer_type == NULL || out_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_arena must be non-null");
    }

    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    return gsx_arena_init(out_arena, buffer_type, &arena_desc);
}

static gsx_error gsx_metal_render_make_tensor(
    gsx_arena_t arena,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_tensor_t *out_tensor)
{
    gsx_tensor_desc desc = { 0 };

    if(arena == NULL || shape == NULL || out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena, shape, and out_tensor must be non-null");
    }

    memset(&desc, 0, sizeof(desc));
    desc.rank = rank;
    for(gsx_index_t i = 0; i < rank; ++i) {
        desc.shape[i] = shape[i];
    }
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    return gsx_tensor_init(out_tensor, &desc);
}

gsx_error gsx_metal_render_context_init(
    gsx_metal_render_context *metal_context,
    gsx_backend_buffer_type_t buffer_type,
    gsx_index_t width,
    gsx_index_t height)
{
    gsx_index_t image_shape[3] = { 3, height, width };
    gsx_index_t alpha_shape[2] = { height, width };
    gsx_size_t pixel_count = 0;
    gsx_size_t image_bytes = 0;
    gsx_size_t alpha_bytes = 0;
    gsx_size_t required_bytes = 0;
    gsx_backend_buffer_type_t unified_buffer_type = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and buffer_type must be non-null");
    }

    metal_context->helper_arena = NULL;
    metal_context->scratch_arena = NULL;
    metal_context->staging_arena = NULL;
    metal_context->helper_image_chw = NULL;
    metal_context->helper_alpha_hw = NULL;
    metal_context->host_visible_pairs = NULL;
    metal_context->host_instance_pairs = NULL;
    metal_context->host_gaussian_capacity = 0;
    metal_context->host_instance_capacity = 0;

    error = gsx_metal_render_init_arena(buffer_type, &metal_context->helper_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_init_arena(buffer_type, &metal_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }
    error = gsx_backend_find_buffer_type(buffer_type->backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED, &unified_buffer_type);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }
    error = gsx_metal_render_init_arena(unified_buffer_type, &metal_context->staging_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    if(gsx_size_mul_overflows((gsx_size_t)width, (gsx_size_t)height, &pixel_count)
        || gsx_size_mul_overflows(pixel_count, 3u * sizeof(float), &image_bytes)
        || gsx_size_mul_overflows(pixel_count, sizeof(float), &alpha_bytes)
        || gsx_size_add_overflows(image_bytes, alpha_bytes, &required_bytes)
        || gsx_size_add_overflows(required_bytes, 4096u, &required_bytes)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render helper tensor storage size overflow");
    }

    error = gsx_arena_reserve(metal_context->helper_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    error = gsx_metal_render_make_tensor(metal_context->helper_arena, 3, image_shape, &metal_context->helper_image_chw);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    error = gsx_metal_render_make_tensor(metal_context->helper_arena, 2, alpha_shape, &metal_context->helper_alpha_hw);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_render_context_dispose(gsx_metal_render_context *metal_context)
{
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    free(metal_context->host_instance_pairs);
    free(metal_context->host_visible_pairs);
    metal_context->host_instance_pairs = NULL;
    metal_context->host_visible_pairs = NULL;
    metal_context->host_gaussian_capacity = 0;
    metal_context->host_instance_capacity = 0;

    if(metal_context->helper_alpha_hw != NULL) {
        error = gsx_tensor_free(metal_context->helper_alpha_hw);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->helper_alpha_hw = NULL;
    }
    if(metal_context->helper_image_chw != NULL) {
        error = gsx_tensor_free(metal_context->helper_image_chw);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->helper_image_chw = NULL;
    }
    if(metal_context->helper_arena != NULL) {
        error = gsx_arena_free(metal_context->helper_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->helper_arena = NULL;
    }
    if(metal_context->scratch_arena != NULL) {
        error = gsx_arena_free(metal_context->scratch_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->scratch_arena = NULL;
    }
    if(metal_context->staging_arena != NULL) {
        error = gsx_arena_free(metal_context->staging_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->staging_arena = NULL;
    }

    return first_error;
}
