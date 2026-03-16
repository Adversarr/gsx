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

static void gsx_metal_render_free_tensor_handle(gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        (void)gsx_tensor_free(*tensor);
        *tensor = NULL;
    }
}

static gsx_error gsx_metal_render_clone_tensor(gsx_tensor_t src, gsx_arena_t arena, gsx_tensor_t *out_clone)
{
    gsx_tensor_desc desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src == NULL || arena == NULL || out_clone == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src, arena, and out_clone must be non-null");
    }

    *out_clone = NULL;
    error = gsx_tensor_get_desc(src, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    desc.arena = arena;

    error = gsx_tensor_init(out_clone, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_tensor_copy(src, *out_clone);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(*out_clone);
        *out_clone = NULL;
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_accumulate_tensor_size(gsx_tensor_t tensor, gsx_size_t *io_total)
{
    if(io_total == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "io_total must be non-null");
    }
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_add_overflows(*io_total, tensor->size_bytes, io_total)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal retain train-state size overflow");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_render_context_clear_train_state(gsx_metal_render_context *metal_context)
{
    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    if(!metal_context->train_state_borrowed) {
        gsx_metal_render_free_tensor_handle(&metal_context->saved_mean3d);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_rotation);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_logscale);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_sh0);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_sh1);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_sh2);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_sh3);
        gsx_metal_render_free_tensor_handle(&metal_context->saved_opacity);
    }

    gsx_metal_render_free_tensor_handle(&metal_context->saved_mean2d);
    gsx_metal_render_free_tensor_handle(&metal_context->saved_conic_opacity);
    gsx_metal_render_free_tensor_handle(&metal_context->saved_color);
    gsx_metal_render_free_tensor_handle(&metal_context->saved_instance_primitive_ids);
    gsx_metal_render_free_tensor_handle(&metal_context->saved_tile_ranges);

    metal_context->saved_mean3d = NULL;
    metal_context->saved_rotation = NULL;
    metal_context->saved_logscale = NULL;
    metal_context->saved_sh0 = NULL;
    metal_context->saved_sh1 = NULL;
    metal_context->saved_sh2 = NULL;
    metal_context->saved_sh3 = NULL;
    metal_context->saved_opacity = NULL;
    metal_context->saved_intrinsics = (gsx_camera_intrinsics){ 0 };
    metal_context->saved_pose = (gsx_camera_pose){ 0 };
    metal_context->saved_background_color = (gsx_vec3){ 0 };
    metal_context->saved_near_plane = 0.0f;
    metal_context->saved_far_plane = 0.0f;
    metal_context->saved_sh_degree = 0;
    metal_context->has_train_state = false;
    metal_context->train_state_borrowed = false;

    if(metal_context->retain_arena != NULL) {
        return gsx_arena_reset(metal_context->retain_arena);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_render_context_snapshot_train_state(
    gsx_metal_render_context *metal_context,
    const gsx_render_forward_request *request,
    gsx_tensor_t mean2d,
    gsx_tensor_t conic_opacity,
    gsx_tensor_t color,
    gsx_tensor_t instance_primitive_ids,
    gsx_tensor_t tile_ranges)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t required_bytes = 0;

    if(metal_context == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and request must be non-null");
    }

    error = gsx_metal_render_context_clear_train_state(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_render_accumulate_tensor_size(request->gs_mean3d, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_rotation, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_logscale, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_sh0, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_sh1, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_sh2, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_sh3, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(request->gs_opacity, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(mean2d, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(conic_opacity, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(color, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(instance_primitive_ids, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_render_accumulate_tensor_size(tile_ranges, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(gsx_size_add_overflows(required_bytes, 4096u, &required_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal retain train-state size overflow");
    }
    error = gsx_arena_reserve(metal_context->retain_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(request->borrow_train_state) {
        metal_context->saved_mean3d = request->gs_mean3d;
        metal_context->saved_rotation = request->gs_rotation;
        metal_context->saved_logscale = request->gs_logscale;
        metal_context->saved_sh0 = request->gs_sh0;
        metal_context->saved_sh1 = request->gs_sh1;
        metal_context->saved_sh2 = request->gs_sh2;
        metal_context->saved_sh3 = request->gs_sh3;
        metal_context->saved_opacity = request->gs_opacity;
        metal_context->train_state_borrowed = true;
    } else {
        error = gsx_metal_render_clone_tensor(request->gs_mean3d, metal_context->retain_arena, &metal_context->saved_mean3d);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
        error = gsx_metal_render_clone_tensor(request->gs_rotation, metal_context->retain_arena, &metal_context->saved_rotation);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
        error = gsx_metal_render_clone_tensor(request->gs_logscale, metal_context->retain_arena, &metal_context->saved_logscale);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
        error = gsx_metal_render_clone_tensor(request->gs_sh0, metal_context->retain_arena, &metal_context->saved_sh0);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
        if(request->gs_sh1 != NULL) {
            error = gsx_metal_render_clone_tensor(request->gs_sh1, metal_context->retain_arena, &metal_context->saved_sh1);
            if(!gsx_error_is_success(error)) {
                gsx_metal_render_context_clear_train_state(metal_context);
                return error;
            }
        }
        if(request->gs_sh2 != NULL) {
            error = gsx_metal_render_clone_tensor(request->gs_sh2, metal_context->retain_arena, &metal_context->saved_sh2);
            if(!gsx_error_is_success(error)) {
                gsx_metal_render_context_clear_train_state(metal_context);
                return error;
            }
        }
        if(request->gs_sh3 != NULL) {
            error = gsx_metal_render_clone_tensor(request->gs_sh3, metal_context->retain_arena, &metal_context->saved_sh3);
            if(!gsx_error_is_success(error)) {
                gsx_metal_render_context_clear_train_state(metal_context);
                return error;
            }
        }
        error = gsx_metal_render_clone_tensor(request->gs_opacity, metal_context->retain_arena, &metal_context->saved_opacity);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
        metal_context->train_state_borrowed = false;
    }

    if(mean2d != NULL) {
        error = gsx_metal_render_clone_tensor(mean2d, metal_context->retain_arena, &metal_context->saved_mean2d);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
    }
    if(conic_opacity != NULL) {
        error = gsx_metal_render_clone_tensor(conic_opacity, metal_context->retain_arena, &metal_context->saved_conic_opacity);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
    }
    if(color != NULL) {
        error = gsx_metal_render_clone_tensor(color, metal_context->retain_arena, &metal_context->saved_color);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
    }
    if(instance_primitive_ids != NULL) {
        error = gsx_metal_render_clone_tensor(instance_primitive_ids, metal_context->retain_arena, &metal_context->saved_instance_primitive_ids);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
    }
    if(tile_ranges != NULL) {
        error = gsx_metal_render_clone_tensor(tile_ranges, metal_context->retain_arena, &metal_context->saved_tile_ranges);
        if(!gsx_error_is_success(error)) {
            gsx_metal_render_context_clear_train_state(metal_context);
            return error;
        }
    }

    metal_context->saved_intrinsics = *request->intrinsics;
    metal_context->saved_pose = *request->pose;
    metal_context->saved_background_color = request->background_color;
    metal_context->saved_near_plane = request->near_plane;
    metal_context->saved_far_plane = request->far_plane;
    metal_context->saved_sh_degree = request->sh_degree;
    metal_context->has_train_state = true;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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
    metal_context->retain_arena = NULL;
    metal_context->helper_image_chw = NULL;
    metal_context->helper_alpha_hw = NULL;
    metal_context->saved_mean3d = NULL;
    metal_context->saved_rotation = NULL;
    metal_context->saved_logscale = NULL;
    metal_context->saved_sh0 = NULL;
    metal_context->saved_sh1 = NULL;
    metal_context->saved_sh2 = NULL;
    metal_context->saved_sh3 = NULL;
    metal_context->saved_opacity = NULL;
    metal_context->saved_mean2d = NULL;
    metal_context->saved_conic_opacity = NULL;
    metal_context->saved_color = NULL;
    metal_context->saved_instance_primitive_ids = NULL;
    metal_context->saved_tile_ranges = NULL;
    metal_context->saved_intrinsics = (gsx_camera_intrinsics){ 0 };
    metal_context->saved_pose = (gsx_camera_pose){ 0 };
    metal_context->saved_background_color = (gsx_vec3){ 0 };
    metal_context->saved_near_plane = 0.0f;
    metal_context->saved_far_plane = 0.0f;
    metal_context->saved_sh_degree = 0;
    metal_context->has_train_state = false;
    metal_context->train_state_borrowed = false;
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
    error = gsx_metal_render_init_arena(buffer_type, &metal_context->retain_arena);
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
        error = gsx_metal_render_context_clear_train_state(metal_context);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }

        if(metal_context->retain_arena != NULL) {
            error = gsx_arena_free(metal_context->retain_arena);
            if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
                first_error = error;
            }
            metal_context->retain_arena = NULL;
        }

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
