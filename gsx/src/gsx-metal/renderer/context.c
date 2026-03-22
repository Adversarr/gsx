#include "../internal.h"

#include <stdlib.h>
#include <string.h>

static gsx_error gsx_metal_render_init_arena(gsx_backend_buffer_type_t buffer_type, gsx_arena_t *out_arena)
{
    gsx_arena_desc arena_desc = { 0 };

    if(buffer_type == NULL || out_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_arena must be non-null");
    }

    return gsx_arena_init(out_arena, buffer_type, &arena_desc);
}

static gsx_error gsx_metal_render_make_f32_tensor(
    gsx_arena_t arena,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_tensor_t *out_tensor)
{
    return gsx_metal_render_make_tensor(arena, GSX_DATA_TYPE_F32, rank, shape, out_tensor);
}

static gsx_error gsx_metal_render_clone_tensor_aligned(
    gsx_tensor_t src,
    gsx_arena_t arena,
    gsx_size_t min_requested_alignment_bytes,
    gsx_tensor_t *out_clone)
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
    if(desc.requested_alignment_bytes < min_requested_alignment_bytes) {
        desc.requested_alignment_bytes = min_requested_alignment_bytes;
    }

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

static gsx_error gsx_metal_render_clone_tensor(gsx_tensor_t src, gsx_arena_t arena, gsx_tensor_t *out_clone)
{
    return gsx_metal_render_clone_tensor_aligned(src, arena, sizeof(float), out_clone);
}

static gsx_error gsx_metal_render_plan_clone_tensor_aligned(
    gsx_tensor_t src,
    gsx_arena_t arena,
    gsx_size_t min_requested_alignment_bytes,
    gsx_tensor_t *out_clone)
{
    gsx_tensor_desc desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || out_clone == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_clone must be non-null");
    }

    *out_clone = NULL;
    if(src == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_tensor_get_desc(src, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    desc.arena = arena;
    if(desc.requested_alignment_bytes < min_requested_alignment_bytes) {
        desc.requested_alignment_bytes = min_requested_alignment_bytes;
    }
    error = gsx_tensor_init(out_clone, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_plan_clone_tensor(gsx_tensor_t src, gsx_arena_t arena, gsx_tensor_t *out_clone)
{
    return gsx_metal_render_plan_clone_tensor_aligned(src, arena, sizeof(float), out_clone);
}

typedef struct gsx_metal_render_snapshot_plan {
    const gsx_render_forward_request *request;
    gsx_tensor_t mean2d;
    gsx_tensor_t conic_opacity;
    gsx_tensor_t color;
    gsx_tensor_t instance_primitive_ids;
    gsx_tensor_t tile_ranges;
    gsx_tensor_t tile_bucket_offsets;
    gsx_tensor_t bucket_tile_index;
    gsx_tensor_t bucket_color_transmittance;
    gsx_tensor_t tile_max_n_contributions;
    gsx_tensor_t tile_n_contributions;
} gsx_metal_render_snapshot_plan;

static gsx_error gsx_metal_render_measure_snapshot_required_bytes(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_metal_render_snapshot_plan *plan = (gsx_metal_render_snapshot_plan *)user_data;
    gsx_tensor_t planned[18] = { NULL };
    gsx_size_t planned_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || plan == NULL || plan->request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "snapshot dry-run plan requires non-null inputs");
    }

    if(!plan->request->borrow_train_state) {
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_mean3d, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor_aligned(plan->request->gs_rotation, dry_run_arena, 16u, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_logscale, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_sh0, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_sh1, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_sh2, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_sh3, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_plan_clone_tensor(plan->request->gs_opacity, dry_run_arena, &planned[planned_count++]);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }

cleanup:
    while(planned_count > 0) {
        planned_count -= 1;
        gsx_metal_render_release_tensor(&planned[planned_count]);
    }
    return error;
}

typedef struct gsx_metal_render_helper_plan {
    gsx_index_t width;
    gsx_index_t height;
} gsx_metal_render_helper_plan;

static gsx_error gsx_metal_render_measure_helper_required_bytes(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_metal_render_helper_plan *plan = (gsx_metal_render_helper_plan *)user_data;
    gsx_index_t image_shape[3] = { 3, 0, 0 };
    gsx_index_t alpha_shape[2] = { 0, 0 };
    gsx_index_t dummy_shape[1] = { 1 };
    gsx_tensor_t image = NULL;
    gsx_tensor_t alpha = NULL;
    gsx_tensor_t dummy = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || plan == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "helper dry-run plan requires non-null inputs");
    }

    image_shape[1] = plan->height;
    image_shape[2] = plan->width;
    alpha_shape[0] = plan->height;
    alpha_shape[1] = plan->width;

    error = gsx_metal_render_make_f32_tensor(dry_run_arena, 3, image_shape, &image);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_metal_render_make_f32_tensor(dry_run_arena, 2, alpha_shape, &alpha);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_metal_render_make_f32_tensor(dry_run_arena, 1, dummy_shape, &dummy);

cleanup:
    gsx_metal_render_release_tensor(&dummy);
    gsx_metal_render_release_tensor(&alpha);
    gsx_metal_render_release_tensor(&image);
    return error;
}

gsx_error gsx_metal_render_context_clear_train_state(gsx_metal_render_context *metal_context)
{
    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    if(!metal_context->train_state_borrowed) {
        gsx_metal_render_release_tensor(&metal_context->saved_mean3d);
        gsx_metal_render_release_tensor(&metal_context->saved_rotation);
        gsx_metal_render_release_tensor(&metal_context->saved_logscale);
        gsx_metal_render_release_tensor(&metal_context->saved_sh0);
        gsx_metal_render_release_tensor(&metal_context->saved_sh1);
        gsx_metal_render_release_tensor(&metal_context->saved_sh2);
        gsx_metal_render_release_tensor(&metal_context->saved_sh3);
        gsx_metal_render_release_tensor(&metal_context->saved_opacity);
    }

    gsx_metal_render_release_tensor(&metal_context->saved_mean2d);
    gsx_metal_render_release_tensor(&metal_context->saved_conic_opacity);
    gsx_metal_render_release_tensor(&metal_context->saved_color);
    gsx_metal_render_release_tensor(&metal_context->saved_instance_primitive_ids);
    gsx_metal_render_release_tensor(&metal_context->saved_tile_ranges);
    gsx_metal_render_release_tensor(&metal_context->saved_tile_bucket_offsets);
    gsx_metal_render_release_tensor(&metal_context->saved_bucket_tile_index);
    gsx_metal_render_release_tensor(&metal_context->saved_bucket_color_transmittance);
    gsx_metal_render_release_tensor(&metal_context->saved_tile_max_n_contributions);
    gsx_metal_render_release_tensor(&metal_context->saved_tile_n_contributions);

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
    metal_context->saved_tile_bucket_offsets = NULL;
    metal_context->saved_bucket_tile_index = NULL;
    metal_context->saved_bucket_color_transmittance = NULL;
    metal_context->saved_tile_max_n_contributions = NULL;
    metal_context->saved_tile_n_contributions = NULL;
    metal_context->saved_bucket_count = 0u;
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
    gsx_tensor_t tile_ranges,
    gsx_tensor_t tile_bucket_offsets,
    gsx_tensor_t bucket_tile_index,
    gsx_tensor_t bucket_color_transmittance,
    gsx_tensor_t tile_max_n_contributions,
    gsx_tensor_t tile_n_contributions,
    uint32_t bucket_count)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_metal_render_snapshot_plan dry_run_plan = { 0 };

    if(metal_context == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and request must be non-null");
    }

    error = gsx_metal_render_context_clear_train_state(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    dry_run_plan.request = request;
    dry_run_plan.mean2d = mean2d;
    dry_run_plan.conic_opacity = conic_opacity;
    dry_run_plan.color = color;
    dry_run_plan.instance_primitive_ids = instance_primitive_ids;
    dry_run_plan.tile_ranges = tile_ranges;
    dry_run_plan.tile_bucket_offsets = tile_bucket_offsets;
    dry_run_plan.bucket_tile_index = bucket_tile_index;
    dry_run_plan.bucket_color_transmittance = bucket_color_transmittance;
    dry_run_plan.tile_max_n_contributions = tile_max_n_contributions;
    dry_run_plan.tile_n_contributions = tile_n_contributions;

    error = gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->retain_arena,
        gsx_metal_render_measure_snapshot_required_bytes,
        &dry_run_plan);
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
        error = gsx_metal_render_clone_tensor_aligned(request->gs_rotation, metal_context->retain_arena, 16u, &metal_context->saved_rotation);
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

    metal_context->saved_mean2d = mean2d;
    metal_context->saved_conic_opacity = conic_opacity;
    metal_context->saved_color = color;
    metal_context->saved_instance_primitive_ids = instance_primitive_ids;
    metal_context->saved_tile_ranges = tile_ranges;
    metal_context->saved_tile_bucket_offsets = tile_bucket_offsets;
    metal_context->saved_bucket_tile_index = bucket_tile_index;
    metal_context->saved_bucket_color_transmittance = bucket_color_transmittance;
    metal_context->saved_tile_max_n_contributions = tile_max_n_contributions;
    metal_context->saved_tile_n_contributions = tile_n_contributions;

    metal_context->saved_intrinsics = *request->intrinsics;
    metal_context->saved_pose = *request->pose;
    metal_context->saved_background_color = request->background_color;
    metal_context->saved_near_plane = request->near_plane;
    metal_context->saved_far_plane = request->far_plane;
    metal_context->saved_sh_degree = request->sh_degree;
    metal_context->saved_bucket_count = bucket_count;
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
    gsx_index_t dummy_shape[1] = { 1 };
    gsx_backend_buffer_type_t unified_buffer_type = NULL;
    gsx_metal_render_helper_plan helper_plan = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and buffer_type must be non-null");
    }

    metal_context->helper_arena = NULL;
    metal_context->scratch_arena = NULL;
    metal_context->forward_per_primitive_arena = NULL;
    metal_context->forward_per_tile_arena = NULL;
    metal_context->forward_per_instance_arena = NULL;
    metal_context->forward_per_bucket_arena = NULL;
    metal_context->retain_arena = NULL;
    metal_context->helper_image_chw = NULL;
    metal_context->helper_alpha_hw = NULL;
    metal_context->optional_dummy_f32 = NULL;
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
    metal_context->saved_tile_bucket_offsets = NULL;
    metal_context->saved_bucket_tile_index = NULL;
    metal_context->saved_bucket_color_transmittance = NULL;
    metal_context->saved_tile_max_n_contributions = NULL;
    metal_context->saved_tile_n_contributions = NULL;
    metal_context->saved_bucket_count = 0u;
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
    error = gsx_metal_render_init_arena(unified_buffer_type, &metal_context->forward_per_primitive_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }
    error = gsx_metal_render_init_arena(unified_buffer_type, &metal_context->forward_per_tile_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }
    error = gsx_metal_render_init_arena(unified_buffer_type, &metal_context->forward_per_instance_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }
    error = gsx_metal_render_init_arena(buffer_type, &metal_context->forward_per_bucket_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }
    error = gsx_metal_render_init_arena(buffer_type, &metal_context->retain_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    helper_plan.width = width;
    helper_plan.height = height;
    error = gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->helper_arena,
        gsx_metal_render_measure_helper_required_bytes,
        &helper_plan);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    error = gsx_metal_render_make_f32_tensor(metal_context->helper_arena, 3, image_shape, &metal_context->helper_image_chw);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    error = gsx_metal_render_make_f32_tensor(metal_context->helper_arena, 2, alpha_shape, &metal_context->helper_alpha_hw);
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_render_context_dispose(metal_context);
        return error;
    }

    error = gsx_metal_render_make_f32_tensor(metal_context->helper_arena, 1, dummy_shape, &metal_context->optional_dummy_f32);
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
    if(metal_context->optional_dummy_f32 != NULL) {
        error = gsx_tensor_free(metal_context->optional_dummy_f32);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->optional_dummy_f32 = NULL;
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
    if(metal_context->forward_per_primitive_arena != NULL) {
        error = gsx_arena_free(metal_context->forward_per_primitive_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->forward_per_primitive_arena = NULL;
    }
    if(metal_context->forward_per_tile_arena != NULL) {
        error = gsx_arena_free(metal_context->forward_per_tile_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->forward_per_tile_arena = NULL;
    }
    if(metal_context->forward_per_instance_arena != NULL) {
        error = gsx_arena_free(metal_context->forward_per_instance_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->forward_per_instance_arena = NULL;
    }
    if(metal_context->forward_per_bucket_arena != NULL) {
        error = gsx_arena_free(metal_context->forward_per_bucket_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        metal_context->forward_per_bucket_arena = NULL;
    }

    return first_error;
}
