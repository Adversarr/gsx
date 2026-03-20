#include "../objc-helpers.h"

#include <string.h>

typedef struct gsx_metal_backward_scratch {
    gsx_tensor_t grad_mean2d;
    gsx_tensor_t absgrad_mean2d;
    gsx_tensor_t grad_conic;
    gsx_tensor_t grad_raw_opacity_partial;
    gsx_tensor_t grad_color;
} gsx_metal_backward_scratch;

static void gsx_metal_render_cleanup_backward_scratch(gsx_metal_backward_scratch *scratch)
{
    gsx_metal_render_release_tensor(&scratch->grad_color);
    gsx_metal_render_release_tensor(&scratch->grad_raw_opacity_partial);
    gsx_metal_render_release_tensor(&scratch->grad_conic);
    gsx_metal_render_release_tensor(&scratch->absgrad_mean2d);
    gsx_metal_render_release_tensor(&scratch->grad_mean2d);
}

static gsx_error gsx_metal_render_plan_backward_scratch(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_size_t gaussian_count = *(const gsx_size_t *)user_data;
    gsx_tensor_t dry_grad_mean2d = NULL;
    gsx_tensor_t dry_absgrad_mean2d = NULL;
    gsx_tensor_t dry_grad_conic = NULL;
    gsx_tensor_t dry_grad_raw_opacity_partial = NULL;
    gsx_tensor_t dry_grad_color = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || user_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dry_run_arena and user_data must be non-null");
    }
    if(gaussian_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        gsx_index_t shape_n[1] = { (gsx_index_t)gaussian_count };
        gsx_index_t shape_n2[2] = { (gsx_index_t)gaussian_count, 2 };
        gsx_index_t shape_n3[2] = { (gsx_index_t)gaussian_count, 3 };

        error = gsx_metal_render_make_tensor_aligned(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n2, 8u, &dry_grad_mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor_aligned(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n2, 8u, &dry_absgrad_mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &dry_grad_conic);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 1, shape_n, &dry_grad_raw_opacity_partial);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &dry_grad_color);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }

cleanup:
    gsx_metal_render_release_tensor(&dry_grad_color);
    gsx_metal_render_release_tensor(&dry_grad_raw_opacity_partial);
    gsx_metal_render_release_tensor(&dry_grad_conic);
    gsx_metal_render_release_tensor(&dry_absgrad_mean2d);
    gsx_metal_render_release_tensor(&dry_grad_mean2d);
    return error;
}

static gsx_error gsx_metal_render_prepare_backward_context(gsx_metal_render_context *metal_context, gsx_size_t gaussian_count)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_arena_reset(metal_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->scratch_arena,
        gsx_metal_render_plan_backward_scratch,
        &gaussian_count);
}

static gsx_error gsx_metal_render_zero_backward_outputs(const gsx_render_backward_request *request)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_tensor_set_zero(request->grad_gs_mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_rotation);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_logscale);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_sh0);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_gs_sh1 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh1);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->grad_gs_sh2 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh2);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->grad_gs_sh3 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh3);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_tensor_set_zero(request->grad_gs_opacity);
}

gsx_error gsx_metal_renderer_backward_impl(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request)
{
    gsx_metal_render_context *metal_context = (gsx_metal_render_context *)context;
    gsx_metal_backward_scratch scratch;
    gsx_backend_tensor_view saved_mean3d_view = { 0 };
    gsx_backend_tensor_view saved_rotation_view = { 0 };
    gsx_backend_tensor_view saved_logscale_view = { 0 };
    gsx_backend_tensor_view saved_sh0_view = { 0 };
    gsx_backend_tensor_view saved_sh1_view = { 0 };
    gsx_backend_tensor_view saved_sh2_view = { 0 };
    gsx_backend_tensor_view saved_sh3_view = { 0 };
    gsx_backend_tensor_view saved_opacity_view = { 0 };
    gsx_backend_tensor_view saved_mean2d_view = { 0 };
    gsx_backend_tensor_view saved_conic_opacity_view = { 0 };
    gsx_backend_tensor_view saved_color_view = { 0 };
    gsx_backend_tensor_view helper_image_view = { 0 };
    gsx_backend_tensor_view helper_alpha_view = { 0 };
    gsx_backend_tensor_view saved_instance_primitive_ids_view = { 0 };
    gsx_backend_tensor_view saved_tile_ranges_view = { 0 };
    gsx_backend_tensor_view saved_tile_bucket_offsets_view = { 0 };
    gsx_backend_tensor_view saved_bucket_tile_index_view = { 0 };
    gsx_backend_tensor_view saved_bucket_color_transmittance_view = { 0 };
    gsx_backend_tensor_view saved_tile_max_n_contributions_view = { 0 };
    gsx_backend_tensor_view saved_tile_n_contributions_view = { 0 };
    gsx_backend_tensor_view optional_dummy_view = { 0 };
    gsx_backend_tensor_view grad_rgb_view = { 0 };
    gsx_backend_tensor_view grad_mean2d_view = { 0 };
    gsx_backend_tensor_view absgrad_mean2d_view = { 0 };
    gsx_backend_tensor_view grad_conic_view = { 0 };
    gsx_backend_tensor_view grad_raw_opacity_partial_view = { 0 };
    gsx_backend_tensor_view grad_color_view = { 0 };
    gsx_backend_tensor_view grad_mean3d_view = { 0 };
    gsx_backend_tensor_view grad_rotation_view = { 0 };
    gsx_backend_tensor_view grad_logscale_view = { 0 };
    gsx_backend_tensor_view grad_sh0_view = { 0 };
    gsx_backend_tensor_view grad_sh1_view = { 0 };
    gsx_backend_tensor_view grad_sh2_view = { 0 };
    gsx_backend_tensor_view grad_sh3_view = { 0 };
    gsx_backend_tensor_view grad_opacity_view = { 0 };
    gsx_backend_tensor_view grad_acc_aux_view = { 0 };
    gsx_backend_tensor_view absgrad_acc_aux_view = { 0 };
    gsx_metal_render_blend_backward_params blend_params = { 0 };
    gsx_metal_render_preprocess_backward_params preprocess_params = { 0 };
    gsx_size_t gaussian_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    memset(&scratch, 0, sizeof(scratch));
    gaussian_count = (gsx_size_t)metal_context->saved_mean3d->shape[0];

    error = gsx_metal_render_prepare_backward_context(metal_context, gaussian_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gaussian_count > 0) {
        gsx_index_t shape_n[1] = { (gsx_index_t)gaussian_count };
        gsx_index_t shape_n2[2] = { (gsx_index_t)gaussian_count, 2 };
        gsx_index_t shape_n3[2] = { (gsx_index_t)gaussian_count, 3 };

        error = gsx_metal_render_make_tensor_aligned(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n2, 8u, &scratch.grad_mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor_aligned(
            metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n2, 8u, &scratch.absgrad_mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &scratch.grad_conic);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 1, shape_n, &scratch.grad_raw_opacity_partial);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &scratch.grad_color);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_set_zero(scratch.grad_mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_set_zero(scratch.absgrad_mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_set_zero(scratch.grad_conic);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_set_zero(scratch.grad_raw_opacity_partial);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_set_zero(scratch.grad_color);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }

    error = gsx_metal_render_zero_backward_outputs(request);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    if(gaussian_count == 0) {
        error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        goto cleanup;
    }

    if(metal_context->saved_instance_primitive_ids == NULL || metal_context->saved_tile_ranges == NULL
        || metal_context->saved_tile_bucket_offsets == NULL || metal_context->saved_bucket_tile_index == NULL
        || metal_context->saved_bucket_color_transmittance == NULL
        || metal_context->saved_tile_max_n_contributions == NULL
        || metal_context->helper_image_chw == NULL || metal_context->helper_alpha_hw == NULL
        || metal_context->saved_tile_n_contributions == NULL) {
        error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        goto cleanup;
    }

    gsx_metal_render_make_tensor_view(metal_context->saved_mean3d, &saved_mean3d_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_rotation, &saved_rotation_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_logscale, &saved_logscale_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_sh0, &saved_sh0_view);
    if(metal_context->saved_sh1 != NULL) {
        gsx_metal_render_make_tensor_view(metal_context->saved_sh1, &saved_sh1_view);
    }
    if(metal_context->saved_sh2 != NULL) {
        gsx_metal_render_make_tensor_view(metal_context->saved_sh2, &saved_sh2_view);
    }
    if(metal_context->saved_sh3 != NULL) {
        gsx_metal_render_make_tensor_view(metal_context->saved_sh3, &saved_sh3_view);
    }
    gsx_metal_render_make_tensor_view(metal_context->saved_opacity, &saved_opacity_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_mean2d, &saved_mean2d_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_conic_opacity, &saved_conic_opacity_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_color, &saved_color_view);
    gsx_metal_render_make_tensor_view(metal_context->helper_image_chw, &helper_image_view);
    gsx_metal_render_make_tensor_view(metal_context->helper_alpha_hw, &helper_alpha_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_instance_primitive_ids, &saved_instance_primitive_ids_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_tile_ranges, &saved_tile_ranges_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_tile_bucket_offsets, &saved_tile_bucket_offsets_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_bucket_tile_index, &saved_bucket_tile_index_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_bucket_color_transmittance, &saved_bucket_color_transmittance_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_tile_max_n_contributions, &saved_tile_max_n_contributions_view);
    gsx_metal_render_make_tensor_view(metal_context->saved_tile_n_contributions, &saved_tile_n_contributions_view);
    gsx_metal_render_make_tensor_view(metal_context->optional_dummy_f32, &optional_dummy_view);
    gsx_metal_render_make_tensor_view(request->grad_rgb, &grad_rgb_view);
    gsx_metal_render_make_tensor_view(scratch.grad_mean2d, &grad_mean2d_view);
    gsx_metal_render_make_tensor_view(scratch.absgrad_mean2d, &absgrad_mean2d_view);
    gsx_metal_render_make_tensor_view(scratch.grad_conic, &grad_conic_view);
    gsx_metal_render_make_tensor_view(scratch.grad_raw_opacity_partial, &grad_raw_opacity_partial_view);
    gsx_metal_render_make_tensor_view(scratch.grad_color, &grad_color_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_mean3d, &grad_mean3d_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_rotation, &grad_rotation_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_logscale, &grad_logscale_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_sh0, &grad_sh0_view);
    if(request->grad_gs_sh1 != NULL) {
        gsx_metal_render_make_tensor_view(request->grad_gs_sh1, &grad_sh1_view);
    }
    if(request->grad_gs_sh2 != NULL) {
        gsx_metal_render_make_tensor_view(request->grad_gs_sh2, &grad_sh2_view);
    }
    if(request->grad_gs_sh3 != NULL) {
        gsx_metal_render_make_tensor_view(request->grad_gs_sh3, &grad_sh3_view);
    }
    gsx_metal_render_make_tensor_view(request->grad_gs_opacity, &grad_opacity_view);
    if(request->gs_grad_acc != NULL) {
        gsx_metal_render_make_tensor_view(request->gs_grad_acc, &grad_acc_aux_view);
    }
    if(request->gs_absgrad_acc != NULL) {
        gsx_metal_render_make_tensor_view(request->gs_absgrad_acc, &absgrad_acc_aux_view);
    }

    blend_params.gaussian_count = (uint32_t)gaussian_count;
    blend_params.width = (uint32_t)renderer->info.width;
    blend_params.height = (uint32_t)renderer->info.height;
    blend_params.grid_width = (uint32_t)((renderer->info.width + 15) / 16);
    blend_params.grid_height = (uint32_t)((renderer->info.height + 15) / 16);
    blend_params.tile_count = blend_params.grid_width * blend_params.grid_height;
    blend_params.total_bucket_count = metal_context->saved_bucket_count;
    blend_params.channel_stride = (uint32_t)((gsx_size_t)renderer->info.width * (gsx_size_t)renderer->info.height);
    blend_params.background_r = metal_context->saved_background_color.x;
    blend_params.background_g = metal_context->saved_background_color.y;
    blend_params.background_b = metal_context->saved_background_color.z;
    error = gsx_metal_backend_dispatch_render_blend_backward(
        renderer->backend,
        &saved_tile_ranges_view,
        &saved_tile_bucket_offsets_view,
        &saved_bucket_tile_index_view,
        &saved_instance_primitive_ids_view,
        &saved_mean2d_view,
        &saved_conic_opacity_view,
        &saved_color_view,
        &helper_image_view,
        &helper_alpha_view,
        &saved_tile_max_n_contributions_view,
        &saved_tile_n_contributions_view,
        &saved_bucket_color_transmittance_view,
        &grad_rgb_view,
        &grad_mean2d_view,
        &absgrad_mean2d_view,
        &grad_conic_view,
        &grad_raw_opacity_partial_view,
        &grad_color_view,
        &blend_params);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    preprocess_params.gaussian_count = (uint32_t)gaussian_count;
    preprocess_params.width = (uint32_t)renderer->info.width;
    preprocess_params.height = (uint32_t)renderer->info.height;
    preprocess_params.sh_degree = (uint32_t)metal_context->saved_sh_degree;
    preprocess_params.has_grad_acc = request->gs_grad_acc != NULL ? 1u : 0u;
    preprocess_params.has_absgrad_acc = request->gs_absgrad_acc != NULL ? 1u : 0u;
    preprocess_params.fx = metal_context->saved_intrinsics.fx;
    preprocess_params.fy = metal_context->saved_intrinsics.fy;
    preprocess_params.cx = metal_context->saved_intrinsics.cx;
    preprocess_params.cy = metal_context->saved_intrinsics.cy;
    preprocess_params.near_plane = metal_context->saved_near_plane;
    preprocess_params.far_plane = metal_context->saved_far_plane;
    preprocess_params.pose_qx = metal_context->saved_pose.rot.x;
    preprocess_params.pose_qy = metal_context->saved_pose.rot.y;
    preprocess_params.pose_qz = metal_context->saved_pose.rot.z;
    preprocess_params.pose_qw = metal_context->saved_pose.rot.w;
    preprocess_params.pose_tx = metal_context->saved_pose.transl.x;
    preprocess_params.pose_ty = metal_context->saved_pose.transl.y;
    preprocess_params.pose_tz = metal_context->saved_pose.transl.z;
    error = gsx_metal_backend_dispatch_render_preprocess_backward(
        renderer->backend,
        &saved_mean3d_view,
        &saved_rotation_view,
        &saved_logscale_view,
        &saved_sh0_view,
        metal_context->saved_sh1 != NULL ? &saved_sh1_view : &optional_dummy_view,
        metal_context->saved_sh2 != NULL ? &saved_sh2_view : &optional_dummy_view,
        metal_context->saved_sh3 != NULL ? &saved_sh3_view : &optional_dummy_view,
        &saved_opacity_view,
        &saved_mean2d_view,
        &saved_conic_opacity_view,
        &grad_mean2d_view,
        &absgrad_mean2d_view,
        &grad_conic_view,
        &grad_raw_opacity_partial_view,
        &grad_color_view,
        &grad_mean3d_view,
        &grad_rotation_view,
        &grad_logscale_view,
        &grad_sh0_view,
        request->grad_gs_sh1 != NULL ? &grad_sh1_view : &optional_dummy_view,
        request->grad_gs_sh2 != NULL ? &grad_sh2_view : &optional_dummy_view,
        request->grad_gs_sh3 != NULL ? &grad_sh3_view : &optional_dummy_view,
        &grad_opacity_view,
        request->gs_grad_acc != NULL ? &grad_acc_aux_view : &optional_dummy_view,
        request->gs_absgrad_acc != NULL ? &absgrad_acc_aux_view : &optional_dummy_view,
        &preprocess_params);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

cleanup:
    gsx_metal_render_cleanup_backward_scratch(&scratch);
    (void)gsx_arena_reset(metal_context->scratch_arena);
    return error;
}
