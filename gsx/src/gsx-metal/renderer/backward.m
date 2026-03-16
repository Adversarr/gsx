#include "../objc-helpers.h"

#include <string.h>

typedef struct gsx_metal_backward_scratch {
    gsx_tensor_t grad_mean2d;
    gsx_tensor_t grad_conic;
    gsx_tensor_t grad_raw_opacity_partial;
    gsx_tensor_t grad_color;
} gsx_metal_backward_scratch;

static bool gsx_metal_render_tensor_is_device_f32(gsx_tensor_t tensor)
{
    return tensor != NULL
        && tensor->data_type == GSX_DATA_TYPE_F32
        && tensor->backing_buffer != NULL
        && gsx_metal_backend_buffer_get_type_class(tensor->backing_buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static bool gsx_metal_render_tensor_is_optional_device_f32(gsx_tensor_t tensor)
{
    return tensor == NULL || gsx_metal_render_tensor_is_device_f32(tensor);
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

static void gsx_metal_render_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    out_view->buffer = tensor->backing_buffer;
    out_view->offset_bytes = tensor->offset_bytes;
    out_view->size_bytes = tensor->size_bytes;
    out_view->effective_alignment_bytes = tensor->effective_alignment_bytes;
    out_view->data_type = tensor->data_type;
}

static void gsx_metal_render_release_tensor(gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        (void)gsx_tensor_free(*tensor);
        *tensor = NULL;
    }
}

static void gsx_metal_render_cleanup_backward_scratch(gsx_metal_backward_scratch *scratch)
{
    gsx_metal_render_release_tensor(&scratch->grad_color);
    gsx_metal_render_release_tensor(&scratch->grad_raw_opacity_partial);
    gsx_metal_render_release_tensor(&scratch->grad_conic);
    gsx_metal_render_release_tensor(&scratch->grad_mean2d);
}

gsx_error gsx_metal_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request)
{
    gsx_metal_render_context *metal_context = (gsx_metal_render_context *)context;
    gsx_metal_backward_scratch scratch;
    gsx_backend_tensor_view saved_mean3d_view = { 0 };
    gsx_backend_tensor_view saved_rotation_view = { 0 };
    gsx_backend_tensor_view saved_logscale_view = { 0 };
    gsx_backend_tensor_view saved_sh0_view = { 0 };
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
    gsx_backend_tensor_view grad_rgb_view = { 0 };
    gsx_backend_tensor_view grad_mean2d_view = { 0 };
    gsx_backend_tensor_view grad_conic_view = { 0 };
    gsx_backend_tensor_view grad_raw_opacity_partial_view = { 0 };
    gsx_backend_tensor_view grad_color_view = { 0 };
    gsx_backend_tensor_view grad_mean3d_view = { 0 };
    gsx_backend_tensor_view grad_rotation_view = { 0 };
    gsx_backend_tensor_view grad_logscale_view = { 0 };
    gsx_backend_tensor_view grad_sh0_view = { 0 };
    gsx_backend_tensor_view grad_opacity_view = { 0 };
    gsx_metal_render_blend_backward_params blend_params = { 0 };
    gsx_metal_render_preprocess_backward_params preprocess_params = { 0 };
    gsx_size_t gaussian_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    memset(&scratch, 0, sizeof(scratch));
    if(renderer == NULL || context == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer, context, and request must be non-null");
    }
    if(!metal_context->has_train_state) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backward requires a retained TRAIN forward on the same context");
    }
    if(request->grad_invdepth != NULL || request->grad_alpha != NULL || request->grad_gs_cov3d != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer backward does not support invdepth/alpha/cov3d gradients yet");
    }
    if(metal_context->saved_sh_degree != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer backward currently supports only sh_degree=0");
    }
    if(metal_context->saved_mean3d == NULL || metal_context->saved_rotation == NULL || metal_context->saved_logscale == NULL
        || metal_context->saved_sh0 == NULL || metal_context->saved_opacity == NULL || metal_context->saved_mean2d == NULL
        || metal_context->saved_conic_opacity == NULL || metal_context->saved_color == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal renderer backward train state is incomplete");
    }

    if(!gsx_metal_render_tensor_is_device_f32(request->grad_rgb)
        || !gsx_metal_render_tensor_is_device_f32(request->grad_gs_mean3d)
        || !gsx_metal_render_tensor_is_device_f32(request->grad_gs_rotation)
        || !gsx_metal_render_tensor_is_device_f32(request->grad_gs_logscale)
        || !gsx_metal_render_tensor_is_device_f32(request->grad_gs_sh0)
        || !gsx_metal_render_tensor_is_optional_device_f32(request->grad_gs_sh1)
        || !gsx_metal_render_tensor_is_optional_device_f32(request->grad_gs_sh2)
        || !gsx_metal_render_tensor_is_optional_device_f32(request->grad_gs_sh3)
        || !gsx_metal_render_tensor_is_device_f32(request->grad_gs_opacity)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer backward currently requires device-backed float32 tensors");
    }

    gaussian_count = (gsx_size_t)metal_context->saved_mean3d->shape[0];

    error = gsx_arena_reset(metal_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gaussian_count > 0) {
        gsx_index_t shape_n[1] = { (gsx_index_t)gaussian_count };
        gsx_index_t shape_n2[2] = { (gsx_index_t)gaussian_count, 2 };
        gsx_index_t shape_n3[2] = { (gsx_index_t)gaussian_count, 3 };

        error = gsx_metal_render_make_tensor(metal_context->scratch_arena, GSX_DATA_TYPE_F32, 2, shape_n2, &scratch.grad_mean2d);
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

    error = gsx_tensor_set_zero(request->grad_gs_mean3d);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_set_zero(request->grad_gs_rotation);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_set_zero(request->grad_gs_logscale);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_set_zero(request->grad_gs_sh0);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(request->grad_gs_sh1 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh1);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }
    if(request->grad_gs_sh2 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh2);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }
    if(request->grad_gs_sh3 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh3);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }
    error = gsx_tensor_set_zero(request->grad_gs_opacity);
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
    gsx_metal_render_make_tensor_view(request->grad_rgb, &grad_rgb_view);
    gsx_metal_render_make_tensor_view(scratch.grad_mean2d, &grad_mean2d_view);
    gsx_metal_render_make_tensor_view(scratch.grad_conic, &grad_conic_view);
    gsx_metal_render_make_tensor_view(scratch.grad_raw_opacity_partial, &grad_raw_opacity_partial_view);
    gsx_metal_render_make_tensor_view(scratch.grad_color, &grad_color_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_mean3d, &grad_mean3d_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_rotation, &grad_rotation_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_logscale, &grad_logscale_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_sh0, &grad_sh0_view);
    gsx_metal_render_make_tensor_view(request->grad_gs_opacity, &grad_opacity_view);

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
        &saved_opacity_view,
        &saved_mean2d_view,
        &saved_conic_opacity_view,
        &grad_mean2d_view,
        &grad_conic_view,
        &grad_raw_opacity_partial_view,
        &grad_color_view,
        &grad_mean3d_view,
        &grad_rotation_view,
        &grad_logscale_view,
        &grad_sh0_view,
        &grad_opacity_view,
        &preprocess_params);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_backend_major_stream_sync(renderer->backend);

cleanup:
    gsx_metal_render_cleanup_backward_scratch(&scratch);
    (void)gsx_arena_reset(metal_context->scratch_arena);
    return error;
}
