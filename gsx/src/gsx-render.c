#include "gsx-impl.h"

#include <math.h>
#include <string.h>

static bool gsx_renderer_feature_flags_are_known(gsx_renderer_feature_flags feature_flags)
{
    const gsx_renderer_feature_flags known_flags = GSX_RENDERER_FEATURE_ANTIALIASING | GSX_RENDERER_FEATURE_DEBUG;

    return (feature_flags & ~known_flags) == 0;
}

static bool gsx_render_precision_is_valid(gsx_render_precision precision)
{
    return precision == GSX_RENDER_PRECISION_FLOAT32 || precision == GSX_RENDER_PRECISION_FLOAT16;
}

static bool gsx_render_forward_type_is_valid(gsx_render_forward_type forward_type)
{
    return forward_type == GSX_RENDER_FORWARD_TYPE_INFERENCE
        || forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN
        || forward_type == GSX_RENDER_FORWARD_TYPE_METRIC;
}

static bool gsx_data_type_is_valid(gsx_data_type data_type)
{
    switch(data_type) {
    case GSX_DATA_TYPE_F32:
    case GSX_DATA_TYPE_F16:
    case GSX_DATA_TYPE_BF16:
    case GSX_DATA_TYPE_U8:
    case GSX_DATA_TYPE_I16:
    case GSX_DATA_TYPE_I32:
    case GSX_DATA_TYPE_U32:
        return true;
    }

    return false;
}

static gsx_error gsx_renderer_require_handle(gsx_renderer_t renderer)
{
    if(renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer must be non-null");
    }
    if(renderer->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "renderer implementation is missing an interface");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_context_require_handle(gsx_render_context_t context)
{
    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }
    if(context->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "render context implementation is missing an interface");
    }
    if(context->renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "render context is detached from its renderer");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_validate_bound_tensor(
    gsx_backend_t backend,
    gsx_tensor_t tensor,
    bool allow_null,
    const char *null_message)
{
    if(tensor == NULL) {
        if(allow_null) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, null_message);
    }
    if(tensor->arena == NULL || tensor->arena->dry_run || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "render tensors must reference accessible storage");
    }
    if(tensor->backing_buffer->buffer_type == NULL || tensor->backing_buffer->buffer_type->backend != backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render tensors must belong to the renderer backend");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_require_tensor(gsx_backend_t backend, gsx_tensor_t tensor, const char *null_message)
{
    return gsx_render_validate_bound_tensor(backend, tensor, false, null_message);
}

static gsx_error gsx_render_validate_optional_tensor(gsx_backend_t backend, gsx_tensor_t tensor, const char *null_message)
{
    return gsx_render_validate_bound_tensor(backend, tensor, true, null_message);
}

static gsx_error gsx_render_validate_tensor_shape(
    gsx_tensor_t tensor,
    gsx_data_type data_type,
    gsx_storage_format storage_format,
    gsx_index_t rank,
    const gsx_index_t *shape,
    const char *message)
{
    gsx_index_t dim = 0;

    if(tensor->data_type != data_type || tensor->storage_format != storage_format || tensor->rank != rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    for(dim = 0; dim < rank; ++dim) {
        if(tensor->shape[dim] != shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_render_tensors_overlap(gsx_tensor_t lhs, gsx_tensor_t rhs)
{
    gsx_size_t lhs_end_bytes = 0;
    gsx_size_t rhs_end_bytes = 0;

    if(lhs == NULL || rhs == NULL || lhs->backing_buffer == NULL || rhs->backing_buffer == NULL) {
        return false;
    }
    if(lhs->backing_buffer != rhs->backing_buffer) {
        return false;
    }
    if(gsx_size_add_overflows(lhs->offset_bytes, lhs->size_bytes, &lhs_end_bytes)
        || gsx_size_add_overflows(rhs->offset_bytes, rhs->size_bytes, &rhs_end_bytes)) {
        return true;
    }

    return lhs->offset_bytes < rhs_end_bytes && rhs->offset_bytes < lhs_end_bytes;
}

static gsx_error gsx_render_validate_no_alias(gsx_tensor_t lhs, gsx_tensor_t rhs, const char *message)
{
    if(gsx_render_tensors_overlap(lhs, rhs)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_validate_no_alias_list(gsx_tensor_t *tensors, gsx_size_t tensor_count, const char *message)
{
    gsx_size_t left = 0;
    gsx_size_t right = 0;

    for(left = 0; left < tensor_count; ++left) {
        if(tensors[left] == NULL) {
            continue;
        }
        for(right = left + 1; right < tensor_count; ++right) {
            if(tensors[right] == NULL) {
                continue;
            }
            if(gsx_render_tensors_overlap(tensors[left], tensors[right])) {
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_validate_input_shapes(const gsx_render_forward_request *request, gsx_size_t *out_count)
{
    gsx_index_t mean_shape[2] = { 0 };
    gsx_index_t pair_shape[2] = { 0 };
    gsx_index_t opacity_shape[1] = { 0 };
    gsx_index_t sh0_shape[2] = { 0 };
    gsx_index_t sh_shape[3] = { 0 };
    gsx_size_t count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    count = (gsx_size_t)request->gs_mean3d->shape[0];
    if(count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render input tensors must have a positive Gaussian count");
    }

    mean_shape[0] = request->gs_mean3d->shape[0];
    mean_shape[1] = 3;
    error = gsx_render_validate_tensor_shape(
        request->gs_mean3d, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, mean_shape, "gs_mean3d must be float32 CHW with shape [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    pair_shape[0] = request->gs_mean3d->shape[0];
    pair_shape[1] = 4;
    error = gsx_render_validate_tensor_shape(
        request->gs_rotation, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, pair_shape, "gs_rotation must be float32 CHW with shape [N,4]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    pair_shape[1] = 3;
    error = gsx_render_validate_tensor_shape(
        request->gs_logscale, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, pair_shape, "gs_logscale must be float32 CHW with shape [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    opacity_shape[0] = request->gs_mean3d->shape[0];
    error = gsx_render_validate_tensor_shape(
        request->gs_opacity, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 1, opacity_shape, "gs_opacity must be float32 CHW with shape [N]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    sh0_shape[0] = request->gs_mean3d->shape[0];
    sh0_shape[1] = 3;
    error = gsx_render_validate_tensor_shape(
        request->gs_sh0, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, sh0_shape, "gs_sh0 must be float32 CHW with shape [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(request->sh_degree >= 1) {
        sh_shape[0] = request->gs_mean3d->shape[0];
        sh_shape[1] = 3;
        sh_shape[2] = 3;
        error = gsx_render_validate_tensor_shape(
            request->gs_sh1, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "gs_sh1 must be float32 CHW with shape [N,3,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->gs_sh1 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs_sh1 must be null when sh_degree is 0");
    }

    if(request->sh_degree >= 2) {
        sh_shape[0] = request->gs_mean3d->shape[0];
        sh_shape[1] = 5;
        sh_shape[2] = 3;
        error = gsx_render_validate_tensor_shape(
            request->gs_sh2, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "gs_sh2 must be float32 CHW with shape [N,5,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->gs_sh2 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs_sh2 must be null when sh_degree is less than 2");
    }

    if(request->sh_degree >= 3) {
        sh_shape[0] = request->gs_mean3d->shape[0];
        sh_shape[1] = 7;
        sh_shape[2] = 3;
        error = gsx_render_validate_tensor_shape(
            request->gs_sh3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "gs_sh3 must be float32 CHW with shape [N,7,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->gs_sh3 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs_sh3 must be null when sh_degree is less than 3");
    }

    if(request->gs_visible_counter != NULL) {
        error = gsx_render_validate_tensor_shape(
            request->gs_visible_counter,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_CHW,
            1,
            opacity_shape,
            "gs_visible_counter must be float32 CHW with shape [N]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->gs_max_screen_radius != NULL) {
        error = gsx_render_validate_tensor_shape(
            request->gs_max_screen_radius,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_CHW,
            1,
            opacity_shape,
            "gs_max_screen_radius must be float32 CHW with shape [N]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    *out_count = count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_validate_output_rgb(const gsx_renderer *renderer, gsx_tensor_t out_rgb)
{
    gsx_index_t output_shape[3];

    if(out_rgb == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_rgb must be non-null for inference and train forwards");
    }

    output_shape[0] = 3;
    output_shape[1] = renderer->info.height;
    output_shape[2] = renderer->info.width;
    return gsx_render_validate_tensor_shape(
        out_rgb, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, output_shape, "out_rgb must be float32 CHW with shape [3,H,W]");
}

static gsx_error gsx_render_validate_backward_output_shapes(const gsx_renderer *renderer, const gsx_render_backward_request *request)
{
    gsx_index_t shape_n[1] = { 0 };
    gsx_index_t shape_n3[2] = { 0 };
    gsx_index_t shape_n4[2] = { 0 };
    gsx_index_t sh_shape[3] = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(renderer == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer and request must be non-null");
    }

    shape_n[0] = request->grad_gs_opacity->shape[0];
    shape_n3[0] = shape_n[0];
    shape_n3[1] = 3;
    shape_n4[0] = shape_n[0];
    shape_n4[1] = 4;

    error = gsx_render_validate_output_rgb(renderer, request->grad_rgb);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_tensor_shape(
        request->grad_gs_mean3d, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, shape_n3, "grad_gs_mean3d must be float32 CHW with shape [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_tensor_shape(
        request->grad_gs_rotation, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, shape_n4, "grad_gs_rotation must be float32 CHW with shape [N,4]");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_tensor_shape(
        request->grad_gs_logscale, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, shape_n3, "grad_gs_logscale must be float32 CHW with shape [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_tensor_shape(
        request->grad_gs_sh0, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, shape_n3, "grad_gs_sh0 must be float32 CHW with shape [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_gs_sh1 != NULL) {
        sh_shape[0] = shape_n[0];
        sh_shape[1] = 3;
        sh_shape[2] = 3;
        error = gsx_render_validate_tensor_shape(
            request->grad_gs_sh1, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "grad_gs_sh1 must be float32 CHW with shape [N,3,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->grad_gs_sh2 != NULL) {
        sh_shape[0] = shape_n[0];
        sh_shape[1] = 5;
        sh_shape[2] = 3;
        error = gsx_render_validate_tensor_shape(
            request->grad_gs_sh2, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "grad_gs_sh2 must be float32 CHW with shape [N,5,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->grad_gs_sh3 != NULL) {
        sh_shape[0] = shape_n[0];
        sh_shape[1] = 7;
        sh_shape[2] = 3;
        error = gsx_render_validate_tensor_shape(
            request->grad_gs_sh3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "grad_gs_sh3 must be float32 CHW with shape [N,7,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    error = gsx_render_validate_tensor_shape(
        request->grad_gs_opacity, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 1, shape_n, "grad_gs_opacity must be float32 CHW with shape [N]");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->gs_grad_acc != NULL) {
        error = gsx_render_validate_tensor_shape(
            request->gs_grad_acc, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 1, shape_n, "gs_grad_acc must be float32 CHW with shape [N]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->gs_absgrad_acc != NULL) {
        error = gsx_render_validate_tensor_shape(
            request->gs_absgrad_acc, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 1, shape_n, "gs_absgrad_acc must be float32 CHW with shape [N]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_render_validate_backward_request(const gsx_renderer *renderer, const gsx_render_backward_request *request)
{
    gsx_tensor_t tensors[14];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(renderer == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer and request must be non-null");
    }

    error = gsx_render_require_tensor(renderer->backend, request->grad_rgb, "grad_rgb must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->grad_invdepth, "grad_invdepth must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->grad_alpha, "grad_alpha must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_require_tensor(renderer->backend, request->grad_gs_mean3d, "grad_gs_mean3d must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_require_tensor(renderer->backend, request->grad_gs_rotation, "grad_gs_rotation must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_require_tensor(renderer->backend, request->grad_gs_logscale, "grad_gs_logscale must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->grad_gs_cov3d, "grad_gs_cov3d must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_require_tensor(renderer->backend, request->grad_gs_sh0, "grad_gs_sh0 must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->grad_gs_sh1, "grad_gs_sh1 must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->grad_gs_sh2, "grad_gs_sh2 must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->grad_gs_sh3, "grad_gs_sh3 must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_require_tensor(renderer->backend, request->grad_gs_opacity, "grad_gs_opacity must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->gs_grad_acc, "gs_grad_acc must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_optional_tensor(renderer->backend, request->gs_absgrad_acc, "gs_absgrad_acc must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_invdepth == NULL && request->grad_alpha == NULL && request->grad_gs_cov3d == NULL) {
        error = gsx_render_validate_backward_output_shapes(renderer, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    tensors[0] = request->grad_rgb;
    tensors[1] = request->grad_invdepth;
    tensors[2] = request->grad_alpha;
    tensors[3] = request->grad_gs_mean3d;
    tensors[4] = request->grad_gs_rotation;
    tensors[5] = request->grad_gs_logscale;
    tensors[6] = request->grad_gs_cov3d;
    tensors[7] = request->grad_gs_sh0;
    tensors[8] = request->grad_gs_sh1;
    tensors[9] = request->grad_gs_sh2;
    tensors[10] = request->grad_gs_sh3;
    tensors[11] = request->grad_gs_opacity;
    tensors[12] = request->gs_grad_acc;
    tensors[13] = request->gs_absgrad_acc;
    return gsx_render_validate_no_alias_list(tensors, (gsx_size_t)(sizeof(tensors) / sizeof(tensors[0])), "render backward tensors must not alias each other");
}

static gsx_error gsx_render_validate_forward_request(const gsx_renderer *renderer, const gsx_render_forward_request *request)
{
    gsx_size_t count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(renderer == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer and request must be non-null");
    }
    if(request->intrinsics == NULL || request->pose == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "intrinsics and pose must be non-null");
    }
    if(request->intrinsics->model != GSX_CAMERA_MODEL_PINHOLE) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "only pinhole camera intrinsics are supported");
    }
    if(request->intrinsics->width != renderer->info.width || request->intrinsics->height != renderer->info.height) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "intrinsics geometry must match the renderer output geometry");
    }
    if(isfinite((double)request->near_plane) == 0 || isfinite((double)request->far_plane) == 0
        || request->near_plane <= 0.0f || request->far_plane <= request->near_plane) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "near_plane and far_plane must be finite with 0 < near < far");
    }
    if(!gsx_render_precision_is_valid(request->precision)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render precision is out of range");
    }
    if(!gsx_render_forward_type_is_valid(request->forward_type)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render forward_type is out of range");
    }
    if(request->sh_degree < 0 || request->sh_degree > 3) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render sh_degree must be in [0,3]");
    }

    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_mean3d, false, "gs_mean3d must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_rotation, false, "gs_rotation must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_logscale, false, "gs_logscale must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_sh0, false, "gs_sh0 must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_sh1, true, "gs_sh1 must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_sh2, true, "gs_sh2 must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_sh3, true, "gs_sh3 must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->gs_opacity, false, "gs_opacity must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(
        renderer->backend,
        request->gs_cov3d,
        true,
        "gs_cov3d must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(
        renderer->backend,
        request->out_rgb,
        request->forward_type == GSX_RENDER_FORWARD_TYPE_METRIC,
        "out_rgb must be non-null for inference and train forwards");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->out_invdepth, true, "out_invdepth must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->out_alpha, true, "out_alpha must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(renderer->backend, request->metric_map, true, "metric_map must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(
        renderer->backend,
        request->gs_metric_accumulator,
        true,
        "gs_metric_accumulator must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(
        renderer->backend,
        request->gs_visible_counter,
        true,
        "gs_visible_counter must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_validate_bound_tensor(
        renderer->backend,
        request->gs_max_screen_radius,
        true,
        "gs_max_screen_radius must reference renderer storage");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_render_validate_input_shapes(request, &count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->out_rgb != NULL) {
        error = gsx_render_validate_output_rgb(renderer, request->out_rgb);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    (void)count;
    if(request->out_rgb != NULL) {
        error = gsx_render_validate_no_alias(request->gs_mean3d, request->out_rgb, "out_rgb must not alias render inputs");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_render_validate_no_alias(request->gs_rotation, request->out_rgb, "out_rgb must not alias render inputs");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_render_validate_no_alias(request->gs_logscale, request->out_rgb, "out_rgb must not alias render inputs");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_render_validate_no_alias(request->gs_opacity, request->out_rgb, "out_rgb must not alias render inputs");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_render_validate_no_alias(request->gs_sh0, request->out_rgb, "out_rgb must not alias render inputs");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(request->gs_sh1 != NULL) {
            error = gsx_render_validate_no_alias(request->gs_sh1, request->out_rgb, "out_rgb must not alias render inputs");
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        if(request->gs_sh2 != NULL) {
            error = gsx_render_validate_no_alias(request->gs_sh2, request->out_rgb, "out_rgb must not alias render inputs");
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        if(request->gs_sh3 != NULL) {
            error = gsx_render_validate_no_alias(request->gs_sh3, request->out_rgb, "out_rgb must not alias render inputs");
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_renderer_validate_desc(gsx_backend_t backend, const gsx_renderer_desc *desc)
{
    if(backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and desc must be non-null");
    }
    if(desc->width <= 0 || desc->height <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer width and height must be positive");
    }
    if(!gsx_data_type_is_valid(desc->output_data_type)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "renderer output_data_type is out of range");
    }
    if(!gsx_renderer_feature_flags_are_known(desc->feature_flags)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer feature_flags contain unsupported bits");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_renderer_base_init(
    gsx_renderer *renderer,
    const gsx_renderer_i *iface,
    gsx_backend_t backend,
    const gsx_renderer_desc *desc,
    const gsx_renderer_capabilities *capabilities)
{
    if(renderer == NULL || iface == NULL || backend == NULL || desc == NULL || capabilities == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer, iface, backend, desc, and capabilities must be non-null");
    }

    memset(renderer, 0, sizeof(*renderer));
    renderer->iface = iface;
    renderer->backend = backend;
    renderer->backend->live_renderer_count += 1;
    renderer->info.width = desc->width;
    renderer->info.height = desc->height;
    renderer->info.output_data_type = desc->output_data_type;
    renderer->info.feature_flags = desc->feature_flags;
    renderer->info.enable_invdepth_output = desc->enable_invdepth_output;
    renderer->info.enable_alpha_output = desc->enable_alpha_output;
    renderer->capabilities = *capabilities;
    renderer->live_context_count = 0;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_renderer_base_deinit(gsx_renderer *renderer)
{
    if(renderer == NULL) {
        return;
    }

    if(renderer->backend != NULL && renderer->backend->live_renderer_count != 0) {
        renderer->backend->live_renderer_count -= 1;
    }
    memset(renderer, 0, sizeof(*renderer));
}

gsx_error gsx_render_context_base_init(gsx_render_context *context, const gsx_render_context_i *iface, gsx_renderer_t renderer)
{
    if(context == NULL || iface == NULL || renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context, iface, and renderer must be non-null");
    }

    memset(context, 0, sizeof(*context));
    context->iface = iface;
    context->renderer = renderer;
    renderer->live_context_count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_render_context_base_deinit(gsx_render_context *context)
{
    if(context == NULL) {
        return;
    }
    if(context->renderer != NULL && context->renderer->live_context_count != 0) {
        context->renderer->live_context_count -= 1;
    }

    memset(context, 0, sizeof(*context));
}

GSX_API gsx_error gsx_renderer_init(gsx_renderer_t *out_renderer, gsx_backend_t backend, const gsx_renderer_desc *desc)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_renderer == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_renderer, backend, and desc must be non-null");
    }

    *out_renderer = NULL;
    if(backend->iface == NULL || backend->iface->create_renderer == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend does not implement renderer creation");
    }

    error = gsx_renderer_validate_desc(backend, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return backend->iface->create_renderer(backend, desc, out_renderer);
}

GSX_API gsx_error gsx_renderer_free(gsx_renderer_t renderer)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(renderer->live_context_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot free renderer while render contexts still exist");
    }
    if(renderer->iface->destroy == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "renderer destroy is not implemented");
    }

    return renderer->iface->destroy(renderer);
}

GSX_API gsx_error gsx_renderer_get_info(gsx_renderer_t renderer, gsx_renderer_info *out_info)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    *out_info = renderer->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_renderer_get_capabilities(gsx_renderer_t renderer, gsx_renderer_capabilities *out_capabilities)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_capabilities == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_capabilities must be non-null");
    }

    *out_capabilities = renderer->capabilities;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_renderer_get_output_data_type(gsx_renderer_t renderer, gsx_render_precision precision, gsx_data_type *out_data_type)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_data_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_data_type must be non-null");
    }
    if(!gsx_render_precision_is_valid(precision)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render precision is out of range");
    }
    if(precision == GSX_RENDER_PRECISION_FLOAT32) {
        if((renderer->capabilities.supported_precisions & GSX_RENDER_PRECISION_FLAG_FLOAT32) == 0) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "requested render precision is not supported");
        }
        *out_data_type = GSX_DATA_TYPE_F32;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if((renderer->capabilities.supported_precisions & GSX_RENDER_PRECISION_FLAG_FLOAT16) == 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "requested render precision is not supported");
    }
    *out_data_type = GSX_DATA_TYPE_F16;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_render_context_init(gsx_render_context_t *out_context, gsx_renderer_t renderer)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_context must be non-null");
    }
    *out_context = NULL;
    if(renderer->iface->create_context == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "renderer context creation is not implemented");
    }

    return renderer->iface->create_context(renderer, out_context);
}

GSX_API gsx_error gsx_render_context_free(gsx_render_context_t context)
{
    gsx_error error = gsx_render_context_require_handle(context);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(context->iface->destroy == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "render context destroy is not implemented");
    }

    return context->iface->destroy(context);
}

GSX_API gsx_error gsx_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_context_require_handle(context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "request must be non-null");
    }
    if(context->renderer != renderer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render context does not belong to the provided renderer");
    }
    if(renderer->iface->render == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "renderer forward is not implemented");
    }

    error = gsx_render_validate_forward_request(renderer, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return renderer->iface->render(renderer, context, request);
}

GSX_API gsx_error gsx_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request)
{
    gsx_error error = gsx_renderer_require_handle(renderer);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_render_context_require_handle(context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "request must be non-null");
    }
    if(context->renderer != renderer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render context does not belong to the provided renderer");
    }
    if(renderer->iface->backward == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "renderer backward is not implemented");
    }
    error = gsx_render_validate_backward_request(renderer, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return renderer->iface->backward(renderer, context, request);
}
