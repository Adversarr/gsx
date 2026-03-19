#include "internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct gsx_metal_loss {
    struct gsx_loss base;
} gsx_metal_loss;

typedef struct gsx_metal_loss_context {
    struct gsx_loss_context base;
    gsx_arena_t ssim_arena;
    gsx_tensor_t ssim_dummy_buffer;
    gsx_tensor_t ssim_buffer_a;
    gsx_tensor_t ssim_buffer_b;
    gsx_size_t ssim_capacity_elements;
} gsx_metal_loss_context;

static gsx_error gsx_metal_loss_destroy(gsx_loss_t loss);
static gsx_error gsx_metal_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context);
static gsx_error gsx_metal_loss_context_destroy(gsx_loss_context_t context);
static gsx_error gsx_metal_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request);
static gsx_error gsx_metal_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request);
static gsx_error gsx_metal_loss_execute_forward(
    gsx_metal_loss *metal_loss,
    gsx_metal_loss_context *metal_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t loss_map_tensor,
    bool train,
    gsx_float_t loss_scale);
static gsx_error gsx_metal_loss_execute_backward(
    gsx_metal_loss *metal_loss,
    gsx_metal_loss_context *metal_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t grad_tensor,
    gsx_float_t grad_scale);

static const gsx_loss_i gsx_metal_loss_iface = {
    gsx_metal_loss_destroy,
    gsx_metal_loss_create_context,
    gsx_metal_loss_forward,
    gsx_metal_loss_backward
};

static const gsx_loss_context_i gsx_metal_loss_context_iface = {
    gsx_metal_loss_context_destroy
};

static bool gsx_metal_loss_buffer_is_device(gsx_backend_buffer_t buffer)
{
    return buffer != NULL && gsx_metal_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static gsx_error gsx_metal_loss_validate_tensor_f32_device(gsx_tensor_t tensor)
{
    if(!gsx_metal_loss_buffer_is_device(tensor->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal loss requires device-backed tensors");
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal loss tensors must use float32 storage");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_metal_loss_grad_scale(const gsx_metal_loss *metal_loss, gsx_size_t element_count, gsx_float_t scale)
{
    if(metal_loss->base.grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN) {
        return scale / (float)element_count;
    }

    return scale;
}

static void gsx_metal_loss_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    out_view->buffer = tensor->backing_buffer;
    out_view->offset_bytes = tensor->offset_bytes;
    out_view->size_bytes = tensor->size_bytes;
    out_view->effective_alignment_bytes = tensor->effective_alignment_bytes;
    out_view->data_type = tensor->data_type;
}

static gsx_error gsx_metal_loss_release_ssim_scratch_buffers(gsx_metal_loss_context *metal_context)
{
    gsx_error first_error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }
    if(metal_context->ssim_buffer_a != NULL) {
        gsx_error error = gsx_tensor_free(metal_context->ssim_buffer_a);

        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        if(gsx_error_is_success(error)) {
            metal_context->ssim_buffer_a = NULL;
        }
    }
    if(metal_context->ssim_buffer_b != NULL) {
        gsx_error error = gsx_tensor_free(metal_context->ssim_buffer_b);

        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        if(gsx_error_is_success(error)) {
            metal_context->ssim_buffer_b = NULL;
        }
    }
    if(gsx_error_is_success(first_error)) {
        metal_context->ssim_capacity_elements = 0;
    }

    return first_error;
}

static gsx_error gsx_metal_loss_ensure_ssim_dummy_buffer(gsx_metal_loss_context *metal_context)
{
    gsx_tensor_desc desc = { 0 };

    if(metal_context == NULL || metal_context->base.loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and loss must be non-null");
    }
    if(metal_context->ssim_dummy_buffer != NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(metal_context->ssim_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "ssim_arena must be initialized before creating dummy buffer");
    }

    desc.rank = 1;
    desc.shape[0] = 1;
    desc.requested_alignment_bytes = 0;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = metal_context->ssim_arena;
    return gsx_tensor_init(&metal_context->ssim_dummy_buffer, &desc);
}

static gsx_error gsx_metal_loss_ensure_ssim_scratch_arena(gsx_metal_loss_context *metal_context)
{
    gsx_backend_buffer_type_t device_buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || metal_context->base.loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and loss must be non-null");
    }
    if(metal_context->ssim_arena != NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_backend_find_buffer_type(
        metal_context->base.loss->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    error = gsx_arena_init(&metal_context->ssim_arena, device_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_loss_fill_ssim_scratch_tensor_desc(gsx_tensor_desc *out_desc, gsx_arena_t arena, gsx_index_t element_count)
{
    out_desc->rank = 1;
    out_desc->shape[0] = element_count;
    out_desc->requested_alignment_bytes = 0;
    out_desc->data_type = GSX_DATA_TYPE_F32;
    out_desc->storage_format = GSX_STORAGE_FORMAT_CHW;
    out_desc->arena = arena;
}

static void gsx_metal_loss_cleanup_ssim_sizing_work(gsx_arena_t *arena, gsx_tensor_t *tensor_a, gsx_tensor_t *tensor_b)
{
    if(tensor_a != NULL && *tensor_a != NULL) {
        (void)gsx_tensor_free(*tensor_a);
        *tensor_a = NULL;
    }
    if(tensor_b != NULL && *tensor_b != NULL) {
        (void)gsx_tensor_free(*tensor_b);
        *tensor_b = NULL;
    }
    if(arena != NULL && *arena != NULL) {
        (void)gsx_arena_free(*arena);
        *arena = NULL;
    }
}

static gsx_error gsx_metal_loss_prepare_ssim_scratch_shapes(
    gsx_size_t element_count,
    gsx_size_t *out_buffer_a_elements,
    gsx_index_t *out_shape_a,
    gsx_index_t *out_shape_b)
{
    gsx_size_t buffer_a_elements = 0;

    if(out_buffer_a_elements == NULL || out_shape_a == NULL || out_shape_b == NULL || element_count == 0) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "element_count and output shape pointers must be valid");
    }
    if(gsx_size_mul_overflows(element_count, (gsx_size_t)2, &buffer_a_elements)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "ssim scratch element count overflowed");
    }
    if(buffer_a_elements > (gsx_size_t)INT_MAX || element_count > (gsx_size_t)INT_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "ssim scratch exceeds tensor shape limits");
    }

    *out_buffer_a_elements = buffer_a_elements;
    *out_shape_a = (gsx_index_t)buffer_a_elements;
    *out_shape_b = (gsx_index_t)element_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_compute_ssim_scratch_required_bytes(
    gsx_backend_t backend,
    gsx_size_t element_count,
    gsx_size_t *out_required_bytes)
{
    gsx_backend_buffer_type_t device_buffer_type = NULL;
    gsx_arena_t dry_run_arena = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_tensor_desc desc = { 0 };
    gsx_tensor_t tensor_dummy = NULL;
    gsx_tensor_t tensor_a = NULL;
    gsx_tensor_t tensor_b = NULL;
    gsx_size_t buffer_a_elements = 0;
    gsx_index_t shape_a = 0;
    gsx_index_t shape_b = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || out_required_bytes == NULL || element_count == 0) {
        error = gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, element_count, and out_required_bytes must be valid");
        goto cleanup;
    }
    error = gsx_metal_loss_prepare_ssim_scratch_shapes(element_count, &buffer_a_elements, &shape_a, &shape_b);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, device_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    gsx_metal_loss_fill_ssim_scratch_tensor_desc(&desc, dry_run_arena, shape_a);
    error = gsx_arena_reset(dry_run_arena);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    desc.shape[0] = 1;
    error = gsx_tensor_init(&tensor_dummy, &desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    desc.shape[0] = shape_a;
    error = gsx_tensor_init(&tensor_a, &desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    desc.shape[0] = shape_b;
    error = gsx_tensor_init(&tensor_b, &desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);

cleanup:
    if(tensor_dummy != NULL) {
        (void)gsx_tensor_free(tensor_dummy);
    }
    gsx_metal_loss_cleanup_ssim_sizing_work(&dry_run_arena, &tensor_a, &tensor_b);
    return error;
}

static gsx_error gsx_metal_loss_ensure_ssim_scratch(gsx_metal_loss_context *metal_context, gsx_size_t element_count)
{
    gsx_tensor_desc desc = { 0 };
    gsx_size_t buffer_a_elements = 0;
    gsx_size_t required_bytes = 0;
    gsx_index_t shape_a = 0;
    gsx_index_t shape_b = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || metal_context->base.loss == NULL || element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context and element_count must be valid");
    }
    /* Keep a valid dummy slot available even when scratch capacity is already sufficient and we early-return. */
    error = gsx_metal_loss_ensure_ssim_scratch_arena(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_ensure_ssim_dummy_buffer(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(metal_context->ssim_buffer_a != NULL && metal_context->ssim_buffer_b != NULL
        && metal_context->ssim_capacity_elements >= element_count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_metal_loss_release_ssim_scratch_buffers(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(metal_context->ssim_dummy_buffer != NULL) {
        error = gsx_tensor_free(metal_context->ssim_dummy_buffer);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        metal_context->ssim_dummy_buffer = NULL;
    }
    error = gsx_arena_reset(metal_context->ssim_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_loss_compute_ssim_scratch_required_bytes(metal_context->base.loss->backend, element_count, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reserve(metal_context->ssim_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_prepare_ssim_scratch_shapes(element_count, &buffer_a_elements, &shape_a, &shape_b);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_loss_ensure_ssim_dummy_buffer(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gsx_metal_loss_fill_ssim_scratch_tensor_desc(&desc, metal_context->ssim_arena, shape_a);
    error = gsx_tensor_init(&metal_context->ssim_buffer_a, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    desc.shape[0] = shape_b;
    error = gsx_tensor_init(&metal_context->ssim_buffer_b, &desc);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(metal_context->ssim_buffer_a);
        metal_context->ssim_buffer_a = NULL;
        if(metal_context->ssim_dummy_buffer != NULL) {
            (void)gsx_tensor_free(metal_context->ssim_dummy_buffer);
            metal_context->ssim_dummy_buffer = NULL;
        }
        (void)gsx_arena_reset(metal_context->ssim_arena);
        return error;
    }

    metal_context->ssim_capacity_elements = element_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_metal_loss_ssim_extract_layout(
    gsx_tensor_t prediction,
    gsx_size_t *out_outer_count,
    gsx_index_t *out_channels,
    gsx_index_t *out_height,
    gsx_index_t *out_width)
{
    gsx_size_t outer_count = 1;
    gsx_index_t axis = 0;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;

    if(prediction == NULL || out_outer_count == NULL || out_channels == NULL || out_height == NULL || out_width == NULL) {
        return false;
    }
    if(prediction->rank < 3) {
        return false;
    }
    for(axis = 0; axis < prediction->rank - 3; ++axis) {
        gsx_size_t dim_extent = (gsx_size_t)prediction->shape[axis];
        gsx_size_t next_outer_count = 0;

        if(dim_extent == 0 || gsx_size_mul_overflows(outer_count, dim_extent, &next_outer_count)) {
            return false;
        }
        outer_count = next_outer_count;
    }
    if(prediction->storage_format == GSX_STORAGE_FORMAT_HWC) {
        height = prediction->shape[prediction->rank - 3];
        width = prediction->shape[prediction->rank - 2];
        channels = prediction->shape[prediction->rank - 1];
    } else {
        channels = prediction->shape[prediction->rank - 3];
        height = prediction->shape[prediction->rank - 2];
        width = prediction->shape[prediction->rank - 1];
    }
    if(channels <= 0 || height <= 0 || width <= 0) {
        return false;
    }

    *out_outer_count = outer_count;
    *out_channels = channels;
    *out_height = height;
    *out_width = width;
    return true;
}

static gsx_error gsx_metal_loss_validate_ssim_tensors(
    gsx_tensor_t prediction,
    gsx_tensor_t target,
    gsx_tensor_t loss_map_accumulator,
    gsx_tensor_t grad_prediction_accumulator)
{
    gsx_storage_format storage_format = prediction->storage_format;
    gsx_size_t outer_count = 0;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;

    if(prediction->rank < 3 || target->rank < 3) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM requires rank>=3 tensors");
    }
    if(loss_map_accumulator != NULL && loss_map_accumulator->rank < 3) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM requires rank>=3 tensors");
    }
    if(storage_format != GSX_STORAGE_FORMAT_CHW && storage_format != GSX_STORAGE_FORMAT_HWC) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM supports only CHW and HWC tensors");
    }
    if(target->storage_format != storage_format) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM tensors must share one supported storage format");
    }
    if(loss_map_accumulator != NULL && loss_map_accumulator->storage_format != storage_format) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM tensors must share one supported storage format");
    }
    if(grad_prediction_accumulator != NULL) {
        if(grad_prediction_accumulator->rank < 3 || grad_prediction_accumulator->storage_format != storage_format) {
            return gsx_make_error(
                GSX_ERROR_NOT_SUPPORTED, "metal SSIM grad_prediction_accumulator must match the image storage format");
        }
    }
    if(!gsx_metal_loss_ssim_extract_layout(prediction, &outer_count, &channels, &height, &width)) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
    }
    (void)outer_count;
    (void)channels;
    (void)height;
    (void)width;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_execute_pointwise(
    gsx_loss_algorithm algorithm,
    bool backward,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t accumulator_tensor,
    gsx_float_t scale)
{
    gsx_backend_tensor_view prediction_view = { 0 };
    gsx_backend_tensor_view target_view = { 0 };
    gsx_backend_tensor_view accumulator_view = { 0 };
    gsx_metal_loss_pointwise_params params = { 0 };
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_loss_validate_tensor_f32_device(prediction_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_tensor_f32_device(target_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_tensor_f32_device(accumulator_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = prediction_tensor->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal loss tensors must contain at least one element");
    }
    if(element_count > (gsx_size_t)UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal loss element count exceeds uint32 dispatch limits");
    }

    gsx_metal_loss_make_tensor_view(prediction_tensor, &prediction_view);
    gsx_metal_loss_make_tensor_view(target_tensor, &target_view);
    gsx_metal_loss_make_tensor_view(accumulator_tensor, &accumulator_view);

    params.element_count = (uint32_t)element_count;
    params.scale = (float)scale;

    switch(algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        if(backward) {
            return gsx_metal_backend_dispatch_loss_mse_backward_f32(
                prediction_tensor->backing_buffer->buffer_type->backend,
                &prediction_view,
                &target_view,
                &accumulator_view,
                &params);
        }
        return gsx_metal_backend_dispatch_loss_mse_f32(
            prediction_tensor->backing_buffer->buffer_type->backend,
            &prediction_view,
            &target_view,
            &accumulator_view,
            &params);
    case GSX_LOSS_ALGORITHM_L1:
        if(backward) {
            return gsx_metal_backend_dispatch_loss_l1_backward_f32(
                prediction_tensor->backing_buffer->buffer_type->backend,
                &prediction_view,
                &target_view,
                &accumulator_view,
                &params);
        }
        return gsx_metal_backend_dispatch_loss_l1_f32(
            prediction_tensor->backing_buffer->buffer_type->backend,
            &prediction_view,
            &target_view,
            &accumulator_view,
            &params);
    case GSX_LOSS_ALGORITHM_SSIM:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal SSIM loss is not implemented");
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

static gsx_error gsx_metal_loss_execute_ssim_forward(
    gsx_metal_loss_context *metal_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t loss_map_tensor,
    bool train,
    gsx_float_t loss_scale)
{
    gsx_backend_tensor_view prediction_view = { 0 };
    gsx_backend_tensor_view target_view = { 0 };
    gsx_backend_tensor_view loss_map_view = { 0 };
    gsx_backend_tensor_view scratch_a_view = { 0 };
    gsx_backend_tensor_view scratch_b_view = { 0 };
    gsx_metal_loss_ssim_params params = { 0 };
    gsx_size_t element_count = 0;
    gsx_size_t outer_count = 0;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_loss_validate_tensor_f32_device(prediction_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_tensor_f32_device(target_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_tensor_f32_device(loss_map_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_ssim_tensors(prediction_tensor, target_tensor, loss_map_tensor, NULL);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_metal_loss_ssim_extract_layout(prediction_tensor, &outer_count, &channels, &height, &width)) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
    }

    element_count = prediction_tensor->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal loss tensors must contain at least one element");
    }
    if(element_count > (gsx_size_t)UINT32_MAX || outer_count > (gsx_size_t)UINT32_MAX || (gsx_size_t)channels > (gsx_size_t)UINT32_MAX
        || (gsx_size_t)height > (gsx_size_t)UINT32_MAX || (gsx_size_t)width > (gsx_size_t)UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal SSIM dispatch dimensions exceed uint32 limits");
    }

    error = gsx_metal_loss_ensure_ssim_scratch_arena(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_ensure_ssim_dummy_buffer(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(train) {
        error = gsx_metal_loss_ensure_ssim_scratch(metal_context, element_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        gsx_metal_loss_make_tensor_view(metal_context->ssim_buffer_a, &scratch_a_view);
        gsx_metal_loss_make_tensor_view(metal_context->ssim_buffer_b, &scratch_b_view);
    } else {
        gsx_metal_loss_make_tensor_view(metal_context->ssim_dummy_buffer, &scratch_a_view);
        gsx_metal_loss_make_tensor_view(metal_context->ssim_dummy_buffer, &scratch_b_view);
    }

    gsx_metal_loss_make_tensor_view(prediction_tensor, &prediction_view);
    gsx_metal_loss_make_tensor_view(target_tensor, &target_view);
    gsx_metal_loss_make_tensor_view(loss_map_tensor, &loss_map_view);

    params.outer_count = (uint32_t)outer_count;
    params.channels = (uint32_t)channels;
    params.height = (uint32_t)height;
    params.width = (uint32_t)width;
    params.element_count = (uint32_t)element_count;
    params.has_scratch = train ? 1u : 0u;
    params.scale = (float)loss_scale;

    if(prediction_tensor->storage_format == GSX_STORAGE_FORMAT_HWC) {
        return gsx_metal_backend_dispatch_loss_ssim_hwc_f32(
            prediction_tensor->backing_buffer->buffer_type->backend,
            &prediction_view,
            &target_view,
            &loss_map_view,
            &scratch_a_view,
            &scratch_b_view,
            &params);
    }

    return gsx_metal_backend_dispatch_loss_ssim_chw_f32(
        prediction_tensor->backing_buffer->buffer_type->backend,
        &prediction_view,
        &target_view,
        &loss_map_view,
        &scratch_a_view,
        &scratch_b_view,
        &params);
}

static gsx_error gsx_metal_loss_execute_ssim_backward(
    gsx_metal_loss_context *metal_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t grad_tensor,
    gsx_float_t grad_scale)
{
    gsx_backend_tensor_view prediction_view = { 0 };
    gsx_backend_tensor_view target_view = { 0 };
    gsx_backend_tensor_view grad_view = { 0 };
    gsx_backend_tensor_view scratch_a_view = { 0 };
    gsx_backend_tensor_view scratch_b_view = { 0 };
    gsx_metal_loss_ssim_params params = { 0 };
    gsx_size_t element_count = 0;
    gsx_size_t outer_count = 0;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_loss_validate_tensor_f32_device(prediction_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_tensor_f32_device(target_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_tensor_f32_device(grad_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_loss_validate_ssim_tensors(prediction_tensor, target_tensor, NULL, grad_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(metal_context->ssim_buffer_a == NULL || metal_context->ssim_buffer_b == NULL) {
        return gsx_make_error(
            GSX_ERROR_INVALID_STATE, "metal SSIM backward requires a train forward pass on the same context");
    }
    if(!gsx_metal_loss_ssim_extract_layout(prediction_tensor, &outer_count, &channels, &height, &width)) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
    }

    element_count = prediction_tensor->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal loss tensors must contain at least one element");
    }
    if(element_count > (gsx_size_t)UINT32_MAX || outer_count > (gsx_size_t)UINT32_MAX || (gsx_size_t)channels > (gsx_size_t)UINT32_MAX
        || (gsx_size_t)height > (gsx_size_t)UINT32_MAX || (gsx_size_t)width > (gsx_size_t)UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal SSIM dispatch dimensions exceed uint32 limits");
    }

    gsx_metal_loss_make_tensor_view(prediction_tensor, &prediction_view);
    gsx_metal_loss_make_tensor_view(target_tensor, &target_view);
    gsx_metal_loss_make_tensor_view(grad_tensor, &grad_view);
    gsx_metal_loss_make_tensor_view(metal_context->ssim_buffer_a, &scratch_a_view);
    gsx_metal_loss_make_tensor_view(metal_context->ssim_buffer_b, &scratch_b_view);

    params.outer_count = (uint32_t)outer_count;
    params.channels = (uint32_t)channels;
    params.height = (uint32_t)height;
    params.width = (uint32_t)width;
    params.element_count = (uint32_t)element_count;
    params.has_scratch = 1u;
    params.scale = (float)grad_scale;

    if(prediction_tensor->storage_format == GSX_STORAGE_FORMAT_HWC) {
        return gsx_metal_backend_dispatch_loss_ssim_backward_hwc_f32(
            prediction_tensor->backing_buffer->buffer_type->backend,
            &prediction_view,
            &target_view,
            &grad_view,
            &scratch_a_view,
            &scratch_b_view,
            &params);
    }

    return gsx_metal_backend_dispatch_loss_ssim_backward_chw_f32(
        prediction_tensor->backing_buffer->buffer_type->backend,
        &prediction_view,
        &target_view,
        &grad_view,
        &scratch_a_view,
        &scratch_b_view,
        &params);
}

static gsx_error gsx_metal_loss_execute_forward(
    gsx_metal_loss *metal_loss,
    gsx_metal_loss_context *metal_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t loss_map_tensor,
    bool train,
    gsx_float_t loss_scale)
{
    switch(metal_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
    case GSX_LOSS_ALGORITHM_L1:
        return gsx_metal_loss_execute_pointwise(
            metal_loss->base.algorithm,
            false,
            prediction_tensor,
            target_tensor,
            loss_map_tensor,
            loss_scale);
    case GSX_LOSS_ALGORITHM_SSIM:
        return gsx_metal_loss_execute_ssim_forward(
            metal_context,
            prediction_tensor,
            target_tensor,
            loss_map_tensor,
            train,
            loss_scale);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

static gsx_error gsx_metal_loss_execute_backward(
    gsx_metal_loss *metal_loss,
    gsx_metal_loss_context *metal_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t grad_tensor,
    gsx_float_t grad_scale)
{
    switch(metal_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
    case GSX_LOSS_ALGORITHM_L1:
        return gsx_metal_loss_execute_pointwise(
            metal_loss->base.algorithm,
            true,
            prediction_tensor,
            target_tensor,
            grad_tensor,
            grad_scale);
    case GSX_LOSS_ALGORITHM_SSIM:
        return gsx_metal_loss_execute_ssim_backward(
            metal_context,
            prediction_tensor,
            target_tensor,
            grad_tensor,
            grad_scale);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

gsx_error gsx_metal_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss)
{
    gsx_metal_loss *metal_loss = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_loss == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_loss and desc must be non-null");
    }

    *out_loss = NULL;
    metal_loss = (gsx_metal_loss *)calloc(1, sizeof(*metal_loss));
    if(metal_loss == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal loss");
    }

    error = gsx_loss_base_init(&metal_loss->base, &gsx_metal_loss_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(metal_loss);
        return error;
    }

    *out_loss = &metal_loss->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_destroy(gsx_loss_t loss)
{
    gsx_metal_loss *metal_loss = (gsx_metal_loss *)loss;

    if(loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss must be non-null");
    }

    gsx_loss_base_deinit(&metal_loss->base);
    free(metal_loss);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context)
{
    gsx_metal_loss_context *metal_context = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(loss == NULL || out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss and out_context must be non-null");
    }

    *out_context = NULL;
    metal_context = (gsx_metal_loss_context *)calloc(1, sizeof(*metal_context));
    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal loss context");
    }
    error = gsx_loss_context_base_init(&metal_context->base, &gsx_metal_loss_context_iface, loss);
    if(!gsx_error_is_success(error)) {
        free(metal_context);
        return error;
    }
    metal_context->ssim_arena = NULL;
    metal_context->ssim_dummy_buffer = NULL;
    metal_context->ssim_buffer_a = NULL;
    metal_context->ssim_buffer_b = NULL;
    metal_context->ssim_capacity_elements = 0u;

    *out_context = &metal_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_context_destroy(gsx_loss_context_t context)
{
    gsx_metal_loss_context *metal_context = (gsx_metal_loss_context *)context;
    gsx_error first_error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    error = gsx_metal_loss_release_ssim_scratch_buffers(metal_context);
    if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
        first_error = error;
    }
    if(metal_context->ssim_dummy_buffer != NULL) {
        error = gsx_tensor_free(metal_context->ssim_dummy_buffer);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }
    if(metal_context->ssim_arena != NULL) {
        error = gsx_arena_free(metal_context->ssim_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }
    gsx_loss_context_base_deinit(&metal_context->base);
    free(metal_context);
    if(!gsx_error_is_success(first_error)) {
        return first_error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request)
{
    gsx_metal_loss *metal_loss = (gsx_metal_loss *)loss;
    gsx_metal_loss_context *metal_context = (gsx_metal_loss_context *)context;

    return gsx_metal_loss_execute_forward(
        metal_loss,
        metal_context,
        request->prediction,
        request->target,
        request->loss_map_accumulator,
        request->train,
        request->scale);
}

static gsx_error gsx_metal_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request)
{
    gsx_metal_loss *metal_loss = (gsx_metal_loss *)loss;
    gsx_metal_loss_context *metal_context = (gsx_metal_loss_context *)context;
    gsx_size_t element_count = context->retained_prediction->size_bytes / sizeof(float);

    return gsx_metal_loss_execute_backward(
        metal_loss,
        metal_context,
        context->retained_prediction,
        context->retained_target,
        request->grad_prediction_accumulator,
        gsx_metal_loss_grad_scale(metal_loss, element_count, request->scale));
}
