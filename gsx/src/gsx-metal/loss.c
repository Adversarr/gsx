#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

typedef struct gsx_metal_loss {
    struct gsx_loss base;
} gsx_metal_loss;

typedef struct gsx_metal_loss_context {
    struct gsx_loss_context base;
} gsx_metal_loss_context;

static gsx_error gsx_metal_loss_destroy(gsx_loss_t loss);
static gsx_error gsx_metal_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context);
static gsx_error gsx_metal_loss_context_destroy(gsx_loss_context_t context);
static gsx_error gsx_metal_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request);
static gsx_error gsx_metal_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request);

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

    *out_context = &metal_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_context_destroy(gsx_loss_context_t context)
{
    gsx_metal_loss_context *metal_context = (gsx_metal_loss_context *)context;

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    gsx_loss_context_base_deinit(&metal_context->base);
    free(metal_context);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request)
{
    gsx_metal_loss *metal_loss = (gsx_metal_loss *)loss;

    (void)context;
    return gsx_metal_loss_execute_pointwise(
        metal_loss->base.algorithm,
        false,
        request->prediction,
        request->target,
        request->loss_map_accumulator,
        request->scale);
}

static gsx_error gsx_metal_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request)
{
    gsx_metal_loss *metal_loss = (gsx_metal_loss *)loss;
    gsx_size_t element_count = context->retained_prediction->size_bytes / sizeof(float);

    return gsx_metal_loss_execute_pointwise(
        metal_loss->base.algorithm,
        true,
        context->retained_prediction,
        context->retained_target,
        request->grad_prediction_accumulator,
        gsx_metal_loss_grad_scale(metal_loss, element_count, request->scale));
}
