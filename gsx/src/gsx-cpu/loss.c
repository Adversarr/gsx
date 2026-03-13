#include "internal.h"

#include <math.h>
#include <stdlib.h>

typedef struct gsx_cpu_loss {
    struct gsx_loss base;
} gsx_cpu_loss;

static gsx_error gsx_cpu_loss_destroy(gsx_loss_t loss);
static gsx_error gsx_cpu_loss_evaluate(gsx_loss_t loss, const gsx_loss_request *request);
static gsx_error gsx_cpu_loss_evaluate_mse(
    const gsx_cpu_loss *cpu_loss,
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t scale
);
static gsx_error gsx_cpu_loss_evaluate_l1(
    const gsx_cpu_loss *cpu_loss,
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t scale
);
static gsx_error gsx_cpu_loss_evaluate_ssim(const gsx_cpu_loss *cpu_loss, const gsx_loss_request *request);

static const gsx_loss_i gsx_cpu_loss_iface = {
    gsx_cpu_loss_destroy,
    gsx_cpu_loss_evaluate
};

/* CPU-private access boundary: only these helpers may dereference cpu_buffer->data. */
static unsigned char *gsx_cpu_loss_tensor_data_bytes(gsx_tensor_t tensor)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)tensor->backing_buffer;

    return (unsigned char *)cpu_buffer->data + (size_t)tensor->offset_bytes;
}

static float *gsx_cpu_loss_tensor_data_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cpu_loss_tensor_data_bytes(tensor);
}

static gsx_error gsx_cpu_loss_validate_tensor_f32(gsx_backend_t backend, gsx_tensor_t tensor, const char *name)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, name);
    }
    if(tensor->arena == NULL || tensor->arena->dry_run || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cpu loss tensors must reference accessible storage");
    }
    if(tensor->backing_buffer->buffer_type == NULL || tensor->backing_buffer->buffer_type->backend != backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss tensors must belong to the owning backend");
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu loss tensors must use float32 storage");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_validate_request_f32(const gsx_cpu_loss *cpu_loss, const gsx_loss_request *request)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cpu_loss_validate_tensor_f32(cpu_loss->base.backend, request->prediction, "prediction must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_loss_validate_tensor_f32(cpu_loss->base.backend, request->target, "target must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_loss_validate_tensor_f32(
        cpu_loss->base.backend, request->loss_map_accumulator, "loss_map_accumulator must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_prediction_accumulator != NULL) {
        error = gsx_cpu_loss_validate_tensor_f32(
            cpu_loss->base.backend,
            request->grad_prediction_accumulator,
            "grad_prediction_accumulator must reference accessible storage");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_cpu_loss_grad_scale(const gsx_cpu_loss *cpu_loss, gsx_size_t element_count, gsx_float_t scale)
{
    if(cpu_loss->base.grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN) {
        return scale / (float)element_count;
    }

    return scale;
}

gsx_error gsx_cpu_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss)
{
    gsx_cpu_loss *cpu_loss = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_loss == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_loss and desc must be non-null");
    }

    *out_loss = NULL;
    if(desc->algorithm == GSX_LOSS_ALGORITHM_SSIM) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu backend does not implement SSIM loss yet");
    }

    cpu_loss = (gsx_cpu_loss *)calloc(1, sizeof(*cpu_loss));
    if(cpu_loss == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu loss");
    }

    error = gsx_loss_base_init(&cpu_loss->base, &gsx_cpu_loss_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(cpu_loss);
        return error;
    }

    *out_loss = &cpu_loss->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_destroy(gsx_loss_t loss)
{
    gsx_cpu_loss *cpu_loss = (gsx_cpu_loss *)loss;

    if(loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss must be non-null");
    }

    gsx_loss_base_deinit(&cpu_loss->base);
    free(cpu_loss);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_evaluate(gsx_loss_t loss, const gsx_loss_request *request)
{
    gsx_cpu_loss *cpu_loss = (gsx_cpu_loss *)loss;
    const float *prediction_values = NULL;
    const float *target_values = NULL;
    float *loss_map_values = NULL;
    float *grad_values = NULL;
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(loss == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss and request must be non-null");
    }

    error = gsx_cpu_loss_validate_request_f32(cpu_loss, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = request->prediction->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu loss tensors must contain at least one element");
    }

    prediction_values = gsx_cpu_loss_tensor_data_f32(request->prediction);
    target_values = gsx_cpu_loss_tensor_data_f32(request->target);
    loss_map_values = gsx_cpu_loss_tensor_data_f32(request->loss_map_accumulator);
    if(request->grad_prediction_accumulator != NULL) {
        grad_values = gsx_cpu_loss_tensor_data_f32(request->grad_prediction_accumulator);
    }

    switch(cpu_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        return gsx_cpu_loss_evaluate_mse(
            cpu_loss, prediction_values, target_values, loss_map_values, grad_values, element_count, request->scale);
    case GSX_LOSS_ALGORITHM_L1:
        return gsx_cpu_loss_evaluate_l1(
            cpu_loss, prediction_values, target_values, loss_map_values, grad_values, element_count, request->scale);
    case GSX_LOSS_ALGORITHM_SSIM:
        return gsx_cpu_loss_evaluate_ssim(cpu_loss, request);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

static gsx_error gsx_cpu_loss_evaluate_mse(
    const gsx_cpu_loss *cpu_loss,
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t scale
)
{
    gsx_size_t element_index = 0;
    float grad_scale = gsx_cpu_loss_grad_scale(cpu_loss, element_count, scale);

    for(element_index = 0; element_index < element_count; ++element_index) {
        float diff = prediction_values[element_index] - target_values[element_index];

        loss_map_values[element_index] += scale * (diff * diff);
        if(grad_values != NULL) {
            grad_values[element_index] += 2.0f * diff * grad_scale;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_evaluate_l1(
    const gsx_cpu_loss *cpu_loss,
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t scale
)
{
    gsx_size_t element_index = 0;
    float grad_scale = gsx_cpu_loss_grad_scale(cpu_loss, element_count, scale);

    for(element_index = 0; element_index < element_count; ++element_index) {
        float diff = prediction_values[element_index] - target_values[element_index];
        float sign = 0.0f;

        if(diff > 0.0f) {
            sign = 1.0f;
        } else if(diff < 0.0f) {
            sign = -1.0f;
        }
        loss_map_values[element_index] += scale * fabsf(diff);
        if(grad_values != NULL) {
            grad_values[element_index] += sign * grad_scale;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_evaluate_ssim(const gsx_cpu_loss *cpu_loss, const gsx_loss_request *request)
{
    (void)cpu_loss;
    (void)request;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu backend does not implement SSIM loss yet");
}
