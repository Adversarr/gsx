#include "gsx-impl.h"

#include <math.h>
#include <string.h>

bool gsx_loss_algorithm_is_valid(gsx_loss_algorithm algorithm)
{
    switch(algorithm) {
    case GSX_LOSS_ALGORITHM_L1:
    case GSX_LOSS_ALGORITHM_MSE:
    case GSX_LOSS_ALGORITHM_SSIM:
        return true;
    }

    return false;
}

bool gsx_loss_grad_normalization_type_is_valid(gsx_loss_grad_normalization_type normalization_type)
{
    switch(normalization_type) {
    case GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN:
    case GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM:
        return true;
    }

    return false;
}

static bool gsx_metric_algorithm_is_valid(gsx_metric_algorithm algorithm)
{
    switch(algorithm) {
    case GSX_METRIC_ALGORITHM_PSNR:
    case GSX_METRIC_ALGORITHM_SSIM:
        return true;
    }

    return false;
}

static gsx_error gsx_loss_require_handle(gsx_loss_t loss)
{
    if(loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss must be non-null");
    }
    if(loss->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "loss implementation is missing an interface");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_loss_validate_bound_tensor(gsx_backend_t backend, gsx_tensor_t tensor, const char *null_message)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, null_message);
    }
    if(tensor->arena == NULL || tensor->arena->dry_run || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "loss tensors must reference accessible storage");
    }
    if(tensor->backing_buffer->buffer_type == NULL || tensor->backing_buffer->buffer_type->backend != backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss tensors must belong to the requested backend");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_loss_validate_tensor_match(gsx_tensor_t lhs, gsx_tensor_t rhs, const char *message)
{
    gsx_index_t dim = 0;

    if(lhs->rank != rhs->rank
        || lhs->data_type != rhs->data_type
        || lhs->storage_format != rhs->storage_format
        || lhs->size_bytes != rhs->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    for(dim = 0; dim < lhs->rank; ++dim) {
        if(lhs->shape[dim] != rhs->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_loss_validate_accumulator_layout(gsx_tensor_t reference, gsx_tensor_t accumulator, const char *message)
{
    gsx_index_t dim = 0;

    if(reference->rank != accumulator->rank || reference->storage_format != accumulator->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    for(dim = 0; dim < reference->rank; ++dim) {
        if(reference->shape[dim] != accumulator->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_loss_tensors_overlap(gsx_tensor_t lhs, gsx_tensor_t rhs)
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

static gsx_error gsx_loss_validate_request(const gsx_loss *loss, const gsx_loss_request *request)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(loss == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss and request must be non-null");
    }

    error = gsx_loss_validate_bound_tensor(loss->backend, request->prediction, "prediction must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_loss_validate_bound_tensor(loss->backend, request->target, "target must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_loss_validate_bound_tensor(loss->backend, request->loss_map_accumulator, "loss_map_accumulator must be non-null");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_prediction_accumulator != NULL) {
        error = gsx_loss_validate_bound_tensor(
            loss->backend, request->grad_prediction_accumulator, "grad_prediction_accumulator must reference accessible storage");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(isfinite((double)request->scale) == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss scale must be finite");
    }

    error = gsx_loss_validate_tensor_match(
        request->prediction, request->target, "loss prediction and target tensors must be compatible");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_loss_validate_accumulator_layout(
        request->prediction, request->loss_map_accumulator, "loss_map_accumulator must match the prediction tensor layout");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_prediction_accumulator != NULL) {
        error = gsx_loss_validate_accumulator_layout(
            request->prediction,
            request->grad_prediction_accumulator,
            "grad_prediction_accumulator must match the prediction tensor layout");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(gsx_loss_tensors_overlap(request->prediction, request->loss_map_accumulator)
        || gsx_loss_tensors_overlap(request->target, request->loss_map_accumulator)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss_map_accumulator must not alias prediction or target");
    }
    if(request->grad_prediction_accumulator != NULL
        && (gsx_loss_tensors_overlap(request->prediction, request->grad_prediction_accumulator)
            || gsx_loss_tensors_overlap(request->target, request->grad_prediction_accumulator))) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_prediction_accumulator must not alias prediction or target");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_loss_validate_desc(gsx_backend_t backend, const gsx_loss_desc *desc)
{
    if(backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and desc must be non-null");
    }
    if(!gsx_loss_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
    }
    if(!gsx_loss_grad_normalization_type_is_valid(desc->grad_normalization)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss grad_normalization is out of range");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_loss_base_init(gsx_loss *loss, const gsx_loss_i *iface, gsx_backend_t backend, const gsx_loss_desc *desc)
{
    if(loss == NULL || iface == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss, iface, backend, and desc must be non-null");
    }

    memset(loss, 0, sizeof(*loss));
    loss->iface = iface;
    loss->backend = backend;
    loss->algorithm = desc->algorithm;
    loss->grad_normalization = desc->grad_normalization;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_loss_base_deinit(gsx_loss *loss)
{
    if(loss == NULL) {
        return;
    }

    loss->iface = NULL;
    loss->backend = NULL;
    loss->algorithm = 0;
    loss->grad_normalization = 0;
}

GSX_API gsx_error gsx_loss_init(gsx_loss_t *out_loss, gsx_backend_t backend, const gsx_loss_desc *desc)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_loss == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_loss, backend, and desc must be non-null");
    }

    *out_loss = NULL;
    if(backend->iface == NULL || backend->iface->create_loss == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend does not implement loss creation");
    }

    error = gsx_loss_validate_desc(backend, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return backend->iface->create_loss(backend, desc, out_loss);
}

GSX_API gsx_error gsx_loss_free(gsx_loss_t loss)
{
    gsx_error error = gsx_loss_require_handle(loss);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(loss->iface->destroy == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "loss destroy is not implemented");
    }

    return loss->iface->destroy(loss);
}

GSX_API gsx_error gsx_loss_get_desc(gsx_loss_t loss, gsx_loss_desc *out_desc)
{
    gsx_error error = gsx_loss_require_handle(loss);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }

    out_desc->algorithm = loss->algorithm;
    out_desc->grad_normalization = loss->grad_normalization;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_loss_evaluate(gsx_loss_t loss, const gsx_loss_request *request)
{
    gsx_error error = gsx_loss_require_handle(loss);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(loss->iface->evaluate == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "loss evaluate is not implemented");
    }

    error = gsx_loss_validate_request(loss, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return loss->iface->evaluate(loss, request);
}

GSX_API gsx_error gsx_loss_get_algorithm_name(gsx_loss_algorithm algorithm, const char **out_name)
{
    if(out_name == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_name must be non-null");
    }

    switch(algorithm) {
    case GSX_LOSS_ALGORITHM_L1:
        *out_name = "l1";
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_MSE:
        *out_name = "mse";
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_SSIM:
        *out_name = "ssim";
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

GSX_API gsx_error gsx_metric_init(gsx_metric_t *out_metric, gsx_backend_t backend, const gsx_metric_desc *desc)
{
    if(out_metric == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_metric, backend, and desc must be non-null");
    }

    *out_metric = NULL;
    if(!gsx_metric_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metric algorithm is out of range");
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metrics are not implemented in this round");
}

GSX_API gsx_error gsx_metric_free(gsx_metric_t metric)
{
    if(metric == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metric must be non-null");
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metrics are not implemented in this round");
}

GSX_API gsx_error gsx_metric_get_desc(gsx_metric_t metric, gsx_metric_desc *out_desc)
{
    if(metric == NULL || out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metric and out_desc must be non-null");
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metrics are not implemented in this round");
}

GSX_API gsx_error gsx_metric_evaluate(gsx_metric_t metric, const gsx_metric_request *request, gsx_float_t *out_value)
{
    if(metric == NULL || request == NULL || out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metric, request, and out_value must be non-null");
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metrics are not implemented in this round");
}

GSX_API gsx_error gsx_metric_get_algorithm_name(gsx_metric_algorithm algorithm, const char **out_name)
{
    if(out_name == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_name must be non-null");
    }

    switch(algorithm) {
    case GSX_METRIC_ALGORITHM_PSNR:
        *out_name = "psnr";
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_METRIC_ALGORITHM_SSIM:
        *out_name = "ssim";
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metric algorithm is out of range");
}
