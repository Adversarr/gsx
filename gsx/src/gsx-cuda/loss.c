#include "internal.h"

#include <math.h>
#include <stdlib.h>

typedef struct gsx_cuda_loss {
    struct gsx_loss base;
} gsx_cuda_loss;

static gsx_error gsx_cuda_loss_destroy(gsx_loss_t loss);
static gsx_error gsx_cuda_loss_evaluate(gsx_loss_t loss, const gsx_loss_request *request);

static const gsx_loss_i gsx_cuda_loss_iface = {
    gsx_cuda_loss_destroy,
    gsx_cuda_loss_evaluate
};

static bool gsx_cuda_loss_buffer_is_device(gsx_backend_buffer_t buffer)
{
    return buffer != NULL && gsx_cuda_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static unsigned char *gsx_cuda_loss_tensor_device_bytes(gsx_tensor_t tensor)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(tensor->backing_buffer);

    return (unsigned char *)cuda_buffer->ptr + (size_t)tensor->offset_bytes;
}

static float *gsx_cuda_loss_tensor_device_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cuda_loss_tensor_device_bytes(tensor);
}

static const float *gsx_cuda_loss_tensor_device_const_f32(gsx_tensor_t tensor)
{
    return (const float *)gsx_cuda_loss_tensor_device_bytes(tensor);
}

static gsx_error gsx_cuda_loss_validate_tensor_f32_device(gsx_tensor_t tensor)
{
    if(!gsx_cuda_loss_buffer_is_device(tensor->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda loss requires device-backed tensors");
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda loss tensors must use float32 storage");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_loss_validate_request_f32_device(const gsx_loss_request *request)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_loss_validate_tensor_f32_device(request->prediction);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_validate_tensor_f32_device(request->target);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_validate_tensor_f32_device(request->loss_map_accumulator);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_prediction_accumulator != NULL) {
        error = gsx_cuda_loss_validate_tensor_f32_device(request->grad_prediction_accumulator);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_cuda_loss_grad_scale(const gsx_cuda_loss *cuda_loss, gsx_size_t element_count, gsx_float_t scale)
{
    if(cuda_loss->base.grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN) {
        return scale / (float)element_count;
    }

    return scale;
}

/*
 * TODO(adversarr): gsx_loss_validate_request() runs before this backend-specific
 * SSIM capability check, so some unsupported layout/rank cases currently surface
 * as INVALID_ARGUMENT instead of NOT_SUPPORTED. Revisit if we want stable
 * backend-capability error codes across CPU/CUDA SSIM implementations.
 */
static gsx_error gsx_cuda_loss_validate_ssim_request(const gsx_loss_request *request)
{
    gsx_storage_format storage_format = request->prediction->storage_format;

    if(request->prediction->rank != 3 || request->target->rank != 3 || request->loss_map_accumulator->rank != 3) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM requires rank-3 tensors");
    }
    if(storage_format != GSX_STORAGE_FORMAT_CHW && storage_format != GSX_STORAGE_FORMAT_HWC) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM supports only CHW and HWC tensors");
    }
    if(request->target->storage_format != storage_format || request->loss_map_accumulator->storage_format != storage_format) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM tensors must share one supported storage format");
    }
    if(request->grad_prediction_accumulator != NULL) {
        if(request->grad_prediction_accumulator->rank != 3
            || request->grad_prediction_accumulator->storage_format != storage_format) {
            return gsx_make_error(
                GSX_ERROR_NOT_SUPPORTED, "cuda SSIM grad_prediction_accumulator must match the image storage format");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss)
{
    gsx_cuda_loss *cuda_loss = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_loss == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_loss and desc must be non-null");
    }

    *out_loss = NULL;
    cuda_loss = (gsx_cuda_loss *)calloc(1, sizeof(*cuda_loss));
    if(cuda_loss == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda loss");
    }

    error = gsx_loss_base_init(&cuda_loss->base, &gsx_cuda_loss_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(cuda_loss);
        return error;
    }

    *out_loss = &cuda_loss->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_loss_destroy(gsx_loss_t loss)
{
    gsx_cuda_loss *cuda_loss = (gsx_cuda_loss *)loss;

    if(loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss must be non-null");
    }

    gsx_loss_base_deinit(&cuda_loss->base);
    free(cuda_loss);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_loss_evaluate(gsx_loss_t loss, const gsx_loss_request *request)
{
    gsx_cuda_loss *cuda_loss = (gsx_cuda_loss *)loss;
    float *loss_map = NULL;
    float *grad_prediction = NULL;
    const float *prediction = NULL;
    const float *target = NULL;
    gsx_size_t element_count = 0;
    float grad_scale = 0.0f;
    void *stream = NULL;
    cudaError_t cuda_error = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_loss_validate_request_f32_device(request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = request->prediction->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda loss tensors must contain at least one element");
    }

    prediction = gsx_cuda_loss_tensor_device_const_f32(request->prediction);
    target = gsx_cuda_loss_tensor_device_const_f32(request->target);
    loss_map = gsx_cuda_loss_tensor_device_f32(request->loss_map_accumulator);
    if(request->grad_prediction_accumulator != NULL) {
        grad_prediction = gsx_cuda_loss_tensor_device_f32(request->grad_prediction_accumulator);
    }
    grad_scale = gsx_cuda_loss_grad_scale(cuda_loss, element_count, request->scale);
    error = gsx_backend_get_major_stream(cuda_loss->base.backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    switch(cuda_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        cuda_error = gsx_cuda_loss_mse_f32_kernel_launch(
            loss_map,
            grad_prediction,
            prediction,
            target,
            element_count,
            request->scale,
            grad_scale,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda MSE loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_L1:
        cuda_error = gsx_cuda_loss_l1_f32_kernel_launch(
            loss_map,
            grad_prediction,
            prediction,
            target,
            element_count,
            request->scale,
            grad_scale,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda L1 loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_SSIM:
        error = gsx_cuda_loss_validate_ssim_request(request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        /* CHW uses [C,H,W]; HWC uses [H,W,C]. */
        if(request->prediction->storage_format == GSX_STORAGE_FORMAT_CHW) {
            cuda_error = gsx_cuda_loss_ssim_chw_f32_kernel_launch(
                loss_map,
                grad_prediction,
                prediction,
                target,
                request->prediction->shape[0],
                request->prediction->shape[1],
                request->prediction->shape[2],
                request->scale,
                grad_scale,
                (cudaStream_t)stream
            );
        } else {
            cuda_error = gsx_cuda_loss_ssim_hwc_f32_kernel_launch(
                loss_map,
                grad_prediction,
                prediction,
                target,
                request->prediction->shape[2],
                request->prediction->shape[0],
                request->prediction->shape[1],
                request->scale,
                grad_scale,
                (cudaStream_t)stream
            );
        }
        error = gsx_cuda_make_error(cuda_error, "cuda SSIM loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}
