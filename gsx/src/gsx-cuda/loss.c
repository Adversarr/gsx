#include "internal.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>

typedef struct gsx_cuda_loss {
    struct gsx_loss base;
} gsx_cuda_loss;

typedef struct gsx_cuda_loss_context {
    struct gsx_loss_context base;
    gsx_arena_t ssim_arena;
    gsx_tensor_t ssim_buffer_a;
    gsx_tensor_t ssim_buffer_b;
    gsx_size_t ssim_capacity_elements;
} gsx_cuda_loss_context;

static gsx_error gsx_cuda_loss_destroy(gsx_loss_t loss);
static gsx_error gsx_cuda_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context);
static gsx_error gsx_cuda_loss_context_destroy(gsx_loss_context_t context);
static gsx_error gsx_cuda_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request);
static gsx_error gsx_cuda_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request);
static gsx_error gsx_cuda_loss_execute_forward(
    gsx_cuda_loss *cuda_loss,
    gsx_cuda_loss_context *cuda_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t loss_map_tensor,
    bool train,
    gsx_float_t loss_scale
);
static gsx_error gsx_cuda_loss_execute_backward(
    gsx_cuda_loss *cuda_loss,
    gsx_cuda_loss_context *cuda_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t grad_tensor,
    gsx_float_t grad_scale
);

static const gsx_loss_i gsx_cuda_loss_iface = {
    gsx_cuda_loss_destroy,
    gsx_cuda_loss_create_context,
    gsx_cuda_loss_forward,
    gsx_cuda_loss_backward
};

static const gsx_loss_context_i gsx_cuda_loss_context_iface = {
    gsx_cuda_loss_context_destroy
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

static float gsx_cuda_loss_grad_scale(const gsx_cuda_loss *cuda_loss, gsx_size_t element_count, gsx_float_t scale)
{
    if(cuda_loss->base.grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN) {
        return scale / (float)element_count;
    }

    return scale;
}

static gsx_error gsx_cuda_loss_release_ssim_scratch_buffers(gsx_cuda_loss_context *cuda_context)
{
    gsx_error first_error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);

    if(cuda_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda_context must be non-null");
    }
    if(cuda_context->ssim_buffer_a != NULL) {
        gsx_error error = gsx_tensor_free(cuda_context->ssim_buffer_a);

        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        if(gsx_error_is_success(error)) {
            cuda_context->ssim_buffer_a = NULL;
        }
    }
    if(cuda_context->ssim_buffer_b != NULL) {
        gsx_error error = gsx_tensor_free(cuda_context->ssim_buffer_b);

        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        if(gsx_error_is_success(error)) {
            cuda_context->ssim_buffer_b = NULL;
        }
    }
    if(gsx_error_is_success(first_error)) {
        cuda_context->ssim_capacity_elements = 0;
    }

    return first_error;
}

static gsx_error gsx_cuda_loss_ensure_ssim_scratch_arena(gsx_cuda_loss_context *cuda_context)
{
    gsx_backend_buffer_type_t device_buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda_context must be non-null");
    }
    if(cuda_context->ssim_arena != NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_cuda_backend_find_buffer_type(
        cuda_context->base.loss->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    error = gsx_arena_init(&cuda_context->ssim_arena, device_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cuda_loss_cleanup_ssim_sizing_work(gsx_arena_t *arena, gsx_tensor_t *tensor_a, gsx_tensor_t *tensor_b)
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

static gsx_error gsx_cuda_loss_prepare_ssim_scratch_shapes(
    gsx_size_t element_count, gsx_size_t *out_buffer_a_elements, gsx_index_t *out_shape_a, gsx_index_t *out_shape_b)
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

static void gsx_cuda_loss_fill_ssim_scratch_tensor_desc(gsx_tensor_desc *out_desc, gsx_arena_t arena, gsx_index_t element_count)
{
    out_desc->rank = 1;
    out_desc->shape[0] = element_count;
    out_desc->requested_alignment_bytes = 0;
    out_desc->data_type = GSX_DATA_TYPE_F32;
    out_desc->storage_format = GSX_STORAGE_FORMAT_CHW;
    out_desc->arena = arena;
}

static gsx_error gsx_cuda_loss_compute_ssim_scratch_required_bytes(
    gsx_backend_t backend, gsx_size_t element_count, gsx_size_t *out_required_bytes)
{
    gsx_backend_buffer_type_t device_buffer_type = NULL;
    gsx_arena_t dry_run_arena = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_tensor_desc desc = { 0 };
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
    error = gsx_cuda_loss_prepare_ssim_scratch_shapes(element_count, &buffer_a_elements, &shape_a, &shape_b);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_cuda_backend_find_buffer_type(
        backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, device_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    gsx_cuda_loss_fill_ssim_scratch_tensor_desc(&desc, dry_run_arena, shape_a);
    error = gsx_arena_reset(dry_run_arena);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
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
    gsx_cuda_loss_cleanup_ssim_sizing_work(&dry_run_arena, &tensor_a, &tensor_b);
    return error;
}

static gsx_error gsx_cuda_loss_ensure_ssim_scratch(gsx_cuda_loss_context *cuda_context, gsx_size_t element_count)
{
    gsx_size_t buffer_a_elements = 0;
    gsx_size_t required_bytes = 0;
    gsx_tensor_desc desc = { 0 };
    gsx_index_t shape_a = 0;
    gsx_index_t shape_b = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_context == NULL || cuda_context->base.loss == NULL || element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda_context and element_count must be valid");
    }
    if(cuda_context->ssim_buffer_a != NULL && cuda_context->ssim_buffer_b != NULL
        && cuda_context->ssim_capacity_elements >= element_count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_cuda_loss_ensure_ssim_scratch_arena(cuda_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_release_ssim_scratch_buffers(cuda_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(cuda_context->ssim_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_compute_ssim_scratch_required_bytes(cuda_context->base.loss->backend, element_count, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reserve(cuda_context->ssim_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_prepare_ssim_scratch_shapes(element_count, &buffer_a_elements, &shape_a, &shape_b);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gsx_cuda_loss_fill_ssim_scratch_tensor_desc(&desc, cuda_context->ssim_arena, shape_a);
    error = gsx_tensor_init(&cuda_context->ssim_buffer_a, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    desc.shape[0] = shape_b;
    error = gsx_tensor_init(&cuda_context->ssim_buffer_b, &desc);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(cuda_context->ssim_buffer_a);
        cuda_context->ssim_buffer_a = NULL;
        (void)gsx_arena_reset(cuda_context->ssim_arena);
        return error;
    }

    cuda_context->ssim_capacity_elements = element_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

/*
 * TODO(adversarr): gsx_loss_validate_request() runs before this backend-specific
 * SSIM capability check, so some unsupported layout/rank cases currently surface
 * as INVALID_ARGUMENT instead of NOT_SUPPORTED. Revisit if we want stable
 * backend-capability error codes across CPU/CUDA SSIM implementations.
 */
static bool gsx_cuda_loss_ssim_extract_layout(
    gsx_tensor_t prediction,
    gsx_size_t *out_outer_count,
    gsx_index_t *out_channels,
    gsx_index_t *out_height,
    gsx_index_t *out_width
)
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

static gsx_error gsx_cuda_loss_validate_ssim_tensors(
    gsx_tensor_t prediction, gsx_tensor_t target, gsx_tensor_t loss_map_accumulator, gsx_tensor_t grad_prediction_accumulator)
{
    gsx_storage_format storage_format = prediction->storage_format;
    gsx_size_t outer_count = 0;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;

    if(prediction->rank < 3 || target->rank < 3) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM requires rank>=3 tensors");
    }
    if(loss_map_accumulator != NULL && loss_map_accumulator->rank < 3) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM requires rank>=3 tensors");
    }
    if(storage_format != GSX_STORAGE_FORMAT_CHW && storage_format != GSX_STORAGE_FORMAT_HWC) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM supports only CHW and HWC tensors");
    }
    if(target->storage_format != storage_format) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM tensors must share one supported storage format");
    }
    if(loss_map_accumulator != NULL && loss_map_accumulator->storage_format != storage_format) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda SSIM tensors must share one supported storage format");
    }
    if(grad_prediction_accumulator != NULL) {
        if(grad_prediction_accumulator->rank < 3 || grad_prediction_accumulator->storage_format != storage_format) {
            return gsx_make_error(
                GSX_ERROR_NOT_SUPPORTED, "cuda SSIM grad_prediction_accumulator must match the image storage format");
        }
    }
    if(!gsx_cuda_loss_ssim_extract_layout(prediction, &outer_count, &channels, &height, &width)) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
    }
    (void)outer_count;
    (void)channels;
    (void)height;
    (void)width;

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

static gsx_error gsx_cuda_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context)
{
    gsx_cuda_loss_context *cuda_context = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(loss == NULL || out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss and out_context must be non-null");
    }

    *out_context = NULL;
    cuda_context = (gsx_cuda_loss_context *)calloc(1, sizeof(*cuda_context));
    if(cuda_context == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda loss context");
    }
    error = gsx_loss_context_base_init(&cuda_context->base, &gsx_cuda_loss_context_iface, loss);
    if(!gsx_error_is_success(error)) {
        free(cuda_context);
        return error;
    }

    *out_context = &cuda_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_loss_context_destroy(gsx_loss_context_t context)
{
    gsx_cuda_loss_context *cuda_context = (gsx_cuda_loss_context *)context;
    gsx_error first_error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    error = gsx_cuda_loss_release_ssim_scratch_buffers(cuda_context);
    if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
        first_error = error;
    }
    if(cuda_context->ssim_arena != NULL) {
        error = gsx_arena_free(cuda_context->ssim_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }
    gsx_loss_context_base_deinit(&cuda_context->base);
    free(cuda_context);
    if(!gsx_error_is_success(first_error)) {
        return first_error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_loss_execute_forward(
    gsx_cuda_loss *cuda_loss,
    gsx_cuda_loss_context *cuda_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t loss_map_tensor,
    bool train,
    gsx_float_t loss_scale)
{
    float *loss_map = NULL;
    float *ssim_buffer_a = NULL;
    float *ssim_buffer_b = NULL;
    const float *prediction = NULL;
    const float *target = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t ssim_outer_count = 0;
    gsx_index_t ssim_channels = 0;
    gsx_index_t ssim_height = 0;
    gsx_index_t ssim_width = 0;
    void *stream = NULL;
    cudaError_t cuda_error = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_loss_validate_tensor_f32_device(prediction_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_validate_tensor_f32_device(target_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_validate_tensor_f32_device(loss_map_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = prediction_tensor->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda loss tensors must contain at least one element");
    }

    prediction = gsx_cuda_loss_tensor_device_const_f32(prediction_tensor);
    target = gsx_cuda_loss_tensor_device_const_f32(target_tensor);
    loss_map = gsx_cuda_loss_tensor_device_f32(loss_map_tensor);
    error = gsx_backend_get_major_stream(cuda_loss->base.backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    switch(cuda_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        cuda_error = gsx_cuda_loss_mse_f32_forward_kernel_launch(
            loss_map,
            prediction,
            target,
            element_count,
            loss_scale,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda MSE loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_L1:
        cuda_error = gsx_cuda_loss_l1_f32_forward_kernel_launch(
            loss_map,
            prediction,
            target,
            element_count,
            loss_scale,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda L1 loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_SSIM:
        error = gsx_cuda_loss_validate_ssim_tensors(prediction_tensor, target_tensor, loss_map_tensor, NULL);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(!gsx_cuda_loss_ssim_extract_layout(
                prediction_tensor, &ssim_outer_count, &ssim_channels, &ssim_height, &ssim_width)) {
            return gsx_make_error(
                GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
        }
        if(train) {
            error = gsx_cuda_loss_ensure_ssim_scratch(cuda_context, element_count);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            ssim_buffer_a = gsx_cuda_loss_tensor_device_f32(cuda_context->ssim_buffer_a);
            ssim_buffer_b = gsx_cuda_loss_tensor_device_f32(cuda_context->ssim_buffer_b);
        }
        if(prediction_tensor->storage_format == GSX_STORAGE_FORMAT_CHW) {
            cuda_error = gsx_cuda_loss_ssim_chw_f32_forward_kernel_launch(
                loss_map,
                prediction,
                target,
                ssim_outer_count,
                ssim_channels,
                ssim_height,
                ssim_width,
                loss_scale,
                ssim_buffer_a,
                ssim_buffer_b,
                (cudaStream_t)stream
            );
        } else {
            cuda_error = gsx_cuda_loss_ssim_hwc_f32_forward_kernel_launch(
                loss_map,
                prediction,
                target,
                ssim_outer_count,
                ssim_channels,
                ssim_height,
                ssim_width,
                loss_scale,
                ssim_buffer_a,
                ssim_buffer_b,
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

static gsx_error gsx_cuda_loss_execute_backward(
    gsx_cuda_loss *cuda_loss,
    gsx_cuda_loss_context *cuda_context,
    gsx_tensor_t prediction_tensor,
    gsx_tensor_t target_tensor,
    gsx_tensor_t grad_tensor,
    gsx_float_t grad_scale)
{
    float *grad_prediction = NULL;
    const float *prediction = NULL;
    const float *target = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t ssim_outer_count = 0;
    gsx_index_t ssim_channels = 0;
    gsx_index_t ssim_height = 0;
    gsx_index_t ssim_width = 0;
    float *ssim_buffer_a = NULL;
    float *ssim_buffer_b = NULL;
    void *stream = NULL;
    cudaError_t cuda_error = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_loss_validate_tensor_f32_device(prediction_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_validate_tensor_f32_device(target_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_loss_validate_tensor_f32_device(grad_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = prediction_tensor->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda loss tensors must contain at least one element");
    }

    prediction = gsx_cuda_loss_tensor_device_const_f32(prediction_tensor);
    target = gsx_cuda_loss_tensor_device_const_f32(target_tensor);
    grad_prediction = gsx_cuda_loss_tensor_device_f32(grad_tensor);
    error = gsx_backend_get_major_stream(cuda_loss->base.backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    switch(cuda_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        cuda_error = gsx_cuda_loss_mse_f32_backward_kernel_launch(
            grad_prediction,
            prediction,
            target,
            element_count,
            grad_scale,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda MSE loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_L1:
        cuda_error = gsx_cuda_loss_l1_f32_backward_kernel_launch(
            grad_prediction,
            prediction,
            target,
            element_count,
            grad_scale,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda L1 loss kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_LOSS_ALGORITHM_SSIM:
        error = gsx_cuda_loss_validate_ssim_tensors(prediction_tensor, target_tensor, NULL, grad_tensor);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(cuda_context->ssim_buffer_a == NULL || cuda_context->ssim_buffer_b == NULL) {
            return gsx_make_error(
                GSX_ERROR_INVALID_STATE, "cuda SSIM backward requires a train forward pass on the same context");
        }
        if(!gsx_cuda_loss_ssim_extract_layout(
                prediction_tensor, &ssim_outer_count, &ssim_channels, &ssim_height, &ssim_width)) {
            return gsx_make_error(
                GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
        }
        ssim_buffer_a = gsx_cuda_loss_tensor_device_f32(cuda_context->ssim_buffer_a);
        ssim_buffer_b = gsx_cuda_loss_tensor_device_f32(cuda_context->ssim_buffer_b);
        if(prediction_tensor->storage_format == GSX_STORAGE_FORMAT_CHW) {
            cuda_error = gsx_cuda_loss_ssim_chw_f32_backward_kernel_launch(
                grad_prediction,
                prediction,
                target,
                ssim_outer_count,
                ssim_channels,
                ssim_height,
                ssim_width,
                grad_scale,
                ssim_buffer_a,
                ssim_buffer_b,
                (cudaStream_t)stream
            );
        } else {
            cuda_error = gsx_cuda_loss_ssim_hwc_f32_backward_kernel_launch(
                grad_prediction,
                prediction,
                target,
                ssim_outer_count,
                ssim_channels,
                ssim_height,
                ssim_width,
                grad_scale,
                ssim_buffer_a,
                ssim_buffer_b,
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

static gsx_error gsx_cuda_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request)
{
    gsx_cuda_loss *cuda_loss = (gsx_cuda_loss *)loss;
    gsx_cuda_loss_context *cuda_context = (gsx_cuda_loss_context *)context;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_loss_execute_forward(
        cuda_loss, cuda_context, request->prediction, request->target, request->loss_map_accumulator, request->train, request->scale);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request)
{
    gsx_cuda_loss *cuda_loss = (gsx_cuda_loss *)loss;
    gsx_cuda_loss_context *cuda_context = (gsx_cuda_loss_context *)context;
    gsx_size_t element_count = context->retained_prediction->size_bytes / sizeof(float);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_loss_execute_backward(
        cuda_loss,
        cuda_context,
        context->retained_prediction,
        context->retained_target,
        request->grad_prediction_accumulator,
        gsx_cuda_loss_grad_scale(cuda_loss, element_count, request->scale));
    return error;
}
