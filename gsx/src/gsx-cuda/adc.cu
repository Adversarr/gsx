#include "internal.h"

#include <stddef.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

typedef struct gsx_cuda_adc {
    struct gsx_adc base;
} gsx_cuda_adc;

static gsx_error gsx_cuda_adc_destroy(gsx_adc_t adc);
static gsx_error gsx_cuda_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);

static const gsx_adc_i gsx_cuda_adc_iface = {
    gsx_cuda_adc_destroy,
    gsx_cuda_adc_step
};

typedef struct gsx_cuda_adc_refine_data {
    gsx_size_t count;
    float *grad_acc;
    float *logscale;
    float *opacity;
    float *rotation;
    float *max_screen_radius;
    bool has_max_screen_radius;
} gsx_cuda_adc_refine_data;

static gsx_size_t gsx_cuda_adc_non_negative_index(gsx_index_t value)
{
    if(value <= 0) {
        return 0;
    }
    return (gsx_size_t)value;
}

static void gsx_cuda_adc_zero_result(gsx_adc_result *out_result)
{
    out_result->gaussians_before = 0;
    out_result->gaussians_after = 0;
    out_result->pruned_count = 0;
    out_result->duplicated_count = 0;
    out_result->grown_count = 0;
    out_result->reset_count = 0;
    out_result->mutated = false;
}

static bool gsx_cuda_adc_optim_enabled(const gsx_adc_request *request)
{
    return request != NULL && request->optim != NULL && request->optim->iface != NULL;
}

static bool gsx_cuda_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_cuda_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_cuda_adc_non_negative_index(desc->end_refine);

    if(desc->refine_every == 0) {
        return false;
    }
    if(global_step % desc->refine_every != 0) {
        return false;
    }
    if(global_step < start_refine || global_step > end_refine) {
        return false;
    }
    return true;
}

static bool gsx_cuda_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_cuda_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_cuda_adc_non_negative_index(desc->end_refine);

    if(desc->reset_every == 0) {
        return false;
    }
    if(global_step % desc->reset_every != 0) {
        return false;
    }
    if(global_step < start_refine || global_step >= end_refine) {
        return false;
    }
    return true;
}

static gsx_error gsx_cuda_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count)
{
    gsx_gs_info info = {};
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_count must be non-null");
    }

    error = gsx_gs_get_info(gs, &info);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_count = info.count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float *gsx_cuda_adc_tensor_device_f32(gsx_tensor_t tensor)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(tensor->backing_buffer);

    return (float *)((unsigned char *)cuda_buffer->ptr + (size_t)tensor->offset_bytes);
}

static int32_t *gsx_cuda_adc_tensor_device_i32(gsx_tensor_t tensor)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(tensor->backing_buffer);

    return (int32_t *)((unsigned char *)cuda_buffer->ptr + (size_t)tensor->offset_bytes);
}

static gsx_error gsx_cuda_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    float **out_values
)
{
    gsx_tensor_t tensor = NULL;
    gsx_size_t expected_count = 0;
    gsx_size_t byte_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_values == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_values must be non-null");
    }
    *out_values = NULL;

    error = gsx_gs_get_field(gs, field, &tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda default adc currently supports only float32 gs fields");
    }
    if(tensor->rank != 1 && tensor->rank != 2) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "unexpected gs field rank for cuda adc");
    }
    if((gsx_size_t)tensor->shape[0] != count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field leading dimension does not match gs count");
    }
    expected_count = count * expected_dim1;
    if(expected_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field expected element count must be non-zero");
    }
    byte_count = expected_count * sizeof(float);
    if(byte_count != tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field byte size does not match expected shape");
    }
    *out_values = gsx_cuda_adc_tensor_device_f32(tensor);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cuda_adc_free_refine_data(gsx_cuda_adc_refine_data *data)
{
    if(data == NULL) {
        return;
    }
    memset(data, 0, sizeof(*data));
}

static gsx_error gsx_cuda_adc_load_refine_data(gsx_gs_t gs, gsx_size_t count, gsx_cuda_adc_refine_data *out_data)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_tensor_t max_screen_radius = NULL;

    if(out_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_data must be non-null");
    }
    memset(out_data, 0, sizeof(*out_data));
    out_data->count = count;

    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_GRAD_ACC, count, 1, &out_data->grad_acc);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_GRAD_ACC auxiliary field"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_LOGSCALE, count, 3, &out_data->logscale);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_LOGSCALE access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_OPACITY, count, 1, &out_data->opacity);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_OPACITY access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_ROTATION, count, 4, &out_data->rotation);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_ROTATION access"
        );
    }
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, &max_screen_radius);
    if(gsx_error_is_success(error)) {
        error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, count, 1, &out_data->max_screen_radius);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_adc_free_refine_data(out_data);
            return error;
        }
        out_data->has_max_screen_radius = true;
    } else {
        out_data->has_max_screen_radius = false;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_cuda_adc_probability_to_logit(float probability)
{
    float clamped = probability;

    if(clamped <= 1e-6f) {
        clamped = 1e-6f;
    }
    if(clamped >= 1.0f - 1e-6f) {
        clamped = 1.0f - 1e-6f;
    }
    return logf(clamped / (1.0f - clamped));
}

static gsx_error gsx_cuda_adc_apply_reset(const gsx_adc_desc *desc, const gsx_adc_request *request)
{
    gsx_tensor_t opacity = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    float min_opacity = -20.0f;
    float max_opacity = 0.0f;

    if(desc == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc and request must be non-null");
    }
    error = gsx_gs_get_field(request->gs, GSX_GS_FIELD_OPACITY, &opacity);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda default adc reset requires gs opacity access");
    }
    if(opacity->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda default adc reset supports only float32 opacity");
    }
    max_opacity = gsx_cuda_adc_probability_to_logit(desc->opacity_clamp_value);
    error = gsx_tensor_clamp_inplace(opacity, &min_opacity, &max_opacity);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_cuda_adc_optim_enabled(request)) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_optim_reset(request->optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_adc_init_index_tensor(
    gsx_tensor_t reference_tensor,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor
)
{
    gsx_backend_buffer_desc buffer_desc = {};
    gsx_size_t byte_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(reference_tensor == NULL || out_buffer == NULL || out_index_tensor == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index tensor inputs must be non-null and count must be positive");
    }
    if(index_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "index count exceeds supported int32 range");
    }
    byte_count = index_count * sizeof(int32_t);
    memset(out_index_tensor, 0, sizeof(*out_index_tensor));

    buffer_desc.buffer_type = reference_tensor->backing_buffer->buffer_type;
    buffer_desc.size_bytes = byte_count;
    error = gsx_backend_buffer_init(out_buffer, &buffer_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    out_index_tensor->arena = reference_tensor->arena;
    out_index_tensor->backing_buffer = *out_buffer;
    out_index_tensor->offset_bytes = 0;
    out_index_tensor->size_bytes = byte_count;
    out_index_tensor->alloc_span_bytes = byte_count;
    out_index_tensor->requested_alignment_bytes = sizeof(int32_t);
    out_index_tensor->effective_alignment_bytes = sizeof(int32_t);
    out_index_tensor->alloc_start_bytes = 0;
    out_index_tensor->alloc_end_bytes = byte_count;
    out_index_tensor->rank = 1;
    out_index_tensor->shape[0] = (gsx_index_t)index_count;
    out_index_tensor->data_type = GSX_DATA_TYPE_I32;
    out_index_tensor->storage_format = GSX_STORAGE_FORMAT_CHW;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_adc_apply_gs_and_optim_gather_tensor(const gsx_adc_request *request, gsx_tensor_t index_tensor)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(request == NULL || index_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "request and index_tensor must be non-null");
    }
    if(gsx_cuda_adc_optim_enabled(request)) {
        error = gsx_optim_gather(request->optim, index_tensor);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    return gsx_gs_gather(request->gs, index_tensor);
}

struct gsx_cuda_adc_duplicate_predicate {
    const float *grad_acc;
    const float *logscale;
    float duplicate_grad_threshold;
    float duplicate_scale_threshold;

    __device__ bool operator()(int32_t idx) const
    {
        float grad = grad_acc[idx];
        float sx = 0.0f;
        float sy = 0.0f;
        float sz = 0.0f;
        float max_scale = 0.0f;

        if(grad <= duplicate_grad_threshold) {
            return false;
        }
        sx = expf(logscale[(size_t)idx * 3 + 0]);
        sy = expf(logscale[(size_t)idx * 3 + 1]);
        sz = expf(logscale[(size_t)idx * 3 + 2]);
        max_scale = sx;
        if(sy > max_scale) {
            max_scale = sy;
        }
        if(sz > max_scale) {
            max_scale = sz;
        }
        if(duplicate_scale_threshold > 0.0f && max_scale > duplicate_scale_threshold) {
            return false;
        }
        return true;
    }
};

struct gsx_cuda_adc_keep_predicate {
    const float *opacity;
    const float *logscale;
    const float *rotation;
    const float *max_screen_radius;
    bool has_max_screen_radius;
    float pruning_opacity_threshold;
    float max_world_scale;
    float max_screen_scale;
    bool prune_degenerate_rotation;

    __device__ bool operator()(int32_t idx) const
    {
        float opacity_value = 0.0f;
        float sx = 0.0f;
        float sy = 0.0f;
        float sz = 0.0f;
        float max_scale = 0.0f;
        float q0 = 0.0f;
        float q1 = 0.0f;
        float q2 = 0.0f;
        float q3 = 0.0f;
        float rotation_norm = 0.0f;

        opacity_value = 1.0f / (1.0f + expf(-opacity[idx]));
        if(opacity_value <= pruning_opacity_threshold) {
            return false;
        }
        sx = expf(logscale[(size_t)idx * 3 + 0]);
        sy = expf(logscale[(size_t)idx * 3 + 1]);
        sz = expf(logscale[(size_t)idx * 3 + 2]);
        max_scale = sx;
        if(sy > max_scale) {
            max_scale = sy;
        }
        if(sz > max_scale) {
            max_scale = sz;
        }
        if(max_world_scale > 0.0f && max_scale > max_world_scale) {
            return false;
        }
        if(max_screen_scale > 0.0f && has_max_screen_radius && max_screen_radius != NULL && max_screen_radius[idx] > max_screen_scale) {
            return false;
        }
        if(prune_degenerate_rotation) {
            q0 = rotation[(size_t)idx * 4 + 0];
            q1 = rotation[(size_t)idx * 4 + 1];
            q2 = rotation[(size_t)idx * 4 + 2];
            q3 = rotation[(size_t)idx * 4 + 3];
            rotation_norm = fabsf(q0) + fabsf(q1) + fabsf(q2) + fabsf(q3);
            if(rotation_norm <= FLT_EPSILON) {
                return false;
            }
        }
        return true;
    }
};

static gsx_error gsx_cuda_adc_apply_refine(
    const gsx_adc_desc *desc,
    gsx_backend_t backend,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
)
{
    gsx_cuda_adc_refine_data refine_data = {};
    gsx_tensor_t mean3d = NULL;
    gsx_size_t count_before_refine = 0;
    gsx_size_t max_capacity = 0;
    gsx_size_t duplicate_budget = 0;
    gsx_size_t duplicate_count = 0;
    gsx_size_t keep_count = 0;
    gsx_backend_buffer_t index_buffer = NULL;
    struct gsx_tensor index_tensor = {};
    void *stream = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc, request, and out_result must be non-null");
    }
    error = gsx_gs_get_field(request->gs, GSX_GS_FIELD_MEAN3D, &mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_backend_get_major_stream(backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_cuda_adc_load_count(request->gs, &count_before_refine);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(count_before_refine == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_cuda_adc_load_refine_data(request->gs, count_before_refine, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    max_capacity = gsx_cuda_adc_non_negative_index(desc->max_num_gaussians);
    if(max_capacity > count_before_refine) {
        duplicate_budget = max_capacity - count_before_refine;
    } else {
        duplicate_budget = 0;
    }
    if(duplicate_budget > 0) {
        thrust::device_vector<int32_t> duplicate_sources((size_t)duplicate_budget);
        thrust::device_ptr<int32_t> duplicate_begin = duplicate_sources.data();
        thrust::device_ptr<int32_t> duplicate_end = duplicate_begin;
        auto exec = thrust::cuda::par.on((cudaStream_t)stream);

        duplicate_end = thrust::copy_if(
            exec,
            thrust::make_counting_iterator<int32_t>(0),
            thrust::make_counting_iterator<int32_t>((int32_t)count_before_refine),
            duplicate_begin,
            gsx_cuda_adc_duplicate_predicate{
                refine_data.grad_acc,
                refine_data.logscale,
                desc->duplicate_grad_threshold,
                desc->duplicate_scale_threshold
            }
        );
        duplicate_count = (gsx_size_t)(duplicate_end - duplicate_begin);
        if(duplicate_count > duplicate_budget) {
            gsx_cuda_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "cuda adc duplicate count overflow");
        }

        if(duplicate_count > 0) {
            gsx_size_t gathered_count = count_before_refine + duplicate_count;

            error = gsx_cuda_adc_init_index_tensor(mean3d, gathered_count, &index_buffer, &index_tensor);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_adc_free_refine_data(&refine_data);
                return error;
            }
            thrust::device_ptr<int32_t> gather_begin = thrust::device_pointer_cast(gsx_cuda_adc_tensor_device_i32(&index_tensor));
            thrust::sequence(exec, gather_begin, gather_begin + (ptrdiff_t)count_before_refine, (int32_t)0);
            thrust::copy_n(duplicate_begin, (ptrdiff_t)duplicate_count, gather_begin + (ptrdiff_t)count_before_refine);

            error = gsx_cuda_adc_apply_gs_and_optim_gather_tensor(request, &index_tensor);
            (void)gsx_backend_buffer_free(index_buffer);
            index_buffer = NULL;
            if(!gsx_error_is_success(error)) {
                gsx_cuda_adc_free_refine_data(&refine_data);
                return error;
            }
            out_result->duplicated_count += duplicate_count;
            out_result->mutated = true;
        }
    }
    gsx_cuda_adc_free_refine_data(&refine_data);

    error = gsx_cuda_adc_load_count(request->gs, &count_before_refine);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(count_before_refine == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_cuda_adc_load_refine_data(request->gs, count_before_refine, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    {
        thrust::device_vector<int32_t> keep_indices((size_t)count_before_refine);
        thrust::device_ptr<int32_t> keep_begin = keep_indices.data();
        thrust::device_ptr<int32_t> keep_end = keep_begin;
        auto exec = thrust::cuda::par.on((cudaStream_t)stream);

        keep_end = thrust::copy_if(
            exec,
            thrust::make_counting_iterator<int32_t>(0),
            thrust::make_counting_iterator<int32_t>((int32_t)count_before_refine),
            keep_begin,
            gsx_cuda_adc_keep_predicate{
                refine_data.opacity,
                refine_data.logscale,
                refine_data.rotation,
                refine_data.max_screen_radius,
                refine_data.has_max_screen_radius,
                desc->pruning_opacity_threshold,
                desc->max_world_scale,
                desc->max_screen_scale,
                desc->prune_degenerate_rotation
            }
        );
        keep_count = (gsx_size_t)(keep_end - keep_begin);
        if(keep_count == 0) {
            keep_count = 1;
            error = gsx_cuda_adc_init_index_tensor(mean3d, keep_count, &index_buffer, &index_tensor);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_adc_free_refine_data(&refine_data);
                return error;
            }
            thrust::fill_n(exec, thrust::device_pointer_cast(gsx_cuda_adc_tensor_device_i32(&index_tensor)), 1, (int32_t)0);
        } else {
            error = gsx_cuda_adc_init_index_tensor(mean3d, keep_count, &index_buffer, &index_tensor);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_adc_free_refine_data(&refine_data);
                return error;
            }
            thrust::copy_n(keep_begin, (ptrdiff_t)keep_count, thrust::device_pointer_cast(gsx_cuda_adc_tensor_device_i32(&index_tensor)));
        }
    }
    if(keep_count < count_before_refine) {
        error = gsx_cuda_adc_apply_gs_and_optim_gather_tensor(request, &index_tensor);
        (void)gsx_backend_buffer_free(index_buffer);
        index_buffer = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_cuda_adc_free_refine_data(&refine_data);
            return error;
        }
        out_result->pruned_count += count_before_refine - keep_count;
        out_result->mutated = true;
    } else if(index_buffer != NULL) {
        (void)gsx_backend_buffer_free(index_buffer);
        index_buffer = NULL;
    }
    gsx_cuda_adc_free_refine_data(&refine_data);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc)
{
    gsx_cuda_adc *cuda_adc = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_adc == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_adc and desc must be non-null");
    }

    *out_adc = NULL;
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda adc currently supports only GSX_ADC_ALGORITHM_DEFAULT");
    }

    cuda_adc = (gsx_cuda_adc *)calloc(1, sizeof(*cuda_adc));
    if(cuda_adc == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda adc");
    }

    error = gsx_adc_base_init(&cuda_adc->base, &gsx_cuda_adc_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(cuda_adc);
        return error;
    }

    *out_adc = &cuda_adc->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_adc_destroy(gsx_adc_t adc)
{
    gsx_cuda_adc *cuda_adc = (gsx_cuda_adc *)adc;

    if(adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc must be non-null");
    }

    gsx_adc_base_deinit(&cuda_adc->base);
    free(cuda_adc);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result)
{
    gsx_size_t count_before = 0;
    gsx_size_t count_after = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    bool refine_window = false;
    bool reset_window = false;

    if(adc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }

    gsx_cuda_adc_zero_result(out_result);

    error = gsx_cuda_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda default adc requires gs info/query support");
    }

    out_result->gaussians_before = count_before;
    out_result->gaussians_after = count_before;
    refine_window = gsx_cuda_adc_in_refine_window(&adc->desc, request->global_step);
    reset_window = gsx_cuda_adc_in_reset_window(&adc->desc, request->global_step);

    if(refine_window) {
        error = gsx_cuda_adc_apply_refine(&adc->desc, adc->backend, request, out_result);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(reset_window) {
        error = gsx_cuda_adc_apply_reset(&adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        out_result->reset_count = 1;
        out_result->mutated = true;
    }

    error = gsx_cuda_adc_load_count(request->gs, &count_after);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda default adc requires gs info/query support");
    }

    out_result->gaussians_after = count_after;
    if(out_result->gaussians_after != out_result->gaussians_before) {
        out_result->mutated = true;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
