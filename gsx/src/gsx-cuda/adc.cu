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
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

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
    float *mean3d;
    float *grad_acc;
    float *absgrad_acc;
    float *visible_counter;
    float *logscale;
    float *opacity;
    float *rotation;
    float *sh0;
    float *sh1;
    float *sh2;
    float *sh3;
    float *max_screen_radius;
    bool has_absgrad_acc;
    bool has_visible_counter;
    bool has_max_screen_radius;
} gsx_cuda_adc_refine_data;

typedef enum gsx_cuda_adc_grow_mode {
    GSX_CUDA_ADC_GROW_NONE = 0,
    GSX_CUDA_ADC_GROW_DUPLICATE = 1,
    GSX_CUDA_ADC_GROW_SPLIT = 2
} gsx_cuda_adc_grow_mode;

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
    bool optional,
    float **out_values
)
{
    gsx_tensor_t tensor = NULL;
    gsx_size_t expected_count = 0;
    gsx_size_t actual_count = 1;
    gsx_size_t byte_count = 0;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_values == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_values must be non-null");
    }
    *out_values = NULL;

    error = gsx_gs_get_field(gs, field, &tensor);
    if(!gsx_error_is_success(error)) {
        if(optional) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return error;
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda default adc currently supports only float32 gs fields");
    }
    if(tensor->rank < 1 || tensor->rank > GSX_TENSOR_MAX_DIM) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "unexpected gs field rank for cuda adc");
    }
    if((gsx_size_t)tensor->shape[0] != count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field leading dimension does not match gs count");
    }
    expected_count = count * expected_dim1;
    if(expected_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field expected element count must be non-zero");
    }
    actual_count = 1;
    for(dim = 0; dim < tensor->rank; ++dim) {
        if(tensor->shape[dim] <= 0) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field shape dimensions must be positive");
        }
        if(gsx_size_mul_overflows(actual_count, (gsx_size_t)tensor->shape[dim], &actual_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gs field element count overflow");
        }
    }
    if(actual_count != expected_count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field shape does not match expected element count");
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

    if(out_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_data must be non-null");
    }
    memset(out_data, 0, sizeof(*out_data));
    out_data->count = count;

    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_MEAN3D, count, 3, false, &out_data->mean3d);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_MEAN3D access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_GRAD_ACC, count, 1, true, &out_data->grad_acc);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_ABSGRAD_ACC, count, 1, true, &out_data->absgrad_acc);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_absgrad_acc = out_data->absgrad_acc != NULL;
    if(out_data->grad_acc == NULL && !out_data->has_absgrad_acc) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_GRAD_ACC or GSX_GS_FIELD_ABSGRAD_ACC auxiliary field"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_VISIBLE_COUNTER, count, 1, true, &out_data->visible_counter);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_visible_counter = out_data->visible_counter != NULL;
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_LOGSCALE, count, 3, false, &out_data->logscale);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_LOGSCALE access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_OPACITY, count, 1, false, &out_data->opacity);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_OPACITY access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_ROTATION, count, 4, false, &out_data->rotation);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_ROTATION access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_SH0, count, 3, false, &out_data->sh0);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cuda default adc refine requires GSX_GS_FIELD_SH0 access"
        );
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_SH1, count, 9, true, &out_data->sh1);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_SH2, count, 15, true, &out_data->sh2);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_SH3, count, 21, true, &out_data->sh3);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cuda_adc_load_refine_field(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, count, 1, true, &out_data->max_screen_radius);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_max_screen_radius = out_data->max_screen_radius != NULL;
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
    float min_opacity = -FLT_MAX;
    float max_opacity = 0.0f;
    float clamp_threshold = 0.0f;

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
    clamp_threshold = desc->pruning_opacity_threshold;
    if(clamp_threshold > 1.0f) {
        clamp_threshold = 1.0f;
    }
    if(clamp_threshold < 1e-6f) {
        clamp_threshold = 1e-6f;
    }
    max_opacity = gsx_cuda_adc_probability_to_logit(clamp_threshold);
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
    gsx_gs_t gs,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor
)
{
    gsx_tensor_t mean3d = NULL;
    gsx_backend_buffer_desc buffer_desc = {};
    gsx_size_t byte_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(gs == NULL || out_buffer == NULL || out_index_tensor == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index tensor inputs must be non-null and count must be positive");
    }
    if(index_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "index count exceeds supported int32 range");
    }
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    byte_count = index_count * sizeof(int32_t);
    memset(out_index_tensor, 0, sizeof(*out_index_tensor));

    buffer_desc.buffer_type = mean3d->backing_buffer->buffer_type;
    buffer_desc.size_bytes = byte_count;
    buffer_desc.alignment_bytes = sizeof(int32_t);
    error = gsx_backend_buffer_init(out_buffer, &buffer_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    out_index_tensor->arena = mean3d->arena;
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
    gsx_index_t group_index = 0;

    if(request == NULL || index_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "request and index_tensor must be non-null");
    }

    error = gsx_gs_gather(request->gs, index_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gsx_cuda_adc_optim_enabled(request)) {
        for(group_index = 0; group_index < request->optim->param_group_count; ++group_index) {
            gsx_optim_param_group_desc *group = &request->optim->param_groups[group_index];
            gsx_gs_field param_field = GSX_GS_FIELD_MEAN3D;
            gsx_gs_field grad_field = GSX_GS_FIELD_GRAD_MEAN3D;

            switch(group->role) {
            case GSX_OPTIM_PARAM_ROLE_MEAN3D:
                param_field = GSX_GS_FIELD_MEAN3D;
                grad_field = GSX_GS_FIELD_GRAD_MEAN3D;
                break;
            case GSX_OPTIM_PARAM_ROLE_LOGSCALE:
                param_field = GSX_GS_FIELD_LOGSCALE;
                grad_field = GSX_GS_FIELD_GRAD_LOGSCALE;
                break;
            case GSX_OPTIM_PARAM_ROLE_ROTATION:
                param_field = GSX_GS_FIELD_ROTATION;
                grad_field = GSX_GS_FIELD_GRAD_ROTATION;
                break;
            case GSX_OPTIM_PARAM_ROLE_OPACITY:
                param_field = GSX_GS_FIELD_OPACITY;
                grad_field = GSX_GS_FIELD_GRAD_OPACITY;
                break;
            case GSX_OPTIM_PARAM_ROLE_SH0:
                param_field = GSX_GS_FIELD_SH0;
                grad_field = GSX_GS_FIELD_GRAD_SH0;
                break;
            case GSX_OPTIM_PARAM_ROLE_SH1:
                param_field = GSX_GS_FIELD_SH1;
                grad_field = GSX_GS_FIELD_GRAD_SH1;
                break;
            case GSX_OPTIM_PARAM_ROLE_SH2:
                param_field = GSX_GS_FIELD_SH2;
                grad_field = GSX_GS_FIELD_GRAD_SH2;
                break;
            case GSX_OPTIM_PARAM_ROLE_SH3:
                param_field = GSX_GS_FIELD_SH3;
                grad_field = GSX_GS_FIELD_GRAD_SH3;
                break;
            default:
                return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer role is not supported by cuda adc rebinding");
            }

            error = gsx_gs_get_field(request->gs, param_field, &group->parameter);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_gs_get_field(request->gs, grad_field, &group->gradient);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }

        error = gsx_optim_gather(request->optim, index_tensor);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

struct gsx_cuda_adc_duplicate_predicate {
    const float *grad_acc;
    const float *absgrad_acc;
    const float *visible_counter;
    bool has_absgrad_acc;
    bool has_visible_counter;
    float duplicate_absgrad_threshold;
    const float *logscale;
    float duplicate_grad_threshold;
    float duplicate_scale_threshold;
    float scene_scale;

    __device__ uint8_t operator()(int32_t idx) const
    {
        float counter = 1.0f;
        float accum = 0.0f;
        float grad = 0.0f;
        bool use_absgrad = false;
        float threshold = 0.0f;
        float sx = 0.0f;
        float sy = 0.0f;
        float sz = 0.0f;
        float max_scale = 0.0f;
        float split_scale = 0.0f;

        if(has_visible_counter && visible_counter != NULL) {
            counter = visible_counter[idx];
        }
        if(counter <= 0.0f) {
            return (uint8_t)GSX_CUDA_ADC_GROW_NONE;
        }
        use_absgrad = has_absgrad_acc && absgrad_acc != NULL && duplicate_absgrad_threshold > 0.0f;
        if(use_absgrad) {
            accum = absgrad_acc[idx];
            threshold = duplicate_absgrad_threshold;
        } else {
            if(grad_acc == NULL) {
                return (uint8_t)GSX_CUDA_ADC_GROW_NONE;
            }
            accum = grad_acc[idx];
            threshold = duplicate_grad_threshold;
        }
        grad = accum / (counter > 1.0f ? counter : 1.0f);
        if(grad <= threshold) {
            return (uint8_t)GSX_CUDA_ADC_GROW_NONE;
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
        split_scale = duplicate_scale_threshold * scene_scale;
        if(max_scale > split_scale) {
            return (uint8_t)GSX_CUDA_ADC_GROW_SPLIT;
        }
        return (uint8_t)GSX_CUDA_ADC_GROW_DUPLICATE;
    }
};

struct gsx_cuda_adc_nonzero_mode_predicate {
    __device__ bool operator()(uint8_t mode) const
    {
        return mode != (uint8_t)GSX_CUDA_ADC_GROW_NONE;
    }
};

struct gsx_cuda_adc_split_mode_count {
    __device__ int operator()(uint8_t mode) const
    {
        return mode == (uint8_t)GSX_CUDA_ADC_GROW_SPLIT ? 1 : 0;
    }
};

struct gsx_cuda_adc_mode_lookup {
    const uint8_t *mode_table;

    __device__ uint8_t operator()(int32_t idx) const
    {
        return mode_table[idx];
    }
};

struct gsx_cuda_adc_keep_predicate {
    const float *opacity;
    const float *logscale;
    const float *rotation;
    const float *max_screen_radius;
    bool has_max_screen_radius;
    float scene_scale;
    gsx_size_t count_before_growth;
    bool prune_large;
    float pruning_opacity_threshold;
    float max_world_scale;
    float max_screen_scale;

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
        bool not_large_ws = true;
        bool not_large_ss = true;
        bool not_transparent = false;
        bool not_degenerate = false;

        opacity_value = 1.0f / (1.0f + expf(-opacity[idx]));
        not_transparent = opacity_value > pruning_opacity_threshold;
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
        if(max_world_scale > 0.0f) {
            not_large_ws = max_scale < (max_world_scale * scene_scale);
        }
        if(max_screen_scale > 0.0f && has_max_screen_radius && max_screen_radius != NULL && (gsx_size_t)idx < count_before_growth) {
            not_large_ss = max_screen_radius[idx] < max_screen_scale;
        }
        q0 = rotation[(size_t)idx * 4 + 0];
        q1 = rotation[(size_t)idx * 4 + 1];
        q2 = rotation[(size_t)idx * 4 + 2];
        q3 = rotation[(size_t)idx * 4 + 3];
        rotation_norm = fabsf(q0) + fabsf(q1) + fabsf(q2) + fabsf(q3);
        not_degenerate = rotation_norm > FLT_EPSILON;
        if(!prune_large) {
            return not_transparent && not_degenerate;
        }
        return not_transparent && not_large_ws && not_large_ss && not_degenerate;
    }
};

struct gsx_cuda_adc_apply_growth_mutation {
    float *mean3d;
    float *logscale;
    float *opacity;
    float *rotation;
    const int32_t *grow_sources;
    const uint8_t *grow_modes;
    gsx_size_t count_before_refine;
    gsx_size_t seed;
    gsx_size_t global_step;

    __device__ static uint32_t hash32(uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    __device__ float sample_logistic(gsx_size_t grow_index, uint32_t lane) const
    {
        uint32_t key = (uint32_t)(seed ^ (global_step * (gsx_size_t)0x9e3779b97f4a7c15ULL));
        uint32_t mixed = hash32(key ^ (uint32_t)grow_index ^ (lane * 0x9e3779b9U + 0x85ebca6bU));
        float u = ((float)mixed + 1.0f) / 4294967297.0f;
        if(u <= 1e-6f) {
            u = 1e-6f;
        }
        if(u >= 1.0f - 1e-6f) {
            u = 1.0f - 1e-6f;
        }
        return logf(u / (1.0f - u));
    }

    __device__ void operator()(int32_t i) const
    {
        gsx_size_t grow_index = (gsx_size_t)i;
        gsx_size_t target = count_before_refine + grow_index;
        gsx_size_t src = (gsx_size_t)grow_sources[grow_index];
        uint8_t mode = grow_modes[grow_index];

        if(mode != (uint8_t)GSX_CUDA_ADC_GROW_SPLIT || src >= count_before_refine) {
            return;
        }

        float qx = 0.0f;
        float qy = 0.0f;
        float qz = 0.0f;
        float qw = 0.0f;
        float q_norm = 0.0f;
        float inv_q = 0.0f;
        float sx = expf(logscale[src * 3 + 0]);
        float sy = expf(logscale[src * 3 + 1]);
        float sz = expf(logscale[src * 3 + 2]);
        float source_opacity = 1.0f / (1.0f + expf(-opacity[src]));
        float split_opacity = 0.0f;
        float rnd1x = sample_logistic(grow_index, 0U);
        float rnd1y = sample_logistic(grow_index, 1U);
        float rnd1z = sample_logistic(grow_index, 2U);
        float rnd2x = sample_logistic(grow_index, 3U);
        float rnd2y = sample_logistic(grow_index, 4U);
        float rnd2z = sample_logistic(grow_index, 5U);
        float m00 = 1.0f;
        float m01 = 0.0f;
        float m02 = 0.0f;
        float m10 = 0.0f;
        float m11 = 1.0f;
        float m12 = 0.0f;
        float m20 = 0.0f;
        float m21 = 0.0f;
        float m22 = 1.0f;
        float t1x = rnd1x * (sx + 1e-5f);
        float t1y = rnd1y * (sy + 1e-5f);
        float t1z = rnd1z * (sz + 1e-5f);
        float t2x = rnd2x * (sx + 1e-5f);
        float t2y = rnd2y * (sy + 1e-5f);
        float t2z = rnd2z * (sz + 1e-5f);
        float off1x = 0.0f;
        float off1y = 0.0f;
        float off1z = 0.0f;
        float off2x = 0.0f;
        float off2y = 0.0f;
        float off2z = 0.0f;
        float new_scale_x = sx / 1.6f;
        float new_scale_y = sy / 1.6f;
        float new_scale_z = sz / 1.6f;

        qx = rotation[src * 4 + 0];
        qy = rotation[src * 4 + 1];
        qz = rotation[src * 4 + 2];
        qw = rotation[src * 4 + 3];
        q_norm = sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
        if(q_norm > 1e-8f) {
            inv_q = 1.0f / q_norm;
            qx *= inv_q;
            qy *= inv_q;
            qz *= inv_q;
            qw *= inv_q;
        }
        m00 = 1.0f - 2.0f * (qy * qy + qz * qz);
        m01 = 2.0f * (qx * qy - qw * qz);
        m02 = 2.0f * (qx * qz + qw * qy);
        m10 = 2.0f * (qx * qy + qw * qz);
        m11 = 1.0f - 2.0f * (qx * qx + qz * qz);
        m12 = 2.0f * (qy * qz - qw * qx);
        m20 = 2.0f * (qx * qz - qw * qy);
        m21 = 2.0f * (qy * qz + qw * qx);
        m22 = 1.0f - 2.0f * (qx * qx + qy * qy);
        off1x = m00 * t1x + m01 * t1y + m02 * t1z;
        off1y = m10 * t1x + m11 * t1y + m12 * t1z;
        off1z = m20 * t1x + m21 * t1y + m22 * t1z;
        off2x = m00 * t2x + m01 * t2y + m02 * t2z;
        off2y = m10 * t2x + m11 * t2y + m12 * t2z;
        off2z = m20 * t2x + m21 * t2y + m22 * t2z;
        if(source_opacity < 0.0f) {
            source_opacity = 0.0f;
        }
        if(source_opacity > 1.0f - 1e-6f) {
            source_opacity = 1.0f - 1e-6f;
        }
        split_opacity = 1.0f - sqrtf(1.0f - source_opacity);
        if(split_opacity <= 1e-6f) {
            split_opacity = 1e-6f;
        }
        if(split_opacity >= 1.0f - 1e-6f) {
            split_opacity = 1.0f - 1e-6f;
        }
        mean3d[target * 3 + 0] = mean3d[src * 3 + 0] + off1x;
        mean3d[target * 3 + 1] = mean3d[src * 3 + 1] + off1y;
        mean3d[target * 3 + 2] = mean3d[src * 3 + 2] + off1z;
        logscale[target * 3 + 0] = logf(new_scale_x);
        logscale[target * 3 + 1] = logf(new_scale_y);
        logscale[target * 3 + 2] = logf(new_scale_z);
        opacity[target] = logf(split_opacity / (1.0f - split_opacity));
        mean3d[src * 3 + 0] = mean3d[src * 3 + 0] + off2x;
        mean3d[src * 3 + 1] = mean3d[src * 3 + 1] + off2y;
        mean3d[src * 3 + 2] = mean3d[src * 3 + 2] + off2z;
        logscale[src * 3 + 0] = logf(new_scale_x);
        logscale[src * 3 + 1] = logf(new_scale_y);
        logscale[src * 3 + 2] = logf(new_scale_z);
        opacity[src] = logf(split_opacity / (1.0f - split_opacity));
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
    gsx_size_t count_before_refine = 0;
    gsx_size_t max_capacity = 0;
    gsx_size_t grow_budget = 0;
    gsx_size_t grow_count = 0;
    gsx_size_t split_count = 0;
    gsx_size_t duplicate_count = 0;
    gsx_size_t count_after_growth = 0;
    gsx_size_t keep_count = 0;
    bool prune_large = false;
    gsx_backend_buffer_t index_buffer = NULL;
    struct gsx_tensor index_tensor = {};
    void *stream = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc, request, and out_result must be non-null");
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
        grow_budget = max_capacity - count_before_refine;
    } else {
        grow_budget = 0;
    }

    {
        auto exec = thrust::cuda::par.on((cudaStream_t)stream);
        if(grow_budget > 0) {
            thrust::device_vector<uint8_t> all_modes((size_t)count_before_refine, 0);
            thrust::device_vector<int32_t> grow_candidates((size_t)count_before_refine, 0);
            thrust::device_vector<uint8_t> selected_modes;
            thrust::device_ptr<int32_t> candidate_begin = grow_candidates.data();
            thrust::device_ptr<int32_t> candidate_end = candidate_begin;

            thrust::transform(
                exec,
                thrust::make_counting_iterator<int32_t>(0),
                thrust::make_counting_iterator<int32_t>((int32_t)count_before_refine),
                all_modes.begin(),
                gsx_cuda_adc_duplicate_predicate{
                    refine_data.grad_acc,
                    refine_data.absgrad_acc,
                    refine_data.visible_counter,
                    refine_data.has_absgrad_acc,
                    refine_data.has_visible_counter,
                    desc->duplicate_absgrad_threshold,
                    refine_data.logscale,
                    desc->duplicate_grad_threshold,
                    desc->duplicate_scale_threshold,
                    request->scene_scale
                }
            );

            candidate_end = thrust::copy_if(
                exec,
                thrust::make_counting_iterator<int32_t>(0),
                thrust::make_counting_iterator<int32_t>((int32_t)count_before_refine),
                all_modes.begin(),
                candidate_begin,
                gsx_cuda_adc_nonzero_mode_predicate{}
            );
            grow_count = (gsx_size_t)(candidate_end - candidate_begin);
            if(grow_count > grow_budget) {
                grow_count = grow_budget;
            }
            if(grow_count > 0) {
                gsx_size_t gathered_count = count_before_refine + grow_count;

                selected_modes.resize((size_t)grow_count);
                thrust::transform(
                    exec,
                    candidate_begin,
                    candidate_begin + (ptrdiff_t)grow_count,
                    selected_modes.begin(),
                    gsx_cuda_adc_mode_lookup{ thrust::raw_pointer_cast(all_modes.data()) }
                );
                split_count = (gsx_size_t)thrust::transform_reduce(
                    exec,
                    selected_modes.begin(),
                    selected_modes.end(),
                    gsx_cuda_adc_split_mode_count{},
                    0,
                    thrust::plus<int>()
                );
                duplicate_count = grow_count - split_count;

                error = gsx_cuda_adc_init_index_tensor(request->gs, gathered_count, &index_buffer, &index_tensor);
                if(!gsx_error_is_success(error)) {
                    gsx_cuda_adc_free_refine_data(&refine_data);
                    return error;
                }
                thrust::device_ptr<int32_t> gather_begin = thrust::device_pointer_cast(gsx_cuda_adc_tensor_device_i32(&index_tensor));
                thrust::sequence(exec, gather_begin, gather_begin + (ptrdiff_t)count_before_refine, (int32_t)0);
                thrust::copy_n(candidate_begin, (ptrdiff_t)grow_count, gather_begin + (ptrdiff_t)count_before_refine);

                error = gsx_cuda_adc_apply_gs_and_optim_gather_tensor(request, &index_tensor);
                (void)gsx_backend_buffer_free(index_buffer);
                index_buffer = NULL;
                if(!gsx_error_is_success(error)) {
                    gsx_cuda_adc_free_refine_data(&refine_data);
                    return error;
                }

                error = gsx_cuda_adc_load_count(request->gs, &count_after_growth);
                if(!gsx_error_is_success(error)) {
                    gsx_cuda_adc_free_refine_data(&refine_data);
                    return error;
                }
                if(count_after_growth != count_before_refine + grow_count) {
                    gsx_cuda_adc_free_refine_data(&refine_data);
                    return gsx_make_error(GSX_ERROR_INVALID_STATE, "cuda default adc growth produced unexpected gaussian count");
                }
                error = gsx_cuda_adc_load_refine_data(request->gs, count_after_growth, &refine_data);
                if(!gsx_error_is_success(error)) {
                    return error;
                }
                thrust::for_each(
                    exec,
                    thrust::make_counting_iterator<int32_t>(0),
                    thrust::make_counting_iterator<int32_t>((int32_t)grow_count),
                    gsx_cuda_adc_apply_growth_mutation{
                        refine_data.mean3d,
                        refine_data.logscale,
                        refine_data.opacity,
                        refine_data.rotation,
                        thrust::raw_pointer_cast(grow_candidates.data()),
                        thrust::raw_pointer_cast(selected_modes.data()),
                        count_before_refine,
                        desc->seed,
                        request->global_step
                    }
                );
                out_result->duplicated_count += duplicate_count;
                out_result->grown_count += split_count;
                out_result->mutated = true;
            }
        }
    }

    error = gsx_cuda_adc_load_count(request->gs, &count_after_growth);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_adc_free_refine_data(&refine_data);
        return error;
    }
    if(count_after_growth == 0) {
        gsx_cuda_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_cuda_adc_load_refine_data(request->gs, count_after_growth, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    prune_large = request->global_step > gsx_cuda_adc_non_negative_index(desc->reset_every);

    {
        auto exec = thrust::cuda::par.on((cudaStream_t)stream);
        thrust::device_vector<int32_t> keep_indices((size_t)count_after_growth);
        thrust::device_ptr<int32_t> keep_begin = keep_indices.data();
        thrust::device_ptr<int32_t> keep_end = keep_begin;

        keep_end = thrust::copy_if(
            exec,
            thrust::make_counting_iterator<int32_t>(0),
            thrust::make_counting_iterator<int32_t>((int32_t)count_after_growth),
            keep_begin,
            gsx_cuda_adc_keep_predicate{
                refine_data.opacity,
                refine_data.logscale,
                refine_data.rotation,
                refine_data.max_screen_radius,
                refine_data.has_max_screen_radius,
                request->scene_scale,
                count_before_refine,
                prune_large,
                desc->pruning_opacity_threshold,
                desc->max_world_scale,
                desc->max_screen_scale
            }
        );
        keep_count = (gsx_size_t)(keep_end - keep_begin);
        if(keep_count == 0) {
            keep_count = 1;
            error = gsx_cuda_adc_init_index_tensor(request->gs, keep_count, &index_buffer, &index_tensor);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_adc_free_refine_data(&refine_data);
                return error;
            }
            thrust::fill_n(exec, thrust::device_pointer_cast(gsx_cuda_adc_tensor_device_i32(&index_tensor)), 1, (int32_t)0);
        } else {
            error = gsx_cuda_adc_init_index_tensor(request->gs, keep_count, &index_buffer, &index_tensor);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_adc_free_refine_data(&refine_data);
                return error;
            }
            thrust::copy_n(keep_begin, (ptrdiff_t)keep_count, thrust::device_pointer_cast(gsx_cuda_adc_tensor_device_i32(&index_tensor)));
        }
    }
    if(keep_count < count_after_growth) {
        error = gsx_cuda_adc_apply_gs_and_optim_gather_tensor(request, &index_tensor);
        (void)gsx_backend_buffer_free(index_buffer);
        index_buffer = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_cuda_adc_free_refine_data(&refine_data);
            return error;
        }
        out_result->pruned_count += count_after_growth - keep_count;
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
