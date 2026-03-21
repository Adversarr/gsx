#include "internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_metal_adc {
    struct gsx_adc base;
} gsx_metal_adc;

static gsx_error gsx_metal_adc_destroy(gsx_adc_t adc);
static gsx_error gsx_metal_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);

static const gsx_adc_i gsx_metal_adc_iface = {
    gsx_metal_adc_destroy,
    gsx_metal_adc_step
};

typedef struct gsx_metal_adc_refine_data {
    gsx_size_t count;
    gsx_tensor_t mean3d_tensor;
    gsx_tensor_t grad_acc_tensor;
    gsx_tensor_t visible_counter_tensor;
    gsx_tensor_t logscale_tensor;
    gsx_tensor_t opacity_tensor;
    gsx_tensor_t rotation_tensor;
    gsx_tensor_t sh0_tensor;
    gsx_tensor_t sh1_tensor;
    gsx_tensor_t sh2_tensor;
    gsx_tensor_t sh3_tensor;
    gsx_tensor_t max_screen_radius_tensor;
    float *mean3d;
    float *grad_acc;
    float *visible_counter;
    float *logscale;
    float *opacity;
    float *rotation;
    float *sh0;
    float *sh1;
    float *sh2;
    float *sh3;
    float *max_screen_radius;
    bool has_visible_counter;
    bool has_max_screen_radius;
} gsx_metal_adc_refine_data;

typedef enum gsx_metal_adc_grow_mode {
    GSX_METAL_ADC_GROW_NONE = 0,
    GSX_METAL_ADC_GROW_DUPLICATE = 1,
    GSX_METAL_ADC_GROW_SPLIT = 2
} gsx_metal_adc_grow_mode;

static gsx_size_t gsx_metal_adc_non_negative_index(gsx_index_t value)
{
    if(value <= 0) {
        return 0;
    }
    return (gsx_size_t)value;
}

static void gsx_metal_adc_zero_result(gsx_adc_result *out_result)
{
    out_result->gaussians_before = 0;
    out_result->gaussians_after = 0;
    out_result->pruned_count = 0;
    out_result->duplicated_count = 0;
    out_result->grown_count = 0;
    out_result->reset_count = 0;
    out_result->mutated = false;
}

static bool gsx_metal_adc_optim_enabled(const gsx_adc_request *request)
{
    return request != NULL && request->optim != NULL && request->optim->iface != NULL;
}

static bool gsx_metal_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_metal_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_metal_adc_non_negative_index(desc->end_refine);

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

static bool gsx_metal_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_metal_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_metal_adc_non_negative_index(desc->end_refine);

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

static gsx_error gsx_metal_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count)
{
    gsx_gs_info info = { 0 };
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

static gsx_error gsx_metal_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    bool optional,
    gsx_tensor_t *out_tensor,
    float **out_values)
{
    gsx_tensor_t tensor = NULL;
    gsx_size_t expected_count = 0;
    gsx_size_t actual_count = 1;
    gsx_size_t byte_count = 0;
    gsx_index_t dim = 0;
    float *values = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_values == NULL || out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor and out_values must be non-null");
    }
    *out_tensor = NULL;
    *out_values = NULL;

    error = gsx_gs_get_field(gs, field, &tensor);
    if(!gsx_error_is_success(error)) {
        if(optional) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return error;
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal default adc currently supports only float32 gs fields");
    }
    if(tensor->rank < 1 || tensor->rank > GSX_TENSOR_MAX_DIM) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "unexpected gs field rank for metal adc");
    }
    if((gsx_size_t)tensor->shape[0] != count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field leading dimension does not match gs count");
    }
    if(gsx_size_mul_overflows(count, expected_dim1, &expected_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gs field expected element count overflow");
    }
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
    if(gsx_size_mul_overflows(expected_count, sizeof(float), &byte_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gs field byte size overflow");
    }
    if(byte_count != tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field byte size does not match expected shape");
    }
    values = (float *)malloc((size_t)byte_count);
    if(values == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate host staging for metal adc field");
    }
    error = gsx_tensor_download(tensor, values, byte_count);
    if(!gsx_error_is_success(error)) {
        free(values);
        return error;
    }

    *out_tensor = tensor;
    *out_values = values;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_adc_free_refine_data(gsx_metal_adc_refine_data *data)
{
    if(data == NULL) {
        return;
    }

    free(data->mean3d);
    free(data->grad_acc);
    free(data->visible_counter);
    free(data->logscale);
    free(data->opacity);
    free(data->rotation);
    free(data->sh0);
    free(data->sh1);
    free(data->sh2);
    free(data->sh3);
    free(data->max_screen_radius);
    memset(data, 0, sizeof(*data));
}

static gsx_error gsx_metal_adc_load_refine_data(gsx_gs_t gs, gsx_size_t count, gsx_metal_adc_refine_data *out_data)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_data must be non-null");
    }
    gsx_metal_adc_free_refine_data(out_data);
    out_data->count = count;

    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_MEAN3D, count, 3, false, &out_data->mean3d_tensor, &out_data->mean3d);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_MEAN3D access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_GRAD_ACC, count, 1, true, &out_data->grad_acc_tensor, &out_data->grad_acc);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    if(out_data->grad_acc == NULL) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_GRAD_ACC auxiliary field"
        );
    }
    error = gsx_metal_adc_load_refine_field(
        gs,
        GSX_GS_FIELD_VISIBLE_COUNTER,
        count,
        1,
        true,
        &out_data->visible_counter_tensor,
        &out_data->visible_counter);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_visible_counter = out_data->visible_counter != NULL;
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_LOGSCALE, count, 3, false, &out_data->logscale_tensor, &out_data->logscale);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_LOGSCALE access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_OPACITY, count, 1, false, &out_data->opacity_tensor, &out_data->opacity);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_OPACITY access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_ROTATION, count, 4, false, &out_data->rotation_tensor, &out_data->rotation);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_ROTATION access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH0, count, 3, false, &out_data->sh0_tensor, &out_data->sh0);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_SH0 access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH1, count, 9, true, &out_data->sh1_tensor, &out_data->sh1);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH2, count, 15, true, &out_data->sh2_tensor, &out_data->sh2);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH3, count, 21, true, &out_data->sh3_tensor, &out_data->sh3);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_metal_adc_load_refine_field(
        gs,
        GSX_GS_FIELD_MAX_SCREEN_RADIUS,
        count,
        1,
        true,
        &out_data->max_screen_radius_tensor,
        &out_data->max_screen_radius);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_max_screen_radius = out_data->max_screen_radius != NULL;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_upload_refine_field(gsx_tensor_t tensor, const float *values, gsx_size_t element_count)
{
    gsx_size_t byte_count = 0;

    if(tensor == NULL || values == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_mul_overflows(element_count, sizeof(float), &byte_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal adc refine upload size overflow");
    }
    return gsx_tensor_upload(tensor, values, byte_count);
}

static gsx_error gsx_metal_adc_upload_growth_mutations(const gsx_metal_adc_refine_data *data)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "data must be non-null");
    }

    error = gsx_metal_adc_upload_refine_field(data->mean3d_tensor, data->mean3d, data->count * 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->logscale_tensor, data->logscale, data->count * 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->opacity_tensor, data->opacity, data->count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->rotation_tensor, data->rotation, data->count * 4);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->sh0_tensor, data->sh0, data->count * 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->sh1_tensor, data->sh1, data->count * 9);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->sh2_tensor, data->sh2, data->count * 15);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_upload_refine_field(data->sh3_tensor, data->sh3, data->count * 21);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_metal_adc_probability_to_logit(float probability)
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

static gsx_error gsx_metal_adc_apply_reset(const gsx_adc_desc *desc, const gsx_adc_request *request)
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
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal default adc reset requires gs opacity access");
    }
    if(opacity->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal default adc reset supports only float32 opacity");
    }
    clamp_threshold = desc->opacity_clamp_value;
    if(clamp_threshold > 1.0f) {
        clamp_threshold = 1.0f;
    }
    if(clamp_threshold < 1e-6f) {
        clamp_threshold = 1e-6f;
    }
    max_opacity = gsx_metal_adc_probability_to_logit(clamp_threshold);
    error = gsx_tensor_clamp_inplace(opacity, &min_opacity, &max_opacity);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_metal_adc_optim_enabled(request)) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_optim_reset(request->optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_build_index_tensor(
    gsx_tensor_t reference_tensor,
    const int32_t *indices,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor)
{
    gsx_backend_buffer_desc buffer_desc = { 0 };
    gsx_size_t byte_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(reference_tensor == NULL || indices == NULL || out_buffer == NULL || out_index_tensor == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index tensor inputs must be non-null and count must be positive");
    }
    if(index_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "index count exceeds supported int32 range");
    }
    if(gsx_size_mul_overflows(index_count, sizeof(int32_t), &byte_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "index tensor byte size overflow");
    }
    memset(out_index_tensor, 0, sizeof(*out_index_tensor));

    buffer_desc.buffer_type = reference_tensor->backing_buffer->buffer_type;
    buffer_desc.size_bytes = byte_count;
    buffer_desc.alignment_bytes = sizeof(int32_t);
    error = gsx_backend_buffer_init(out_buffer, &buffer_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_backend_buffer_upload(*out_buffer, 0, indices, byte_count);
    if(!gsx_error_is_success(error)) {
        gsx_backend_buffer_free(*out_buffer);
        *out_buffer = NULL;
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

static gsx_error gsx_metal_adc_apply_gs_and_optim_gather(const gsx_adc_request *request, const int32_t *indices, gsx_size_t index_count)
{
    gsx_tensor_t mean3d = NULL;
    gsx_backend_buffer_t index_buffer = NULL;
    struct gsx_tensor index_tensor = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(request == NULL || indices == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gather inputs must be non-null and count must be positive");
    }

    error = gsx_gs_get_field(request->gs, GSX_GS_FIELD_MEAN3D, &mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_build_index_tensor(mean3d, indices, index_count, &index_buffer, &index_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_gs_gather(request->gs, &index_tensor);
    if(!gsx_error_is_success(error)) {
        gsx_backend_buffer_free(index_buffer);
        return error;
    }

    if(gsx_metal_adc_optim_enabled(request)) {
        error = gsx_optim_rebind_param_groups_from_gs(request->optim, request->gs);
        if(!gsx_error_is_success(error)) {
            gsx_backend_buffer_free(index_buffer);
            return error;
        }
        error = gsx_optim_gather(request->optim, &index_tensor);
        if(!gsx_error_is_success(error)) {
            gsx_backend_buffer_free(index_buffer);
            return error;
        }
    }

    gsx_backend_buffer_free(index_buffer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_copy_slice(float *dst, gsx_size_t dst_index, const float *src, gsx_size_t src_index, gsx_size_t width)
{
    if(dst == NULL || src == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy slice tensors must be non-null");
    }
    memcpy(dst + dst_index * width, src + src_index * width, width * sizeof(float));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_copy_optional_slice(
    float *dst,
    gsx_size_t dst_index,
    const float *src,
    gsx_size_t src_index,
    gsx_size_t width)
{
    if(dst == NULL || src == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_metal_adc_copy_slice(dst, dst_index, src, src_index, width);
}

static float gsx_metal_adc_clamp_probability(float value)
{
    if(value < 0.0f) {
        return 0.0f;
    }
    if(value > 1.0f - 1e-6f) {
        return 1.0f - 1e-6f;
    }
    return value;
}

static uint32_t gsx_metal_adc_hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static float gsx_metal_adc_sample_logistic(gsx_size_t seed, gsx_size_t global_step, gsx_size_t grow_index, uint32_t lane)
{
    uint32_t key = (uint32_t)(seed ^ (global_step * (gsx_size_t)0x9e3779b97f4a7c15ULL));
    uint32_t mixed = gsx_metal_adc_hash32(key ^ (uint32_t)grow_index ^ (lane * 0x9e3779b9U + 0x85ebca6bU));
    float u = ((float)mixed + 1.0f) / 4294967297.0f;

    if(u <= 1e-6f) {
        u = 1e-6f;
    }
    if(u >= 1.0f - 1e-6f) {
        u = 1.0f - 1e-6f;
    }
    return logf(u / (1.0f - u));
}

static void gsx_metal_adc_normalize_quaternion(float *qx, float *qy, float *qz, float *qw)
{
    float q_norm = 0.0f;
    float inv_q = 0.0f;

    if(qx == NULL || qy == NULL || qz == NULL || qw == NULL) {
        return;
    }
    q_norm = sqrtf((*qx) * (*qx) + (*qy) * (*qy) + (*qz) * (*qz) + (*qw) * (*qw));
    if(q_norm <= 1e-8f) {
        return;
    }
    inv_q = 1.0f / q_norm;
    *qx *= inv_q;
    *qy *= inv_q;
    *qz *= inv_q;
    *qw *= inv_q;
}

static void gsx_metal_adc_build_rotation_matrix(
    float qx,
    float qy,
    float qz,
    float qw,
    float *m00,
    float *m01,
    float *m02,
    float *m10,
    float *m11,
    float *m12,
    float *m20,
    float *m21,
    float *m22)
{
    *m00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    *m01 = 2.0f * (qx * qy - qw * qz);
    *m02 = 2.0f * (qx * qz + qw * qy);
    *m10 = 2.0f * (qx * qy + qw * qz);
    *m11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    *m12 = 2.0f * (qy * qz - qw * qx);
    *m20 = 2.0f * (qx * qz - qw * qy);
    *m21 = 2.0f * (qy * qz + qw * qx);
    *m22 = 1.0f - 2.0f * (qx * qx + qy * qy);
}

static gsx_error gsx_metal_adc_copy_shared_growth_fields(gsx_metal_adc_refine_data *data, gsx_size_t target, gsx_size_t src)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_adc_copy_slice(data->rotation, target, data->rotation, src, 4);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_copy_slice(data->sh0, target, data->sh0, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_copy_optional_slice(data->sh1, target, data->sh1, src, 9);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_copy_optional_slice(data->sh2, target, data->sh2, src, 15);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_copy_optional_slice(data->sh3, target, data->sh3, src, 21);
    return error;
}

static gsx_error gsx_metal_adc_apply_duplicate_mutation(gsx_metal_adc_refine_data *data, gsx_size_t target, gsx_size_t src)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_adc_copy_slice(data->mean3d, target, data->mean3d, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_copy_slice(data->logscale, target, data->logscale, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    data->opacity[target] = data->opacity[src];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_apply_split_mutation(
    gsx_metal_adc_refine_data *data,
    gsx_size_t target,
    gsx_size_t src,
    gsx_size_t seed,
    gsx_size_t global_step,
    gsx_size_t grow_index)
{
    float qx = data->rotation[src * 4 + 0];
    float qy = data->rotation[src * 4 + 1];
    float qz = data->rotation[src * 4 + 2];
    float qw = data->rotation[src * 4 + 3];
    float sx = expf(data->logscale[src * 3 + 0]);
    float sy = expf(data->logscale[src * 3 + 1]);
    float sz = expf(data->logscale[src * 3 + 2]);
    float source_opacity = 1.0f / (1.0f + expf(-data->opacity[src]));
    float split_opacity = 0.0f;
    float rnd1x = gsx_metal_adc_sample_logistic(seed, global_step, grow_index, 0U);
    float rnd1y = gsx_metal_adc_sample_logistic(seed, global_step, grow_index, 1U);
    float rnd1z = gsx_metal_adc_sample_logistic(seed, global_step, grow_index, 2U);
    float rnd2x = gsx_metal_adc_sample_logistic(seed, global_step, grow_index, 3U);
    float rnd2y = gsx_metal_adc_sample_logistic(seed, global_step, grow_index, 4U);
    float rnd2z = gsx_metal_adc_sample_logistic(seed, global_step, grow_index, 5U);
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

    gsx_metal_adc_normalize_quaternion(&qx, &qy, &qz, &qw);
    gsx_metal_adc_build_rotation_matrix(qx, qy, qz, qw, &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22);
    off1x = m00 * t1x + m01 * t1y + m02 * t1z;
    off1y = m10 * t1x + m11 * t1y + m12 * t1z;
    off1z = m20 * t1x + m21 * t1y + m22 * t1z;
    off2x = m00 * t2x + m01 * t2y + m02 * t2z;
    off2y = m10 * t2x + m11 * t2y + m12 * t2z;
    off2z = m20 * t2x + m21 * t2y + m22 * t2z;
    source_opacity = gsx_metal_adc_clamp_probability(source_opacity);
    split_opacity = 1.0f - sqrtf(1.0f - source_opacity);
    if(split_opacity <= 1e-6f) {
        split_opacity = 1e-6f;
    }
    if(split_opacity >= 1.0f - 1e-6f) {
        split_opacity = 1.0f - 1e-6f;
    }

    data->mean3d[target * 3 + 0] = data->mean3d[src * 3 + 0] + off1x;
    data->mean3d[target * 3 + 1] = data->mean3d[src * 3 + 1] + off1y;
    data->mean3d[target * 3 + 2] = data->mean3d[src * 3 + 2] + off1z;
    data->logscale[target * 3 + 0] = logf(new_scale_x);
    data->logscale[target * 3 + 1] = logf(new_scale_y);
    data->logscale[target * 3 + 2] = logf(new_scale_z);
    data->opacity[target] = logf(split_opacity / (1.0f - split_opacity));
    data->mean3d[src * 3 + 0] = data->mean3d[src * 3 + 0] + off2x;
    data->mean3d[src * 3 + 1] = data->mean3d[src * 3 + 1] + off2y;
    data->mean3d[src * 3 + 2] = data->mean3d[src * 3 + 2] + off2z;
    data->logscale[src * 3 + 0] = logf(new_scale_x);
    data->logscale[src * 3 + 1] = logf(new_scale_y);
    data->logscale[src * 3 + 2] = logf(new_scale_z);
    data->opacity[src] = logf(split_opacity / (1.0f - split_opacity));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_metal_adc_grow_mode gsx_metal_adc_grow_mode_for_index(
    const gsx_adc_desc *desc,
    const gsx_metal_adc_refine_data *data,
    float scene_scale,
    gsx_size_t index)
{
    float counter = 1.0f;
    float accum = 0.0f;
    float grad = 0.0f;
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    float max_scale = 0.0f;
    float split_scale = 0.0f;

    if(desc == NULL || data == NULL || data->logscale == NULL || index >= data->count) {
        return GSX_METAL_ADC_GROW_NONE;
    }
    if(data->has_visible_counter && data->visible_counter != NULL) {
        counter = data->visible_counter[index];
    }
    if(counter <= 0.0f) {
        return GSX_METAL_ADC_GROW_NONE;
    }
    if(data->grad_acc == NULL) {
        return GSX_METAL_ADC_GROW_NONE;
    }
    accum = data->grad_acc[index];
    float threshold = desc->duplicate_grad_threshold;
    grad = accum / (counter > 1.0f ? counter : 1.0f);
    if(grad <= threshold) {
        return GSX_METAL_ADC_GROW_NONE;
    }

    sx = expf(data->logscale[index * 3 + 0]);
    sy = expf(data->logscale[index * 3 + 1]);
    sz = expf(data->logscale[index * 3 + 2]);
    max_scale = sx;
    if(sy > max_scale) {
        max_scale = sy;
    }
    if(sz > max_scale) {
        max_scale = sz;
    }
    split_scale = desc->duplicate_scale_threshold * scene_scale;
    if(max_scale > split_scale) {
        return GSX_METAL_ADC_GROW_SPLIT;
    }
    return GSX_METAL_ADC_GROW_DUPLICATE;
}

static bool gsx_metal_adc_should_keep(
    const gsx_adc_desc *desc,
    const gsx_metal_adc_refine_data *data,
    float scene_scale,
    gsx_size_t index,
    gsx_size_t count_before_growth,
    bool prune_large)
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

    if(desc == NULL || data == NULL || data->opacity == NULL || data->logscale == NULL || data->rotation == NULL || index >= data->count) {
        return false;
    }

    opacity_value = 1.0f / (1.0f + expf(-data->opacity[index]));
    not_transparent = opacity_value > desc->pruning_opacity_threshold;
    sx = expf(data->logscale[index * 3 + 0]);
    sy = expf(data->logscale[index * 3 + 1]);
    sz = expf(data->logscale[index * 3 + 2]);
    max_scale = sx;
    if(sy > max_scale) {
        max_scale = sy;
    }
    if(sz > max_scale) {
        max_scale = sz;
    }
    if(desc->max_world_scale > 0.0f) {
        not_large_ws = max_scale < (desc->max_world_scale * scene_scale);
    }
    if(desc->max_screen_scale > 0.0f && data->has_max_screen_radius && data->max_screen_radius != NULL && index < count_before_growth) {
        not_large_ss = data->max_screen_radius[index] < desc->max_screen_scale;
    }
    q0 = data->rotation[index * 4 + 0];
    q1 = data->rotation[index * 4 + 1];
    q2 = data->rotation[index * 4 + 2];
    q3 = data->rotation[index * 4 + 3];
    rotation_norm = fabsf(q0) + fabsf(q1) + fabsf(q2) + fabsf(q3);
    not_degenerate = rotation_norm > FLT_EPSILON;
    if(!prune_large) {
        return not_transparent && not_degenerate;
    }
    return not_transparent && not_large_ws && not_large_ss && not_degenerate;
}

static gsx_error gsx_metal_adc_apply_refine(const gsx_adc_desc *desc, const gsx_adc_request *request, gsx_adc_result *out_result)
{
    gsx_metal_adc_refine_data refine_data = { 0 };
    gsx_size_t count_before_refine = 0;
    gsx_size_t count_after_growth = 0;
    gsx_size_t max_capacity = 0;
    gsx_size_t grow_budget = 0;
    gsx_size_t grow_count = 0;
    gsx_size_t split_count = 0;
    gsx_size_t duplicate_count = 0;
    gsx_size_t keep_count = 0;
    gsx_size_t index = 0;
    int32_t *grow_sources = NULL;
    uint8_t *grow_modes = NULL;
    int32_t *gather_indices = NULL;
    int32_t *keep_indices = NULL;
    bool prune_large = false;
    gsx_error result = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc, request, and out_result must be non-null");
    }

    error = gsx_metal_adc_load_count(request->gs, &count_before_refine);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(count_before_refine == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_metal_adc_load_refine_data(request->gs, count_before_refine, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    max_capacity = gsx_metal_adc_non_negative_index(desc->max_num_gaussians);
    if(max_capacity > count_before_refine) {
        grow_budget = max_capacity - count_before_refine;
    }

    if(grow_budget > 0) {
        grow_sources = (int32_t *)malloc(sizeof(int32_t) * (size_t)grow_budget);
        grow_modes = (uint8_t *)malloc(sizeof(uint8_t) * (size_t)grow_budget);
        if(grow_sources == NULL || grow_modes == NULL) {
            result = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate duplicate source index buffer");
            goto cleanup;
        }
        for(index = 0; index < count_before_refine && grow_count < grow_budget; ++index) {
            gsx_metal_adc_grow_mode mode = gsx_metal_adc_grow_mode_for_index(desc, &refine_data, request->scene_scale, index);
            if(mode == GSX_METAL_ADC_GROW_NONE) {
                continue;
            }
            grow_sources[grow_count] = (int32_t)index;
            grow_modes[grow_count] = (uint8_t)mode;
            if(mode == GSX_METAL_ADC_GROW_SPLIT) {
                split_count += 1;
            } else {
                duplicate_count += 1;
            }
            grow_count += 1;
        }
    }

    if(grow_count > 0) {
        gsx_size_t gathered_count = count_before_refine + grow_count;

        gather_indices = (int32_t *)malloc(sizeof(int32_t) * (size_t)gathered_count);
        if(gather_indices == NULL) {
            result = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate duplicate gather index buffer");
            goto cleanup;
        }
        for(index = 0; index < count_before_refine; ++index) {
            gather_indices[index] = (int32_t)index;
        }
        for(index = 0; index < grow_count; ++index) {
            gather_indices[count_before_refine + index] = grow_sources[index];
        }

        error = gsx_metal_adc_apply_gs_and_optim_gather(request, gather_indices, gathered_count);
        if(!gsx_error_is_success(error)) {
            result = error;
            goto cleanup;
        }
        free(gather_indices);
        gather_indices = NULL;

        error = gsx_metal_adc_load_count(request->gs, &count_after_growth);
        if(!gsx_error_is_success(error)) {
            result = error;
            goto cleanup;
        }
        if(count_after_growth != gathered_count) {
            result = gsx_make_error(GSX_ERROR_INVALID_STATE, "metal default adc growth produced unexpected gaussian count");
            goto cleanup;
        }

        error = gsx_metal_adc_load_refine_data(request->gs, count_after_growth, &refine_data);
        if(!gsx_error_is_success(error)) {
            result = error;
            goto cleanup;
        }
        for(index = 0; index < grow_count; ++index) {
            gsx_size_t src = (gsx_size_t)grow_sources[index];
            gsx_size_t target = count_before_refine + index;

            if(src >= count_before_refine || target >= count_after_growth) {
                continue;
            }
            error = gsx_metal_adc_copy_shared_growth_fields(&refine_data, target, src);
            if(!gsx_error_is_success(error)) {
                result = error;
                goto cleanup;
            }
            if(grow_modes[index] == (uint8_t)GSX_METAL_ADC_GROW_DUPLICATE) {
                error = gsx_metal_adc_apply_duplicate_mutation(&refine_data, target, src);
            } else {
                error = gsx_metal_adc_apply_split_mutation(&refine_data, target, src, desc->seed, request->global_step, index);
            }
            if(!gsx_error_is_success(error)) {
                result = error;
                goto cleanup;
            }
        }
        error = gsx_metal_adc_upload_growth_mutations(&refine_data);
        if(!gsx_error_is_success(error)) {
            result = error;
            goto cleanup;
        }

        out_result->duplicated_count += duplicate_count;
        out_result->grown_count += split_count;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_load_count(request->gs, &count_after_growth);
    if(!gsx_error_is_success(error)) {
        result = error;
        goto cleanup;
    }
    if(count_after_growth == 0) {
        result = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        goto cleanup;
    }
    error = gsx_metal_adc_load_refine_data(request->gs, count_after_growth, &refine_data);
    if(!gsx_error_is_success(error)) {
        result = error;
        goto cleanup;
    }

    keep_indices = (int32_t *)malloc(sizeof(int32_t) * (size_t)count_after_growth);
    if(keep_indices == NULL) {
        result = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate prune keep-index buffer");
        goto cleanup;
    }
    prune_large = request->global_step > gsx_metal_adc_non_negative_index(desc->reset_every);
    for(index = 0; index < count_after_growth; ++index) {
        if(gsx_metal_adc_should_keep(desc, &refine_data, request->scene_scale, index, count_before_refine, prune_large)) {
            keep_indices[keep_count] = (int32_t)index;
            keep_count += 1;
        }
    }
    if(keep_count == 0) {
        keep_indices[0] = 0;
        keep_count = 1;
    }
    if(keep_count < count_after_growth) {
        error = gsx_metal_adc_apply_gs_and_optim_gather(request, keep_indices, keep_count);
        if(!gsx_error_is_success(error)) {
            result = error;
            goto cleanup;
        }
        out_result->pruned_count += count_after_growth - keep_count;
        out_result->mutated = true;
    }

    result = gsx_make_error(GSX_ERROR_SUCCESS, NULL);

cleanup:
    free(grow_sources);
    free(grow_modes);
    free(gather_indices);
    free(keep_indices);
    gsx_metal_adc_free_refine_data(&refine_data);
    return result;
}

gsx_error gsx_metal_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc)
{
    gsx_metal_adc *metal_adc = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_adc == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_adc and desc must be non-null");
    }

    *out_adc = NULL;
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc currently supports only GSX_ADC_ALGORITHM_DEFAULT");
    }

    metal_adc = (gsx_metal_adc *)calloc(1, sizeof(*metal_adc));
    if(metal_adc == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal adc");
    }

    error = gsx_adc_base_init(&metal_adc->base, &gsx_metal_adc_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(metal_adc);
        return error;
    }

    *out_adc = &metal_adc->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_destroy(gsx_adc_t adc)
{
    gsx_metal_adc *metal_adc = (gsx_metal_adc *)adc;

    if(adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc must be non-null");
    }

    gsx_adc_base_deinit(&metal_adc->base);
    free(metal_adc);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result)
{
    gsx_size_t count_before = 0;
    gsx_size_t count_after = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    bool refine_window = false;
    bool reset_window = false;

    if(adc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }

    gsx_metal_adc_zero_result(out_result);

    error = gsx_metal_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal default adc requires gs info/query support");
    }

    out_result->gaussians_before = count_before;
    out_result->gaussians_after = count_before;
    refine_window = gsx_metal_adc_in_refine_window(&adc->desc, request->global_step);
    reset_window = gsx_metal_adc_in_reset_window(&adc->desc, request->global_step);

    if(refine_window) {
        error = gsx_metal_adc_apply_refine(&adc->desc, request, out_result);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(reset_window) {
        error = gsx_metal_adc_apply_reset(&adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        out_result->reset_count = 1;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_load_count(request->gs, &count_after);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal default adc requires gs info/query support");
    }

    out_result->gaussians_after = count_after;
    if(out_result->gaussians_after != out_result->gaussians_before) {
        out_result->mutated = true;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
