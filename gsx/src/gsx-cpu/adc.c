#include "internal.h"
#include "adc/internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static gsx_error gsx_cpu_adc_destroy(gsx_adc_t adc);
static gsx_error gsx_cpu_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);
static gsx_error gsx_cpu_adc_build_index_tensor(
    gsx_tensor_t reference_tensor,
    const int32_t *indices,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor
);

static const gsx_adc_i gsx_cpu_adc_iface = {
    gsx_cpu_adc_destroy,
    gsx_cpu_adc_step
};

gsx_size_t gsx_cpu_adc_non_negative_index(gsx_index_t value)
{
    if(value <= 0) {
        return 0;
    }
    return (gsx_size_t)value;
}

void gsx_cpu_adc_zero_result(gsx_adc_result *out_result)
{
    out_result->gaussians_before = 0;
    out_result->gaussians_after = 0;
    out_result->pruned_count = 0;
    out_result->duplicated_count = 0;
    out_result->grown_count = 0;
    out_result->reset_count = 0;
    out_result->mutated = false;
}

bool gsx_cpu_adc_optim_enabled(const gsx_adc_request *request)
{
    return request != NULL && request->optim != NULL && request->optim->iface != NULL;
}

bool gsx_cpu_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_cpu_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_cpu_adc_non_negative_index(desc->end_refine);

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

bool gsx_cpu_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_cpu_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_cpu_adc_non_negative_index(desc->end_refine);

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

gsx_error gsx_cpu_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count)
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

gsx_error gsx_cpu_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    bool optional,
    float **out_values
)
{
    gsx_tensor_t tensor = NULL;
    void *native_handle = NULL;
    gsx_size_t offset_bytes = 0;
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
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc currently supports only float32 gs fields");
    }
    error = gsx_adc_validate_gs_field_shape(tensor, count, expected_dim1);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_tensor_get_native_handle(tensor, &native_handle, &offset_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    *out_values = (float *)((unsigned char *)native_handle + (size_t)offset_bytes);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_cpu_adc_free_refine_data(gsx_cpu_adc_refine_data *data)
{
    if(data == NULL) {
        return;
    }
    memset(data, 0, sizeof(*data));
}

gsx_error gsx_cpu_adc_load_refine_data(gsx_gs_t gs, gsx_size_t count, bool require_grad_acc, gsx_cpu_adc_refine_data *out_data)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_data must be non-null");
    }
    memset(out_data, 0, sizeof(*out_data));
    out_data->count = count;

    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_MEAN3D, count, 3, false, &out_data->mean3d);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc refine requires GSX_GS_FIELD_MEAN3D access");
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_GRAD_ACC, count, 1, true, &out_data->grad_acc);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return error;
    }
    if(require_grad_acc && out_data->grad_acc == NULL) {
        gsx_cpu_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc refine requires GSX_GS_FIELD_GRAD_ACC auxiliary field");
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_VISIBLE_COUNTER, count, 1, true, &out_data->visible_counter);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_visible_counter = out_data->visible_counter != NULL;
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_LOGSCALE, count, 3, false, &out_data->logscale);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc refine requires GSX_GS_FIELD_LOGSCALE access");
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_OPACITY, count, 1, false, &out_data->opacity);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc refine requires GSX_GS_FIELD_OPACITY access");
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_ROTATION, count, 4, false, &out_data->rotation);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc refine requires GSX_GS_FIELD_ROTATION access");
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_SH0, count, 3, false, &out_data->sh0);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc refine requires GSX_GS_FIELD_SH0 access");
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_SH1, count, 9, true, &out_data->sh1);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_SH2, count, 15, true, &out_data->sh2);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_SH3, count, 21, true, &out_data->sh3);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_cpu_adc_load_refine_field(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, count, 1, true, &out_data->max_screen_radius);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_max_screen_radius = out_data->max_screen_radius != NULL;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

float gsx_cpu_adc_probability_to_logit(float probability)
{
    float clamped = probability;

    if(clamped <= 1e-6f) {
        clamped = 1e-6f;
    }
    if(clamped >= 1.0f - 1e-6f) {
        clamped = 1.0f - 1e-6f;
    }
    return gsx_logit(clamped);
}

gsx_error gsx_cpu_adc_sample_normal(gsx_pcg32_t rng, float *out_value)
{
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }
    return gsx_pcg32_next_normal(rng, out_value);
}

gsx_error gsx_cpu_adc_copy_slice(float *dst, gsx_size_t dst_index, const float *src, gsx_size_t src_index, gsx_size_t width)
{
    if(dst == NULL || src == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy slice tensors must be non-null");
    }
    memcpy(dst + dst_index * width, src + src_index * width, width * sizeof(float));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cpu_adc_copy_optional_slice(
    float *dst,
    gsx_size_t dst_index,
    const float *src,
    gsx_size_t src_index,
    gsx_size_t width
)
{
    if(dst == NULL || src == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_cpu_adc_copy_slice(dst, dst_index, src, src_index, width);
}

float gsx_cpu_adc_clamp_probability(float value)
{
    if(value < 0.0f) {
        return 0.0f;
    }
    if(value > 1.0f - 1e-6f) {
        return 1.0f - 1e-6f;
    }
    return value;
}

void gsx_cpu_adc_normalize_quaternion(float *qx, float *qy, float *qz, float *qw)
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

void gsx_cpu_adc_build_rotation_matrix(
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
    float *m22
)
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

static gsx_error gsx_cpu_adc_build_index_tensor(
    gsx_tensor_t reference_tensor,
    const int32_t *indices,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor
)
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
    byte_count = index_count * sizeof(int32_t);
    memset(out_index_tensor, 0, sizeof(*out_index_tensor));

    buffer_desc.buffer_type = reference_tensor->backing_buffer->buffer_type;
    buffer_desc.size_bytes = byte_count;
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

gsx_error gsx_cpu_adc_apply_gs_and_optim_gather(const gsx_adc_request *request, const int32_t *indices, gsx_size_t index_count)
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
    error = gsx_cpu_adc_build_index_tensor(mean3d, indices, index_count, &index_buffer, &index_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_gs_gather(request->gs, &index_tensor);
    if(!gsx_error_is_success(error)) {
        gsx_backend_buffer_free(index_buffer);
        return error;
    }

    if(gsx_cpu_adc_optim_enabled(request)) {
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

gsx_error gsx_cpu_adc_zero_growth_optim_state(const gsx_adc_request *request, gsx_size_t old_count, gsx_size_t new_count)
{
    if(!gsx_cpu_adc_optim_enabled(request) || new_count <= old_count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_cpu_optim_zero_appended_rows(request->optim, old_count, new_count);
}

gsx_error gsx_cpu_adc_reset_post_refine_aux(gsx_gs_t gs)
{
    return gsx_gs_zero_aux_tensors(
        gs,
        GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER | GSX_GS_AUX_MAX_SCREEN_RADIUS
    );
}

gsx_error gsx_cpu_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc)
{
    gsx_cpu_adc *cpu_adc = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_adc == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_adc and desc must be non-null");
    }

    *out_adc = NULL;
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT && desc->algorithm != GSX_ADC_ALGORITHM_MCMC) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc currently supports only GSX_ADC_ALGORITHM_DEFAULT and GSX_ADC_ALGORITHM_MCMC");
    }

    cpu_adc = (gsx_cpu_adc *)calloc(1, sizeof(*cpu_adc));
    if(cpu_adc == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu adc");
    }

    error = gsx_adc_base_init(&cpu_adc->base, &gsx_cpu_adc_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(cpu_adc);
        return error;
    }

    error = gsx_pcg32_init(&cpu_adc->rng, (gsx_pcg32_state_t)desc->seed);
    if(!gsx_error_is_success(error)) {
        gsx_adc_base_deinit(&cpu_adc->base);
        free(cpu_adc);
        return error;
    }

    *out_adc = &cpu_adc->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_destroy(gsx_adc_t adc)
{
    gsx_cpu_adc *cpu_adc = (gsx_cpu_adc *)adc;

    if(adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc must be non-null");
    }

    gsx_pcg32_free(cpu_adc->rng);
    gsx_adc_base_deinit(&cpu_adc->base);
    free(cpu_adc);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result)
{
    gsx_size_t count_before = 0;
    gsx_size_t count_after = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    bool refine_window = false;
    bool reset_window = false;

    if(adc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }

    gsx_cpu_adc_zero_result(out_result);

    error = gsx_cpu_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc requires gs info/query support");
    }

    out_result->gaussians_before = count_before;
    out_result->gaussians_after = count_before;
    refine_window = gsx_cpu_adc_in_refine_window(&adc->desc, request->global_step);
    reset_window = gsx_cpu_adc_in_reset_window(&adc->desc, request->global_step);

    if(adc->desc.algorithm == GSX_ADC_ALGORITHM_MCMC) {
        error = gsx_cpu_adc_apply_mcmc_noise((gsx_cpu_adc *)adc, &adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(refine_window) {
        switch(adc->desc.algorithm) {
        case GSX_ADC_ALGORITHM_DEFAULT:
            error = gsx_cpu_adc_apply_default_refine((gsx_cpu_adc *)adc, &adc->desc, request, out_result);
            break;
        case GSX_ADC_ALGORITHM_MCMC:
            error = gsx_cpu_adc_apply_mcmc_refine((gsx_cpu_adc *)adc, &adc->desc, request, out_result);
            break;
        default:
            error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc algorithm is not supported");
            break;
        }
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(reset_window) {
        error = gsx_cpu_adc_apply_default_reset(&adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        out_result->reset_count = 1;
        out_result->mutated = true;
    }

    error = gsx_cpu_adc_load_count(request->gs, &count_after);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc requires gs info/query support");
    }

    out_result->gaussians_after = count_after;
    if(out_result->gaussians_after != out_result->gaussians_before) {
        out_result->mutated = true;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
