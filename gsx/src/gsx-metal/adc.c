#include "internal.h"

#include "gsx/gsx-random.h"

#include "../pcg32.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_metal_adc {
    struct gsx_adc base;
    gsx_pcg32_t rng;
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
    gsx_backend_tensor_view mean3d_view;
    gsx_tensor_t grad_acc_tensor;
    gsx_backend_tensor_view grad_acc_view;
    gsx_tensor_t visible_counter_tensor;
    gsx_backend_tensor_view visible_counter_view;
    gsx_tensor_t logscale_tensor;
    gsx_backend_tensor_view logscale_view;
    gsx_tensor_t opacity_tensor;
    gsx_backend_tensor_view opacity_view;
    gsx_tensor_t rotation_tensor;
    gsx_backend_tensor_view rotation_view;
    gsx_tensor_t max_screen_radius_tensor;
    gsx_backend_tensor_view max_screen_radius_view;
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

static void gsx_metal_adc_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    out_view->buffer = tensor->backing_buffer;
    out_view->offset_bytes = tensor->offset_bytes;
    out_view->size_bytes = tensor->size_bytes;
    out_view->effective_alignment_bytes = tensor->effective_alignment_bytes;
    out_view->data_type = tensor->data_type;
}

static gsx_error gsx_metal_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    bool optional,
    gsx_tensor_t *out_tensor,
    gsx_backend_tensor_view *out_view)
{
    gsx_tensor_t tensor = NULL;
    gsx_size_t expected_count = 0;
    gsx_size_t actual_count = 1;
    gsx_size_t byte_count = 0;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_tensor == NULL || out_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor and out_view must be non-null");
    }

    *out_tensor = NULL;
    memset(out_view, 0, sizeof(*out_view));

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

    gsx_metal_adc_make_tensor_view(tensor, out_view);
    *out_tensor = tensor;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_adc_free_refine_data(gsx_metal_adc_refine_data *data)
{
    if(data == NULL) {
        return;
    }
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

    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_MEAN3D, count, 3, false, &out_data->mean3d_tensor, &out_data->mean3d_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_MEAN3D access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_GRAD_ACC, count, 1, true, &out_data->grad_acc_tensor, &out_data->grad_acc_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    if(out_data->grad_acc_tensor == NULL) {
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
        &out_data->visible_counter_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_visible_counter = out_data->visible_counter_tensor != NULL;
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_LOGSCALE, count, 3, false, &out_data->logscale_tensor, &out_data->logscale_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_LOGSCALE access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_OPACITY, count, 1, false, &out_data->opacity_tensor, &out_data->opacity_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_OPACITY access"
        );
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_ROTATION, count, 4, false, &out_data->rotation_tensor, &out_data->rotation_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal default adc refine requires GSX_GS_FIELD_ROTATION access"
        );
    }
    error = gsx_metal_adc_load_refine_field(
        gs,
        GSX_GS_FIELD_MAX_SCREEN_RADIUS,
        count,
        1,
        true,
        &out_data->max_screen_radius_tensor,
        &out_data->max_screen_radius_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_max_screen_radius = out_data->max_screen_radius_tensor != NULL;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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
    if(clamp_threshold > 1.0f - 1e-6f) {
        clamp_threshold = 1.0f - 1e-6f;
    }
    max_opacity = gsx_logit(clamp_threshold);
    error = gsx_tensor_clamp_inplace(opacity, &min_opacity, &max_opacity);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_metal_adc_optim_enabled(request)) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_optim_reset_param_group_by_role(request->optim, GSX_OPTIM_PARAM_ROLE_OPACITY);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_init_temp_buffer_for_tensor(
    gsx_tensor_t reference_tensor,
    gsx_size_t byte_count,
    gsx_backend_buffer_t *out_buffer)
{
    gsx_backend_buffer_desc buffer_desc = { 0 };

    if(reference_tensor == NULL || out_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reference_tensor and out_buffer must be non-null");
    }

    *out_buffer = NULL;
    buffer_desc.buffer_type = reference_tensor->backing_buffer->buffer_type;
    buffer_desc.size_bytes = byte_count;
    buffer_desc.alignment_bytes = sizeof(uint32_t);
    return gsx_backend_buffer_init(out_buffer, &buffer_desc);
}

static gsx_error gsx_metal_adc_build_index_tensor(
    gsx_tensor_t reference_tensor,
    const int32_t *indices,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor)
{
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
    error = gsx_metal_adc_init_temp_buffer_for_tensor(reference_tensor, byte_count, out_buffer);
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

static gsx_error gsx_metal_adc_zero_growth_optim_state(const gsx_adc_request *request, gsx_size_t old_count, gsx_size_t new_count)
{
    if(!gsx_metal_adc_optim_enabled(request) || new_count <= old_count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_metal_optim_zero_appended_rows(request->optim, old_count, new_count);
}

static gsx_error gsx_metal_adc_reset_post_refine_aux(gsx_gs_t gs)
{
    return gsx_gs_zero_aux_tensors(
        gs,
        GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER | GSX_GS_AUX_MAX_SCREEN_RADIUS
    );
}

static gsx_error gsx_metal_adc_download_temp_buffer(gsx_backend_buffer_t buffer, void *dst_bytes, gsx_size_t byte_count)
{
    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }
    if(byte_count != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero byte_count");
    }
    return gsx_backend_buffer_download(buffer, 0, dst_bytes, byte_count);
}

static gsx_error gsx_metal_adc_upload_temp_buffer(gsx_backend_buffer_t buffer, const void *src_bytes, gsx_size_t byte_count)
{
    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }
    if(byte_count != 0 && src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero byte_count");
    }
    return gsx_backend_buffer_upload(buffer, 0, src_bytes, byte_count);
}

static gsx_error gsx_metal_adc_advance_rng_after_splits(gsx_metal_adc *metal_adc, gsx_size_t split_count)
{
    gsx_size_t advance_count = 0;

    if(metal_adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc must be non-null");
    }
    if(split_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_mul_overflows(split_count, (gsx_size_t)12, &advance_count) || advance_count > (gsx_size_t)INT64_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc split rng advance exceeds supported range");
    }
    return gsx_pcg32_advance(metal_adc->rng, (gsx_pcg32_statediff_t)advance_count);
}

static gsx_error gsx_metal_adc_apply_refine(
    gsx_metal_adc *metal_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result)
{
    gsx_metal_adc_refine_data refine_data = { 0 };
    gsx_size_t count_before_refine = 0;
    gsx_size_t max_capacity = 0;
    gsx_size_t grow_budget = 0;
    gsx_size_t grow_count = 0;
    gsx_size_t split_count = 0;
    gsx_size_t duplicate_count = 0;
    gsx_size_t count_after_growth = 0;
    gsx_size_t keep_count = 0;
    gsx_size_t index = 0;
    gsx_backend_buffer_t mode_buffer = NULL;
    gsx_backend_buffer_t split_source_buffer = NULL;
    gsx_backend_buffer_t split_target_buffer = NULL;
    gsx_backend_buffer_t keep_mask_buffer = NULL;
    uint8_t *modes = NULL;
    int32_t *gather_indices = NULL;
    int32_t *split_sources = NULL;
    int32_t *split_targets = NULL;
    uint8_t *keep_mask = NULL;
    int32_t *keep_indices = NULL;
    bool prune_large = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc, desc, request, and out_result must be non-null");
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
        gsx_size_t mode_byte_count = 0;
        gsx_metal_adc_classify_growth_params params = { 0 };

        if(gsx_size_mul_overflows(count_before_refine, sizeof(uint8_t), &mode_byte_count)) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "growth mode buffer size overflow");
            goto cleanup;
        }
        modes = (uint8_t *)malloc((size_t)mode_byte_count);
        if(modes == NULL) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate growth mode staging");
            goto cleanup;
        }
        error = gsx_metal_adc_init_temp_buffer_for_tensor(refine_data.mean3d_tensor, mode_byte_count, &mode_buffer);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        params.gaussian_count = (uint32_t)count_before_refine;
        params.has_visible_counter = refine_data.has_visible_counter ? 1u : 0u;
        params.duplicate_grad_threshold = desc->duplicate_grad_threshold;
        params.duplicate_scale_threshold = desc->duplicate_scale_threshold;
        params.scene_scale = request->scene_scale;
        error = gsx_metal_backend_dispatch_adc_classify_growth(
            refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
            &refine_data.grad_acc_view,
            refine_data.has_visible_counter ? &refine_data.visible_counter_view : NULL,
            &refine_data.logscale_view,
            mode_buffer,
            &params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_download_temp_buffer(mode_buffer, modes, mode_byte_count);
        // Although current metal backend's download function is synchronous, we
        // should still treat it as potentially asynchronous and ensure proper
        // synchronization before reading the downloaded data.
        error = gsx_backend_major_stream_sync(refine_data.mean3d_tensor->backing_buffer->buffer_type->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        
        // A potential acceleration to this is to use a unified tensor/buffer.
        for(index = 0; index < count_before_refine; ++index) {
            if(modes[index] != (uint8_t)GSX_METAL_ADC_GROW_NONE) {
                grow_count += 1;
                if(modes[index] == (uint8_t)GSX_METAL_ADC_GROW_SPLIT) {
                    split_count += 1;
                } else {
                    duplicate_count += 1;
                }
            }
        }
        if(grow_count > grow_budget) {
            gsx_size_t allowed = 0;

            grow_count = 0;
            split_count = 0;
            duplicate_count = 0;
            for(index = 0; index < count_before_refine && allowed < grow_budget; ++index) {
                if(modes[index] != (uint8_t)GSX_METAL_ADC_GROW_NONE) {
                    allowed += 1;
                    grow_count += 1;
                    if(modes[index] == (uint8_t)GSX_METAL_ADC_GROW_SPLIT) {
                        split_count += 1;
                    } else {
                        duplicate_count += 1;
                    }
                }
            }
        }
    }

    if(grow_count > 0) {
        gsx_size_t gathered_count = count_before_refine + grow_count;
        gsx_size_t gather_byte_count = 0;
        gsx_size_t split_byte_count = 0;
        gsx_size_t grow_index = 0;
        gsx_size_t split_index = 0;
        gsx_metal_adc_apply_split_params split_params = { 0 };
        gsx_pcg32 *rng_state = NULL;

        if(gsx_size_mul_overflows(gathered_count, sizeof(int32_t), &gather_byte_count)) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "growth gather index size overflow");
            goto cleanup;
        }
        gather_indices = (int32_t *)malloc((size_t)gather_byte_count);
        if(gather_indices == NULL) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate growth gather indices");
            goto cleanup;
        }
        for(index = 0; index < count_before_refine; ++index) {
            gather_indices[index] = (int32_t)index;
        }
        if(split_count > 0) {
            if(gsx_size_mul_overflows(split_count, sizeof(int32_t), &split_byte_count)) {
                error = gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "split index size overflow");
                goto cleanup;
            }
            split_sources = (int32_t *)malloc((size_t)split_byte_count);
            split_targets = (int32_t *)malloc((size_t)split_byte_count);
            if(split_sources == NULL || split_targets == NULL) {
                error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate split staging");
                goto cleanup;
            }
        }
        for(index = 0; index < count_before_refine && grow_index < grow_count; ++index) {
            if(modes[index] == (uint8_t)GSX_METAL_ADC_GROW_NONE) {
                continue;
            }
            gather_indices[count_before_refine + grow_index] = (int32_t)index;
            if(modes[index] == (uint8_t)GSX_METAL_ADC_GROW_SPLIT) {
                split_sources[split_index] = (int32_t)index;
                split_targets[split_index] = (int32_t)(count_before_refine + grow_index);
                split_index += 1;
            }
            grow_index += 1;
        }

        error = gsx_metal_adc_apply_gs_and_optim_gather(request, gather_indices, gathered_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_load_count(request->gs, &count_after_growth);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(count_after_growth != gathered_count) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "metal default adc growth produced unexpected gaussian count");
            goto cleanup;
        }
        error = gsx_metal_adc_zero_growth_optim_state(request, count_before_refine, count_after_growth);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_load_refine_data(request->gs, count_after_growth, &refine_data);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        if(split_count > 0) {
            error = gsx_metal_adc_init_temp_buffer_for_tensor(refine_data.mean3d_tensor, split_count * sizeof(int32_t), &split_source_buffer);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_adc_init_temp_buffer_for_tensor(refine_data.mean3d_tensor, split_count * sizeof(int32_t), &split_target_buffer);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_adc_upload_temp_buffer(split_source_buffer, split_sources, split_count * sizeof(int32_t));
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_adc_upload_temp_buffer(split_target_buffer, split_targets, split_count * sizeof(int32_t));
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            rng_state = (gsx_pcg32 *)metal_adc->rng;
            split_params.split_count = (uint32_t)split_count;
            split_params.rng_state = rng_state->state;
            split_params.rng_inc = rng_state->inc;
            error = gsx_metal_backend_dispatch_adc_apply_split(
                refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
                &refine_data.mean3d_view,
                &refine_data.logscale_view,
                &refine_data.opacity_view,
                &refine_data.rotation_view,
                split_source_buffer,
                split_target_buffer,
                &split_params);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_adc_advance_rng_after_splits(metal_adc, split_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        out_result->duplicated_count += duplicate_count;
        out_result->grown_count += split_count;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_load_count(request->gs, &count_after_growth);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(count_after_growth == 0) {
        error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        goto cleanup;
    }
    error = gsx_metal_adc_load_refine_data(request->gs, count_after_growth, &refine_data);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    {
        gsx_size_t keep_mask_byte_count = 0;
        gsx_metal_adc_keep_mask_params keep_params = { 0 };

        if(gsx_size_mul_overflows(count_after_growth, sizeof(uint8_t), &keep_mask_byte_count)) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "keep-mask buffer size overflow");
            goto cleanup;
        }
        keep_mask = (uint8_t *)malloc((size_t)keep_mask_byte_count);
        keep_indices = (int32_t *)malloc((size_t)(count_after_growth * sizeof(int32_t)));
        if(keep_mask == NULL || keep_indices == NULL) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate prune staging");
            goto cleanup;
        }
        error = gsx_metal_adc_init_temp_buffer_for_tensor(refine_data.mean3d_tensor, keep_mask_byte_count, &keep_mask_buffer);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        prune_large = request->global_step > gsx_metal_adc_non_negative_index(desc->reset_every);
        keep_params.gaussian_count = (uint32_t)count_after_growth;
        keep_params.has_max_screen_radius = refine_data.has_max_screen_radius ? 1u : 0u;
        keep_params.count_before_growth = (uint32_t)count_before_refine;
        keep_params.prune_large = prune_large ? 1u : 0u;
        keep_params.scene_scale = request->scene_scale;
        keep_params.pruning_opacity_threshold = desc->pruning_opacity_threshold;
        keep_params.max_world_scale = desc->max_world_scale;
        keep_params.max_screen_scale = desc->max_screen_scale;
        error = gsx_metal_backend_dispatch_adc_keep_mask(
            refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
            &refine_data.opacity_view,
            &refine_data.logscale_view,
            &refine_data.rotation_view,
            refine_data.has_max_screen_radius ? &refine_data.max_screen_radius_view : NULL,
            keep_mask_buffer,
            &keep_params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_download_temp_buffer(keep_mask_buffer, keep_mask, keep_mask_byte_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        for(index = 0; index < count_after_growth; ++index) {
            if(keep_mask[index] != 0) {
                keep_indices[keep_count] = (int32_t)index;
                keep_count += 1;
            }
        }
        if(keep_count == 0) {
            keep_indices[0] = 0;
            keep_count = 1;
        }
    }

    if(keep_count < count_after_growth) {
        error = gsx_metal_adc_apply_gs_and_optim_gather(request, keep_indices, keep_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        out_result->pruned_count += count_after_growth - keep_count;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_reset_post_refine_aux(request->gs);

cleanup:
    if(mode_buffer != NULL) {
        (void)gsx_backend_buffer_free(mode_buffer);
    }
    if(split_source_buffer != NULL) {
        (void)gsx_backend_buffer_free(split_source_buffer);
    }
    if(split_target_buffer != NULL) {
        (void)gsx_backend_buffer_free(split_target_buffer);
    }
    if(keep_mask_buffer != NULL) {
        (void)gsx_backend_buffer_free(keep_mask_buffer);
    }
    free(modes);
    free(gather_indices);
    free(split_sources);
    free(split_targets);
    free(keep_mask);
    free(keep_indices);
    gsx_metal_adc_free_refine_data(&refine_data);
    return error;
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

    error = gsx_pcg32_init(&metal_adc->rng, (gsx_pcg32_state_t)desc->seed);
    if(!gsx_error_is_success(error)) {
        gsx_adc_base_deinit(&metal_adc->base);
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

    gsx_pcg32_free(metal_adc->rng);
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
        error = gsx_metal_adc_apply_refine((gsx_metal_adc *)adc, &adc->desc, request, out_result);
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
