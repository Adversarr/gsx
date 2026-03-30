#include "adc/internal.h"

#include "../pcg32.h"

#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static gsx_error gsx_metal_adc_destroy(gsx_adc_t adc);
static gsx_error gsx_metal_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);

static const gsx_adc_i gsx_metal_adc_iface = {
    gsx_metal_adc_destroy,
    gsx_metal_adc_step
};

gsx_size_t gsx_metal_adc_non_negative_index(gsx_index_t value)
{
    if(value <= 0) {
        return 0;
    }
    return (gsx_size_t)value;
}

void gsx_metal_adc_zero_result(gsx_adc_result *out_result)
{
    out_result->gaussians_before = 0;
    out_result->gaussians_after = 0;
    out_result->pruned_count = 0;
    out_result->duplicated_count = 0;
    out_result->grown_count = 0;
    out_result->reset_count = 0;
    out_result->mutated = false;
}

bool gsx_metal_adc_optim_enabled(const gsx_adc_request *request)
{
    return request != NULL && request->optim != NULL && request->optim->iface != NULL;
}

bool gsx_metal_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step)
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

bool gsx_metal_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step)
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

gsx_error gsx_metal_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count)
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

void gsx_metal_adc_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    gsx_tensor_fill_backend_view(tensor, out_view);
}

gsx_error gsx_metal_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    bool optional,
    gsx_tensor_t *out_tensor,
    gsx_backend_tensor_view *out_view)
{
    gsx_tensor_t tensor = NULL;
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
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc currently supports only float32 gs fields");
    }
    error = gsx_adc_validate_gs_field_shape(tensor, count, expected_dim1);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gsx_metal_adc_make_tensor_view(tensor, out_view);
    *out_tensor = tensor;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_metal_adc_free_refine_data(gsx_metal_adc_refine_data *data)
{
    if(data == NULL) {
        return;
    }
    memset(data, 0, sizeof(*data));
}

gsx_error gsx_metal_adc_load_refine_data(gsx_gs_t gs, gsx_size_t count, bool require_grad_acc, gsx_metal_adc_refine_data *out_data)
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
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc refine requires GSX_GS_FIELD_MEAN3D access");
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_GRAD_ACC, count, 1, true, &out_data->grad_acc_tensor, &out_data->grad_acc_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_ABSGRAD_ACC, count, 1, true, &out_data->absgrad_acc_tensor, &out_data->absgrad_acc_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    if(require_grad_acc && out_data->grad_acc_tensor == NULL) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc refine requires GSX_GS_FIELD_GRAD_ACC auxiliary field");
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_VISIBLE_COUNTER, count, 1, true, &out_data->visible_counter_tensor, &out_data->visible_counter_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_visible_counter = out_data->visible_counter_tensor != NULL;
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_LOGSCALE, count, 3, false, &out_data->logscale_tensor, &out_data->logscale_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc refine requires GSX_GS_FIELD_LOGSCALE access");
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_OPACITY, count, 1, false, &out_data->opacity_tensor, &out_data->opacity_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc refine requires GSX_GS_FIELD_OPACITY access");
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_ROTATION, count, 4, false, &out_data->rotation_tensor, &out_data->rotation_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc refine requires GSX_GS_FIELD_ROTATION access");
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH0, count, 3, false, &out_data->sh0_tensor, &out_data->sh0_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc refine requires GSX_GS_FIELD_SH0 access");
    }
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH1, count, 9, true, &out_data->sh1_tensor, &out_data->sh1_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_sh1 = out_data->sh1_tensor != NULL;
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH2, count, 15, true, &out_data->sh2_tensor, &out_data->sh2_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_sh2 = out_data->sh2_tensor != NULL;
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_SH3, count, 21, true, &out_data->sh3_tensor, &out_data->sh3_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_sh3 = out_data->sh3_tensor != NULL;
    error = gsx_metal_adc_load_refine_field(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, count, 1, true, &out_data->max_screen_radius_tensor, &out_data->max_screen_radius_view);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(out_data);
        return error;
    }
    out_data->has_max_screen_radius = out_data->max_screen_radius_tensor != NULL;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_adc_resolve_staging_buffer_type(
    gsx_tensor_t reference_tensor,
    gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(reference_tensor == NULL || reference_tensor->backing_buffer == NULL || reference_tensor->backing_buffer->buffer_type == NULL
        || out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reference_tensor and out_buffer_type must be non-null");
    }

    error = gsx_backend_find_buffer_type(reference_tensor->backing_buffer->buffer_type->backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED, &buffer_type);
    if(!gsx_error_is_success(error)) {
        if(error.code != GSX_ERROR_NOT_SUPPORTED) {
            return error;
        }
        buffer_type = reference_tensor->backing_buffer->buffer_type;
    }

    *out_buffer_type = buffer_type;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_begin_staging_cycle(gsx_metal_adc *metal_adc, gsx_tensor_t reference_tensor)
{
    gsx_backend_buffer_type_t staging_buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || reference_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc and reference_tensor must be non-null");
    }

    error = gsx_metal_adc_resolve_staging_buffer_type(reference_tensor, &staging_buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(metal_adc->staging_arena != NULL && metal_adc->staging_buffer_type != staging_buffer_type) {
        error = gsx_arena_free(metal_adc->staging_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        metal_adc->staging_arena = NULL;
        metal_adc->staging_buffer_type = NULL;
    }
    if(metal_adc->staging_arena == NULL) {
        arena_desc.initial_capacity_bytes = 4096;
        error = gsx_arena_init(&metal_adc->staging_arena, staging_buffer_type, &arena_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        metal_adc->staging_buffer_type = staging_buffer_type;
    }

    error = gsx_backend_major_stream_sync(reference_tensor->backing_buffer->buffer_type->backend);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_arena_reset(metal_adc->staging_arena);
}

gsx_error gsx_metal_adc_make_linear_staging_desc(
    gsx_data_type data_type,
    gsx_size_t element_count,
    gsx_size_t requested_alignment_bytes,
    gsx_tensor_desc *out_desc)
{
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }
    if(element_count == 0 || element_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "staging tensor element count must be positive and fit in int32");
    }

    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->rank = 1;
    out_desc->shape[0] = (gsx_index_t)element_count;
    out_desc->requested_alignment_bytes = requested_alignment_bytes;
    out_desc->data_type = data_type;
    out_desc->storage_format = GSX_STORAGE_FORMAT_CHW;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_prepare_staging_tensors(
    gsx_metal_adc *metal_adc,
    gsx_tensor_t reference_tensor,
    gsx_tensor_t *out_tensors,
    const gsx_tensor_desc *descs,
    gsx_index_t tensor_count)
{
    gsx_arena_mark mark = { 0 };
    gsx_arena_t sizing_arena = NULL;
    gsx_arena_desc sizing_desc = { 0 };
    gsx_tensor_t cursor_tensor = NULL;
    gsx_tensor_t *planned_tensors = NULL;
    gsx_tensor_desc cursor_desc = { 0 };
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error cleanup_error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || reference_tensor == NULL || out_tensors == NULL || descs == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc, reference_tensor, out_tensors, and descs must be non-null");
    }
    if(tensor_count <= 0 || tensor_count > GSX_TENSOR_MAX_DIM) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor_count must be in [1, GSX_TENSOR_MAX_DIM]");
    }
    if(metal_adc->staging_arena == NULL || metal_adc->staging_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal adc staging arena must be initialized before allocating staging tensors");
    }

    error = gsx_arena_get_mark(metal_adc->staging_arena, &mark);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    sizing_desc.dry_run = true;
    error = gsx_arena_init(&sizing_arena, metal_adc->staging_buffer_type, &sizing_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(mark.offset_bytes > 0) {
        error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_U8, mark.offset_bytes, 1, &cursor_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        cursor_desc.arena = sizing_arena;
        error = gsx_tensor_init(&cursor_tensor, &cursor_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }
    planned_tensors = (gsx_tensor_t *)calloc((size_t)tensor_count, sizeof(*planned_tensors));
    if(planned_tensors == NULL) {
        error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate staging plan tensor handles");
        goto cleanup;
    }
    error = gsx_tensor_init_many(planned_tensors, sizing_arena, descs, tensor_count);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_arena_get_required_bytes(sizing_arena, &required_bytes);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_free_many(planned_tensors, tensor_count);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    free(planned_tensors);
    planned_tensors = NULL;
    if(cursor_tensor != NULL) {
        error = gsx_tensor_free(cursor_tensor);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        cursor_tensor = NULL;
    }
    cleanup_error = gsx_arena_free(sizing_arena);
    if(!gsx_error_is_success(cleanup_error)) {
        return cleanup_error;
    }

    error = gsx_arena_reserve(metal_adc->staging_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_tensor_init_many(out_tensors, metal_adc->staging_arena, descs, tensor_count);

cleanup:
    free(planned_tensors);
    if(cursor_tensor != NULL) {
        cleanup_error = gsx_tensor_free(cursor_tensor);
        if(gsx_error_is_success(error)) {
            error = cleanup_error;
        }
    }
    if(sizing_arena != NULL) {
        cleanup_error = gsx_arena_free(sizing_arena);
        if(gsx_error_is_success(error)) {
            error = cleanup_error;
        }
    }
    return error;
}

gsx_error gsx_metal_adc_tensor_host_bytes(gsx_tensor_t tensor, void **out_bytes)
{
    void *base_bytes = NULL;

    if(tensor == NULL || tensor->backing_buffer == NULL || out_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and out_bytes must be non-null");
    }
    base_bytes = gsx_metal_backend_buffer_get_host_bytes(tensor->backing_buffer);
    if(base_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor storage is not host-visible");
    }

    *out_bytes = (unsigned char *)base_bytes + (size_t)tensor->offset_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_build_index_tensor(
    gsx_metal_adc *metal_adc,
    gsx_tensor_t reference_tensor,
    const int32_t *indices,
    gsx_size_t index_count,
    gsx_tensor_t *out_index_tensor)
{
    gsx_tensor_desc index_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || reference_tensor == NULL || indices == NULL || out_index_tensor == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index tensor inputs must be non-null and count must be positive");
    }

    error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_I32, index_count, sizeof(int32_t), &index_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_prepare_staging_tensors(metal_adc, reference_tensor, out_index_tensor, &index_desc, 1);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_upload(*out_index_tensor, indices, index_count * sizeof(*indices));
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(*out_index_tensor);
        *out_index_tensor = NULL;
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_apply_gs_and_optim_gather(
    gsx_metal_adc *metal_adc,
    const gsx_adc_request *request,
    const int32_t *indices,
    gsx_size_t index_count)
{
    gsx_tensor_t mean3d = NULL;
    gsx_tensor_t index_tensor = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || request == NULL || indices == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc, request, and gather inputs must be non-null and count must be positive");
    }
    error = gsx_gs_get_field(request->gs, GSX_GS_FIELD_MEAN3D, &mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_build_index_tensor(metal_adc, mean3d, indices, index_count, &index_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_gather(request->gs, index_tensor);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(index_tensor);
        return error;
    }
    if(gsx_metal_adc_optim_enabled(request)) {
        error = gsx_optim_rebind_param_groups_from_gs(request->optim, request->gs);
        if(!gsx_error_is_success(error)) {
            (void)gsx_tensor_free(index_tensor);
            return error;
        }
        error = gsx_optim_gather(request->optim, index_tensor);
        if(!gsx_error_is_success(error)) {
            (void)gsx_tensor_free(index_tensor);
            return error;
        }
    }
    (void)gsx_tensor_free(index_tensor);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_apply_gs_gather_and_rebind_optim(
    gsx_metal_adc *metal_adc,
    const gsx_adc_request *request,
    const int32_t *indices,
    gsx_size_t index_count)
{
    gsx_tensor_t mean3d = NULL;
    gsx_tensor_t index_tensor = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || request == NULL || indices == NULL || index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc, request, and gather inputs must be non-null and count must be positive");
    }
    error = gsx_gs_get_field(request->gs, GSX_GS_FIELD_MEAN3D, &mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_build_index_tensor(metal_adc, mean3d, indices, index_count, &index_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_gather(request->gs, index_tensor);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(index_tensor);
        return error;
    }
    if(gsx_metal_adc_optim_enabled(request)) {
        error = gsx_optim_rebind_param_groups_from_gs(request->optim, request->gs);
        if(!gsx_error_is_success(error)) {
            (void)gsx_tensor_free(index_tensor);
            return error;
        }
    }
    (void)gsx_tensor_free(index_tensor);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_zero_growth_optim_state(const gsx_adc_request *request, gsx_size_t old_count, gsx_size_t new_count)
{
    if(!gsx_metal_adc_optim_enabled(request) || new_count <= old_count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_metal_optim_zero_appended_rows(request->optim, old_count, new_count);
}

gsx_error gsx_metal_adc_reset_post_refine_aux(gsx_gs_t gs)
{
    return gsx_gs_zero_aux_tensors(
        gs,
        GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_ABSGRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER | GSX_GS_AUX_MAX_SCREEN_RADIUS);
}

gsx_error gsx_metal_adc_advance_rng(gsx_metal_adc *metal_adc, gsx_size_t draw_count, const char *context)
{
    if(metal_adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc must be non-null");
    }
    if(draw_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(draw_count > (gsx_size_t)INT64_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, context);
    }
    return gsx_pcg32_advance(metal_adc->rng, (gsx_pcg32_statediff_t)draw_count);
}

gsx_error gsx_metal_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc)
{
    gsx_metal_adc *metal_adc = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_adc == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_adc and desc must be non-null");
    }
    *out_adc = NULL;
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT && desc->algorithm != GSX_ADC_ALGORITHM_ABSGS
        && desc->algorithm != GSX_ADC_ALGORITHM_MCMC) {
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "metal adc currently supports only GSX_ADC_ALGORITHM_DEFAULT, GSX_ADC_ALGORITHM_ABSGS, and GSX_ADC_ALGORITHM_MCMC");
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

    if(metal_adc->staging_arena != NULL) {
        gsx_error arena_error = gsx_arena_free(metal_adc->staging_arena);

        if(!gsx_error_is_success(arena_error)) {
            return arena_error;
        }
        metal_adc->staging_arena = NULL;
        metal_adc->staging_buffer_type = NULL;
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
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc requires gs info/query support");
    }

    out_result->gaussians_before = count_before;
    out_result->gaussians_after = count_before;
    refine_window = gsx_metal_adc_in_refine_window(&adc->desc, request->global_step);
    reset_window = gsx_metal_adc_in_reset_window(&adc->desc, request->global_step);

    if(adc->desc.algorithm == GSX_ADC_ALGORITHM_MCMC) {
        error = gsx_metal_adc_apply_mcmc_noise((gsx_metal_adc *)adc, &adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(refine_window) {
        switch(adc->desc.algorithm) {
        case GSX_ADC_ALGORITHM_DEFAULT:
        case GSX_ADC_ALGORITHM_ABSGS:
            error = gsx_metal_adc_apply_default_refine((gsx_metal_adc *)adc, &adc->desc, request, out_result);
            break;
        case GSX_ADC_ALGORITHM_MCMC:
            error = gsx_metal_adc_apply_mcmc_refine((gsx_metal_adc *)adc, &adc->desc, request, out_result);
            break;
        default:
            error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc algorithm is not supported");
            break;
        }
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(reset_window) {
        error = gsx_metal_adc_apply_default_reset(&adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        out_result->reset_count = 1;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_load_count(request->gs, &count_after);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal adc requires gs info/query support");
    }
    out_result->gaussians_after = count_after;
    if(out_result->gaussians_after != out_result->gaussians_before) {
        out_result->mutated = true;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
