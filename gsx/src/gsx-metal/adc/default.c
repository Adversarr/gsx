#include "internal.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

gsx_error gsx_metal_adc_apply_default_reset(const gsx_adc_desc *desc, const gsx_adc_request *request)
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

gsx_error gsx_metal_adc_apply_default_refine(
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
    error = gsx_metal_adc_load_refine_data(request->gs, count_before_refine, true, &refine_data);
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
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_backend_major_stream_sync(refine_data.mean3d_tensor->backing_buffer->buffer_type->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

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
        error = gsx_metal_adc_load_refine_data(request->gs, count_after_growth, true, &refine_data);
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
            error = gsx_metal_adc_advance_rng(metal_adc, split_count * 12u, "adc split rng advance exceeds supported range");
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
    error = gsx_metal_adc_load_refine_data(request->gs, count_after_growth, true, &refine_data);
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
