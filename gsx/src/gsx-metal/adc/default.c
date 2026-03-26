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
    gsx_tensor_t mode_tensor = NULL;
    gsx_tensor_t split_source_tensor = NULL;
    gsx_tensor_t split_target_tensor = NULL;
    gsx_tensor_t keep_mask_tensor = NULL;
    int32_t *gather_indices = NULL;
    int32_t *split_sources = NULL;
    int32_t *split_targets = NULL;
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
    error = gsx_metal_adc_load_refine_data(
        request->gs,
        count_before_refine,
        desc->algorithm != GSX_ADC_ALGORITHM_ABSGS,
        &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(desc->algorithm == GSX_ADC_ALGORITHM_ABSGS && refine_data.absgrad_acc_tensor == NULL) {
        gsx_metal_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal absgs adc refine requires GSX_GS_FIELD_ABSGRAD_ACC auxiliary field");
    }
    error = gsx_metal_adc_begin_staging_cycle(metal_adc, refine_data.mean3d_tensor);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(&refine_data);
        return error;
    }

    max_capacity = gsx_metal_adc_non_negative_index(desc->max_num_gaussians);
    if(max_capacity > count_before_refine) {
        grow_budget = max_capacity - count_before_refine;
    }

    if(grow_budget > 0) {
        gsx_tensor_desc mode_desc = { 0 };
        gsx_backend_tensor_view mode_view = { 0 };
        uint8_t *modes = NULL;
        gsx_metal_adc_classify_growth_params params = { 0 };
        const gsx_backend_tensor_view *growth_grad_view = NULL;

        error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_U8, count_before_refine, sizeof(uint8_t), &mode_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_prepare_staging_tensors(metal_adc, refine_data.mean3d_tensor, &mode_tensor, &mode_desc, 1);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        gsx_metal_adc_make_tensor_view(mode_tensor, &mode_view);

        params.gaussian_count = (uint32_t)count_before_refine;
        params.has_visible_counter = refine_data.has_visible_counter ? 1u : 0u;
        if(desc->algorithm == GSX_ADC_ALGORITHM_ABSGS) {
            growth_grad_view = &refine_data.absgrad_acc_view;
            params.growth_grad_threshold = desc->duplicate_absgrad_threshold;
        } else {
            if(refine_data.grad_acc_tensor == NULL) {
                error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal default adc refine requires GSX_GS_FIELD_GRAD_ACC auxiliary field");
                goto cleanup;
            }
            growth_grad_view = &refine_data.grad_acc_view;
            params.growth_grad_threshold = desc->duplicate_grad_threshold;
        }
        params.duplicate_scale_threshold = desc->duplicate_scale_threshold;
        params.scene_scale = request->scene_scale;
        error = gsx_metal_backend_dispatch_adc_classify_growth(
            refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
            growth_grad_view,
            refine_data.has_visible_counter ? &refine_data.visible_counter_view : NULL,
            &refine_data.logscale_view,
            &mode_view,
            &params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_backend_major_stream_sync(refine_data.mean3d_tensor->backing_buffer->buffer_type->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_tensor_host_bytes(mode_tensor, (void **)&modes);
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
        {
            uint8_t *modes = NULL;

            error = gsx_metal_adc_tensor_host_bytes(mode_tensor, (void **)&modes);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            for(index = 0; index < count_before_refine && grow_index < grow_count; ++index) {
                uint8_t mode_value = modes[index];

                if(mode_value == (uint8_t)GSX_METAL_ADC_GROW_NONE) {
                    continue;
                }
                gather_indices[count_before_refine + grow_index] = (int32_t)index;
                if(mode_value == (uint8_t)GSX_METAL_ADC_GROW_SPLIT) {
                    split_sources[split_index] = (int32_t)index;
                    split_targets[split_index] = (int32_t)(count_before_refine + grow_index);
                    split_index += 1;
                }
                grow_index += 1;
            }
        }
        (void)gsx_tensor_free(mode_tensor);
        mode_tensor = NULL;

        error = gsx_metal_adc_apply_gs_and_optim_gather(metal_adc, request, gather_indices, gathered_count);
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
        error = gsx_metal_adc_load_refine_data(
            request->gs,
            count_after_growth,
            desc->algorithm != GSX_ADC_ALGORITHM_ABSGS,
            &refine_data);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        if(split_count > 0) {
            gsx_tensor_desc split_descs[2] = { 0 };
            gsx_tensor_t split_tensors[2] = { NULL, NULL };
            gsx_backend_tensor_view split_source_view = { 0 };
            gsx_backend_tensor_view split_target_view = { 0 };

            error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_I32, split_count, sizeof(int32_t), &split_descs[0]);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            split_descs[1] = split_descs[0];
            error = gsx_metal_adc_prepare_staging_tensors(metal_adc, refine_data.mean3d_tensor, split_tensors, split_descs, 2);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            split_source_tensor = split_tensors[0];
            split_target_tensor = split_tensors[1];
            error = gsx_tensor_upload(split_source_tensor, split_sources, split_count * sizeof(int32_t));
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_upload(split_target_tensor, split_targets, split_count * sizeof(int32_t));
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            gsx_metal_adc_make_tensor_view(split_source_tensor, &split_source_view);
            gsx_metal_adc_make_tensor_view(split_target_tensor, &split_target_view);

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
                &split_source_view,
                &split_target_view,
                &split_params);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_adc_advance_rng(metal_adc, split_count * 12u, "adc split rng advance exceeds supported range");
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            (void)gsx_tensor_free(split_source_tensor);
            split_source_tensor = NULL;
            (void)gsx_tensor_free(split_target_tensor);
            split_target_tensor = NULL;
        }

        out_result->duplicated_count += duplicate_count;
        out_result->grown_count += split_count;
        out_result->mutated = true;
    }
    if(mode_tensor != NULL) {
        (void)gsx_tensor_free(mode_tensor);
        mode_tensor = NULL;
    }

    error = gsx_metal_adc_load_count(request->gs, &count_after_growth);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(count_after_growth == 0) {
        error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        goto cleanup;
    }
    error = gsx_metal_adc_load_refine_data(
        request->gs,
        count_after_growth,
        desc->algorithm != GSX_ADC_ALGORITHM_ABSGS,
        &refine_data);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    {
        gsx_tensor_desc keep_mask_desc = { 0 };
        gsx_backend_tensor_view keep_mask_view = { 0 };
        uint8_t *keep_mask = NULL;
        gsx_metal_adc_keep_mask_params keep_params = { 0 };

        error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_U8, count_after_growth, sizeof(uint8_t), &keep_mask_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        keep_indices = (int32_t *)malloc((size_t)(count_after_growth * sizeof(int32_t)));
        if(keep_indices == NULL) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate prune index staging");
            goto cleanup;
        }
        error = gsx_metal_adc_prepare_staging_tensors(metal_adc, refine_data.mean3d_tensor, &keep_mask_tensor, &keep_mask_desc, 1);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        gsx_metal_adc_make_tensor_view(keep_mask_tensor, &keep_mask_view);

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
            &keep_mask_view,
            &keep_params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_backend_major_stream_sync(refine_data.mean3d_tensor->backing_buffer->buffer_type->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_tensor_host_bytes(keep_mask_tensor, (void **)&keep_mask);
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
        (void)gsx_tensor_free(keep_mask_tensor);
        keep_mask_tensor = NULL;
    }

    if(keep_count < count_after_growth) {
        error = gsx_metal_adc_apply_gs_and_optim_gather(metal_adc, request, keep_indices, keep_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        out_result->pruned_count += count_after_growth - keep_count;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_reset_post_refine_aux(request->gs);

cleanup:
    if(mode_tensor != NULL) {
        (void)gsx_tensor_free(mode_tensor);
    }
    if(split_source_tensor != NULL) {
        (void)gsx_tensor_free(split_source_tensor);
    }
    if(split_target_tensor != NULL) {
        (void)gsx_tensor_free(split_target_tensor);
    }
    if(keep_mask_tensor != NULL) {
        (void)gsx_tensor_free(keep_mask_tensor);
    }
    free(gather_indices);
    free(split_sources);
    free(split_targets);
    free(keep_indices);
    gsx_metal_adc_free_refine_data(&refine_data);
    return error;
}
