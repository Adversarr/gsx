#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GSX_METAL_ADC_MCMC_RELOCATION_MAX_RATIO 51u

static gsx_error gsx_metal_adc_mcmc_sample_weighted(
    gsx_metal_adc *metal_adc,
    gsx_tensor_t reference_tensor,
    const float *weights,
    const gsx_size_t *candidates,
    gsx_size_t count,
    gsx_size_t sample_count,
    gsx_size_t *out_samples)
{
    gsx_tensor_t cdf_tensor = NULL;
    gsx_tensor_t sample_tensor = NULL;
    gsx_tensor_desc staging_descs[2] = { 0 };
    gsx_tensor_t staging_tensors[2] = { NULL, NULL };
    gsx_backend_tensor_view cdf_view = { 0 };
    gsx_backend_tensor_view sample_view = { 0 };
    gsx_metal_tensor_multinomial_i32_params params = { 0 };
    float *cdf_values = NULL;
    int32_t *sample_values = NULL;
    gsx_pcg32 *rng_state = NULL;
    float total_weight = 0.0f;
    gsx_size_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || reference_tensor == NULL || weights == NULL || candidates == NULL || out_samples == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc weighted sampling inputs must be non-null");
    }
    if(count == 0 || sample_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc weighted sampling counts must be positive");
    }
    if(count > (gsx_size_t)UINT32_MAX || sample_count > (gsx_size_t)UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "mcmc weighted sampling exceeds supported uint32 range");
    }

    error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_F32, count, sizeof(float), &staging_descs[0]);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_I32, sample_count, sizeof(int32_t), &staging_descs[1]);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_prepare_staging_tensors(metal_adc, reference_tensor, staging_tensors, staging_descs, 2);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    cdf_tensor = staging_tensors[0];
    sample_tensor = staging_tensors[1];
    error = gsx_metal_adc_tensor_host_bytes(cdf_tensor, (void **)&cdf_values);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    for(index = 0; index < count; ++index) {
        total_weight += weights[index] > 0.0f ? weights[index] : 0.0f;
        cdf_values[index] = total_weight;
    }
    if(total_weight <= 0.0f) {
        error = gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc weighted sampling requires positive opacity mass");
        goto cleanup;
    }

    gsx_metal_adc_make_tensor_view(cdf_tensor, &cdf_view);
    gsx_metal_adc_make_tensor_view(sample_tensor, &sample_view);
    rng_state = (gsx_pcg32 *)metal_adc->rng;
    params.rng_state = rng_state->state;
    params.rng_inc = rng_state->inc;
    params.sample_count = (uint32_t)sample_count;
    params.category_count = (uint32_t)count;
    error = gsx_metal_backend_dispatch_tensor_multinomial_i32(
        reference_tensor->backing_buffer->buffer_type->backend,
        &cdf_view,
        &sample_view,
        &params);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_metal_adc_advance_rng(metal_adc, sample_count, "adc mcmc multinomial rng advance exceeds supported range");
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_backend_major_stream_sync(reference_tensor->backing_buffer->buffer_type->backend);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_metal_adc_tensor_host_bytes(sample_tensor, (void **)&sample_values);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    for(index = 0; index < sample_count; ++index) {
        int32_t sampled_index = sample_values[index];

        if(sampled_index < 0 || (gsx_size_t)sampled_index >= count) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc multinomial produced an out-of-range sample index");
            goto cleanup;
        }
        out_samples[index] = candidates[sampled_index];
    }

cleanup:
    if(sample_tensor != NULL) {
        (void)gsx_tensor_free(sample_tensor);
    }
    if(cdf_tensor != NULL) {
        (void)gsx_tensor_free(cdf_tensor);
    }
    return error;
}

static gsx_error gsx_metal_adc_mcmc_dispatch_relocation(
    gsx_metal_adc *metal_adc,
    const gsx_metal_adc_refine_data *refine_data,
    const gsx_size_t *sample_counts,
    gsx_size_t count,
    float min_opacity)
{
    gsx_tensor_t counts_tensor = NULL;
    gsx_tensor_desc counts_desc = { 0 };
    gsx_backend_tensor_view counts_view = { 0 };
    gsx_metal_adc_mcmc_relocation_params params = { 0 };
    uint32_t *count_values = NULL;
    gsx_pcg32 *rng_state = NULL;
    gsx_size_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || refine_data == NULL || sample_counts == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc relocation inputs must be non-null");
    }
    if(count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_I32, count, sizeof(int32_t), &counts_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_prepare_staging_tensors(metal_adc, refine_data->mean3d_tensor, &counts_tensor, &counts_desc, 1);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_tensor_host_bytes(counts_tensor, (void **)&count_values);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(counts_tensor);
        return error;
    }
    for(index = 0; index < count; ++index) {
        count_values[index] = (uint32_t)sample_counts[index];
    }
    gsx_metal_adc_make_tensor_view(counts_tensor, &counts_view);

    params.gaussian_count = (uint32_t)count;
    params.min_opacity = min_opacity;
    rng_state = (gsx_pcg32 *)metal_adc->rng;
    params.rng_state = rng_state->state;
    params.rng_inc = rng_state->inc;
    error = gsx_metal_backend_dispatch_adc_mcmc_relocation(
        refine_data->mean3d_tensor->backing_buffer->buffer_type->backend,
        &refine_data->logscale_view,
        &refine_data->opacity_view,
        &counts_view,
        &params);
    (void)gsx_tensor_free(counts_tensor);
    return error;
}

static gsx_error gsx_metal_adc_mcmc_stage_opacity_values(
    gsx_metal_adc *metal_adc,
    gsx_tensor_t opacity_tensor,
    gsx_tensor_t *out_staging_tensor,
    float **out_values)
{
    gsx_tensor_desc opacity_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || opacity_tensor == NULL || out_staging_tensor == NULL || out_values == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_adc, opacity_tensor, out_staging_tensor, and out_values must be non-null");
    }

    error = gsx_tensor_get_desc(opacity_tensor, &opacity_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_prepare_staging_tensors(metal_adc, opacity_tensor, out_staging_tensor, &opacity_desc, 1);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_copy(opacity_tensor, *out_staging_tensor);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(*out_staging_tensor);
        *out_staging_tensor = NULL;
        return error;
    }
    error = gsx_backend_major_stream_sync(opacity_tensor->backing_buffer->buffer_type->backend);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(*out_staging_tensor);
        *out_staging_tensor = NULL;
        return error;
    }
    error = gsx_metal_adc_tensor_host_bytes(*out_staging_tensor, (void **)out_values);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(*out_staging_tensor);
        *out_staging_tensor = NULL;
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_adc_apply_mcmc_noise(
    gsx_metal_adc *metal_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request)
{
    gsx_metal_adc_refine_data refine_data = { 0 };
    gsx_size_t count = 0;
    gsx_size_t stop_iter = 0;
    gsx_float_t lr_mean3d = 0.0f;
    gsx_metal_adc_mcmc_noise_params params = { 0 };
    gsx_pcg32 *rng_state = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || desc == NULL || request == NULL || desc->noise_strength <= 0.0f) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    stop_iter = gsx_metal_adc_non_negative_index(desc->end_refine);
    if(stop_iter > 0u && request->global_step >= stop_iter) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_adc_load_count(request->gs, &count);
    if(!gsx_error_is_success(error) || count == 0) {
        return error;
    }
    error = gsx_metal_adc_load_refine_data(request->gs, count, false, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_optim_get_learning_rate_by_role(request->optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &lr_mean3d);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(&refine_data);
        return error;
    }

    rng_state = (gsx_pcg32 *)metal_adc->rng;
    params.gaussian_count = (uint32_t)count;
    params.noise_strength = desc->noise_strength;
    params.learning_rate = lr_mean3d;
    params.rng_state = rng_state->state;
    params.rng_inc = rng_state->inc;
    error = gsx_metal_backend_dispatch_adc_mcmc_noise(
        refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
        &refine_data.mean3d_view,
        &refine_data.logscale_view,
        &refine_data.opacity_view,
        &refine_data.rotation_view,
        &params);
    if(gsx_error_is_success(error)) {
        error = gsx_metal_adc_advance_rng(metal_adc, count * 6u, "adc mcmc noise rng advance exceeds supported range");
    }
    gsx_metal_adc_free_refine_data(&refine_data);
    return error;
}

gsx_error gsx_metal_adc_apply_mcmc_refine(
    gsx_metal_adc *metal_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result)
{
    gsx_metal_adc_refine_data refine_data = { 0 };
    gsx_size_t count_before = 0;
    gsx_size_t count_after_growth = 0;
    gsx_size_t dead_count = 0;
    gsx_size_t target_count = 0;
    gsx_size_t target_growth = 0;
    gsx_size_t index = 0;
    gsx_tensor_t dead_mask_tensor = NULL;
    int32_t *gather_indices = NULL;
    gsx_size_t *dead_indices = NULL;
    gsx_size_t *live_candidates = NULL;
    gsx_size_t *sampled_sources = NULL;
    gsx_size_t *sample_counts = NULL;
    float *sample_weights = NULL;
    gsx_metal_adc_mcmc_dead_mask_params dead_mask_params = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc refine inputs must be non-null");
    }

    error = gsx_metal_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error) || count_before == 0) {
        return error;
    }
    error = gsx_metal_adc_load_refine_data(request->gs, count_before, false, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_adc_begin_staging_cycle(metal_adc, refine_data.mean3d_tensor);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(&refine_data);
        return error;
    }

    dead_indices = (gsx_size_t *)malloc(sizeof(*dead_indices) * count_before);
    sample_counts = (gsx_size_t *)calloc(count_before, sizeof(*sample_counts));
    if(dead_indices == NULL || sample_counts == NULL) {
        error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc dead-mask buffer");
        goto cleanup;
    }
    {
        gsx_tensor_desc dead_mask_desc = { 0 };
        gsx_backend_tensor_view dead_mask_view = { 0 };
        uint8_t *dead_mask = NULL;

        error = gsx_metal_adc_make_linear_staging_desc(GSX_DATA_TYPE_U8, count_before, sizeof(uint8_t), &dead_mask_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_prepare_staging_tensors(metal_adc, refine_data.mean3d_tensor, &dead_mask_tensor, &dead_mask_desc, 1);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        gsx_metal_adc_make_tensor_view(dead_mask_tensor, &dead_mask_view);
        dead_mask_params.gaussian_count = (uint32_t)count_before;
        dead_mask_params.pruning_opacity_threshold = desc->pruning_opacity_threshold;
        error = gsx_metal_backend_dispatch_adc_mcmc_dead_mask(
            refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
            &refine_data.opacity_view,
            &dead_mask_view,
            &dead_mask_params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_backend_major_stream_sync(refine_data.mean3d_tensor->backing_buffer->buffer_type->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_tensor_host_bytes(dead_mask_tensor, (void **)&dead_mask);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        for(index = 0; index < count_before; ++index) {
            if(dead_mask[index] != 0u) {
                dead_indices[dead_count] = index;
                dead_count += 1u;
            }
        }
    }
    out_result->pruned_count = dead_count;
    (void)gsx_tensor_free(dead_mask_tensor);
    dead_mask_tensor = NULL;

    if(dead_count > 0 && dead_count < count_before) {
        gsx_size_t live_count = count_before - dead_count;
        gsx_size_t live_write = 0;
        gsx_size_t dead_read = 0;
        gsx_tensor_t opacity_staging_tensor = NULL;
        float *opacity_values = NULL;

        live_candidates = (gsx_size_t *)malloc(sizeof(*live_candidates) * live_count);
        sample_weights = (float *)malloc(sizeof(*sample_weights) * live_count);
        sampled_sources = (gsx_size_t *)malloc(sizeof(*sampled_sources) * dead_count);
        gather_indices = (int32_t *)malloc(sizeof(*gather_indices) * count_before);
        if(live_candidates == NULL || sample_weights == NULL || sampled_sources == NULL || gather_indices == NULL) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc relocation buffers");
            goto cleanup;
        }
        error = gsx_metal_adc_mcmc_stage_opacity_values(metal_adc, refine_data.opacity_tensor, &opacity_staging_tensor, &opacity_values);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        for(index = 0; index < count_before; ++index) {
            bool is_dead = dead_read < dead_count && dead_indices[dead_read] == index;

            gather_indices[index] = (int32_t)index;
            if(!is_dead) {
                live_candidates[live_write] = index;
                sample_weights[live_write] = gsx_sigmoid(opacity_values[index]);
                live_write += 1u;
            } else {
                dead_read += 1u;
            }
        }
        (void)gsx_tensor_free(opacity_staging_tensor);
        opacity_staging_tensor = NULL;
        error = gsx_metal_adc_mcmc_sample_weighted(
            metal_adc,
            refine_data.mean3d_tensor,
            sample_weights,
            live_candidates,
            live_count,
            dead_count,
            sampled_sources);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        for(index = 0; index < dead_count; ++index) {
            sample_counts[sampled_sources[index]] += 1u;
            gather_indices[dead_indices[index]] = (int32_t)sampled_sources[index];
        }
        error = gsx_metal_adc_mcmc_dispatch_relocation(metal_adc, &refine_data, sample_counts, count_before, desc->pruning_opacity_threshold);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_adc_apply_gs_gather_and_rebind_optim(metal_adc, request, gather_indices, count_before);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(gsx_metal_adc_optim_enabled(request)) {
            error = gsx_metal_optim_zero_rows(request->optim, sampled_sources, dead_count, count_before);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }
        memset(sample_counts, 0, sizeof(*sample_counts) * count_before);
        out_result->mutated = true;
        out_result->grown_count += dead_count;
        free(sampled_sources);
        sampled_sources = NULL;
        free(sample_weights);
        sample_weights = NULL;
        free(live_candidates);
        live_candidates = NULL;
        free(gather_indices);
        gather_indices = NULL;
        error = gsx_metal_adc_load_refine_data(request->gs, count_before, false, &refine_data);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    } else if(dead_count == count_before) {
        error = gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc relocate requires at least one live gaussian");
        goto cleanup;
    }

    if(desc->max_num_gaussians > 0 && desc->grow_ratio > 0.0f) {
        double scaled_target = floor((double)count_before * (1.0 + (double)desc->grow_ratio));

        if(scaled_target > (double)desc->max_num_gaussians) {
            scaled_target = (double)desc->max_num_gaussians;
        }
        if(scaled_target > (double)count_before) {
            target_count = (gsx_size_t)scaled_target;
            target_growth = target_count - count_before;
        }
    }

    if(target_growth > 0) {
        gsx_size_t *all_candidates = NULL;
        float *all_weights = NULL;
        gsx_size_t gathered_count = count_before + target_growth;
        gsx_tensor_t opacity_staging_tensor = NULL;
        float *opacity_values = NULL;

        all_candidates = (gsx_size_t *)malloc(sizeof(*all_candidates) * count_before);
        all_weights = (float *)malloc(sizeof(*all_weights) * count_before);
        sampled_sources = (gsx_size_t *)malloc(sizeof(*sampled_sources) * target_growth);
        gather_indices = (int32_t *)malloc(sizeof(*gather_indices) * gathered_count);
        if(all_candidates == NULL || all_weights == NULL || sampled_sources == NULL || gather_indices == NULL) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc growth buffers");
            goto cleanup;
        }
        error = gsx_metal_adc_mcmc_stage_opacity_values(metal_adc, refine_data.opacity_tensor, &opacity_staging_tensor, &opacity_values);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            goto cleanup;
        }
        for(index = 0; index < count_before; ++index) {
            all_candidates[index] = index;
            all_weights[index] = gsx_sigmoid(opacity_values[index]);
        }
        (void)gsx_tensor_free(opacity_staging_tensor);
        opacity_staging_tensor = NULL;
        error = gsx_metal_adc_mcmc_sample_weighted(
            metal_adc,
            refine_data.mean3d_tensor,
            all_weights,
            all_candidates,
            count_before,
            target_growth,
            sampled_sources);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            goto cleanup;
        }
        for(index = 0; index < target_growth; ++index) {
            sample_counts[sampled_sources[index]] += 1u;
        }
        error = gsx_metal_adc_mcmc_dispatch_relocation(metal_adc, &refine_data, sample_counts, count_before, desc->pruning_opacity_threshold);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            goto cleanup;
        }
        for(index = 0; index < count_before; ++index) {
            gather_indices[index] = (int32_t)index;
        }
        for(index = 0; index < target_growth; ++index) {
            gather_indices[count_before + index] = (int32_t)sampled_sources[index];
        }
        error = gsx_metal_adc_apply_gs_and_optim_gather(metal_adc, request, gather_indices, gathered_count);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            goto cleanup;
        }
        error = gsx_metal_adc_load_count(request->gs, &count_after_growth);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            goto cleanup;
        }
        if(count_after_growth != gathered_count) {
            free(gather_indices);
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "metal mcmc adc growth produced unexpected gaussian count");
            goto cleanup;
        }
        error = gsx_metal_adc_zero_growth_optim_state(request, count_before, count_after_growth);
        free(gather_indices);
        gather_indices = NULL;
        free(sampled_sources);
        sampled_sources = NULL;
        free(all_weights);
        free(all_candidates);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        out_result->grown_count += target_growth;
        out_result->mutated = true;
    }

    error = gsx_metal_adc_reset_post_refine_aux(request->gs);

cleanup:
    if(dead_mask_tensor != NULL) {
        (void)gsx_tensor_free(dead_mask_tensor);
    }
    free(gather_indices);
    free(dead_indices);
    free(live_candidates);
    free(sampled_sources);
    free(sample_counts);
    free(sample_weights);
    gsx_metal_adc_free_refine_data(&refine_data);
    return error;
}
