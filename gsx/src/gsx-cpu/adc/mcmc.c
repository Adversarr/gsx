#include "internal.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GSX_CPU_ADC_MCMC_RELOCATION_MAX_RATIO 51u
#define GSX_CPU_ADC_MCMC_NOISE_GATE_K 100.0f
#define GSX_CPU_ADC_MCMC_NOISE_GATE_X0 0.995f

static float gsx_cpu_adc_mcmc_noise_gate(float opacity)
{
    float shifted = (1.0f - opacity) - GSX_CPU_ADC_MCMC_NOISE_GATE_X0;

    return 1.0f / (1.0f + expf(-GSX_CPU_ADC_MCMC_NOISE_GATE_K * shifted));
}

static float gsx_cpu_adc_mcmc_binom(gsx_size_t n, gsx_size_t k)
{
    gsx_size_t index = 0;
    double value = 1.0;

    if(k > n) {
        return 0.0f;
    }
    if(k == 0 || k == n) {
        return 1.0f;
    }
    if(k > n - k) {
        k = n - k;
    }
    for(index = 1; index <= k; ++index) {
        value = value * (double)(n - (k - index)) / (double)index;
    }
    return (float)value;
}

static gsx_size_t gsx_cpu_adc_mcmc_clamp_ratio(gsx_size_t ratio)
{
    if(ratio < 1) {
        return 1;
    }
    if(ratio > GSX_CPU_ADC_MCMC_RELOCATION_MAX_RATIO) {
        return GSX_CPU_ADC_MCMC_RELOCATION_MAX_RATIO;
    }
    return ratio;
}

static float gsx_cpu_adc_mcmc_relocated_opacity(float opacity, gsx_size_t ratio)
{
    float clamped = gsx_cpu_adc_clamp_probability(opacity);
    float root = 1.0f / (float)gsx_cpu_adc_mcmc_clamp_ratio(ratio);

    return 1.0f - powf(1.0f - clamped, root);
}

static float gsx_cpu_adc_mcmc_scale_coeff(float opacity, float new_opacity, gsx_size_t ratio)
{
    float denom_sum = 0.0f;
    gsx_size_t i = 0;

    for(i = 1; i <= gsx_cpu_adc_mcmc_clamp_ratio(ratio); ++i) {
        gsx_size_t k = 0;

        for(k = 0; k < i; ++k) {
            float sign = (k % 2u) == 0u ? 1.0f : -1.0f;
            float coeff = gsx_cpu_adc_mcmc_binom(i - 1u, k);
            float power = powf(new_opacity, (float)(k + 1u));

            denom_sum += coeff * sign * power / sqrtf((float)(k + 1u));
        }
    }
    if(fabsf(denom_sum) <= 1e-8f) {
        return 1.0f;
    }
    return opacity / denom_sum;
}

static gsx_error gsx_cpu_adc_mcmc_sample_weighted(
    gsx_pcg32_t rng,
    const float *weights,
    const gsx_size_t *candidates,
    gsx_size_t count,
    gsx_size_t sample_count,
    gsx_size_t *out_samples
)
{
    float total_weight = 0.0f;
    gsx_size_t index = 0;
    gsx_size_t sample_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(weights == NULL || candidates == NULL || out_samples == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc weighted sampling inputs must be non-null");
    }
    if(count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc candidate count must be positive");
    }

    for(index = 0; index < count; ++index) {
        if(weights[index] > 0.0f) {
            total_weight += weights[index];
        }
    }
    if(total_weight <= 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc weighted sampling requires positive opacity mass");
    }

    for(sample_index = 0; sample_index < sample_count; ++sample_index) {
        float draw = 0.0f;
        float cumulative = 0.0f;

        error = gsx_pcg32_next_float(rng, &draw);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        draw *= total_weight;
        for(index = 0; index < count; ++index) {
            float weight = weights[index] > 0.0f ? weights[index] : 0.0f;

            cumulative += weight;
            if(draw <= cumulative || index + 1u == count) {
                out_samples[sample_index] = candidates[index];
                break;
            }
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_mcmc_copy_row_fields(
    gsx_cpu_adc_refine_data *data,
    gsx_size_t dst,
    gsx_size_t src
)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cpu_adc_copy_slice(data->mean3d, dst, data->mean3d, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_slice(data->logscale, dst, data->logscale, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_slice(data->rotation, dst, data->rotation, src, 4);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_slice(data->sh0, dst, data->sh0, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_optional_slice(data->sh1, dst, data->sh1, src, 9);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_optional_slice(data->sh2, dst, data->sh2, src, 15);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_optional_slice(data->sh3, dst, data->sh3, src, 21);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    data->opacity[dst] = data->opacity[src];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_mcmc_apply_relocation_updates(
    gsx_cpu_adc_refine_data *data,
    const gsx_size_t *sampled_indices,
    const gsx_size_t *sample_counts,
    const gsx_size_t *dead_indices,
    gsx_size_t dead_count,
    gsx_size_t count,
    float min_opacity
)
{
    float *source_opacity = NULL;
    float *source_scale = NULL;
    gsx_size_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    source_opacity = (float *)malloc(sizeof(*source_opacity) * count);
    source_scale = (float *)malloc(sizeof(*source_scale) * count * 3u);
    if(source_opacity == NULL || source_scale == NULL) {
        free(source_opacity);
        free(source_scale);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc relocation snapshot buffers");
    }
    for(index = 0; index < count; ++index) {
        source_opacity[index] = gsx_sigmoid(data->opacity[index]);
        source_scale[index * 3u + 0u] = gsx_expf(data->logscale[index * 3u + 0u]);
        source_scale[index * 3u + 1u] = gsx_expf(data->logscale[index * 3u + 1u]);
        source_scale[index * 3u + 2u] = gsx_expf(data->logscale[index * 3u + 2u]);
    }

    for(index = 0; index < count; ++index) {
        if(sample_counts[index] > 0u) {
            gsx_size_t ratio = sample_counts[index] + 1u;
            float new_opacity = gsx_cpu_adc_mcmc_relocated_opacity(source_opacity[index], ratio);
            float scale_coeff = gsx_cpu_adc_mcmc_scale_coeff(source_opacity[index], new_opacity, ratio);

            if(new_opacity < min_opacity) {
                new_opacity = min_opacity;
            }
            if(new_opacity > 1.0f - 1e-6f) {
                new_opacity = 1.0f - 1e-6f;
            }
            data->opacity[index] = gsx_cpu_adc_probability_to_logit(new_opacity);
            data->logscale[index * 3u + 0u] = gsx_logf(source_scale[index * 3u + 0u] * scale_coeff);
            data->logscale[index * 3u + 1u] = gsx_logf(source_scale[index * 3u + 1u] * scale_coeff);
            data->logscale[index * 3u + 2u] = gsx_logf(source_scale[index * 3u + 2u] * scale_coeff);
        }
    }

    for(index = 0; index < dead_count; ++index) {
        error = gsx_cpu_adc_mcmc_copy_row_fields(data, dead_indices[index], sampled_indices[index]);
        if(!gsx_error_is_success(error)) {
            free(source_opacity);
            free(source_scale);
            return error;
        }
    }

    free(source_opacity);
    free(source_scale);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cpu_adc_apply_mcmc_noise(
    gsx_cpu_adc *cpu_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    bool *out_mutated
)
{
    gsx_cpu_adc_refine_data refine_data = { 0 };
    gsx_size_t count = 0;
    gsx_size_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t stop_iter = 0;

    if(out_mutated == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_mutated must be non-null");
    }
    *out_mutated = false;
    if(cpu_adc == NULL || desc == NULL || request == NULL || desc->noise_strength <= 0.0f) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    stop_iter = gsx_cpu_adc_non_negative_index(desc->end_refine);
    if(stop_iter > 0u && request->global_step >= stop_iter) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_cpu_adc_load_count(request->gs, &count);
    if(!gsx_error_is_success(error) || count == 0) {
        return error;
    }
    error = gsx_cpu_adc_load_refine_data(request->gs, count, false, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    gsx_float_t lr_mean3d = 0.0f;
    error = gsx_optim_get_learning_rate_by_role(request->optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &lr_mean3d);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(&refine_data);
        return error;
    }

    for(index = 0; index < count; ++index) {
        float opacity = gsx_sigmoid(refine_data.opacity[index]);
        float gate = gsx_cpu_adc_mcmc_noise_gate(opacity);
        float scale_x = gsx_expf(refine_data.logscale[index * 3 + 0]);
        float scale_y = gsx_expf(refine_data.logscale[index * 3 + 1]);
        float scale_z = gsx_expf(refine_data.logscale[index * 3 + 2]);
        float qx = refine_data.rotation[index * 4 + 0];
        float qy = refine_data.rotation[index * 4 + 1];
        float qz = refine_data.rotation[index * 4 + 2];
        float qw = refine_data.rotation[index * 4 + 3];
        float m00 = 1.0f;
        float m01 = 0.0f;
        float m02 = 0.0f;
        float m10 = 0.0f;
        float m11 = 1.0f;
        float m12 = 0.0f;
        float m20 = 0.0f;
        float m21 = 0.0f;
        float m22 = 1.0f;
        float c00 = 0.0f;
        float c01 = 0.0f;
        float c02 = 0.0f;
        float c10 = 0.0f;
        float c11 = 0.0f;
        float c12 = 0.0f;
        float c20 = 0.0f;
        float c21 = 0.0f;
        float c22 = 0.0f;
        float noise_x = 0.0f;
        float noise_y = 0.0f;
        float noise_z = 0.0f;
        float scaled_noise_x = 0.0f;
        float scaled_noise_y = 0.0f;
        float scaled_noise_z = 0.0f;

        error = gsx_cpu_adc_sample_normal(cpu_adc->rng, &noise_x);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        error = gsx_cpu_adc_sample_normal(cpu_adc->rng, &noise_y);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        error = gsx_cpu_adc_sample_normal(cpu_adc->rng, &noise_z);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        gsx_cpu_adc_normalize_quaternion(&qx, &qy, &qz, &qw);
        gsx_cpu_adc_build_rotation_matrix(qx, qy, qz, qw, &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22);
        c00 = m00 * m00 * scale_x * scale_x + m01 * m01 * scale_y * scale_y + m02 * m02 * scale_z * scale_z;
        c01 = m00 * m10 * scale_x * scale_x + m01 * m11 * scale_y * scale_y + m02 * m12 * scale_z * scale_z;
        c02 = m00 * m20 * scale_x * scale_x + m01 * m21 * scale_y * scale_y + m02 * m22 * scale_z * scale_z;
        c10 = c01;
        c11 = m10 * m10 * scale_x * scale_x + m11 * m11 * scale_y * scale_y + m12 * m12 * scale_z * scale_z;
        c12 = m10 * m20 * scale_x * scale_x + m11 * m21 * scale_y * scale_y + m12 * m22 * scale_z * scale_z;
        c20 = c02;
        c21 = c12;
        c22 = m20 * m20 * scale_x * scale_x + m21 * m21 * scale_y * scale_y + m22 * m22 * scale_z * scale_z;
        scaled_noise_x = noise_x * gate * (desc->noise_strength * lr_mean3d);
        scaled_noise_y = noise_y * gate * (desc->noise_strength * lr_mean3d);
        scaled_noise_z = noise_z * gate * (desc->noise_strength * lr_mean3d);
        refine_data.mean3d[index * 3 + 0] += c00 * scaled_noise_x + c01 * scaled_noise_y + c02 * scaled_noise_z;
        refine_data.mean3d[index * 3 + 1] += c10 * scaled_noise_x + c11 * scaled_noise_y + c12 * scaled_noise_z;
        refine_data.mean3d[index * 3 + 2] += c20 * scaled_noise_x + c21 * scaled_noise_y + c22 * scaled_noise_z;
    }

    *out_mutated = true;
    gsx_cpu_adc_free_refine_data(&refine_data);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cpu_adc_apply_mcmc_refine(
    gsx_cpu_adc *cpu_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
)
{
    gsx_cpu_adc_refine_data refine_data = { 0 };
    gsx_size_t count_before = 0;
    gsx_size_t count_after_growth = 0;
    gsx_size_t dead_count = 0;
    gsx_size_t target_count = 0;
    gsx_size_t target_growth = 0;
    gsx_size_t index = 0;
    bool *dead_mask = NULL;
    int32_t *gather_indices = NULL;
    gsx_size_t *dead_indices = NULL;
    gsx_size_t *live_candidates = NULL;
    gsx_size_t *sampled_sources = NULL;
    gsx_size_t *sample_counts = NULL;
    float *sample_weights = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_adc == NULL || desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc refine inputs must be non-null");
    }

    error = gsx_cpu_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error) || count_before == 0) {
        return error;
    }
    error = gsx_cpu_adc_load_refine_data(request->gs, count_before, false, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    dead_mask = (bool *)calloc(count_before, sizeof(*dead_mask));
    dead_indices = (gsx_size_t *)malloc(sizeof(*dead_indices) * count_before);
    sample_counts = (gsx_size_t *)calloc(count_before, sizeof(*sample_counts));
    if(dead_mask == NULL || dead_indices == NULL || sample_counts == NULL) {
        free(sample_counts);
        free(dead_indices);
        gsx_cpu_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc dead-mask buffer");
    }

    for(index = 0; index < count_before; ++index) {
        if(gsx_sigmoid(refine_data.opacity[index]) <= desc->pruning_opacity_threshold) {
            dead_mask[index] = true;
            dead_indices[dead_count] = index;
            dead_count += 1;
        }
    }

    if(dead_count > 0 && dead_count < count_before) {
        gsx_size_t live_count = count_before - dead_count;
        gsx_size_t live_write = 0;

        live_candidates = (gsx_size_t *)malloc(sizeof(*live_candidates) * live_count);
        sample_weights = (float *)malloc(sizeof(*sample_weights) * live_count);
        sampled_sources = (gsx_size_t *)malloc(sizeof(*sampled_sources) * dead_count);
        if(live_candidates == NULL || sample_weights == NULL || sampled_sources == NULL) {
            free(sampled_sources);
            free(sample_weights);
            free(live_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc relocation buffers");
        }
        for(index = 0; index < count_before; ++index) {
            if(!dead_mask[index]) {
                live_candidates[live_write] = index;
                sample_weights[live_write] = gsx_sigmoid(refine_data.opacity[index]);
                live_write += 1u;
            }
        }
        error = gsx_cpu_adc_mcmc_sample_weighted(cpu_adc->rng, sample_weights, live_candidates, live_count, dead_count, sampled_sources);
        if(!gsx_error_is_success(error)) {
            free(sampled_sources);
            free(sample_weights);
            free(live_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        for(index = 0; index < dead_count; ++index) {
            sample_counts[sampled_sources[index]] += 1u;
        }
        error = gsx_cpu_adc_mcmc_apply_relocation_updates(
            &refine_data,
            sampled_sources,
            sample_counts,
            dead_indices,
            dead_count,
            count_before,
            desc->pruning_opacity_threshold
        );
        if(!gsx_error_is_success(error)) {
            free(sampled_sources);
            free(sample_weights);
            free(live_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        if(gsx_cpu_adc_optim_enabled(request)) {
            error = gsx_cpu_optim_zero_rows(request->optim, sampled_sources, dead_count, count_before);
            if(!gsx_error_is_success(error)) {
                free(sampled_sources);
                free(sample_weights);
                free(live_candidates);
                free(sample_counts);
                free(dead_indices);
                free(dead_mask);
                gsx_cpu_adc_free_refine_data(&refine_data);
                return error;
            }
        }
        memset(sample_counts, 0, sizeof(*sample_counts) * count_before);
        out_result->mutated = true;
        free(sampled_sources);
        free(sample_weights);
        free(live_candidates);
        sampled_sources = NULL;
        sample_weights = NULL;
        live_candidates = NULL;
    } else if(dead_count == count_before) {
        free(sample_counts);
        free(dead_indices);
        free(dead_mask);
        gsx_cpu_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc relocate requires at least one live gaussian");
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

        all_candidates = (gsx_size_t *)malloc(sizeof(*all_candidates) * count_before);
        all_weights = (float *)malloc(sizeof(*all_weights) * count_before);
        sampled_sources = (gsx_size_t *)malloc(sizeof(*sampled_sources) * target_growth);
        gather_indices = (int32_t *)malloc(sizeof(*gather_indices) * gathered_count);
        if(all_candidates == NULL || all_weights == NULL || sampled_sources == NULL || gather_indices == NULL) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc growth buffers");
        }
        for(index = 0; index < count_before; ++index) {
            all_candidates[index] = index;
            all_weights[index] = gsx_sigmoid(refine_data.opacity[index]);
        }
        error = gsx_cpu_adc_mcmc_sample_weighted(cpu_adc->rng, all_weights, all_candidates, count_before, target_growth, sampled_sources);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        for(index = 0; index < target_growth; ++index) {
            sample_counts[sampled_sources[index]] += 1u;
        }
        error = gsx_cpu_adc_mcmc_apply_relocation_updates(
            &refine_data,
            sampled_sources,
            sample_counts,
            NULL,
            0,
            count_before,
            desc->pruning_opacity_threshold
        );
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        for(index = 0; index < count_before; ++index) {
            gather_indices[index] = (int32_t)index;
        }
        for(index = 0; index < target_growth; ++index) {
            gather_indices[count_before + index] = (int32_t)sampled_sources[index];
        }
        error = gsx_cpu_adc_apply_gs_and_optim_gather(request, gather_indices, gathered_count);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        error = gsx_cpu_adc_load_count(request->gs, &count_after_growth);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        if(count_after_growth != gathered_count) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "cpu mcmc adc growth produced unexpected gaussian count");
        }
        error = gsx_cpu_adc_zero_growth_optim_state(request, count_before, count_after_growth);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            free(sampled_sources);
            free(all_weights);
            free(all_candidates);
            free(sample_counts);
            free(dead_indices);
            free(dead_mask);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        out_result->grown_count += target_growth;
        out_result->mutated = true;
        free(gather_indices);
        free(sampled_sources);
        free(all_weights);
        free(all_candidates);
        gather_indices = NULL;
        sampled_sources = NULL;
        sample_weights = NULL;
        live_candidates = NULL;
    }

    free(sample_counts);
    free(dead_indices);
    free(dead_mask);
    gsx_cpu_adc_free_refine_data(&refine_data);
    return gsx_cpu_adc_reset_post_refine_aux(request->gs);
}
