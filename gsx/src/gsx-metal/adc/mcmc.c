#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GSX_METAL_ADC_MCMC_RELOCATION_MAX_RATIO 51u

typedef struct gsx_metal_adc_mcmc_stats {
    uint64_t total_ns;
    uint64_t initial_opacity_stage_ns;
    uint64_t classify_dead_ns;
    uint64_t relocation_sample_ns;
    uint64_t relocation_dispatch_ns;
    uint64_t relocation_gather_ns;
    uint64_t reload_refine_ns;
    uint64_t growth_opacity_stage_ns;
    uint64_t growth_sample_ns;
    uint64_t growth_dispatch_ns;
    uint64_t growth_gather_ns;
    uint64_t growth_reload_count_ns;
    uint64_t reset_aux_ns;
    gsx_size_t gaussian_count_before;
    gsx_size_t dead_count;
    gsx_size_t live_candidate_count;
    gsx_size_t target_growth;
    bool relocation_applied;
    bool growth_applied;
} gsx_metal_adc_mcmc_stats;

typedef struct gsx_metal_adc_mcmc_noise_stats {
    uint64_t total_ns;
    uint64_t load_count_ns;
    uint64_t load_refine_ns;
    uint64_t load_lr_ns;
    uint64_t dispatch_ns;
    uint64_t rng_advance_ns;
    gsx_size_t gaussian_count;
} gsx_metal_adc_mcmc_noise_stats;

static bool gsx_metal_adc_mcmc_stats_enabled(void)
{
    const char *env = getenv("GSX_METAL_ADC_MCMC_STATS");

    return env != NULL && env[0] != '\0' && env[0] != '0';
}

static uint64_t gsx_metal_adc_mcmc_get_monotonic_time_ns(void)
{
#if defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts = { 0 };

    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

static double gsx_metal_adc_mcmc_ns_to_ms(uint64_t duration_ns)
{
    return (double)duration_ns / 1000000.0;
}

static float gsx_metal_adc_mcmc_clamp_probability(float value)
{
    if(value < 1e-6f) {
        return 1e-6f;
    }
    if(value > 1.0f - 1e-6f) {
        return 1.0f - 1e-6f;
    }
    return value;
}

static float gsx_metal_adc_mcmc_logit(float probability)
{
    float clamped = gsx_metal_adc_mcmc_clamp_probability(probability);

    return logf(clamped / (1.0f - clamped));
}

static gsx_size_t gsx_metal_adc_mcmc_upper_bound(const float *cdf_values, gsx_size_t category_count, float draw)
{
    gsx_size_t low = 0;
    gsx_size_t high = category_count;

    while(low < high) {
        gsx_size_t mid = low + ((high - low) >> 1u);

        if(draw < cdf_values[mid]) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    if(low >= category_count) {
        return category_count - 1u;
    }
    return low;
}

static const char *gsx_metal_adc_mcmc_slowest_stage_label(const gsx_metal_adc_mcmc_stats *stats)
{
    uint64_t slowest_ns = 0;
    const char *label = "none";

    if(stats == NULL) {
        return label;
    }
#define GSX_METAL_ADC_MCMC_SELECT_STAGE(field, name) \
    do { \
        if(stats->field > slowest_ns) { \
            slowest_ns = stats->field; \
            label = name; \
        } \
    } while(0)
    GSX_METAL_ADC_MCMC_SELECT_STAGE(initial_opacity_stage_ns, "initial_opacity_stage");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(classify_dead_ns, "classify_dead");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(relocation_sample_ns, "relocation_sample");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(relocation_dispatch_ns, "relocation_dispatch");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(relocation_gather_ns, "relocation_gather");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(reload_refine_ns, "reload_refine");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(growth_opacity_stage_ns, "growth_opacity_stage");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(growth_sample_ns, "growth_sample");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(growth_dispatch_ns, "growth_dispatch");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(growth_gather_ns, "growth_gather");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(growth_reload_count_ns, "growth_reload_count");
    GSX_METAL_ADC_MCMC_SELECT_STAGE(reset_aux_ns, "reset_aux");
#undef GSX_METAL_ADC_MCMC_SELECT_STAGE
    return label;
}

static void gsx_metal_adc_mcmc_print_stats(const gsx_metal_adc_mcmc_stats *stats, gsx_size_t global_step, gsx_error error)
{
    if(stats == NULL) {
        return;
    }

    fprintf(
        stderr,
        "METAL adc mcmc stats: status=%d step=%zu gaussians=%zu dead=%zu live_candidates=%zu target_growth=%zu relocation=%s growth=%s total=%.3f ms opacity_stage0=%.3f ms classify_dead=%.3f ms relocation_sample=%.3f ms relocation_dispatch=%.3f ms relocation_gather=%.3f ms reload_refine=%.3f ms opacity_stage1=%.3f ms growth_sample=%.3f ms growth_dispatch=%.3f ms growth_gather=%.3f ms growth_reload_count=%.3f ms reset_aux=%.3f ms bottleneck=%s note=%s\n",
        (int)error.code,
        (size_t)global_step,
        (size_t)stats->gaussian_count_before,
        (size_t)stats->dead_count,
        (size_t)stats->live_candidate_count,
        (size_t)stats->target_growth,
        stats->relocation_applied ? "true" : "false",
        stats->growth_applied ? "true" : "false",
        gsx_metal_adc_mcmc_ns_to_ms(stats->total_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->initial_opacity_stage_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->classify_dead_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->relocation_sample_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->relocation_dispatch_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->relocation_gather_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->reload_refine_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->growth_opacity_stage_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->growth_sample_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->growth_dispatch_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->growth_gather_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->growth_reload_count_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->reset_aux_ns),
        gsx_metal_adc_mcmc_slowest_stage_label(stats),
        "Metal MCMC is slow mainly because refinement still serializes around host-visible opacity staging and structural gather/rebind.");
}

static void gsx_metal_adc_mcmc_print_noise_stats(const gsx_metal_adc_mcmc_noise_stats *stats, gsx_size_t global_step, gsx_error error)
{
    if(stats == NULL) {
        return;
    }

    fprintf(
        stderr,
        "METAL adc mcmc noise stats: status=%d step=%zu gaussians=%zu total=%.3f ms load_count=%.3f ms load_refine=%.3f ms load_lr=%.3f ms dispatch=%.3f ms rng_advance=%.3f ms note=%s\n",
        (int)error.code,
        (size_t)global_step,
        (size_t)stats->gaussian_count,
        gsx_metal_adc_mcmc_ns_to_ms(stats->total_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->load_count_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->load_refine_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->load_lr_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->dispatch_ns),
        gsx_metal_adc_mcmc_ns_to_ms(stats->rng_advance_ns),
        "This path runs every ADC step, so even sub-millisecond dispatch cost accumulates heavily.");
}

static gsx_error gsx_metal_adc_mcmc_sample_weighted(
    gsx_metal_adc *metal_adc,
    const float *weights,
    const gsx_size_t *candidates,
    gsx_size_t count,
    gsx_size_t sample_count,
    gsx_size_t *out_samples)
{
    float *cdf_values = NULL;
    gsx_pcg32 *rng_state = NULL;
    float total_weight = 0.0f;
    gsx_size_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || weights == NULL || candidates == NULL || out_samples == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc weighted sampling inputs must be non-null");
    }
    if(count == 0 || sample_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc weighted sampling counts must be positive");
    }
    cdf_values = (float *)malloc(sizeof(*cdf_values) * count);
    if(cdf_values == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc sampling cdf buffer");
    }
    for(index = 0; index < count; ++index) {
        total_weight += weights[index] > 0.0f ? weights[index] : 0.0f;
        cdf_values[index] = total_weight;
    }
    if(total_weight <= 0.0f) {
        error = gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc weighted sampling requires positive opacity mass");
        goto cleanup;
    }

    rng_state = (gsx_pcg32 *)metal_adc->rng;
    for(index = 0; index < sample_count; ++index) {
        gsx_size_t sampled_index = gsx_metal_adc_mcmc_upper_bound(cdf_values, count, pcg32_next_float(rng_state) * total_weight);

        if(sampled_index >= count) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "mcmc multinomial produced an out-of-range sample index");
            goto cleanup;
        }
        out_samples[index] = candidates[sampled_index];
    }

cleanup:
    free(cdf_values);
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

static gsx_size_t gsx_metal_adc_mcmc_clamp_ratio(gsx_size_t ratio)
{
    if(ratio < 1u) {
        return 1u;
    }
    if(ratio > GSX_METAL_ADC_MCMC_RELOCATION_MAX_RATIO) {
        return GSX_METAL_ADC_MCMC_RELOCATION_MAX_RATIO;
    }
    return ratio;
}

static float gsx_metal_adc_mcmc_binom(gsx_size_t n, gsx_size_t k)
{
    gsx_size_t index = 0;
    double value = 1.0;

    if(k > n) {
        return 0.0f;
    }
    if(k == 0u || k == n) {
        return 1.0f;
    }
    if(k > n - k) {
        k = n - k;
    }
    for(index = 1u; index <= k; ++index) {
        value = value * (double)(n - (k - index)) / (double)index;
    }
    return (float)value;
}

static float gsx_metal_adc_mcmc_relocated_opacity(float opacity, gsx_size_t ratio)
{
    float clamped = gsx_metal_adc_mcmc_clamp_probability(opacity);
    float root = 1.0f / (float)gsx_metal_adc_mcmc_clamp_ratio(ratio);

    return 1.0f - powf(1.0f - clamped, root);
}

static float gsx_metal_adc_mcmc_scale_coeff(float opacity, float new_opacity, gsx_size_t ratio)
{
    float denom_sum = 0.0f;
    gsx_size_t i = 0;

    for(i = 1u; i <= gsx_metal_adc_mcmc_clamp_ratio(ratio); ++i) {
        gsx_size_t k = 0;

        for(k = 0u; k < i; ++k) {
            float sign = (k & 1u) == 0u ? 1.0f : -1.0f;
            float coeff = gsx_metal_adc_mcmc_binom(i - 1u, k);
            float power = powf(new_opacity, (float)(k + 1u));

            denom_sum += coeff * sign * power / sqrtf((float)(k + 1u));
        }
    }
    if(fabsf(denom_sum) <= 1e-8f) {
        return 1.0f;
    }
    return opacity / denom_sum;
}

static void gsx_metal_adc_mcmc_apply_cpu_relocation_weights(
    float *opacity_probabilities,
    const gsx_size_t *sample_counts,
    const gsx_size_t *dead_indices,
    const gsx_size_t *sampled_sources,
    gsx_size_t count,
    gsx_size_t dead_count,
    float min_opacity)
{
    gsx_size_t index = 0;

    if(opacity_probabilities == NULL || sample_counts == NULL) {
        return;
    }

    for(index = 0; index < count; ++index) {
        if(sample_counts[index] > 0u) {
            float new_opacity = gsx_metal_adc_mcmc_relocated_opacity(opacity_probabilities[index], sample_counts[index] + 1u);

            (void)gsx_metal_adc_mcmc_scale_coeff(opacity_probabilities[index], new_opacity, sample_counts[index] + 1u);
            if(new_opacity < min_opacity) {
                new_opacity = min_opacity;
            }
            opacity_probabilities[index] = gsx_metal_adc_mcmc_clamp_probability(new_opacity);
        }
    }
    if(dead_indices == NULL || sampled_sources == NULL) {
        return;
    }
    for(index = 0; index < dead_count; ++index) {
        opacity_probabilities[dead_indices[index]] = opacity_probabilities[sampled_sources[index]];
    }
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
    gsx_metal_adc_mcmc_noise_stats stats = { 0 };
    bool stats_enabled = gsx_metal_adc_mcmc_stats_enabled();
    uint64_t total_start_ns = 0;
    uint64_t stage_start_ns = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || desc == NULL || request == NULL || desc->noise_strength <= 0.0f) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(stats_enabled) {
        total_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    stop_iter = gsx_metal_adc_non_negative_index(desc->end_refine);
    if(stop_iter > 0u && request->global_step >= stop_iter) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(stats_enabled) {
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    error = gsx_metal_adc_load_count(request->gs, &count);
    if(!gsx_error_is_success(error) || count == 0) {
        return error;
    }
    if(stats_enabled) {
        stats.load_count_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
    }
    stats.gaussian_count = count;
    if(stats_enabled) {
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    error = gsx_metal_adc_load_refine_data(request->gs, count, false, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(stats_enabled) {
        stats.load_refine_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    error = gsx_optim_get_learning_rate_by_role(request->optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &lr_mean3d);
    if(!gsx_error_is_success(error)) {
        gsx_metal_adc_free_refine_data(&refine_data);
        return error;
    }
    if(stats_enabled) {
        stats.load_lr_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
    }

    rng_state = (gsx_pcg32 *)metal_adc->rng;
    params.gaussian_count = (uint32_t)count;
    params.noise_strength = desc->noise_strength;
    params.learning_rate = lr_mean3d;
    params.rng_state = rng_state->state;
    params.rng_inc = rng_state->inc;
    if(stats_enabled) {
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    error = gsx_metal_backend_dispatch_adc_mcmc_noise(
        refine_data.mean3d_tensor->backing_buffer->buffer_type->backend,
        &refine_data.mean3d_view,
        &refine_data.logscale_view,
        &refine_data.opacity_view,
        &refine_data.rotation_view,
        &params);
    if(stats_enabled) {
        stats.dispatch_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
    }
    if(gsx_error_is_success(error)) {
        if(stats_enabled) {
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
        }
        error = gsx_metal_adc_advance_rng(metal_adc, count * 6u, "adc mcmc noise rng advance exceeds supported range");
        if(stats_enabled) {
            stats.rng_advance_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
        }
    }
    gsx_metal_adc_free_refine_data(&refine_data);
    if(stats_enabled) {
        stats.total_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - total_start_ns;
        gsx_metal_adc_mcmc_print_noise_stats(&stats, request->global_step, error);
    }
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
    gsx_tensor_t opacity_staging_tensor = NULL;
    int32_t *gather_indices = NULL;
    gsx_size_t *dead_indices = NULL;
    gsx_size_t *live_candidates = NULL;
    gsx_size_t *sampled_sources = NULL;
    gsx_size_t *sample_counts = NULL;
    float *sample_weights = NULL;
    float *opacity_values = NULL;
    float *opacity_probabilities = NULL;
    gsx_metal_adc_mcmc_stats stats = { 0 };
    bool stats_enabled = gsx_metal_adc_mcmc_stats_enabled();
    uint64_t total_start_ns = 0;
    uint64_t stage_start_ns = 0;
    float pruning_opacity_logit = 0.0f;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_adc == NULL || desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "mcmc refine inputs must be non-null");
    }
    if(stats_enabled) {
        total_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    pruning_opacity_logit = gsx_metal_adc_mcmc_logit(desc->pruning_opacity_threshold);

    error = gsx_metal_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error) || count_before == 0) {
        return error;
    }
    stats.gaussian_count_before = count_before;
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
    opacity_probabilities = (float *)malloc(sizeof(*opacity_probabilities) * count_before);
    if(dead_indices == NULL || sample_counts == NULL || opacity_probabilities == NULL) {
        error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc dead-mask buffer");
        goto cleanup;
    }
    if(stats_enabled) {
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    error = gsx_metal_adc_mcmc_stage_opacity_values(metal_adc, refine_data.opacity_tensor, &opacity_staging_tensor, &opacity_values);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(stats_enabled) {
        stats.initial_opacity_stage_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    for(index = 0; index < count_before; ++index) {
        opacity_probabilities[index] = gsx_sigmoid(opacity_values[index]);
        if(opacity_values[index] <= pruning_opacity_logit) {
            dead_indices[dead_count] = index;
            dead_count += 1u;
        }
    }
    if(opacity_staging_tensor != NULL) {
        (void)gsx_tensor_free(opacity_staging_tensor);
        opacity_staging_tensor = NULL;
        opacity_values = NULL;
    }
    if(stats_enabled) {
        stats.classify_dead_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
    }
    stats.dead_count = dead_count;
    out_result->pruned_count = dead_count;

    if(dead_count > 0 && dead_count < count_before) {
        gsx_size_t live_count = count_before - dead_count;
        gsx_size_t live_write = 0;
        gsx_size_t dead_read = 0;

        live_candidates = (gsx_size_t *)malloc(sizeof(*live_candidates) * live_count);
        sample_weights = (float *)malloc(sizeof(*sample_weights) * live_count);
        sampled_sources = (gsx_size_t *)malloc(sizeof(*sampled_sources) * dead_count);
        gather_indices = (int32_t *)malloc(sizeof(*gather_indices) * count_before);
        if(live_candidates == NULL || sample_weights == NULL || sampled_sources == NULL || gather_indices == NULL) {
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc relocation buffers");
            goto cleanup;
        }
        for(index = 0; index < count_before; ++index) {
            bool is_dead = dead_read < dead_count && dead_indices[dead_read] == index;

            gather_indices[index] = (int32_t)index;
            if(!is_dead) {
                live_candidates[live_write] = index;
                sample_weights[live_write] = opacity_probabilities[index];
                live_write += 1u;
            } else {
                dead_read += 1u;
            }
        }
        stats.live_candidate_count = live_count;
        if(stats_enabled) {
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
        }
        error = gsx_metal_adc_mcmc_sample_weighted(
            metal_adc,
            sample_weights,
            live_candidates,
            live_count,
            dead_count,
            sampled_sources);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(stats_enabled) {
            stats.relocation_sample_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
        }
        for(index = 0; index < dead_count; ++index) {
            sample_counts[sampled_sources[index]] += 1u;
            gather_indices[dead_indices[index]] = (int32_t)sampled_sources[index];
        }
        if(stats_enabled) {
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
        }
        error = gsx_metal_adc_mcmc_dispatch_relocation(metal_adc, &refine_data, sample_counts, count_before, desc->pruning_opacity_threshold);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(stats_enabled) {
            stats.relocation_dispatch_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
        }
        error = gsx_metal_adc_apply_gs_gather_and_rebind_optim(metal_adc, request, gather_indices, count_before);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(stats_enabled) {
            stats.relocation_gather_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
        }
        if(gsx_metal_adc_optim_enabled(request)) {
            error = gsx_metal_optim_zero_rows(request->optim, sampled_sources, dead_count, count_before);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }
        gsx_metal_adc_mcmc_apply_cpu_relocation_weights(
            opacity_probabilities,
            sample_counts,
            dead_indices,
            sampled_sources,
            count_before,
            dead_count,
            desc->pruning_opacity_threshold);
        memset(sample_counts, 0, sizeof(*sample_counts) * count_before);
        out_result->mutated = true;
        out_result->grown_count += dead_count;
        stats.relocation_applied = true;
        free(sampled_sources);
        sampled_sources = NULL;
        free(sample_weights);
        sample_weights = NULL;
        free(live_candidates);
        live_candidates = NULL;
        free(gather_indices);
        gather_indices = NULL;
        if(stats_enabled) {
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
        }
        error = gsx_metal_adc_load_refine_data(request->gs, count_before, false, &refine_data);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(stats_enabled) {
            stats.reload_refine_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
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
    stats.target_growth = target_growth;

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
            gather_indices = NULL;
            free(sampled_sources);
            sampled_sources = NULL;
            free(all_weights);
            free(all_candidates);
            error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate mcmc growth buffers");
            goto cleanup;
        }
        for(index = 0; index < count_before; ++index) {
            all_candidates[index] = index;
            all_weights[index] = opacity_probabilities[index];
        }
        if(stats_enabled) {
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
        }
        error = gsx_metal_adc_mcmc_sample_weighted(
            metal_adc,
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
        if(stats_enabled) {
            stats.growth_sample_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
        }
        for(index = 0; index < target_growth; ++index) {
            sample_counts[sampled_sources[index]] += 1u;
        }
        if(stats_enabled) {
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
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
        if(stats_enabled) {
            stats.growth_dispatch_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
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
        if(stats_enabled) {
            stats.growth_gather_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
            stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
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
        if(stats_enabled) {
            stats.growth_reload_count_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
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
        stats.growth_applied = true;
    }

    if(stats_enabled) {
        stage_start_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns();
    }
    error = gsx_metal_adc_reset_post_refine_aux(request->gs);
    if(stats_enabled) {
        stats.reset_aux_ns += gsx_metal_adc_mcmc_get_monotonic_time_ns() - stage_start_ns;
    }

cleanup:
    if(opacity_staging_tensor != NULL) {
        (void)gsx_tensor_free(opacity_staging_tensor);
    }
    free(gather_indices);
    free(dead_indices);
    free(live_candidates);
    free(sampled_sources);
    free(sample_counts);
    free(sample_weights);
    free(opacity_probabilities);
    gsx_metal_adc_free_refine_data(&refine_data);
    if(stats_enabled) {
        stats.total_ns = gsx_metal_adc_mcmc_get_monotonic_time_ns() - total_start_ns;
        gsx_metal_adc_mcmc_print_stats(&stats, request->global_step, error);
    }
    return error;
}
