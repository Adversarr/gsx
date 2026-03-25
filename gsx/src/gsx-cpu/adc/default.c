#include "internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum gsx_cpu_adc_grow_mode {
    GSX_CPU_ADC_GROW_NONE = 0,
    GSX_CPU_ADC_GROW_DUPLICATE = 1,
    GSX_CPU_ADC_GROW_SPLIT = 2
} gsx_cpu_adc_grow_mode;

static gsx_error gsx_cpu_adc_copy_shared_growth_fields(gsx_cpu_adc_refine_data *data, gsx_size_t target, gsx_size_t src)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cpu_adc_copy_slice(data->rotation, target, data->rotation, src, 4);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_slice(data->sh0, target, data->sh0, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_optional_slice(data->sh1, target, data->sh1, src, 9);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_optional_slice(data->sh2, target, data->sh2, src, 15);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_optional_slice(data->sh3, target, data->sh3, src, 21);
    return error;
}

static gsx_error gsx_cpu_adc_apply_duplicate_mutation(gsx_cpu_adc_refine_data *data, gsx_size_t target, gsx_size_t src)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cpu_adc_copy_slice(data->mean3d, target, data->mean3d, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_copy_slice(data->logscale, target, data->logscale, src, 3);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    data->opacity[target] = data->opacity[src];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_apply_split_mutation(
    gsx_cpu_adc_refine_data *data,
    gsx_size_t target,
    gsx_size_t src,
    gsx_pcg32_t rng
)
{
    float qx = data->rotation[src * 4 + 0];
    float qy = data->rotation[src * 4 + 1];
    float qz = data->rotation[src * 4 + 2];
    float qw = data->rotation[src * 4 + 3];
    float sx = gsx_expf(data->logscale[src * 3 + 0]);
    float sy = gsx_expf(data->logscale[src * 3 + 1]);
    float sz = gsx_expf(data->logscale[src * 3 + 2]);
    float source_opacity = gsx_sigmoid(data->opacity[src]);
    float split_opacity = 0.0f;
    float rnd1x = 0.0f;
    float rnd1y = 0.0f;
    float rnd1z = 0.0f;
    float rnd2x = 0.0f;
    float rnd2y = 0.0f;
    float rnd2z = 0.0f;
    float m00 = 1.0f;
    float m01 = 0.0f;
    float m02 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;
    float m12 = 0.0f;
    float m20 = 0.0f;
    float m21 = 0.0f;
    float m22 = 1.0f;
    float t1x = 0.0f;
    float t1y = 0.0f;
    float t1z = 0.0f;
    float t2x = 0.0f;
    float t2y = 0.0f;
    float t2z = 0.0f;
    float off1x = 0.0f;
    float off1y = 0.0f;
    float off1z = 0.0f;
    float off2x = 0.0f;
    float off2y = 0.0f;
    float off2z = 0.0f;
    float new_scale_x = sx / 1.6f;
    float new_scale_y = sy / 1.6f;
    float new_scale_z = sz / 1.6f;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cpu_adc_sample_normal(rng, &rnd1x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_sample_normal(rng, &rnd1y);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_sample_normal(rng, &rnd1z);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_sample_normal(rng, &rnd2x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_sample_normal(rng, &rnd2y);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_adc_sample_normal(rng, &rnd2z);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    t1x = rnd1x * (sx + 1e-5f);
    t1y = rnd1y * (sy + 1e-5f);
    t1z = rnd1z * (sz + 1e-5f);
    t2x = rnd2x * (sx + 1e-5f);
    t2y = rnd2y * (sy + 1e-5f);
    t2z = rnd2z * (sz + 1e-5f);

    gsx_cpu_adc_normalize_quaternion(&qx, &qy, &qz, &qw);
    gsx_cpu_adc_build_rotation_matrix(qx, qy, qz, qw, &m00, &m01, &m02, &m10, &m11, &m12, &m20, &m21, &m22);
    off1x = m00 * t1x + m01 * t1y + m02 * t1z;
    off1y = m10 * t1x + m11 * t1y + m12 * t1z;
    off1z = m20 * t1x + m21 * t1y + m22 * t1z;
    off2x = m00 * t2x + m01 * t2y + m02 * t2z;
    off2y = m10 * t2x + m11 * t2y + m12 * t2z;
    off2z = m20 * t2x + m21 * t2y + m22 * t2z;
    source_opacity = gsx_cpu_adc_clamp_probability(source_opacity);
    split_opacity = 1.0f - sqrtf(1.0f - source_opacity);
    data->mean3d[target * 3 + 0] = data->mean3d[src * 3 + 0] + off1x;
    data->mean3d[target * 3 + 1] = data->mean3d[src * 3 + 1] + off1y;
    data->mean3d[target * 3 + 2] = data->mean3d[src * 3 + 2] + off1z;
    data->logscale[target * 3 + 0] = gsx_logf(new_scale_x);
    data->logscale[target * 3 + 1] = gsx_logf(new_scale_y);
    data->logscale[target * 3 + 2] = gsx_logf(new_scale_z);
    data->opacity[target] = gsx_cpu_adc_probability_to_logit(split_opacity);
    data->mean3d[src * 3 + 0] = data->mean3d[src * 3 + 0] + off2x;
    data->mean3d[src * 3 + 1] = data->mean3d[src * 3 + 1] + off2y;
    data->mean3d[src * 3 + 2] = data->mean3d[src * 3 + 2] + off2z;
    data->logscale[src * 3 + 0] = gsx_logf(new_scale_x);
    data->logscale[src * 3 + 1] = gsx_logf(new_scale_y);
    data->logscale[src * 3 + 2] = gsx_logf(new_scale_z);
    data->opacity[src] = gsx_cpu_adc_probability_to_logit(split_opacity);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_cpu_adc_grow_mode gsx_cpu_adc_grow_mode_for_index(
    const gsx_adc_desc *desc,
    const gsx_cpu_adc_refine_data *data,
    float scene_scale,
    gsx_size_t index
)
{
    float counter = 1.0f;
    float accum = 0.0f;
    float grad = 0.0f;
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    float max_scale = 0.0f;
    float grow_grad = 0.0f;
    float split_scale = 0.0f;
    const float *selected_grad_acc = NULL;

    if(desc == NULL || data == NULL || data->logscale == NULL || index >= data->count) {
        return GSX_CPU_ADC_GROW_NONE;
    }
    if(data->has_visible_counter && data->visible_counter != NULL) {
        counter = data->visible_counter[index];
    }
    switch(desc->algorithm) {
    case GSX_ADC_ALGORITHM_ABSGS:
        selected_grad_acc = data->absgrad_acc;
        grow_grad = desc->duplicate_absgrad_threshold;
        break;
    case GSX_ADC_ALGORITHM_DEFAULT:
        selected_grad_acc = data->grad_acc;
        grow_grad = desc->duplicate_grad_threshold;
        break;
    default:
        return GSX_CPU_ADC_GROW_NONE;
    }
    if(counter <= 0.0f || selected_grad_acc == NULL) {
        return GSX_CPU_ADC_GROW_NONE;
    }
    accum = selected_grad_acc[index];
    grad = accum / (counter > 1.0f ? counter : 1.0f);
    if(grad <= grow_grad) {
        return GSX_CPU_ADC_GROW_NONE;
    }
    sx = gsx_expf(data->logscale[index * 3 + 0]);
    sy = gsx_expf(data->logscale[index * 3 + 1]);
    sz = gsx_expf(data->logscale[index * 3 + 2]);
    max_scale = sx;
    if(sy > max_scale) {
        max_scale = sy;
    }
    if(sz > max_scale) {
        max_scale = sz;
    }
    split_scale = desc->duplicate_scale_threshold * scene_scale;
    if(max_scale > split_scale) {
        return GSX_CPU_ADC_GROW_SPLIT;
    }
    return GSX_CPU_ADC_GROW_DUPLICATE;
}

static bool gsx_cpu_adc_should_keep(
    const gsx_adc_desc *desc,
    const gsx_cpu_adc_refine_data *data,
    float scene_scale,
    gsx_size_t index,
    gsx_size_t count_before_growth,
    bool prune_large
)
{
    float opacity = 0.0f;
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
    opacity = gsx_sigmoid(data->opacity[index]);
    not_transparent = opacity > desc->pruning_opacity_threshold;
    sx = gsx_expf(data->logscale[index * 3 + 0]);
    sy = gsx_expf(data->logscale[index * 3 + 1]);
    sz = gsx_expf(data->logscale[index * 3 + 2]);
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
    return not_transparent && ((not_large_ws && not_large_ss) || !prune_large) && not_degenerate;
}

gsx_error gsx_cpu_adc_apply_default_reset(const gsx_adc_desc *desc, const gsx_adc_request *request)
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
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc reset requires gs opacity access");
    }
    if(opacity->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc reset supports only float32 opacity");
    }
    clamp_threshold = desc->opacity_clamp_value;
    if(clamp_threshold > 1.0f) {
        clamp_threshold = 1.0f;
    }
    if(clamp_threshold < 1e-6f) {
        clamp_threshold = 1e-6f;
    }
    max_opacity = gsx_cpu_adc_probability_to_logit(clamp_threshold);
    error = gsx_tensor_clamp_inplace(opacity, &min_opacity, &max_opacity);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_cpu_adc_optim_enabled(request)) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_optim_reset_param_group_by_role(request->optim, GSX_OPTIM_PARAM_ROLE_OPACITY);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cpu_adc_apply_default_refine(
    gsx_cpu_adc *cpu_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
)
{
    gsx_cpu_adc_refine_data refine_data = { 0 };
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
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_adc == NULL || desc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu_adc, desc, request, and out_result must be non-null");
    }

    error = gsx_cpu_adc_load_count(request->gs, &count_before_refine);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(count_before_refine == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_cpu_adc_load_refine_data(request->gs, count_before_refine, desc->algorithm != GSX_ADC_ALGORITHM_ABSGS, &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(desc->algorithm == GSX_ADC_ALGORITHM_ABSGS && refine_data.absgrad_acc == NULL) {
        gsx_cpu_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu absgs adc refine requires GSX_GS_FIELD_ABSGRAD_ACC auxiliary field");
    }

    max_capacity = gsx_cpu_adc_non_negative_index(desc->max_num_gaussians);
    if(max_capacity > count_before_refine) {
        grow_budget = max_capacity - count_before_refine;
    }
    if(grow_budget > 0) {
        grow_sources = (int32_t *)malloc(sizeof(int32_t) * grow_budget);
        grow_modes = (uint8_t *)malloc(sizeof(uint8_t) * grow_budget);
        if(grow_sources == NULL || grow_modes == NULL) {
            free(grow_sources);
            free(grow_modes);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate duplicate source index buffer");
        }
        for(index = 0; index < count_before_refine && grow_count < grow_budget; ++index) {
            gsx_cpu_adc_grow_mode mode = gsx_cpu_adc_grow_mode_for_index(desc, &refine_data, request->scene_scale, index);
            if(mode != GSX_CPU_ADC_GROW_NONE) {
                grow_sources[grow_count] = (int32_t)index;
                grow_modes[grow_count] = (uint8_t)mode;
                if(mode == GSX_CPU_ADC_GROW_SPLIT) {
                    split_count += 1;
                } else {
                    duplicate_count += 1;
                }
                grow_count += 1;
            }
        }
    }

    if(grow_count > 0) {
        gsx_size_t gathered_count = count_before_refine + grow_count;

        gather_indices = (int32_t *)malloc(sizeof(int32_t) * gathered_count);
        if(gather_indices == NULL) {
            free(grow_sources);
            free(grow_modes);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate duplicate gather index buffer");
        }
        for(index = 0; index < count_before_refine; ++index) {
            gather_indices[index] = (int32_t)index;
        }
        for(index = 0; index < grow_count; ++index) {
            gather_indices[count_before_refine + index] = grow_sources[index];
        }
        error = gsx_cpu_adc_apply_gs_and_optim_gather(request, gather_indices, gathered_count);
        if(!gsx_error_is_success(error)) {
            free(gather_indices);
            free(grow_sources);
            free(grow_modes);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        free(gather_indices);
        gather_indices = NULL;
        error = gsx_cpu_adc_load_count(request->gs, &count_after_growth);
        if(!gsx_error_is_success(error)) {
            free(grow_sources);
            free(grow_modes);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        if(count_after_growth != count_before_refine + grow_count) {
            free(grow_sources);
            free(grow_modes);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "cpu default adc growth produced unexpected gaussian count");
        }
        error = gsx_cpu_adc_zero_growth_optim_state(request, count_before_refine, count_after_growth);
        if(!gsx_error_is_success(error)) {
            free(grow_sources);
            free(grow_modes);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        error = gsx_cpu_adc_load_refine_data(
            request->gs,
            count_after_growth,
            desc->algorithm != GSX_ADC_ALGORITHM_ABSGS,
            &refine_data);
        if(!gsx_error_is_success(error)) {
            free(grow_sources);
            free(grow_modes);
            return error;
        }

        for(index = 0; index < grow_count; ++index) {
            gsx_size_t src = (gsx_size_t)grow_sources[index];
            gsx_size_t target = count_before_refine + index;

            if(src >= count_before_refine || target >= count_after_growth) {
                continue;
            }
            error = gsx_cpu_adc_copy_shared_growth_fields(&refine_data, target, src);
            if(!gsx_error_is_success(error)) {
                free(grow_sources);
                free(grow_modes);
                gsx_cpu_adc_free_refine_data(&refine_data);
                return error;
            }
            if(grow_modes[index] == (uint8_t)GSX_CPU_ADC_GROW_DUPLICATE) {
                error = gsx_cpu_adc_apply_duplicate_mutation(&refine_data, target, src);
            } else {
                error = gsx_cpu_adc_apply_split_mutation(&refine_data, target, src, cpu_adc->rng);
            }
            if(!gsx_error_is_success(error)) {
                free(grow_sources);
                free(grow_modes);
                gsx_cpu_adc_free_refine_data(&refine_data);
                return error;
            }
        }
        out_result->duplicated_count += duplicate_count;
        out_result->grown_count += split_count;
        out_result->mutated = true;
    }

    free(grow_sources);
    free(grow_modes);

    error = gsx_cpu_adc_load_count(request->gs, &count_after_growth);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_adc_free_refine_data(&refine_data);
        return error;
    }
    if(count_after_growth == 0) {
        gsx_cpu_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_cpu_adc_load_refine_data(
        request->gs,
        count_after_growth,
        desc->algorithm != GSX_ADC_ALGORITHM_ABSGS,
        &refine_data);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    keep_indices = (int32_t *)malloc(sizeof(int32_t) * count_after_growth);
    if(keep_indices == NULL) {
        gsx_cpu_adc_free_refine_data(&refine_data);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate prune keep-index buffer");
    }
    prune_large = request->global_step > gsx_cpu_adc_non_negative_index(desc->reset_every);
    for(index = 0; index < count_after_growth; ++index) {
        if(gsx_cpu_adc_should_keep(desc, &refine_data, request->scene_scale, index, count_before_refine, prune_large)) {
            keep_indices[keep_count] = (int32_t)index;
            keep_count += 1;
        }
    }
    if(keep_count == 0) {
        keep_indices[keep_count] = 0;
        keep_count = 1;
    }
    if(keep_count < count_after_growth) {
        error = gsx_cpu_adc_apply_gs_and_optim_gather(request, keep_indices, keep_count);
        if(!gsx_error_is_success(error)) {
            free(keep_indices);
            gsx_cpu_adc_free_refine_data(&refine_data);
            return error;
        }
        out_result->pruned_count += count_after_growth - keep_count;
        out_result->mutated = true;
    }

    free(keep_indices);
    gsx_cpu_adc_free_refine_data(&refine_data);
    return gsx_cpu_adc_reset_post_refine_aux(request->gs);
}
