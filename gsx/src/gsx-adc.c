#include "gsx-impl.h"

#include <string.h>

bool gsx_adc_algorithm_is_valid(gsx_adc_algorithm algorithm)
{
    switch(algorithm) {
    case GSX_ADC_ALGORITHM_DEFAULT:
    case GSX_ADC_ALGORITHM_ABSGS:
    case GSX_ADC_ALGORITHM_MCMC:
    case GSX_ADC_ALGORITHM_FASTGS:
        return true;
    }

    return false;
}

static gsx_error gsx_adc_require_handle(gsx_adc_t adc)
{
    if(adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc must be non-null");
    }
    if(adc->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "adc implementation is missing an interface");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_adc_validate_desc(gsx_backend_t backend, const gsx_adc_desc *desc)
{
    gsx_size_t start_refine = 0;
    gsx_size_t end_refine = 0;

    if(backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and desc must be non-null");
    }
    if(!gsx_adc_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc algorithm is out of range");
    }
    if(!gsx_optim_float_is_finite(desc->pruning_opacity_threshold)
        || !gsx_optim_float_is_finite(desc->opacity_clamp_value)
        || !gsx_optim_float_is_finite(desc->max_world_scale)
        || !gsx_optim_float_is_finite(desc->max_screen_scale)
        || !gsx_optim_float_is_finite(desc->duplicate_grad_threshold)
        || !gsx_optim_float_is_finite(desc->duplicate_scale_threshold)
        || !gsx_optim_float_is_finite(desc->duplicate_absgrad_threshold)
        || !gsx_optim_float_is_finite(desc->noise_strength)
        || !gsx_optim_float_is_finite(desc->grow_ratio)
        || !gsx_optim_float_is_finite(desc->loss_threshold)
        || !gsx_optim_float_is_finite(desc->importance_threshold)
        || !gsx_optim_float_is_finite(desc->prune_budget_ratio)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc descriptor must contain finite numeric values");
    }
    if(desc->pruning_opacity_threshold < 0.0f || desc->opacity_clamp_value < 0.0f || desc->opacity_clamp_value > 1.0f) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "opacity thresholds must be in valid range");
    }
    if(desc->max_world_scale < 0.0f || desc->max_screen_scale < 0.0f || desc->duplicate_grad_threshold < 0.0f
        || desc->duplicate_scale_threshold < 0.0f || desc->duplicate_absgrad_threshold < 0.0f || desc->noise_strength < 0.0f
        || desc->grow_ratio < 0.0f || desc->loss_threshold < 0.0f || desc->importance_threshold < 0.0f
        || desc->prune_budget_ratio < 0.0f) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc descriptor thresholds and scales must be non-negative");
    }
    start_refine = desc->start_refine > 0 ? (gsx_size_t)desc->start_refine : 0;
    end_refine = desc->end_refine > 0 ? (gsx_size_t)desc->end_refine : 0;
    if(start_refine > end_refine) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc refine window must satisfy start_refine <= end_refine");
    }
    if(desc->max_num_gaussians < 0 || desc->reset_every < 0 || desc->refine_every < 0 || desc->max_sampled_cameras < 0) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc count and interval fields must be non-negative");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_adc_base_init(gsx_adc *adc, const gsx_adc_i *iface, gsx_backend_t backend, const gsx_adc_desc *desc)
{
    if(adc == NULL || iface == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc, iface, backend, and desc must be non-null");
    }

    memset(adc, 0, sizeof(*adc));
    adc->iface = iface;
    adc->backend = backend;
    adc->desc = *desc;
    adc->backend->live_adc_count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_adc_base_deinit(gsx_adc *adc)
{
    if(adc == NULL) {
        return;
    }

    if(adc->backend != NULL && adc->backend->live_adc_count != 0) {
        adc->backend->live_adc_count -= 1;
    }
    memset(adc, 0, sizeof(*adc));
}

static gsx_error gsx_adc_validate_request(const gsx_adc *adc, const gsx_adc_request *request)
{
    if(adc == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc and request must be non-null");
    }
    if(request->gs == NULL || request->optim == NULL || request->dataloader == NULL || request->renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc request handles must be non-null");
    }
    if(request->optim->backend != adc->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc request optimizer must belong to the adc backend");
    }
    if(request->renderer->backend != adc->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc request renderer must belong to the adc backend");
    }
    if(!gsx_optim_float_is_finite(request->scene_scale) || request->scene_scale <= 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc request scene_scale must be finite and positive");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_adc_init(gsx_adc_t *out_adc, gsx_backend_t backend, const gsx_adc_desc *desc)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_adc == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_adc, backend, and desc must be non-null");
    }

    *out_adc = NULL;
    if(backend->iface == NULL || backend->iface->create_adc == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend does not implement adc creation");
    }

    error = gsx_adc_validate_desc(backend, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "adc currently supports only GSX_ADC_ALGORITHM_DEFAULT");
    }

    return backend->iface->create_adc(backend, desc, out_adc);
}

GSX_API gsx_error gsx_adc_free(gsx_adc_t adc)
{
    gsx_error error = gsx_adc_require_handle(adc);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(adc->iface->destroy == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "adc destroy is not implemented");
    }

    return adc->iface->destroy(adc);
}

GSX_API gsx_error gsx_adc_get_desc(gsx_adc_t adc, gsx_adc_desc *out_desc)
{
    gsx_error error = gsx_adc_require_handle(adc);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }

    *out_desc = adc->desc;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_adc_set_desc(gsx_adc_t adc, const gsx_adc_desc *desc)
{
    gsx_error error = gsx_adc_require_handle(adc);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_adc_validate_desc(adc->backend, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "adc currently supports only GSX_ADC_ALGORITHM_DEFAULT");
    }

    adc->desc = *desc;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result)
{
    gsx_error error = gsx_adc_require_handle(adc);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }
    if(adc->iface->step == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "adc step is not implemented");
    }
    error = gsx_adc_validate_request(adc, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return adc->iface->step(adc, request, out_result);
}

GSX_API gsx_error gsx_adc_get_gs_aux_fields(gsx_adc_t adc, gsx_gs_aux_flags *out_fields)
{
    gsx_error error = gsx_adc_require_handle(adc);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_fields == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_fields must be non-null");
    }

    switch (adc->desc.algorithm) {
    case GSX_ADC_ALGORITHM_DEFAULT:
        *out_fields = GSX_GS_AUX_DEFAULT | GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_MAX_SCREEN_RADIUS;
        break;
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "adc algorithm does not supported");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
