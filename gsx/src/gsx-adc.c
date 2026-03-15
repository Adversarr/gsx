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
    if(backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and desc must be non-null");
    }
    if(!gsx_adc_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "adc algorithm is out of range");
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
