#include "internal.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct gsx_cpu_adc {
    struct gsx_adc base;
} gsx_cpu_adc;

static gsx_error gsx_cpu_adc_destroy(gsx_adc_t adc);
static gsx_error gsx_cpu_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);

static const gsx_adc_i gsx_cpu_adc_iface = {
    gsx_cpu_adc_destroy,
    gsx_cpu_adc_step
};

gsx_error gsx_cpu_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc)
{
    gsx_cpu_adc *cpu_adc = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_adc == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_adc and desc must be non-null");
    }

    *out_adc = NULL;
    if(desc->algorithm != GSX_ADC_ALGORITHM_DEFAULT) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu adc currently supports only GSX_ADC_ALGORITHM_DEFAULT");
    }

    cpu_adc = (gsx_cpu_adc *)calloc(1, sizeof(*cpu_adc));
    if(cpu_adc == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu adc");
    }

    error = gsx_adc_base_init(&cpu_adc->base, &gsx_cpu_adc_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(cpu_adc);
        return error;
    }

    *out_adc = &cpu_adc->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_destroy(gsx_adc_t adc)
{
    gsx_cpu_adc *cpu_adc = (gsx_cpu_adc *)adc;

    if(adc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "adc must be non-null");
    }

    gsx_adc_base_deinit(&cpu_adc->base);
    free(cpu_adc);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result)
{
    (void)adc;
    (void)request;

    if(out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }

    out_result->gaussians_before = 0;
    out_result->gaussians_after = 0;
    out_result->pruned_count = 0;
    out_result->duplicated_count = 0;
    out_result->grown_count = 0;
    out_result->reset_count = 0;
    out_result->mutated = false;
    fprintf(stderr, "gsx_cpu_adc_step placeholder: DEFAULT algorithm is not implemented yet\n");
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
