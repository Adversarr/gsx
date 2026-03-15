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

static gsx_size_t gsx_cpu_adc_non_negative_index(gsx_index_t value)
{
    if(value <= 0) {
        return 0;
    }
    return (gsx_size_t)value;
}

static void gsx_cpu_adc_zero_result(gsx_adc_result *out_result)
{
    out_result->gaussians_before = 0;
    out_result->gaussians_after = 0;
    out_result->pruned_count = 0;
    out_result->duplicated_count = 0;
    out_result->grown_count = 0;
    out_result->reset_count = 0;
    out_result->mutated = false;
}

static bool gsx_cpu_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_cpu_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_cpu_adc_non_negative_index(desc->end_refine);

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

static bool gsx_cpu_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step)
{
    gsx_size_t start_refine = gsx_cpu_adc_non_negative_index(desc->start_refine);
    gsx_size_t end_refine = gsx_cpu_adc_non_negative_index(desc->end_refine);

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

static gsx_error gsx_cpu_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count)
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

static gsx_error gsx_cpu_adc_require_refine_capabilities(gsx_gs_t gs)
{
    gsx_tensor_t tensor = NULL;
    gsx_error error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_ACC, &tensor);

    if(!gsx_error_is_success(error)) {
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "cpu default adc refine requires gs gradient/statistic and structural mutation support"
        );
    }
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc refine path is not fully implemented yet");
}

static gsx_error gsx_cpu_adc_apply_reset(const gsx_adc_desc *desc, const gsx_adc_request *request)
{
    // TODO: use gsx_tensor_clamp_inplace to clamp opacity in-place!
    gsx_error error = { 0 };

    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc reset requires gs opacity clamp support");
    }

    error = gsx_optim_reset(request->optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

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
    gsx_size_t count_before = 0;
    gsx_size_t count_after = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    bool refine_window = false;
    bool reset_window = false;
    bool should_refine = false;

    if(adc == NULL || request == NULL || out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }

    gsx_cpu_adc_zero_result(out_result);

    error = gsx_cpu_adc_load_count(request->gs, &count_before);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc requires gs info/query support");
    }

    out_result->gaussians_before = count_before;
    out_result->gaussians_after = count_before;
    refine_window = gsx_cpu_adc_in_refine_window(&adc->desc, request->global_step);
    reset_window = gsx_cpu_adc_in_reset_window(&adc->desc, request->global_step);
    should_refine = refine_window && count_before < gsx_cpu_adc_non_negative_index(adc->desc.max_num_gaussians);

    if(should_refine) {
        error = gsx_cpu_adc_require_refine_capabilities(request->gs);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(reset_window) {
        error = gsx_cpu_adc_apply_reset(&adc->desc, request);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        out_result->reset_count = 1;
        out_result->mutated = true;
    }

    error = gsx_cpu_adc_load_count(request->gs, &count_after);
    if(!gsx_error_is_success(error)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu default adc requires gs info/query support");
    }

    out_result->gaussians_after = count_after;
    if(out_result->gaussians_after != out_result->gaussians_before) {
        out_result->mutated = true;
    }

    if(refine_window && !should_refine) {
        fprintf(stderr, "gsx_cpu_adc_step: refine skipped due to max_num_gaussians capacity\n");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
