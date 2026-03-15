#include "internal.h"

gsx_error gsx_cuda_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc)
{
    (void)backend;
    (void)desc;
    if(out_adc != NULL) {
        *out_adc = NULL;
    }
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda adc is not implemented");
}
