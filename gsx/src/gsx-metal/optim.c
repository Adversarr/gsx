#include "internal.h"

gsx_error gsx_metal_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim)
{
    (void)backend;
    (void)desc;
    if(out_optim != NULL) {
        *out_optim = NULL;
    }
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optim is not implemented");
}
