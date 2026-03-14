#include "internal.h"

gsx_error gsx_metal_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss)
{
    (void)backend;
    (void)desc;
    if(out_loss != NULL) {
        *out_loss = NULL;
    }
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal loss is not implemented");
}
