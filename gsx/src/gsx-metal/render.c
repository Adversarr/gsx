#include "internal.h"

gsx_error gsx_metal_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer)
{
    (void)backend;
    (void)desc;
    if(out_renderer != NULL) {
        *out_renderer = NULL;
    }
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer is not implemented");
}
