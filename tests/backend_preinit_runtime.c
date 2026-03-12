#include "gsx/gsx.h"

static int require_code(gsx_error error, gsx_error_code expected_code)
{
    return error.code == expected_code ? 0 : 1;
}

int main(void)
{
    gsx_index_t backend_device_count = 0;
    gsx_backend_device_t backend_device = NULL;
    gsx_backend_t backend = NULL;
    gsx_backend_desc backend_desc = { 0 };

    if(require_code(gsx_count_backend_devices(&backend_device_count), GSX_ERROR_INVALID_STATE) != 0) {
        return 1;
    }
    if(require_code(gsx_get_backend_device(0, &backend_device), GSX_ERROR_INVALID_STATE) != 0) {
        return 1;
    }
    if(require_code(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CPU, &backend_device_count), GSX_ERROR_INVALID_STATE) != 0) {
        return 1;
    }
    if(require_code(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device), GSX_ERROR_INVALID_STATE) != 0) {
        return 1;
    }
    if(require_code(gsx_backend_device_get_info(NULL, NULL), GSX_ERROR_INVALID_STATE) != 0) {
        return 1;
    }
    if(require_code(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_INVALID_STATE) != 0) {
        return 1;
    }

    return 0;
}
