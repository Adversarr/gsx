#ifndef GSX_METAL_TENSOR_HELPERS_H
#define GSX_METAL_TENSOR_HELPERS_H

#include "../gsx-tensor-helpers.h"

static inline bool gsx_metal_tensor_buffer_is_device(gsx_backend_buffer_t buffer)
{
    gsx_metal_backend_buffer *metal_buffer = (gsx_metal_backend_buffer *)buffer;

    return metal_buffer != NULL && metal_buffer->base.buffer_type != NULL
        && gsx_metal_backend_buffer_get_type_class(&metal_buffer->base) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static inline bool gsx_metal_tensor_is_device_f32(gsx_tensor_t tensor)
{
    return tensor != NULL
        && tensor->data_type == GSX_DATA_TYPE_F32
        && tensor->backing_buffer != NULL
        && gsx_metal_tensor_buffer_is_device(tensor->backing_buffer);
}

static inline bool gsx_metal_tensor_is_optional_device_f32(gsx_tensor_t tensor)
{
    return tensor == NULL || gsx_metal_tensor_is_device_f32(tensor);
}

static inline bool gsx_metal_tensor_is_backed_f32(gsx_tensor_t tensor)
{
    return tensor != NULL
        && tensor->data_type == GSX_DATA_TYPE_F32
        && tensor->backing_buffer != NULL;
}

static inline bool gsx_metal_tensor_is_backed_i32(gsx_tensor_t tensor)
{
    return tensor != NULL
        && tensor->data_type == GSX_DATA_TYPE_I32
        && tensor->backing_buffer != NULL;
}

static inline gsx_error gsx_metal_tensor_validate_f32_device(gsx_tensor_t tensor, const char *backend_message, const char *dtype_message)
{
    if(!gsx_metal_tensor_buffer_is_device(tensor->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, backend_message);
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, dtype_message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#endif /* GSX_METAL_TENSOR_HELPERS_H */
