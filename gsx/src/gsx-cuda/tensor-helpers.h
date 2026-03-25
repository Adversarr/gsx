#ifndef GSX_CUDA_TENSOR_HELPERS_H
#define GSX_CUDA_TENSOR_HELPERS_H

#include "../gsx-tensor-helpers.h"

static inline bool gsx_cuda_tensor_buffer_is_device(gsx_backend_buffer_t buffer)
{
    gsx_cuda_backend_buffer *cuda_buffer = (gsx_cuda_backend_buffer *)buffer;

    return cuda_buffer != NULL && cuda_buffer->base.buffer_type != NULL
        && gsx_cuda_backend_buffer_get_type_class(&cuda_buffer->base) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static inline unsigned char *gsx_cuda_tensor_device_bytes(gsx_tensor_t tensor)
{
    gsx_cuda_backend_buffer *cuda_buffer = (gsx_cuda_backend_buffer *)tensor->backing_buffer;

    return (unsigned char *)cuda_buffer->ptr + (size_t)tensor->offset_bytes;
}

static inline float *gsx_cuda_tensor_device_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cuda_tensor_device_bytes(tensor);
}

static inline const float *gsx_cuda_tensor_device_const_f32(gsx_tensor_t tensor)
{
    return (const float *)gsx_cuda_tensor_device_bytes(tensor);
}

static inline gsx_error gsx_cuda_tensor_validate_f32_device(gsx_tensor_t tensor, const char *backend_message, const char *dtype_message)
{
    if(!gsx_cuda_tensor_buffer_is_device(tensor->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, backend_message);
    }
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, dtype_message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#endif /* GSX_CUDA_TENSOR_HELPERS_H */
