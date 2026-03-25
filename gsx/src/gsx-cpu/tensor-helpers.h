#ifndef GSX_CPU_TENSOR_HELPERS_H
#define GSX_CPU_TENSOR_HELPERS_H

#include "../gsx-tensor-helpers.h"

static inline unsigned char *gsx_cpu_tensor_data_bytes(gsx_tensor_t tensor)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)tensor->backing_buffer;

    return (unsigned char *)cpu_buffer->data + (size_t)tensor->offset_bytes;
}

static inline float *gsx_cpu_tensor_data_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cpu_tensor_data_bytes(tensor);
}

static inline const float *gsx_cpu_tensor_data_const_f32(gsx_tensor_t tensor)
{
    return (const float *)gsx_cpu_tensor_data_bytes(tensor);
}

static inline gsx_error gsx_cpu_tensor_validate_f32(gsx_tensor_t tensor, const char *message)
{
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#endif /* GSX_CPU_TENSOR_HELPERS_H */
