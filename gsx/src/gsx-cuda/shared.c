#include "internal.h"

#include <string.h>

gsx_cuda_backend_provider gsx_cuda_backend_provider_singleton = { 0 };
gsx_cuda_backend_device *gsx_cuda_backend_devices = NULL;
int gsx_cuda_device_count = 0;
int gsx_cuda_device_capacity = 0;

const gsx_backend_provider_i gsx_cuda_backend_provider_iface = {
    gsx_cuda_backend_provider_discover_devices,
    gsx_cuda_backend_provider_create_backend
};

const gsx_backend_i gsx_cuda_backend_iface = {
    gsx_cuda_backend_free,
    gsx_cuda_backend_get_info,
    gsx_cuda_backend_get_capabilities,
    gsx_cuda_backend_get_major_stream,
    gsx_cuda_backend_count_buffer_types,
    gsx_cuda_backend_get_buffer_type,
    gsx_cuda_backend_find_buffer_type,
    gsx_cuda_backend_create_renderer,
    gsx_cuda_backend_create_loss,
    gsx_cuda_backend_create_optim
};

const gsx_backend_buffer_type_i gsx_cuda_backend_buffer_type_iface = {
    gsx_cuda_backend_buffer_type_get_info,
    gsx_cuda_backend_buffer_type_get_alloc_size,
    gsx_cuda_backend_buffer_type_init_buffer
};

const gsx_backend_buffer_i gsx_cuda_backend_buffer_iface = {
    gsx_cuda_backend_buffer_free,
    gsx_cuda_backend_buffer_get_info,
    gsx_cuda_backend_buffer_upload,
    gsx_cuda_backend_buffer_download,
    gsx_cuda_backend_buffer_set_zero,
    gsx_cuda_backend_buffer_memset_tensor,
    gsx_cuda_backend_buffer_set_tensor,
    gsx_cuda_backend_buffer_get_tensor,
    gsx_cuda_backend_buffer_copy_tensor,
    gsx_cuda_backend_buffer_fill_tensor,
    gsx_cuda_backend_buffer_check_finite_tensor
};

gsx_error gsx_cuda_make_error(cudaError_t cuda_err, const char *context)
{
    if(cuda_err == cudaSuccess) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(cuda_err == cudaErrorMemoryAllocation) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, context);
    }
    if(cuda_err == cudaErrorNoDevice || cuda_err == cudaErrorInvalidDevice) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, context);
    }
    return gsx_make_error(GSX_ERROR_UNKNOWN, context);
}

gsx_cuda_backend *gsx_cuda_backend_from_base(gsx_backend_t backend)
{
    return (gsx_cuda_backend *)backend;
}

gsx_cuda_backend_buffer_type *gsx_cuda_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type)
{
    return (gsx_cuda_backend_buffer_type *)buffer_type;
}

gsx_cuda_backend_buffer *gsx_cuda_backend_buffer_from_base(gsx_backend_buffer_t buffer)
{
    return (gsx_cuda_backend_buffer *)buffer;
}

gsx_backend_buffer_type_class gsx_cuda_backend_buffer_get_type_class(gsx_backend_buffer_t buffer)
{
    return gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type)->info.type;
}

void gsx_cuda_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes)
{
    gsx_size_t offset_bytes = 0;

    for(offset_bytes = 0; offset_bytes < total_bytes; offset_bytes += value_size_bytes) {
        memcpy((unsigned char *)dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
    }
}

bool gsx_cuda_backend_f16_is_finite(uint16_t value)
{
    return ((value >> 10) & 0x1FU) != 0x1FU;
}

bool gsx_cuda_backend_bf16_is_finite(uint16_t value)
{
    return ((value >> 7) & 0xFFU) != 0xFFU;
}

gsx_error gsx_cuda_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count)
{
    gsx_size_t end_offset = 0;

    if(gsx_size_add_overflows(offset_bytes, byte_count, &end_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer range overflow");
    }
    if(end_offset > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer range exceeds buffer size");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_cuda_backend_init_buffer_type(
    gsx_cuda_backend *cuda_backend,
    gsx_cuda_backend_buffer_type *buffer_type,
    gsx_backend_buffer_type_class type,
    const char *name,
    gsx_size_t alignment_bytes
)
{
    buffer_type->base.iface = &gsx_cuda_backend_buffer_type_iface;
    buffer_type->base.backend = &cuda_backend->base;
    buffer_type->base.live_arena_count = 0;
    buffer_type->info.backend = &cuda_backend->base;
    buffer_type->info.type = type;
    buffer_type->info.name = name;
    buffer_type->info.alignment_bytes = alignment_bytes;
    buffer_type->info.max_allocation_size_bytes = 0;
}
