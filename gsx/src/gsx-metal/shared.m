#include "internal.h"

#include <string.h>

gsx_metal_backend_provider gsx_metal_backend_provider_singleton = { 0 };
gsx_metal_backend_device *gsx_metal_backend_devices = NULL;
int gsx_metal_device_count = 0;
int gsx_metal_device_capacity = 0;

const gsx_backend_provider_i gsx_metal_backend_provider_iface = {
    gsx_metal_backend_provider_discover_devices,
    gsx_metal_backend_provider_create_backend
};

const gsx_backend_i gsx_metal_backend_iface = {
    gsx_metal_backend_free,
    gsx_metal_backend_get_info,
    gsx_metal_backend_get_capabilities,
    gsx_metal_backend_get_major_stream,
    gsx_metal_backend_major_stream_sync,
    gsx_metal_backend_count_buffer_types,
    gsx_metal_backend_get_buffer_type,
    gsx_metal_backend_find_buffer_type,
    gsx_metal_backend_create_renderer,
    gsx_metal_backend_create_loss,
    gsx_metal_backend_create_optim,
    gsx_metal_backend_create_adc
};

const gsx_backend_buffer_type_i gsx_metal_backend_buffer_type_iface = {
    gsx_metal_backend_buffer_type_get_info,
    gsx_metal_backend_buffer_type_get_alloc_size,
    gsx_metal_backend_buffer_type_init_buffer
};

const gsx_backend_buffer_i gsx_metal_backend_buffer_iface = {
    gsx_metal_backend_buffer_free,
    gsx_metal_backend_buffer_get_info,
    gsx_metal_backend_buffer_upload,
    gsx_metal_backend_buffer_download,
    gsx_metal_backend_buffer_set_zero,
    gsx_metal_backend_buffer_memset_tensor,
    gsx_metal_backend_buffer_set_tensor,
    gsx_metal_backend_buffer_get_tensor,
    gsx_metal_backend_buffer_copy_tensor,
    gsx_metal_backend_buffer_fill_tensor,
    gsx_metal_backend_buffer_check_finite_tensor,
    gsx_metal_backend_buffer_gather_tensor,
    gsx_metal_backend_buffer_resize_tensor,
    gsx_metal_backend_buffer_exp_tensor
};

gsx_metal_backend *gsx_metal_backend_from_base(gsx_backend_t backend)
{
    return (gsx_metal_backend *)backend;
}

gsx_metal_backend_buffer_type *gsx_metal_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type)
{
    return (gsx_metal_backend_buffer_type *)buffer_type;
}

gsx_metal_backend_buffer *gsx_metal_backend_buffer_from_base(gsx_backend_buffer_t buffer)
{
    return (gsx_metal_backend_buffer *)buffer;
}

gsx_backend_buffer_type_class gsx_metal_backend_buffer_get_type_class(gsx_backend_buffer_t buffer)
{
    return gsx_metal_backend_buffer_type_from_base(buffer->buffer_type)->info.type;
}

void gsx_metal_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes)
{
    gsx_size_t offset_bytes = 0;

    for(offset_bytes = 0; offset_bytes < total_bytes; offset_bytes += value_size_bytes) {
        memcpy((unsigned char *)dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
    }
}

bool gsx_metal_backend_f16_is_finite(uint16_t value)
{
    return ((value >> 10) & 0x1FU) != 0x1FU;
}

bool gsx_metal_backend_bf16_is_finite(uint16_t value)
{
    return ((value >> 7) & 0xFFU) != 0xFFU;
}

gsx_error gsx_metal_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count)
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

void gsx_metal_backend_init_buffer_type(
    gsx_metal_backend *metal_backend,
    gsx_metal_backend_buffer_type *buffer_type,
    gsx_backend_buffer_type_class type,
    const char *name,
    gsx_size_t alignment_bytes
)
{
    buffer_type->base.iface = &gsx_metal_backend_buffer_type_iface;
    buffer_type->base.backend = &metal_backend->base;
    buffer_type->base.live_arena_count = 0;
    buffer_type->info.backend = &metal_backend->base;
    buffer_type->info.type = type;
    buffer_type->info.name = name;
    buffer_type->info.alignment_bytes = alignment_bytes;
    buffer_type->info.max_allocation_size_bytes = 0;
}

gsx_error gsx_metal_backend_tensor_view_validate(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view)
{
    gsx_size_t tensor_end_bytes = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }
    if(gsx_size_add_overflows(tensor_view->offset_bytes, tensor_view->size_bytes, &tensor_end_bytes) || tensor_end_bytes > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor view exceeds backing buffer");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_tensor_view_check_range(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_size_t offset_bytes,
    gsx_size_t byte_count
)
{
    gsx_size_t tensor_end_bytes = 0;
    gsx_size_t absolute_offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(offset_bytes > tensor_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange offset is out of range");
    }
    if(gsx_size_add_overflows(offset_bytes, byte_count, &tensor_end_bytes) || tensor_end_bytes > tensor_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange exceeds tensor size");
    }
    if(gsx_size_add_overflows(tensor_view->offset_bytes, offset_bytes, &absolute_offset_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange absolute offset overflows");
    }

    return gsx_metal_backend_buffer_check_range(buffer, absolute_offset_bytes, byte_count);
}
