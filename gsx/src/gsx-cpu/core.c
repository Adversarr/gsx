#include "internal.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

static gsx_error gsx_cpu_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
static gsx_error gsx_cpu_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);
static gsx_error gsx_cpu_backend_free(gsx_backend_t backend);
static gsx_error gsx_cpu_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
static gsx_error gsx_cpu_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
static gsx_error gsx_cpu_backend_get_major_stream(gsx_backend_t backend, void **out_stream);
static gsx_error gsx_cpu_backend_major_stream_sync(gsx_backend_t backend);
static gsx_error gsx_cpu_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
static gsx_error gsx_cpu_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
static gsx_error gsx_cpu_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);
static gsx_error gsx_cpu_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
static gsx_error gsx_cpu_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
static gsx_error gsx_cpu_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);
static gsx_error gsx_cpu_backend_buffer_free(gsx_backend_buffer_t buffer);
static gsx_error gsx_cpu_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
static gsx_error gsx_cpu_backend_buffer_get_native_handle(gsx_backend_buffer_t buffer, void **out_handle);
static gsx_error gsx_cpu_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
static gsx_error gsx_cpu_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
static gsx_error gsx_cpu_backend_buffer_set_zero(gsx_backend_buffer_t buffer);
static gsx_error gsx_cpu_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
static gsx_error gsx_cpu_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
static gsx_error gsx_cpu_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
static gsx_error gsx_cpu_backend_buffer_copy_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view
);
static gsx_error gsx_cpu_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
);
static gsx_error gsx_cpu_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
);
static gsx_error gsx_cpu_backend_buffer_gather_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape
);
static gsx_error gsx_cpu_backend_buffer_unary_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
);
static gsx_error gsx_cpu_backend_buffer_unary_tensor_inplace(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
);
static gsx_error gsx_cpu_backend_buffer_unary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
);
static gsx_error gsx_cpu_backend_buffer_binary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
);
static gsx_error gsx_cpu_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
);

static const gsx_backend_provider_i gsx_cpu_backend_provider_iface = {
    gsx_cpu_backend_provider_discover_devices,
    gsx_cpu_backend_provider_create_backend
};

static const gsx_backend_i gsx_cpu_backend_iface = {
    gsx_cpu_backend_free,
    gsx_cpu_backend_get_info,
    gsx_cpu_backend_get_capabilities,
    gsx_cpu_backend_get_major_stream,
    gsx_cpu_backend_major_stream_sync,
    gsx_cpu_backend_count_buffer_types,
    gsx_cpu_backend_get_buffer_type,
    gsx_cpu_backend_find_buffer_type,
    gsx_cpu_backend_create_renderer,
    gsx_cpu_backend_create_loss,
    gsx_cpu_backend_create_optim,
    gsx_cpu_backend_create_adc
};

static const gsx_backend_buffer_type_i gsx_cpu_backend_buffer_type_iface = {
    gsx_cpu_backend_buffer_type_get_info,
    gsx_cpu_backend_buffer_type_get_alloc_size,
    gsx_cpu_backend_buffer_type_init_buffer
};

static const gsx_backend_buffer_i gsx_cpu_backend_buffer_iface = {
    gsx_cpu_backend_buffer_free,
    gsx_cpu_backend_buffer_get_info,
    gsx_cpu_backend_buffer_get_native_handle,
    gsx_cpu_backend_buffer_upload,
    gsx_cpu_backend_buffer_download,
    gsx_cpu_backend_buffer_set_zero,
    gsx_cpu_backend_buffer_memset_tensor,
    gsx_cpu_backend_buffer_set_tensor,
    gsx_cpu_backend_buffer_get_tensor,
    gsx_cpu_backend_buffer_copy_tensor,
    gsx_cpu_backend_buffer_fill_tensor,
    gsx_cpu_backend_buffer_check_finite_tensor,
    gsx_cpu_backend_buffer_gather_tensor,
    gsx_cpu_backend_buffer_unary_tensor,
    gsx_cpu_backend_buffer_unary_tensor_inplace,
    gsx_cpu_backend_buffer_unary_reduce_tensor,
    gsx_cpu_backend_buffer_binary_reduce_tensor,
    gsx_cpu_backend_buffer_clamp_inplace_tensor
};

static gsx_cpu_backend_provider gsx_cpu_backend_provider_singleton = { 0 };
static gsx_cpu_backend_device gsx_cpu_backend_device_singleton = { 0 };

static gsx_size_t gsx_cpu_detect_total_memory_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX memory_status = { 0 };

    memory_status.dwLength = sizeof(memory_status);
    if(GlobalMemoryStatusEx(&memory_status) == 0) {
        return 0;
    }
    return (gsx_size_t)memory_status.ullTotalPhys;
#elif defined(__APPLE__)
    uint64_t memory_bytes = 0;
    size_t memory_size = sizeof(memory_bytes);

    if(sysctlbyname("hw.memsize", &memory_bytes, &memory_size, NULL, 0) == 0) {
        return (gsx_size_t)memory_bytes;
    }
    return 0;
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
    uint64_t memory_bytes = 0;
    size_t memory_size = sizeof(memory_bytes);

    if(sysctlbyname("hw.physmem", &memory_bytes, &memory_size, NULL, 0) == 0) {
        return (gsx_size_t)memory_bytes;
    }
    return 0;
#else
    long physical_page_count = sysconf(_SC_PHYS_PAGES);
    long page_size_bytes = sysconf(_SC_PAGESIZE);
    gsx_size_t total_memory_bytes = 0;

    if(physical_page_count <= 0 || page_size_bytes <= 0) {
        return 0;
    }
    if(gsx_size_mul_overflows((gsx_size_t)physical_page_count, (gsx_size_t)page_size_bytes, &total_memory_bytes)) {
        return 0;
    }
    return total_memory_bytes;
#endif
}

static gsx_error gsx_cpu_round_alloc_size(gsx_size_t requested_size_bytes, gsx_size_t alignment_bytes, gsx_size_t *out_alloc_size_bytes)
{
    if(out_alloc_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_alloc_size_bytes must be non-null");
    }
    if(requested_size_bytes == 0) {
        *out_alloc_size_bytes = 0;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_round_up_overflows(requested_size_bytes, alignment_bytes, out_alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "requested allocation size overflows during alignment rounding");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_alloc_aligned(gsx_size_t alloc_size_bytes, gsx_size_t alignment_bytes, void **out_data)
{
    if(out_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_data must be non-null");
    }

    *out_data = NULL;
    if(alloc_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(alloc_size_bytes > (gsx_size_t)SIZE_MAX || alignment_bytes > (gsx_size_t)SIZE_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "aligned allocation cannot fit in size_t");
    }

#if defined(_WIN32)
    *out_data = _aligned_malloc((size_t)alloc_size_bytes, (size_t)alignment_bytes);
    if(*out_data == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "aligned allocation failed");
    }
#else
    *out_data = aligned_alloc((size_t)alignment_bytes, (size_t)alloc_size_bytes);
    if(*out_data == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "aligned allocation failed");
    }
#endif

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cpu_free_aligned(void *data)
{
#if defined(_WIN32)
    _aligned_free(data);
#else
    free(data);
#endif
}

static gsx_error gsx_cpu_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count)
{
    gsx_size_t end_offset_bytes = 0;

    if(offset_bytes > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer offset is out of range");
    }
    if(gsx_size_add_overflows(offset_bytes, byte_count, &end_offset_bytes) || end_offset_bytes > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer byte range exceeds the logical buffer size");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_tensor_view_validate(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view)
{
    gsx_size_t tensor_end_bytes = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference the provided buffer");
    }
    if(gsx_size_add_overflows(tensor_view->offset_bytes, tensor_view->size_bytes, &tensor_end_bytes) || tensor_end_bytes > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor view exceeds the backing buffer");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_tensor_view_check_range(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_size_t offset_bytes,
    gsx_size_t byte_count
)
{
    gsx_size_t tensor_end_bytes = 0;
    gsx_size_t absolute_offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cpu_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(offset_bytes > tensor_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange offset is out of range");
    }
    if(gsx_size_add_overflows(offset_bytes, byte_count, &tensor_end_bytes) || tensor_end_bytes > tensor_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange exceeds the tensor size");
    }
    if(gsx_size_add_overflows(tensor_view->offset_bytes, offset_bytes, &absolute_offset_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange absolute offset overflows");
    }

    return gsx_cpu_backend_buffer_check_range(buffer, absolute_offset_bytes, byte_count);
}

static unsigned char *gsx_cpu_backend_tensor_data(gsx_cpu_backend_buffer *cpu_buffer, const gsx_backend_tensor_view *tensor_view, gsx_size_t offset_bytes)
{
    return (unsigned char *)cpu_buffer->data + (size_t)(tensor_view->offset_bytes + offset_bytes);
}

static gsx_error gsx_cpu_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry)
{
    (void)provider;
    return gsx_builtin_registry_append_device(registry, &gsx_cpu_backend_device_singleton.base);
}

static void gsx_cpu_backend_init_buffer_type(gsx_cpu_backend *cpu_backend, gsx_cpu_backend_buffer_type *buffer_type, gsx_backend_buffer_type_class type, const char *name)
{
    buffer_type->base.iface = &gsx_cpu_backend_buffer_type_iface;
    buffer_type->base.backend = &cpu_backend->base;
    buffer_type->info.backend = &cpu_backend->base;
    buffer_type->info.type = type;
    buffer_type->info.name = name;
    buffer_type->info.alignment_bytes = 64;
    buffer_type->info.max_allocation_size_bytes = 0;
}

static gsx_error gsx_cpu_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend)
{
    gsx_cpu_backend *cpu_backend = NULL;

    if(desc->options_size_bytes != 0 && desc->options == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "non-zero options_size_bytes requires a non-null options pointer");
    }
    if(desc->options != NULL || desc->options_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu backend does not expose backend-specific options yet");
    }

    cpu_backend = (gsx_cpu_backend *)calloc(1, sizeof(*cpu_backend));
    if(cpu_backend == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu backend");
    }

    cpu_backend->base.iface = &gsx_cpu_backend_iface;
    cpu_backend->base.provider = &gsx_cpu_backend_provider_singleton.base;
    cpu_backend->base.device = backend_device;
    cpu_backend->base.live_buffer_count = 0;
    cpu_backend->base.live_arena_count = 0;
    cpu_backend->base.live_renderer_count = 0;
    cpu_backend->base.live_loss_count = 0;
    cpu_backend->base.live_optim_count = 0;
    cpu_backend->base.live_adc_count = 0;
    cpu_backend->capabilities.supported_data_types = GSX_DATA_TYPE_FLAG_F32 | GSX_DATA_TYPE_FLAG_U8 | GSX_DATA_TYPE_FLAG_I32;
    cpu_backend->capabilities.supports_async_prefetch = false;

    gsx_cpu_backend_init_buffer_type(cpu_backend, &cpu_backend->host_buffer_type, GSX_BACKEND_BUFFER_TYPE_HOST, "cpu-host");
    gsx_cpu_backend_init_buffer_type(cpu_backend, &cpu_backend->device_buffer_type, GSX_BACKEND_BUFFER_TYPE_DEVICE, "cpu-device");

    *out_backend = &cpu_backend->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_free(gsx_backend_t backend)
{
    gsx_cpu_backend *cpu_backend = (gsx_cpu_backend *)backend;

    if(backend->live_buffer_count != 0 || backend->live_arena_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "free backend arenas and buffers before freeing the backend");
    }
    if(backend->live_renderer_count != 0 || backend->live_loss_count != 0 || backend->live_optim_count != 0
        || backend->live_adc_count != 0) {
        return gsx_make_error(
            GSX_ERROR_INVALID_STATE, "free backend renderers, losses, optimizers, and adcs before freeing the backend");
    }

    free(cpu_backend);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info)
{
    out_info->backend_type = GSX_BACKEND_TYPE_CPU;
    out_info->device = backend->device;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities)
{
    gsx_cpu_backend *cpu_backend = (gsx_cpu_backend *)backend;

    *out_capabilities = cpu_backend->capabilities;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_get_major_stream(gsx_backend_t backend, void **out_stream)
{
    (void)backend;
    *out_stream = NULL;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_major_stream_sync(gsx_backend_t backend)
{
    (void)backend;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count)
{
    (void)backend;
    *out_count = 2;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_cpu_backend *cpu_backend = (gsx_cpu_backend *)backend;

    if(index == 0) {
        *out_buffer_type = &cpu_backend->host_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(index == 1) {
        *out_buffer_type = &cpu_backend->device_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cpu backend buffer-type index is out of range");
}

static gsx_error gsx_cpu_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_cpu_backend *cpu_backend = (gsx_cpu_backend *)backend;

    switch(type) {
    case GSX_BACKEND_BUFFER_TYPE_HOST:
        *out_buffer_type = &cpu_backend->host_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_DEVICE:
        *out_buffer_type = &cpu_backend->device_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
    case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu backend does not expose that buffer-type class");
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer-type class is out of range");
}

static gsx_error gsx_cpu_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info)
{
    gsx_cpu_backend_buffer_type *cpu_buffer_type = (gsx_cpu_backend_buffer_type *)buffer_type;

    *out_info = cpu_buffer_type->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes)
{
    gsx_cpu_backend_buffer_type *cpu_buffer_type = (gsx_cpu_backend_buffer_type *)buffer_type;

    return gsx_cpu_round_alloc_size(requested_size_bytes, cpu_buffer_type->info.alignment_bytes, out_alloc_size_bytes);
}

static gsx_error gsx_cpu_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer)
{
    gsx_cpu_backend_buffer_type *cpu_buffer_type = (gsx_cpu_backend_buffer_type *)buffer_type;
    gsx_cpu_backend_buffer *cpu_buffer = NULL;
    gsx_size_t requested_alignment_bytes = desc->alignment_bytes;
    gsx_size_t effective_alignment_bytes = cpu_buffer_type->info.alignment_bytes;
    gsx_size_t alloc_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(desc->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->size_bytes must be non-zero");
    }
    if(requested_alignment_bytes != 0 && !gsx_is_power_of_two(requested_alignment_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "explicit buffer alignment must be a power of two");
    }
    if(requested_alignment_bytes > effective_alignment_bytes) {
        effective_alignment_bytes = requested_alignment_bytes;
    }

    error = gsx_cpu_round_alloc_size(desc->size_bytes, effective_alignment_bytes, &alloc_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cpu_buffer = (gsx_cpu_backend_buffer *)calloc(1, sizeof(*cpu_buffer));
    if(cpu_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu backend buffer");
    }

    error = gsx_cpu_alloc_aligned(alloc_size_bytes, effective_alignment_bytes, &cpu_buffer->data);
    if(!gsx_error_is_success(error)) {
        free(cpu_buffer);
        return error;
    }

    cpu_buffer->base.iface = &gsx_cpu_backend_buffer_iface;
    cpu_buffer->base.buffer_type = buffer_type;
    cpu_buffer->base.size_bytes = desc->size_bytes;
    cpu_buffer->base.alignment_bytes = effective_alignment_bytes;
    cpu_buffer->alloc_size_bytes = alloc_size_bytes;
    buffer_type->backend->live_buffer_count += 1;

    *out_buffer = &cpu_buffer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_free(gsx_backend_buffer_t buffer)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    gsx_backend_t backend = buffer->buffer_type->backend;

    if(backend->live_buffer_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_buffer_count underflow in buffer free");
    }

    gsx_cpu_free_aligned(cpu_buffer->data);
    backend->live_buffer_count -= 1;
    free(cpu_buffer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info)
{
    out_info->backend = buffer->buffer_type->backend;
    out_info->buffer_type = buffer->buffer_type;
    out_info->size_bytes = buffer->size_bytes;
    out_info->alignment_bytes = buffer->alignment_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_get_native_handle(gsx_backend_buffer_t buffer, void **out_handle)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;

    if(out_handle == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_handle must be non-null");
    }

    *out_handle = cpu_buffer->data;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(byte_count != 0 && src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null when byte_count is non-zero");
    }

    error = gsx_cpu_backend_buffer_check_range(buffer, dst_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    memcpy((unsigned char *)cpu_buffer->data + (size_t)dst_offset_bytes, src_bytes, (size_t)byte_count);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(byte_count != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when byte_count is non-zero");
    }

    error = gsx_cpu_backend_buffer_check_range(buffer, src_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    memcpy(dst_bytes, (const unsigned char *)cpu_buffer->data + (size_t)src_offset_bytes, (size_t)byte_count);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_set_zero(gsx_backend_buffer_t buffer)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;

    if(cpu_buffer->alloc_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    memset(cpu_buffer->data, 0, (size_t)cpu_buffer->alloc_size_bytes);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    gsx_error error = gsx_cpu_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);

    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    memset(gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, offset_bytes), value, (size_t)size_bytes);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(size_bytes != 0 && src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null when size_bytes is non-zero");
    }

    error = gsx_cpu_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);
    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    memcpy(gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, offset_bytes), src_bytes, (size_t)size_bytes);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(size_bytes != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when size_bytes is non-zero");
    }

    error = gsx_cpu_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);
    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    memcpy(dst_bytes, gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, offset_bytes), (size_t)size_bytes);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_copy_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view
)
{
    gsx_cpu_backend_buffer *src_cpu_buffer = NULL;
    gsx_cpu_backend_buffer *dst_cpu_buffer = NULL;
    gsx_size_t src_begin_bytes = 0;
    gsx_size_t src_end_bytes = 0;
    gsx_size_t dst_begin_bytes = 0;
    gsx_size_t dst_end_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || src_view == NULL || dst_view == NULL || src_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and tensor views must be non-null");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend || dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy requires source and destination to belong to the same backend");
    }

    error = gsx_cpu_backend_tensor_view_validate(src_view->buffer, src_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(dst_buffer, dst_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(src_view->size_bytes != dst_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy requires equal source and destination sizes");
    }
    if(src_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    src_begin_bytes = src_view->offset_bytes;
    src_end_bytes = src_view->offset_bytes + src_view->size_bytes;
    dst_begin_bytes = dst_view->offset_bytes;
    dst_end_bytes = dst_view->offset_bytes + dst_view->size_bytes;
    if(src_view->buffer == dst_buffer) {
        if(src_begin_bytes == dst_begin_bytes && src_end_bytes == dst_end_bytes) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        if(!(dst_end_bytes <= src_begin_bytes || src_end_bytes <= dst_begin_bytes)) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy rejects overlapping source and destination ranges");
        }
    }

    src_cpu_buffer = (gsx_cpu_backend_buffer *)src_view->buffer;
    dst_cpu_buffer = (gsx_cpu_backend_buffer *)dst_buffer;
    memcpy(
        (unsigned char *)dst_cpu_buffer->data + (size_t)dst_begin_bytes,
        (const unsigned char *)src_cpu_buffer->data + (size_t)src_begin_bytes,
        (size_t)src_view->size_bytes
    );
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    unsigned char *dst_bytes = NULL;
    gsx_size_t offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(value_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_size_bytes must be non-zero");
    }
    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->size_bytes != 0 && value_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_bytes must be non-null when the tensor is non-empty");
    }
    if(tensor_view->size_bytes % value_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of value_size_bytes");
    }

    error = gsx_cpu_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error) || tensor_view->size_bytes == 0) {
        return error;
    }

    dst_bytes = gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, 0);
    for(offset_bytes = 0; offset_bytes < tensor_view->size_bytes; offset_bytes += value_size_bytes) {
        memcpy(dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    const float *values = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_is_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_is_finite must be non-null");
    }
    *out_is_finite = true;

    error = gsx_cpu_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "finite check is only implemented for float32 tensors");
    }
    if(tensor_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensors must have a byte size divisible by sizeof(float)");
    }

    values = (const float *)gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, 0);
    element_count = tensor_view->size_bytes / sizeof(float);
    for(element_index = 0; element_index < element_count; ++element_index) {
        if(!isfinite((double)values[element_index])) {
            *out_is_finite = false;
            break;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_tensor_compute_total_bytes(
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_size_t *out_total_bytes,
    gsx_size_t *out_row_bytes
)
{
    gsx_size_t element_size_bytes = 0;
    gsx_size_t element_count = 1;
    gsx_size_t row_elements = 1;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(shape == NULL || out_total_bytes == NULL || out_row_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "shape and output pointers must be non-null");
    }
    if(rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "rank must be at least 1");
    }
    error = gsx_data_type_get_size_bytes(data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    for(dim = 0; dim < rank; ++dim) {
        if(shape[dim] <= 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "shape entries must be positive");
        }
        if(gsx_size_mul_overflows(element_count, (gsx_size_t)shape[dim], &element_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor element count overflows");
        }
        if(dim >= 1 && gsx_size_mul_overflows(row_elements, (gsx_size_t)shape[dim], &row_elements)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor row element count overflows");
        }
    }
    if(gsx_size_mul_overflows(element_count, element_size_bytes, out_total_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor byte size overflows");
    }
    if(gsx_size_mul_overflows(row_elements, element_size_bytes, out_row_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor row byte size overflows");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_gather_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape
)
{
    gsx_cpu_backend_buffer *x_buffer = NULL;
    gsx_cpu_backend_buffer *index_buffer = NULL;
    gsx_cpu_backend_buffer *out_buffer = NULL;
    const int32_t *indices = NULL;
    const unsigned char *x_bytes = NULL;
    unsigned char *out_bytes = NULL;
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_out_bytes = 0;
    gsx_size_t expected_index_bytes = 0;
    gsx_size_t x_row_bytes = 0;
    gsx_size_t out_row_bytes = 0;
    gsx_size_t row_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || x_view == NULL || index_view == NULL || out_view == NULL || x_shape == NULL || out_shape == NULL
        || x_view->buffer == NULL || index_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shapes must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || index_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "all gather tensors must belong to the same backend");
    }
    if(index_view->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index tensor must use int32");
    }
    if(x_rank != out_rank || x_rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out ranks must match and be at least 1");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out data types must match");
    }

    error = gsx_cpu_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(index_view->buffer, index_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_cpu_backend_tensor_compute_total_bytes(x_view->data_type, x_rank, x_shape, &expected_x_bytes, &x_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_compute_total_bytes(out_view->data_type, out_rank, out_shape, &expected_out_bytes, &out_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(expected_x_bytes != x_view->size_bytes || expected_out_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match the provided shape metadata");
    }
    if(x_row_bytes != out_row_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out trailing dimensions must match");
    }
    if(gsx_size_mul_overflows((gsx_size_t)out_shape[0], sizeof(int32_t), &expected_index_bytes) || expected_index_bytes != index_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index view byte size must match out leading dimension");
    }

    x_buffer = (gsx_cpu_backend_buffer *)x_view->buffer;
    index_buffer = (gsx_cpu_backend_buffer *)index_view->buffer;
    out_buffer = (gsx_cpu_backend_buffer *)dst_buffer;
    x_bytes = gsx_cpu_backend_tensor_data(x_buffer, x_view, 0);
    indices = (const int32_t *)gsx_cpu_backend_tensor_data(index_buffer, index_view, 0);
    out_bytes = gsx_cpu_backend_tensor_data(out_buffer, out_view, 0);

    for(row_index = 0; row_index < (gsx_size_t)out_shape[0]; ++row_index) {
        int32_t src_row = indices[row_index];

        if(src_row < 0 || src_row >= x_shape[0]) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather index is out of range");
        }
        memcpy(
            out_bytes + row_index * out_row_bytes,
            x_bytes + (gsx_size_t)src_row * x_row_bytes,
            x_row_bytes
        );
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_apply_unary_tensor_f32(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
)
{
    gsx_cpu_backend_buffer *x_buffer = NULL;
    gsx_cpu_backend_buffer *out_buffer = NULL;
    const float *x_values = NULL;
    float *out_values = NULL;
    gsx_size_t expected_bytes = 0;
    gsx_size_t row_bytes = 0;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || shape == NULL || x_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shape must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must belong to the same backend");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out data types must match");
    }

    error = gsx_cpu_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_compute_total_bytes(x_view->data_type, rank, shape, &expected_bytes, &row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(row_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "row byte size must be non-zero");
    }
    if(expected_bytes != x_view->size_bytes || expected_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match the provided shape metadata");
    }

    if(x_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op only supports float32 tensors on cpu backend");
    }

    x_buffer = (gsx_cpu_backend_buffer *)x_view->buffer;
    out_buffer = (gsx_cpu_backend_buffer *)dst_buffer;
    x_values = (const float *)gsx_cpu_backend_tensor_data(x_buffer, x_view, 0);
    out_values = (float *)gsx_cpu_backend_tensor_data(out_buffer, out_view, 0);
    element_count = expected_bytes / sizeof(float);

    for(element_index = 0; element_index < element_count; ++element_index) {
        float x_value = x_values[element_index];
        switch(op) {
        case GSX_IMPL_UNARY_OP_EXP:
            out_values[element_index] = expf(x_value);
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID:
            out_values[element_index] = 1.0f / (1.0f + expf(-x_value));
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE: {
            float sigmoid_value = 1.0f / (1.0f + expf(-x_value));
            out_values[element_index] = sigmoid_value * (1.0f - sigmoid_value);
            break;
        }
        case GSX_IMPL_UNARY_OP_ABS:
            out_values[element_index] = fabsf(x_value);
            break;
        default:
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_apply_unary_inplace_tensor_f32(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
)
{
    gsx_cpu_backend_buffer *cpu_buffer = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    float *values = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer == NULL || tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference buffer");
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op only supports float32 tensors on cpu backend");
    }

    error = gsx_cpu_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    values = (float *)gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, 0);
    element_count = tensor_view->size_bytes / sizeof(float);

    for(element_index = 0; element_index < element_count; ++element_index) {
        float x_value = values[element_index];
        switch(op) {
        case GSX_IMPL_UNARY_OP_EXP:
            values[element_index] = expf(x_value);
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID:
            values[element_index] = 1.0f / (1.0f + expf(-x_value));
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE: {
            float sigmoid_value = 1.0f / (1.0f + expf(-x_value));
            values[element_index] = sigmoid_value * (1.0f - sigmoid_value);
            break;
        }
        case GSX_IMPL_UNARY_OP_ABS:
            values[element_index] = fabsf(x_value);
            break;
        default:
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_unary_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
)
{
    return gsx_cpu_backend_buffer_apply_unary_tensor_f32(dst_buffer, x_view, out_view, rank, shape, op);
}

static gsx_error gsx_cpu_backend_buffer_unary_tensor_inplace(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
)
{
    return gsx_cpu_backend_buffer_apply_unary_inplace_tensor_f32(buffer, tensor_view, op);
}

static gsx_error gsx_cpu_backend_reduce_validate_shape_contract(
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_reduce_count
)
{
    gsx_index_t dim = 0;
    gsx_size_t outer_count = 1;
    gsx_size_t reduce_count = 1;
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_out_bytes = 0;
    gsx_size_t row_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(x_view == NULL || out_view == NULL || x_shape == NULL || out_shape == NULL || out_outer_count == NULL
        || out_reduce_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce shape inputs must be non-null");
    }
    if(start_axis < 0 || start_axis >= x_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "start_axis must be in range [0, x_rank)");
    }
    if(out_rank != start_axis + 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_rank must equal start_axis + 1");
    }
    if(out_shape[start_axis] != 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out reduced axis extent must be 1");
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(x_shape[dim] != out_shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out prefix shape must match x prefix shape");
        }
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(gsx_size_mul_overflows(outer_count, (gsx_size_t)x_shape[dim], &outer_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "outer_count overflows");
        }
    }
    for(dim = start_axis; dim < x_rank; ++dim) {
        if(gsx_size_mul_overflows(reduce_count, (gsx_size_t)x_shape[dim], &reduce_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce_count overflows");
        }
    }
    error = gsx_cpu_backend_tensor_compute_total_bytes(x_view->data_type, x_rank, x_shape, &expected_x_bytes, &row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_compute_total_bytes(out_view->data_type, out_rank, out_shape, &expected_out_bytes, &row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(expected_x_bytes != x_view->size_bytes || expected_out_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match provided reduce shape metadata");
    }
    *out_outer_count = outer_count;
    *out_reduce_count = reduce_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_unary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
)
{
    gsx_cpu_backend_buffer *x_buffer = NULL;
    gsx_cpu_backend_buffer *out_buffer = NULL;
    const float *x_values = NULL;
    float *out_values = NULL;
    gsx_size_t outer_count = 0;
    gsx_size_t reduce_count = 0;
    gsx_size_t outer_index = 0;
    gsx_size_t reduce_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || workspace_view == NULL || x_shape == NULL || out_shape == NULL
        || x_view->buffer == NULL || workspace_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce buffers, views, and shapes must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || workspace_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce tensors and workspace must belong to the same backend");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x_view and out_view data_type must match");
    }
    if(x_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary_reduce only supports float32 tensors on cpu backend");
    }
    error = gsx_cpu_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(workspace_view->buffer, workspace_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_reduce_validate_shape_contract(
        x_view, out_view, x_rank, x_shape, out_rank, out_shape, start_axis, &outer_count, &reduce_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    x_buffer = (gsx_cpu_backend_buffer *)x_view->buffer;
    out_buffer = (gsx_cpu_backend_buffer *)dst_buffer;
    x_values = (const float *)gsx_cpu_backend_tensor_data(x_buffer, x_view, 0);
    out_values = (float *)gsx_cpu_backend_tensor_data(out_buffer, out_view, 0);

    for(outer_index = 0; outer_index < outer_count; ++outer_index) {
        gsx_size_t base_index = outer_index * reduce_count;
        float accum = 0.0f;

        if(op == GSX_IMPL_UNARY_REDUCE_OP_MAX) {
            accum = x_values[base_index];
        }
        for(reduce_index = 0; reduce_index < reduce_count; ++reduce_index) {
            float value = x_values[base_index + reduce_index];
            switch(op) {
            case GSX_IMPL_UNARY_REDUCE_OP_SUM:
            case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
                accum += value;
                break;
            case GSX_IMPL_UNARY_REDUCE_OP_MAX:
                if(value > accum) {
                    accum = value;
                }
                break;
            default:
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary_reduce op");
            }
        }
        if(op == GSX_IMPL_UNARY_REDUCE_OP_MEAN) {
            accum /= (float)reduce_count;
        }
        out_values[outer_index] = accum;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_binary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
)
{
    gsx_cpu_backend_buffer *lhs_buffer = NULL;
    gsx_cpu_backend_buffer *rhs_buffer = NULL;
    gsx_cpu_backend_buffer *out_buffer = NULL;
    const float *lhs_values = NULL;
    const float *rhs_values = NULL;
    float *out_values = NULL;
    gsx_size_t outer_count = 0;
    gsx_size_t reduce_count = 0;
    gsx_size_t outer_index = 0;
    gsx_size_t reduce_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || lhs_view == NULL || rhs_view == NULL || out_view == NULL || workspace_view == NULL || lhs_shape == NULL
        || rhs_shape == NULL || out_shape == NULL || lhs_view->buffer == NULL || rhs_view->buffer == NULL
        || workspace_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce buffers, views, and shapes must be non-null");
    }
    if(rhs_rank != lhs_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs_rank and rhs_rank must match");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(lhs_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || rhs_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || workspace_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce tensors and workspace must belong to the same backend");
    }
    if(lhs_view->data_type != rhs_view->data_type || lhs_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce tensor data_type must match");
    }
    if(lhs_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "binary_reduce only supports float32 tensors on cpu backend");
    }
    error = gsx_cpu_backend_tensor_view_validate(lhs_view->buffer, lhs_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(rhs_view->buffer, rhs_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_tensor_view_validate(workspace_view->buffer, workspace_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_reduce_validate_shape_contract(
        lhs_view, out_view, lhs_rank, lhs_shape, out_rank, out_shape, start_axis, &outer_count, &reduce_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_backend_reduce_validate_shape_contract(
        rhs_view, out_view, rhs_rank, rhs_shape, out_rank, out_shape, start_axis, &outer_count, &reduce_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    lhs_buffer = (gsx_cpu_backend_buffer *)lhs_view->buffer;
    rhs_buffer = (gsx_cpu_backend_buffer *)rhs_view->buffer;
    out_buffer = (gsx_cpu_backend_buffer *)dst_buffer;
    lhs_values = (const float *)gsx_cpu_backend_tensor_data(lhs_buffer, lhs_view, 0);
    rhs_values = (const float *)gsx_cpu_backend_tensor_data(rhs_buffer, rhs_view, 0);
    out_values = (float *)gsx_cpu_backend_tensor_data(out_buffer, out_view, 0);

    for(outer_index = 0; outer_index < outer_count; ++outer_index) {
        gsx_size_t base_index = outer_index * reduce_count;
        float accum = 0.0f;

        for(reduce_index = 0; reduce_index < reduce_count; ++reduce_index) {
            float diff = lhs_values[base_index + reduce_index] - rhs_values[base_index + reduce_index];
            switch(op) {
            case GSX_IMPL_BINARY_REDUCE_OP_MSE:
                accum += diff * diff;
                break;
            case GSX_IMPL_BINARY_REDUCE_OP_MAE:
                accum += fabsf(diff);
                break;
            default:
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown binary_reduce op");
            }
        }
        out_values[outer_index] = accum / (float)reduce_count;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
)
{
    gsx_cpu_backend_buffer *cpu_buffer = NULL;
    gsx_size_t element_size_bytes = 0;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer == NULL || tensor_view == NULL || min_value == NULL || max_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor_view, min_value, and max_value must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference buffer");
    }

    error = gsx_cpu_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_data_type_get_size_bytes(tensor_view->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes % element_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor view byte size is not aligned to element size");
    }

    cpu_buffer = (gsx_cpu_backend_buffer *)buffer;
    element_count = tensor_view->size_bytes / element_size_bytes;

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32: {
        float *values = (float *)gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, 0);
        const float min_bound = *(const float *)min_value;
        const float max_bound = *(const float *)max_value;

        for(element_index = 0; element_index < element_count; ++element_index) {
            if(values[element_index] < min_bound) {
                values[element_index] = min_bound;
            } else if(values[element_index] > max_bound) {
                values[element_index] = max_bound;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_U8: {
        uint8_t *values = (uint8_t *)gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, 0);
        const uint8_t min_bound = *(const uint8_t *)min_value;
        const uint8_t max_bound = *(const uint8_t *)max_value;

        for(element_index = 0; element_index < element_count; ++element_index) {
            if(values[element_index] < min_bound) {
                values[element_index] = min_bound;
            } else if(values[element_index] > max_bound) {
                values[element_index] = max_bound;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_I32: {
        int32_t *values = (int32_t *)gsx_cpu_backend_tensor_data(cpu_buffer, tensor_view, 0);
        const int32_t min_bound = *(const int32_t *)min_value;
        const int32_t max_bound = *(const int32_t *)max_value;

        for(element_index = 0; element_index < element_count; ++element_index) {
            if(values[element_index] < min_bound) {
                values[element_index] = min_bound;
            } else if(values[element_index] > max_bound) {
                values[element_index] = max_bound;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp_inplace only supports f32, u8, and i32 tensors on cpu backend");
    }
}

gsx_error gsx_cpu_backend_provider_bootstrap(gsx_builtin_registry_state *registry)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    gsx_cpu_backend_provider_singleton.base.iface = &gsx_cpu_backend_provider_iface;
    gsx_cpu_backend_provider_singleton.base.backend_type = GSX_BACKEND_TYPE_CPU;
    gsx_cpu_backend_provider_singleton.base.backend_name = "cpu";

    gsx_cpu_backend_device_singleton.base.provider = &gsx_cpu_backend_provider_singleton.base;
    gsx_cpu_backend_device_singleton.base.info.backend_type = GSX_BACKEND_TYPE_CPU;
    gsx_cpu_backend_device_singleton.base.info.backend_name = gsx_cpu_backend_provider_singleton.base.backend_name;
    gsx_cpu_backend_device_singleton.base.info.device_index = 0;
    gsx_cpu_backend_device_singleton.base.info.name = "cpu0";
    gsx_cpu_backend_device_singleton.base.info.total_memory_bytes = gsx_cpu_detect_total_memory_bytes();

    error = gsx_cpu_backend_provider_singleton.base.iface->discover_devices(&gsx_cpu_backend_provider_singleton.base, registry);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_builtin_registry_append_provider(registry, &gsx_cpu_backend_provider_singleton.base);
}
