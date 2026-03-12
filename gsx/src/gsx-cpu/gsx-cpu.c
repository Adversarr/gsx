#include "../gsx-impl.h"

#include <math.h>
#include <errno.h>
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

typedef struct gsx_cpu_backend_provider {
    struct gsx_backend_provider base;
} gsx_cpu_backend_provider;

typedef struct gsx_cpu_backend_device {
    struct gsx_backend_device base;
} gsx_cpu_backend_device;

typedef struct gsx_cpu_backend_buffer_type {
    struct gsx_backend_buffer_type base;
    gsx_backend_buffer_type_info info;
} gsx_cpu_backend_buffer_type;

typedef struct gsx_cpu_backend {
    struct gsx_backend base;
    gsx_backend_capabilities capabilities;
    gsx_cpu_backend_buffer_type host_buffer_type;
    gsx_cpu_backend_buffer_type device_buffer_type;
} gsx_cpu_backend;

typedef struct gsx_cpu_backend_buffer {
    struct gsx_backend_buffer base;
    void *data;
    gsx_size_t alloc_size_bytes;
} gsx_cpu_backend_buffer;

static gsx_error gsx_cpu_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
static gsx_error gsx_cpu_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);
static gsx_error gsx_cpu_backend_free(gsx_backend_t backend);
static gsx_error gsx_cpu_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
static gsx_error gsx_cpu_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
static gsx_error gsx_cpu_backend_get_major_stream(gsx_backend_t backend, void **out_stream);
static gsx_error gsx_cpu_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
static gsx_error gsx_cpu_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
static gsx_error gsx_cpu_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);
static gsx_error gsx_cpu_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
static gsx_error gsx_cpu_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
static gsx_error gsx_cpu_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);
static gsx_error gsx_cpu_backend_buffer_free(gsx_backend_buffer_t buffer);
static gsx_error gsx_cpu_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
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

static const gsx_backend_provider_i gsx_cpu_backend_provider_iface = {
    gsx_cpu_backend_provider_discover_devices,
    gsx_cpu_backend_provider_create_backend
};

static const gsx_backend_i gsx_cpu_backend_iface = {
    gsx_cpu_backend_free,
    gsx_cpu_backend_get_info,
    gsx_cpu_backend_get_capabilities,
    gsx_cpu_backend_get_major_stream,
    gsx_cpu_backend_count_buffer_types,
    gsx_cpu_backend_get_buffer_type,
    gsx_cpu_backend_find_buffer_type
};

static const gsx_backend_buffer_type_i gsx_cpu_backend_buffer_type_iface = {
    gsx_cpu_backend_buffer_type_get_info,
    gsx_cpu_backend_buffer_type_get_alloc_size,
    gsx_cpu_backend_buffer_type_init_buffer
};

static const gsx_backend_buffer_i gsx_cpu_backend_buffer_iface = {
    gsx_cpu_backend_buffer_free,
    gsx_cpu_backend_buffer_get_info,
    gsx_cpu_backend_buffer_upload,
    gsx_cpu_backend_buffer_download,
    gsx_cpu_backend_buffer_set_zero,
    gsx_cpu_backend_buffer_memset_tensor,
    gsx_cpu_backend_buffer_set_tensor,
    gsx_cpu_backend_buffer_get_tensor,
    gsx_cpu_backend_buffer_copy_tensor,
    gsx_cpu_backend_buffer_fill_tensor,
    gsx_cpu_backend_buffer_check_finite_tensor
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
    {
        void *data = NULL;
        int alloc_status = posix_memalign(&data, (size_t)alignment_bytes, (size_t)alloc_size_bytes);

        if(alloc_status == EINVAL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "aligned allocation arguments are invalid");
        }
        if(alloc_status != 0) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "aligned allocation failed");
        }
        *out_data = data;
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
    cpu_backend->capabilities.supported_data_types = GSX_DATA_TYPE_FLAG_F32;
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
