#include "gsx-impl.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_cuda_backend_provider gsx_cuda_backend_provider;
typedef struct gsx_cuda_backend_device gsx_cuda_backend_device;
typedef struct gsx_cuda_backend gsx_cuda_backend;
typedef struct gsx_cuda_backend_buffer_type gsx_cuda_backend_buffer_type;
typedef struct gsx_cuda_backend_buffer gsx_cuda_backend_buffer;

struct gsx_cuda_backend_provider {
    struct gsx_backend_provider base;
};

struct gsx_cuda_backend_device {
    struct gsx_backend_device base;
    int cuda_device_ordinal;
    int compute_capability_major;
    int compute_capability_minor;
    char device_name[256];
};

struct gsx_cuda_backend_buffer_type {
    struct gsx_backend_buffer_type base;
    gsx_backend_buffer_type_info info;
};

struct gsx_cuda_backend {
    struct gsx_backend base;
    gsx_backend_capabilities capabilities;
    cudaStream_t major_stream;
    gsx_cuda_backend_buffer_type device_buffer_type;
    gsx_cuda_backend_buffer_type host_pinned_buffer_type;
};

struct gsx_cuda_backend_buffer {
    struct gsx_backend_buffer base;
    void *ptr;
    gsx_size_t alloc_size_bytes;
};

static gsx_cuda_backend_provider gsx_cuda_backend_provider_singleton = { 0 };
static gsx_cuda_backend_device *gsx_cuda_backend_devices = NULL;
static int gsx_cuda_device_count = 0;

static gsx_error gsx_cuda_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
static gsx_error gsx_cuda_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);

static const gsx_backend_provider_i gsx_cuda_backend_provider_iface = {
    gsx_cuda_backend_provider_discover_devices,
    gsx_cuda_backend_provider_create_backend
};

static gsx_error gsx_cuda_backend_free(gsx_backend_t backend);
static gsx_error gsx_cuda_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
static gsx_error gsx_cuda_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
static gsx_error gsx_cuda_backend_get_major_stream(gsx_backend_t backend, void **out_stream);
static gsx_error gsx_cuda_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
static gsx_error gsx_cuda_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
static gsx_error gsx_cuda_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);

static const gsx_backend_i gsx_cuda_backend_iface = {
    gsx_cuda_backend_free,
    gsx_cuda_backend_get_info,
    gsx_cuda_backend_get_capabilities,
    gsx_cuda_backend_get_major_stream,
    gsx_cuda_backend_count_buffer_types,
    gsx_cuda_backend_get_buffer_type,
    gsx_cuda_backend_find_buffer_type
};

static gsx_error gsx_cuda_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
static gsx_error gsx_cuda_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
static gsx_error gsx_cuda_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);

static const gsx_backend_buffer_type_i gsx_cuda_backend_buffer_type_iface = {
    gsx_cuda_backend_buffer_type_get_info,
    gsx_cuda_backend_buffer_type_get_alloc_size,
    gsx_cuda_backend_buffer_type_init_buffer
};

static gsx_error gsx_cuda_backend_buffer_free(gsx_backend_buffer_t buffer);
static gsx_error gsx_cuda_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
static gsx_error gsx_cuda_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
static gsx_error gsx_cuda_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
static gsx_error gsx_cuda_backend_buffer_set_zero(gsx_backend_buffer_t buffer);
static gsx_error gsx_cuda_backend_buffer_memset_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, uint8_t value, gsx_size_t offset_bytes, gsx_size_t size_bytes);
static gsx_error gsx_cuda_backend_buffer_set_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, const void *src_bytes, gsx_size_t offset_bytes, gsx_size_t size_bytes);
static gsx_error gsx_cuda_backend_buffer_get_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, void *dst_bytes, gsx_size_t offset_bytes, gsx_size_t size_bytes);
static gsx_error gsx_cuda_backend_buffer_copy_tensor(gsx_backend_buffer_t dst_buffer, const gsx_backend_tensor_view *src_view, const gsx_backend_tensor_view *dst_view);
static gsx_error gsx_cuda_backend_buffer_fill_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, const void *value_bytes, gsx_size_t value_size_bytes);
static gsx_error gsx_cuda_backend_buffer_check_finite_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, bool *out_is_finite);

static const gsx_backend_buffer_i gsx_cuda_backend_buffer_iface = {
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

extern void gsx_cuda_fill_tensor_kernel_launch(void *dst, const void *value, gsx_size_t value_size, gsx_size_t total_bytes, gsx_size_t alignment_bytes, cudaStream_t stream);
extern void gsx_cuda_check_finite_tensor_f32_kernel_launch(const void *src, gsx_size_t total_elements, gsx_size_t alignment_bytes, int *out_has_non_finite, cudaStream_t stream);
extern void gsx_cuda_check_finite_tensor_f16_kernel_launch(const void *src, gsx_size_t total_elements, gsx_size_t alignment_bytes, int *out_has_non_finite, cudaStream_t stream);
extern void gsx_cuda_check_finite_tensor_bf16_kernel_launch(const void *src, gsx_size_t total_elements, gsx_size_t alignment_bytes, int *out_has_non_finite, cudaStream_t stream);

static gsx_error gsx_cuda_make_error(cudaError_t cuda_err, const char *context)
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

static gsx_cuda_backend *gsx_cuda_backend_from_base(gsx_backend_t backend)
{
    return (gsx_cuda_backend *)backend;
}

static gsx_cuda_backend_buffer_type *gsx_cuda_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type)
{
    return (gsx_cuda_backend_buffer_type *)buffer_type;
}

static gsx_cuda_backend_buffer *gsx_cuda_backend_buffer_from_base(gsx_backend_buffer_t buffer)
{
    return (gsx_cuda_backend_buffer *)buffer;
}

static gsx_backend_buffer_type_class gsx_cuda_backend_buffer_get_type_class(gsx_backend_buffer_t buffer)
{
    return gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type)->info.type;
}

static void gsx_cuda_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes)
{
    gsx_size_t offset_bytes = 0;

    for(offset_bytes = 0; offset_bytes < total_bytes; offset_bytes += value_size_bytes) {
        memcpy((unsigned char *)dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
    }
}

static bool gsx_cuda_backend_f16_is_finite(uint16_t value)
{
    return ((value >> 10) & 0x1FU) != 0x1FU;
}

static bool gsx_cuda_backend_bf16_is_finite(uint16_t value)
{
    return ((value >> 7) & 0xFFU) != 0xFFU;
}

static gsx_error gsx_cuda_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count)
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

static gsx_error gsx_cuda_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry)
{
    cudaError_t cuda_err = cudaSuccess;
    int device_count = 0;
    int device_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    struct cudaDeviceProp prop;

    (void)provider;

    cuda_err = cudaGetDeviceCount(&device_count);
    if(cuda_err == cudaErrorNoDevice) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaGetDeviceCount failed");
    }
    if(device_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_cuda_backend_devices = (gsx_cuda_backend_device *)calloc((size_t)device_count, sizeof(*gsx_cuda_backend_devices));
    if(gsx_cuda_backend_devices == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate CUDA device storage");
    }
    gsx_cuda_device_count = device_count;

    for(device_index = 0; device_index < device_count; ++device_index) {
        gsx_cuda_backend_device *dev = &gsx_cuda_backend_devices[device_index];
        size_t total_memory = 0;

        memset(&prop, 0, sizeof(prop));
        cuda_err = cudaGetDeviceProperties(&prop, device_index);
        if(cuda_err != cudaSuccess) {
            continue;
        }

        total_memory = prop.totalGlobalMem;

        dev->base.provider = &gsx_cuda_backend_provider_singleton.base;
        dev->base.info.backend_type = GSX_BACKEND_TYPE_CUDA;
        dev->base.info.backend_name = "cuda";
        dev->base.info.device_index = device_index;
        strncpy(dev->device_name, prop.name, sizeof(dev->device_name) - 1);
        dev->device_name[sizeof(dev->device_name) - 1] = '\0';
        dev->base.info.name = dev->device_name;
        dev->base.info.total_memory_bytes = (gsx_size_t)total_memory;
        dev->cuda_device_ordinal = device_index;
        dev->compute_capability_major = prop.major;
        dev->compute_capability_minor = prop.minor;

        error = gsx_builtin_registry_append_device(registry, &dev->base);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cuda_backend_init_buffer_type(gsx_cuda_backend *cuda_backend, gsx_cuda_backend_buffer_type *buffer_type, gsx_backend_buffer_type_class type, const char *name, gsx_size_t alignment_bytes)
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

static gsx_error gsx_cuda_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend)
{
    gsx_cuda_backend_device *cuda_device = (gsx_cuda_backend_device *)backend_device;
    gsx_cuda_backend *cuda_backend = NULL;
    cudaError_t cuda_err = cudaSuccess;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->options_size_bytes != 0 && desc->options == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "non-zero options_size_bytes requires a non-null options pointer");
    }
    if(desc->options != NULL || desc->options_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda backend does not expose backend-specific options yet");
    }
    if(out_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend must be non-null");
    }
    *out_backend = NULL;

    cuda_backend = (gsx_cuda_backend *)calloc(1, sizeof(*cuda_backend));
    if(cuda_backend == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate CUDA backend");
    }

    cuda_backend->base.iface = &gsx_cuda_backend_iface;
    cuda_backend->base.provider = &gsx_cuda_backend_provider_singleton.base;
    cuda_backend->base.device = backend_device;
    cuda_backend->base.live_buffer_count = 0;
    cuda_backend->base.live_arena_count = 0;

    cuda_err = cudaSetDevice(cuda_device->cuda_device_ordinal);
    if(cuda_err != cudaSuccess) {
        free(cuda_backend);
        return gsx_cuda_make_error(cuda_err, "cudaSetDevice failed");
    }

    cuda_err = cudaStreamCreate(&cuda_backend->major_stream);
    if(cuda_err != cudaSuccess) {
        free(cuda_backend);
        return gsx_cuda_make_error(cuda_err, "cudaStreamCreate failed");
    }

    cuda_backend->capabilities.supported_data_types = GSX_DATA_TYPE_FLAG_F32;
    cuda_backend->capabilities.supports_async_prefetch = true;

    gsx_cuda_backend_init_buffer_type(cuda_backend, &cuda_backend->device_buffer_type, GSX_BACKEND_BUFFER_TYPE_DEVICE, "device", 256);
    gsx_cuda_backend_init_buffer_type(cuda_backend, &cuda_backend->host_pinned_buffer_type, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, "host_pinned", 64);

    *out_backend = &cuda_backend->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_free(gsx_backend_t backend)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend must be non-null");
    }
    if(cuda_backend->base.live_buffer_count > 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend has live buffers");
    }
    if(cuda_backend->base.live_arena_count > 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend has live arenas");
    }

    if(cuda_backend->major_stream != NULL) {
        cudaStreamDestroy(cuda_backend->major_stream);
        cuda_backend->major_stream = NULL;
    }

    free(cuda_backend);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->backend_type = GSX_BACKEND_TYPE_CUDA;
    out_info->device = cuda_backend->base.device;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_capabilities == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_capabilities must be non-null");
    }

    *out_capabilities = cuda_backend->capabilities;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_get_major_stream(gsx_backend_t backend, void **out_stream)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_stream == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_stream must be non-null");
    }

    *out_stream = cuda_backend->major_stream;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count)
{
    (void)backend;

    if(out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_count must be non-null");
    }

    *out_count = 2;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer_type must be non-null");
    }
    if(index < 0 || index >= 2) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer type index out of range");
    }

    if(index == 0) {
        *out_buffer_type = &cuda_backend->device_buffer_type.base;
    } else {
        *out_buffer_type = &cuda_backend->host_pinned_buffer_type.base;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer_type must be non-null");
    }

    switch(type) {
    case GSX_BACKEND_BUFFER_TYPE_DEVICE:
        *out_buffer_type = &cuda_backend->device_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
        *out_buffer_type = &cuda_backend->host_pinned_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_HOST:
    case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "buffer type not supported by CUDA backend");
    }
    return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "invalid buffer type class");
}

static gsx_error gsx_cuda_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info)
{
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer_type);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    *out_info = cuda_buffer_type->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes)
{
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer_type);

    if(out_alloc_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_alloc_size_bytes must be non-null");
    }

    if(gsx_round_up_overflows(requested_size_bytes, cuda_buffer_type->info.alignment_bytes, out_alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer)
{
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer_type);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer_type->backend);
    gsx_cuda_backend_buffer *cuda_buffer = NULL;
    gsx_size_t alloc_size_bytes = 0;
    gsx_size_t effective_alignment = 0;
    cudaError_t cuda_err = cudaSuccess;
    void *ptr = NULL;

    if(out_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer must be non-null");
    }
    *out_buffer = NULL;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->size_bytes must be non-zero");
    }
    if(desc->alignment_bytes != 0 && !gsx_is_power_of_two(desc->alignment_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->alignment_bytes must be a power of two");
    }

    effective_alignment = cuda_buffer_type->info.alignment_bytes;
    if(desc->alignment_bytes > effective_alignment) {
        effective_alignment = desc->alignment_bytes;
    }

    if(gsx_round_up_overflows(desc->size_bytes, effective_alignment, &alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }

    cuda_buffer = (gsx_cuda_backend_buffer *)calloc(1, sizeof(*cuda_buffer));
    if(cuda_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate buffer struct");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMalloc(&ptr, alloc_size_bytes);
        if(cuda_err != cudaSuccess) {
            free(cuda_buffer);
            return gsx_cuda_make_error(cuda_err, "cudaMalloc failed");
        }
    } else if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        cuda_err = cudaMallocHost(&ptr, alloc_size_bytes);
        if(cuda_err != cudaSuccess) {
            free(cuda_buffer);
            return gsx_cuda_make_error(cuda_err, "cudaMallocHost failed");
        }
    } else {
        free(cuda_buffer);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unsupported buffer type");
    }

    cuda_buffer->base.iface = &gsx_cuda_backend_buffer_iface;
    cuda_buffer->base.buffer_type = buffer_type;
    cuda_buffer->base.size_bytes = desc->size_bytes;
    cuda_buffer->base.alignment_bytes = effective_alignment;
    cuda_buffer->ptr = ptr;
    cuda_buffer->alloc_size_bytes = alloc_size_bytes;

    cuda_backend->base.live_buffer_count += 1;
    *out_buffer = &cuda_buffer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_buffer_free(gsx_backend_buffer_t buffer)
{
    gsx_cuda_backend_buffer *cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    cudaError_t cuda_err = cudaSuccess;

    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }
    cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    type = gsx_cuda_backend_buffer_get_type_class(buffer);
    if(cuda_backend->base.live_buffer_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_buffer_count underflow in buffer free");
    }

    if(cuda_buffer->ptr != NULL) {
        if(type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
            cuda_err = cudaFree(cuda_buffer->ptr);
        } else if(type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
            cuda_err = cudaFreeHost(cuda_buffer->ptr);
        }
        cuda_buffer->ptr = NULL;
    }

    cuda_backend->base.live_buffer_count -= 1;
    free(cuda_buffer);

    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "CUDA free failed");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info)
{
    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->backend = buffer->buffer_type->backend;
    out_info->buffer_type = buffer->buffer_type;
    out_info->size_bytes = buffer->size_bytes;
    out_info->alignment_bytes = buffer->alignment_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_backend_buffer_check_range(buffer, dst_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero byte_count");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + dst_offset_bytes,
            src_bytes,
            byte_count,
            cudaMemcpyHostToDevice,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + dst_offset_bytes,
            src_bytes,
            byte_count,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }

    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync upload failed");
}

static gsx_error gsx_cuda_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_backend_buffer_check_range(buffer, src_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero byte_count");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + src_offset_bytes,
            byte_count,
            cudaMemcpyDeviceToHost,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + src_offset_bytes,
            byte_count,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }

    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync download failed");
}

static gsx_error gsx_cuda_backend_buffer_set_zero(gsx_backend_buffer_t buffer)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    cudaError_t cuda_err = cudaSuccess;

    if(gsx_cuda_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        memset(cuda_buffer->ptr, 0, (size_t)cuda_buffer->alloc_size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_err = cudaMemsetAsync(cuda_buffer->ptr, 0, cuda_buffer->alloc_size_bytes, cuda_backend->major_stream);
    return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync failed");
}

static gsx_error gsx_cuda_backend_buffer_memset_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, uint8_t value, gsx_size_t offset_bytes, gsx_size_t size_bytes)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t tensor_offset = 0;
    gsx_size_t total_offset = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    tensor_offset = tensor_view->offset_bytes;
    if(gsx_size_add_overflows(tensor_offset, offset_bytes, &total_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor offset overflow");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, total_offset, size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gsx_cuda_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        memset((char*)cuda_buffer->ptr + total_offset, value, (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_err = cudaMemsetAsync(
        (char*)cuda_buffer->ptr + total_offset,
        value,
        size_bytes,
        cuda_backend->major_stream
    );
    return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync tensor failed");
}

static gsx_error gsx_cuda_backend_buffer_set_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, const void *src_bytes, gsx_size_t offset_bytes, gsx_size_t size_bytes)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t tensor_offset = 0;
    gsx_size_t total_offset = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    tensor_offset = tensor_view->offset_bytes;
    if(gsx_size_add_overflows(tensor_offset, offset_bytes, &total_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor offset overflow");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, total_offset, size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero size_bytes");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + total_offset,
            src_bytes,
            size_bytes,
            cudaMemcpyHostToDevice,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + total_offset,
            src_bytes,
            size_bytes,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }
    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync set_tensor failed");
}

static gsx_error gsx_cuda_backend_buffer_get_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, void *dst_bytes, gsx_size_t offset_bytes, gsx_size_t size_bytes)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t tensor_offset = 0;
    gsx_size_t total_offset = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    tensor_offset = tensor_view->offset_bytes;
    if(gsx_size_add_overflows(tensor_offset, offset_bytes, &total_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor offset overflow");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, total_offset, size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero size_bytes");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + total_offset,
            size_bytes,
            cudaMemcpyDeviceToHost,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + total_offset,
            size_bytes,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }
    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync get_tensor failed");
}

static gsx_error gsx_cuda_backend_buffer_copy_tensor(gsx_backend_buffer_t dst_buffer, const gsx_backend_tensor_view *src_view, const gsx_backend_tensor_view *dst_view)
{
    gsx_cuda_backend_buffer *cuda_dst_buffer = NULL;
    gsx_cuda_backend_buffer *cuda_src_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_cuda_backend_buffer_type *cuda_dst_type = NULL;
    gsx_cuda_backend_buffer_type *cuda_src_type = NULL;
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t copy_size = 0;

    if(src_view == NULL || dst_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_view and dst_view must be non-null");
    }
    if(src_view->buffer == NULL || dst_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views must have valid buffers");
    }
    if(dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_view->buffer must match dst_buffer");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy_tensor requires source and destination to belong to the same backend");
    }

    copy_size = src_view->size_bytes;
    if(dst_view->size_bytes != copy_size) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy_tensor requires equal source and destination sizes");
    }
    if(copy_size == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_cuda_backend_buffer_check_range(src_view->buffer, src_view->offset_bytes, copy_size);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_view->buffer, dst_view->offset_bytes, copy_size);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    {
        gsx_size_t src_end = 0, dst_end = 0;
        gsx_size_add_overflows(src_view->offset_bytes, copy_size, &src_end);
        gsx_size_add_overflows(dst_view->offset_bytes, copy_size, &dst_end);

        if(src_view->buffer == dst_view->buffer) {
            bool overlaps = !(src_end <= dst_view->offset_bytes || dst_end <= src_view->offset_bytes);
            if(overlaps) {
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy_tensor source and destination regions overlap");
            }
        }
    }

    cuda_dst_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    cuda_src_buffer = gsx_cuda_backend_buffer_from_base(src_view->buffer);
    cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
    cuda_dst_type = gsx_cuda_backend_buffer_type_from_base(dst_buffer->buffer_type);
    cuda_src_type = gsx_cuda_backend_buffer_type_from_base(src_view->buffer->buffer_type);

    {
        enum cudaMemcpyKind kind = cudaMemcpyDefault;
        if(cuda_src_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE && cuda_dst_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
            kind = cudaMemcpyDeviceToDevice;
        } else if(cuda_src_type->info.type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED && cuda_dst_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
            kind = cudaMemcpyHostToDevice;
        } else if(cuda_src_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE && cuda_dst_type->info.type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyHostToHost;
        }

        cuda_err = cudaMemcpyAsync(
            (char*)cuda_dst_buffer->ptr + dst_view->offset_bytes,
            (const char*)cuda_src_buffer->ptr + src_view->offset_bytes,
            copy_size,
            kind,
            cuda_backend->major_stream
        );
    }

    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync copy_tensor failed");
}

static gsx_error gsx_cuda_backend_buffer_fill_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, const void *value_bytes, gsx_size_t value_size_bytes)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    void *value_device_bytes = NULL;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }
    if(value_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_size_bytes must be non-zero");
    }
    if(tensor_view->size_bytes != 0 && value_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_bytes must be non-null when the tensor is non-empty");
    }
    if(tensor_view->size_bytes % value_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of value_size_bytes");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_cuda_backend_fill_host_bytes(
            (char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            tensor_view->size_bytes,
            value_bytes,
            value_size_bytes
        );
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_err = cudaMalloc(&value_device_bytes, value_size_bytes);
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaMalloc for fill_tensor value staging failed");
    }

    cuda_err = cudaMemcpyAsync(
        value_device_bytes,
        value_bytes,
        value_size_bytes,
        cudaMemcpyHostToDevice,
        cuda_backend->major_stream
    );
    if(cuda_err != cudaSuccess) {
        cudaFree(value_device_bytes);
        return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync for fill_tensor value staging failed");
    }

    gsx_cuda_fill_tensor_kernel_launch(
        (char*)cuda_buffer->ptr + tensor_view->offset_bytes,
        value_device_bytes,
        value_size_bytes,
        tensor_view->size_bytes,
        tensor_view->effective_alignment_bytes,
        cuda_backend->major_stream
    );
    cuda_err = cudaGetLastError();
    if(cuda_err != cudaSuccess) {
        cudaFree(value_device_bytes);
        return gsx_cuda_make_error(cuda_err, "fill_tensor kernel launch failed");
    }

    cuda_err = cudaFree(value_device_bytes);
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaFree for fill_tensor value staging failed");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_backend_buffer_check_finite_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, bool *out_is_finite)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    void *has_non_finite_dev_ptr = NULL;
    int has_non_finite_host = 0;
    cudaError_t cuda_err = cudaSuccess;
    gsx_size_t element_count = 0;
    gsx_size_t element_size = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }
    if(out_is_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_is_finite must be non-null");
    }
    *out_is_finite = true;

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        *out_is_finite = true;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32:
        element_size = 4;
        break;
    case GSX_DATA_TYPE_F16:
        element_size = 2;
        break;
    case GSX_DATA_TYPE_BF16:
        element_size = 2;
        break;
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "check_finite only supports floating point types");
    }
    if(tensor_view->size_bytes % element_size != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of the checked element size");
    }

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_size_t element_index = 0;

        if(tensor_view->data_type == GSX_DATA_TYPE_F32) {
            const float *values = (const float *)((const char *)cuda_buffer->ptr + tensor_view->offset_bytes);

            element_count = tensor_view->size_bytes / sizeof(float);
            for(element_index = 0; element_index < element_count; ++element_index) {
                if(!isfinite((double)values[element_index])) {
                    *out_is_finite = false;
                    break;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }

        {
            const uint16_t *values = (const uint16_t *)((const char *)cuda_buffer->ptr + tensor_view->offset_bytes);

            element_count = tensor_view->size_bytes / element_size;
            for(element_index = 0; element_index < element_count; ++element_index) {
                bool is_value_finite = tensor_view->data_type == GSX_DATA_TYPE_F16
                    ? gsx_cuda_backend_f16_is_finite(values[element_index])
                    : gsx_cuda_backend_bf16_is_finite(values[element_index]);

                if(!is_value_finite) {
                    *out_is_finite = false;
                    break;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
    }

    element_count = tensor_view->size_bytes / element_size;

    cuda_err = cudaMalloc(&has_non_finite_dev_ptr, sizeof(int));
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaMalloc for check_finite flag failed");
    }

    cuda_err = cudaMemsetAsync(has_non_finite_dev_ptr, 0, sizeof(int), cuda_backend->major_stream);
    if(cuda_err != cudaSuccess) {
        cudaFree(has_non_finite_dev_ptr);
        return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync for check_finite flag failed");
    }

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32:
        gsx_cuda_check_finite_tensor_f32_kernel_launch(
            (const char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            element_count,
            tensor_view->effective_alignment_bytes,
            (int*)has_non_finite_dev_ptr,
            cuda_backend->major_stream
        );
        break;
    case GSX_DATA_TYPE_F16:
        gsx_cuda_check_finite_tensor_f16_kernel_launch(
            (const char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            element_count,
            tensor_view->effective_alignment_bytes,
            (int*)has_non_finite_dev_ptr,
            cuda_backend->major_stream
        );
        break;
    case GSX_DATA_TYPE_BF16:
        gsx_cuda_check_finite_tensor_bf16_kernel_launch(
            (const char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            element_count,
            tensor_view->effective_alignment_bytes,
            (int*)has_non_finite_dev_ptr,
            cuda_backend->major_stream
        );
        break;
    default:
        cudaFree(has_non_finite_dev_ptr);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "check_finite only supports floating point types");
    }
    cuda_err = cudaGetLastError();
    if(cuda_err != cudaSuccess) {
        cudaFree(has_non_finite_dev_ptr);
        return gsx_cuda_make_error(cuda_err, "check_finite kernel launch failed");
    }

    cuda_err = cudaMemcpyAsync(&has_non_finite_host, has_non_finite_dev_ptr, sizeof(int), cudaMemcpyDeviceToHost, cuda_backend->major_stream);
    if(cuda_err != cudaSuccess) {
        cudaFree(has_non_finite_dev_ptr);
        return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync for check_finite result failed");
    }

    cuda_err = cudaStreamSynchronize(cuda_backend->major_stream);
    cudaFree(has_non_finite_dev_ptr);
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaStreamSynchronize for check_finite failed");
    }

    *out_is_finite = (has_non_finite_host == 0);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_provider_bootstrap(gsx_builtin_registry_state *registry)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    int device_count = 0;

    cuda_err = cudaGetDeviceCount(&device_count);
    if(cuda_err == cudaErrorNoDevice || cuda_err == cudaErrorInsufficientDriver) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "No CUDA devices available");
    }
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaGetDeviceCount failed during bootstrap");
    }
    if(device_count == 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "No CUDA devices available");
    }

    gsx_cuda_backend_provider_singleton.base.iface = &gsx_cuda_backend_provider_iface;
    gsx_cuda_backend_provider_singleton.base.backend_type = GSX_BACKEND_TYPE_CUDA;
    gsx_cuda_backend_provider_singleton.base.backend_name = "cuda";

    error = gsx_cuda_backend_provider_singleton.base.iface->discover_devices(
        &gsx_cuda_backend_provider_singleton.base, registry);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_builtin_registry_append_provider(registry, &gsx_cuda_backend_provider_singleton.base);
}
