#include "internal.h"

#include <stdlib.h>
#include <string.h>

gsx_error gsx_cuda_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry)
{
    cudaError_t cuda_err = cudaSuccess;
    int device_count = 0;
    int device_index = 0;
    int registered_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    struct cudaDeviceProp prop;

    (void)provider;

    cuda_err = cudaGetDeviceCount(&device_count);
    if(cuda_err == cudaErrorNoDevice) {
        gsx_cuda_device_count = 0;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaGetDeviceCount failed");
    }
    if(device_count == 0) {
        gsx_cuda_device_count = 0;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gsx_cuda_backend_devices == NULL) {
        gsx_cuda_backend_devices = (gsx_cuda_backend_device *)calloc((size_t)device_count, sizeof(*gsx_cuda_backend_devices));
        if(gsx_cuda_backend_devices == NULL) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate CUDA device storage");
        }
        gsx_cuda_device_capacity = device_count;
    } else if(gsx_cuda_device_capacity < device_count) {
        gsx_cuda_backend_device *resized_devices = (gsx_cuda_backend_device *)realloc(
            gsx_cuda_backend_devices, (size_t)device_count * sizeof(*gsx_cuda_backend_devices));
        if(resized_devices == NULL) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to grow CUDA device storage");
        }
        memset(
            resized_devices + gsx_cuda_device_capacity,
            0,
            (size_t)(device_count - gsx_cuda_device_capacity) * sizeof(*gsx_cuda_backend_devices)
        );
        gsx_cuda_backend_devices = resized_devices;
        gsx_cuda_device_capacity = device_count;
    }

    memset(gsx_cuda_backend_devices, 0, (size_t)device_count * sizeof(*gsx_cuda_backend_devices));

    for(device_index = 0; device_index < device_count; ++device_index) {
        gsx_cuda_backend_device *dev = &gsx_cuda_backend_devices[registered_count];
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
        dev->base.info.device_index = registered_count;
        strncpy(dev->device_name, prop.name, sizeof(dev->device_name) - 1);
        dev->device_name[sizeof(dev->device_name) - 1] = '\0';
        dev->base.info.name = dev->device_name;
        dev->base.info.total_memory_bytes = (gsx_size_t)total_memory;
        dev->cuda_device_ordinal = device_index;
        dev->compute_capability_major = prop.major;
        dev->compute_capability_minor = prop.minor;

        error = gsx_builtin_registry_append_device(registry, &dev->base);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_device_count = registered_count;
            return error;
        }
        registered_count += 1;
    }

    gsx_cuda_device_count = registered_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend)
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
    cuda_backend->base.live_renderer_count = 0;
    cuda_backend->base.live_loss_count = 0;
    cuda_backend->base.live_optim_count = 0;

    cuda_err = cudaSetDevice(cuda_device->cuda_device_ordinal);
    if(cuda_err != cudaSuccess) {
        free(cuda_backend);
        return gsx_cuda_make_error(cuda_err, "cudaSetDevice failed");
    }

    cuda_err = cudaStreamCreateWithFlags(&cuda_backend->major_stream, cudaStreamNonBlocking);
    if(cuda_err != cudaSuccess) {
        free(cuda_backend);
        return gsx_cuda_make_error(cuda_err, "cudaStreamCreateWithFlags failed");
    }

    cuda_backend->capabilities.supported_data_types = GSX_DATA_TYPE_FLAG_F32;
    cuda_backend->capabilities.supports_async_prefetch = true;

    gsx_cuda_backend_init_buffer_type(cuda_backend, &cuda_backend->device_buffer_type, GSX_BACKEND_BUFFER_TYPE_DEVICE, "device", 256);
    gsx_cuda_backend_init_buffer_type(cuda_backend, &cuda_backend->host_pinned_buffer_type, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, "host_pinned", 64);

    *out_backend = &cuda_backend->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_free(gsx_backend_t backend)
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
    if(cuda_backend->base.live_renderer_count > 0 || cuda_backend->base.live_loss_count > 0 || cuda_backend->base.live_optim_count > 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend has live renderers, losses, or optimizers");
    }

    if(cuda_backend->major_stream != NULL) {
        cudaStreamDestroy(cuda_backend->major_stream);
        cuda_backend->major_stream = NULL;
    }

    free(cuda_backend);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->backend_type = GSX_BACKEND_TYPE_CUDA;
    out_info->device = cuda_backend->base.device;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_capabilities == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_capabilities must be non-null");
    }

    *out_capabilities = cuda_backend->capabilities;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_get_major_stream(gsx_backend_t backend, void **out_stream)
{
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(backend);

    if(out_stream == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_stream must be non-null");
    }

    *out_stream = cuda_backend->major_stream;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count)
{
    (void)backend;

    if(out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_count must be non-null");
    }

    *out_count = 2;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type)
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

gsx_error gsx_cuda_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type)
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
