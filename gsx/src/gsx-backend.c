#include "gsx-impl.h"

#include <stdlib.h>
#include <string.h>

static gsx_builtin_registry_state gsx_builtin_registry_state_singleton = { 0 };

static bool gsx_backend_type_is_valid(gsx_backend_type backend_type);

gsx_builtin_registry_state *gsx_builtin_registry_get(void)
{
    return &gsx_builtin_registry_state_singleton;
}

static gsx_error gsx_builtin_registry_reserve_providers(gsx_builtin_registry_state *registry, gsx_index_t required_capacity)
{
    gsx_backend_provider_t *providers = NULL;
    gsx_index_t new_capacity = 4;

    if(registry == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "registry must be non-null");
    }
    if(required_capacity <= registry->backend_provider_capacity) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    while(new_capacity < required_capacity) {
        if(new_capacity > INT32_MAX / 2) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend provider capacity overflowed");
        }
        new_capacity *= 2;
    }

    providers = (gsx_backend_provider_t *)realloc(
        registry->backend_providers,
        (size_t)new_capacity * sizeof(*registry->backend_providers)
    );
    if(providers == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to grow backend provider registry storage");
    }

    registry->backend_providers = providers;
    registry->backend_provider_capacity = new_capacity;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_builtin_registry_reserve_devices(gsx_builtin_registry_state *registry, gsx_index_t required_capacity)
{
    gsx_backend_device_t *devices = NULL;
    gsx_index_t new_capacity = 4;

    if(registry == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "registry must be non-null");
    }
    if(required_capacity <= registry->backend_device_capacity) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    while(new_capacity < required_capacity) {
        if(new_capacity > INT32_MAX / 2) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend device capacity overflowed");
        }
        new_capacity *= 2;
    }

    devices = (gsx_backend_device_t *)realloc(
        registry->backend_devices,
        (size_t)new_capacity * sizeof(*registry->backend_devices)
    );
    if(devices == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to grow backend device registry storage");
    }

    registry->backend_devices = devices;
    registry->backend_device_capacity = new_capacity;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_builtin_registry_reset(gsx_builtin_registry_state *registry)
{
    if(registry == NULL) {
        return;
    }

    free(registry->backend_providers);
    free(registry->backend_devices);
    registry->is_initialized = false;
    registry->backend_provider_count = 0;
    registry->backend_provider_capacity = 0;
    registry->backend_providers = NULL;
    registry->backend_device_count = 0;
    registry->backend_device_capacity = 0;
    registry->backend_devices = NULL;
}

gsx_error gsx_builtin_registry_append_provider(gsx_builtin_registry_state *registry, gsx_backend_provider_t backend_provider)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(registry == NULL || backend_provider == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "registry and backend_provider must be non-null");
    }
    if(backend_provider->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_provider->iface must be non-null");
    }
    if(backend_provider->iface->discover_devices == NULL || backend_provider->iface->create_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_provider callbacks must be non-null");
    }
    if(!gsx_backend_type_is_valid(backend_provider->backend_type)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_provider->backend_type is out of range");
    }
    if(backend_provider->backend_name == NULL || backend_provider->backend_name[0] == '\0') {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_provider->backend_name must be non-empty");
    }

    error = gsx_builtin_registry_reserve_providers(registry, registry->backend_provider_count + 1);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    registry->backend_providers[registry->backend_provider_count] = backend_provider;
    registry->backend_provider_count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_builtin_registry_append_device(gsx_builtin_registry_state *registry, gsx_backend_device_t backend_device)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_backend_provider_t provider = NULL;

    if(registry == NULL || backend_device == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "registry and backend_device must be non-null");
    }
    provider = backend_device->provider;
    if(provider == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device->provider must be non-null");
    }
    if(provider->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device->provider->iface must be non-null");
    }
    if(provider->iface->discover_devices == NULL || provider->iface->create_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device provider callbacks must be non-null");
    }
    if(!gsx_backend_type_is_valid(backend_device->info.backend_type)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device->info.backend_type is out of range");
    }
    if(backend_device->info.backend_type != provider->backend_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device backend_type must match provider");
    }
    if(backend_device->info.backend_name == NULL || backend_device->info.backend_name[0] == '\0') {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device->info.backend_name must be non-empty");
    }
    if(provider->backend_name == NULL || provider->backend_name[0] == '\0') {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device provider backend_name must be non-empty");
    }
    if(strcmp(backend_device->info.backend_name, provider->backend_name) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device backend_name must match provider");
    }
    if(backend_device->info.name == NULL || backend_device->info.name[0] == '\0') {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device->info.name must be non-empty");
    }

    error = gsx_builtin_registry_reserve_devices(registry, registry->backend_device_count + 1);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    registry->backend_devices[registry->backend_device_count] = backend_device;
    registry->backend_device_count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_backend_type_is_valid(gsx_backend_type backend_type)
{
    switch(backend_type) {
    case GSX_BACKEND_TYPE_CPU:
    case GSX_BACKEND_TYPE_CUDA:
    case GSX_BACKEND_TYPE_METAL:
        return true;
    }

    return false;
}

static gsx_error gsx_backend_require_registry_initialized(void)
{
    if(!gsx_builtin_registry_get()->is_initialized) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "call gsx_backend_registry_init before using backend APIs");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_registry_init(void)
{
    gsx_builtin_registry_state *registry = gsx_builtin_registry_get();
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(registry->is_initialized) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_builtin_registry_reset(registry);

    error = gsx_cpu_backend_provider_bootstrap(registry);
    if(error.code == GSX_ERROR_NOT_SUPPORTED) {
        error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(!gsx_error_is_success(error)) {
        gsx_builtin_registry_reset(registry);
        return error;
    }

    registry->is_initialized = true;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_count_backend_devices(gsx_index_t *out_count)
{
    gsx_error error = gsx_backend_require_registry_initialized();

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_count must be non-null");
    }

    *out_count = gsx_builtin_registry_get()->backend_device_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_get_backend_device(gsx_index_t index, gsx_backend_device_t *out_backend_device)
{
    gsx_builtin_registry_state *registry = gsx_builtin_registry_get();
    gsx_error error = gsx_backend_require_registry_initialized();

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_backend_device == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend_device must be non-null");
    }
    if(index < 0 || index >= registry->backend_device_count) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend device index is out of range");
    }

    *out_backend_device = registry->backend_devices[index];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_count_backend_devices_by_type(gsx_backend_type backend_type, gsx_index_t *out_count)
{
    gsx_builtin_registry_state *registry = gsx_builtin_registry_get();
    gsx_index_t backend_device_index = 0;
    gsx_index_t backend_device_count = 0;
    gsx_error error = gsx_backend_require_registry_initialized();

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_count must be non-null");
    }
    if(!gsx_backend_type_is_valid(backend_type)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend_type is out of range");
    }

    for(backend_device_index = 0; backend_device_index < registry->backend_device_count; ++backend_device_index) {
        if(registry->backend_devices[backend_device_index]->info.backend_type == backend_type) {
            backend_device_count += 1;
        }
    }

    *out_count = backend_device_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_get_backend_device_by_type(gsx_backend_type backend_type, gsx_index_t index, gsx_backend_device_t *out_backend_device)
{
    gsx_builtin_registry_state *registry = gsx_builtin_registry_get();
    gsx_index_t backend_device_index = 0;
    gsx_index_t filtered_index = 0;
    gsx_error error = gsx_backend_require_registry_initialized();

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_backend_device == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend_device must be non-null");
    }
    if(!gsx_backend_type_is_valid(backend_type)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend_type is out of range");
    }
    if(index < 0) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend device index is out of range");
    }

    for(backend_device_index = 0; backend_device_index < registry->backend_device_count; ++backend_device_index) {
        if(registry->backend_devices[backend_device_index]->info.backend_type != backend_type) {
            continue;
        }
        if(filtered_index == index) {
            *out_backend_device = registry->backend_devices[backend_device_index];
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        filtered_index += 1;
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "backend device index is out of range");
}

GSX_API gsx_error gsx_backend_device_get_info(gsx_backend_device_t backend_device, gsx_backend_device_info *out_info)
{
    gsx_error error = gsx_backend_require_registry_initialized();

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(backend_device == NULL || out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend_device and out_info must be non-null");
    }

    *out_info = backend_device->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_init(gsx_backend_t *out_backend, const gsx_backend_desc *desc)
{
    gsx_error error = gsx_backend_require_registry_initialized();

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend and desc must be non-null");
    }
    if(desc->device == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->device must be non-null");
    }

    return desc->device->provider->iface->create_backend(desc->device, desc, out_backend);
}

GSX_API gsx_error gsx_backend_free(gsx_backend_t backend)
{
    if(backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend must be non-null");
    }

    return backend->iface->free(backend);
}

GSX_API gsx_error gsx_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info)
{
    if(backend == NULL || out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and out_info must be non-null");
    }

    return backend->iface->get_info(backend, out_info);
}

GSX_API gsx_error gsx_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities)
{
    if(backend == NULL || out_capabilities == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and out_capabilities must be non-null");
    }

    return backend->iface->get_capabilities(backend, out_capabilities);
}

GSX_API gsx_error gsx_backend_get_major_stream(gsx_backend_t backend, void **out_stream)
{
    if(backend == NULL || out_stream == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and out_stream must be non-null");
    }

    return backend->iface->get_major_stream(backend, out_stream);
}

GSX_API gsx_error gsx_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count)
{
    if(backend == NULL || out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and out_count must be non-null");
    }

    return backend->iface->count_buffer_types(backend, out_count);
}

GSX_API gsx_error gsx_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type)
{
    if(backend == NULL || out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and out_buffer_type must be non-null");
    }

    return backend->iface->get_buffer_type(backend, index, out_buffer_type);
}

GSX_API gsx_error gsx_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type)
{
    if(backend == NULL || out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and out_buffer_type must be non-null");
    }

    return backend->iface->find_buffer_type(backend, type, out_buffer_type);
}

GSX_API gsx_error gsx_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info)
{
    if(buffer_type == NULL || out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_info must be non-null");
    }

    return buffer_type->iface->get_info(buffer_type, out_info);
}

GSX_API gsx_error gsx_backend_buffer_type_get_type(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_class *out_type)
{
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || out_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_type must be non-null");
    }

    error = buffer_type->iface->get_info(buffer_type, &buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_type = buffer_type_info.type;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_buffer_type_get_name(gsx_backend_buffer_type_t buffer_type, const char **out_name)
{
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || out_name == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_name must be non-null");
    }

    error = buffer_type->iface->get_info(buffer_type, &buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_name = buffer_type_info.name;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_buffer_type_get_backend(gsx_backend_buffer_type_t buffer_type, gsx_backend_t *out_backend)
{
    if(buffer_type == NULL || out_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_backend must be non-null");
    }

    *out_backend = buffer_type->backend;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_buffer_type_get_alignment(gsx_backend_buffer_type_t buffer_type, gsx_size_t *out_alignment_bytes)
{
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || out_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_alignment_bytes must be non-null");
    }

    error = buffer_type->iface->get_info(buffer_type, &buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_alignment_bytes = buffer_type_info.alignment_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_buffer_type_get_max_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t *out_max_size_bytes)
{
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || out_max_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_max_size_bytes must be non-null");
    }

    error = buffer_type->iface->get_info(buffer_type, &buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_max_size_bytes = buffer_type_info.max_allocation_size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes)
{
    if(buffer_type == NULL || out_alloc_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_alloc_size_bytes must be non-null");
    }

    return buffer_type->iface->get_alloc_size(buffer_type, requested_size_bytes, out_alloc_size_bytes);
}

GSX_API gsx_error gsx_backend_buffer_init(gsx_backend_buffer_t *out_buffer, const gsx_backend_buffer_desc *desc)
{
    if(out_buffer == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer and desc must be non-null");
    }
    if(desc->buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->buffer_type must be non-null");
    }

    return desc->buffer_type->iface->init_buffer(desc->buffer_type, desc, out_buffer);
}

GSX_API gsx_error gsx_backend_buffer_free(gsx_backend_buffer_t buffer)
{
    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }

    return buffer->iface->free(buffer);
}

GSX_API gsx_error gsx_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info)
{
    if(buffer == NULL || out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and out_info must be non-null");
    }

    return buffer->iface->get_info(buffer, out_info);
}

GSX_API gsx_error gsx_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count)
{
    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }

    return buffer->iface->upload(buffer, dst_offset_bytes, src_bytes, byte_count);
}

GSX_API gsx_error gsx_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count)
{
    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }

    return buffer->iface->download(buffer, src_offset_bytes, dst_bytes, byte_count);
}

GSX_API gsx_error gsx_backend_buffer_set_zero(gsx_backend_buffer_t buffer)
{
    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }

    return buffer->iface->set_zero(buffer);
}
