#include "internal.h"

#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

gsx_error gsx_metal_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry)
{
    NSArray<id<MTLDevice>> *devices = nil;
    NSUInteger device_index = 0;
    int registered_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    (void)provider;

    /* TODO(backend-lifetime): release retained device objects before this path rewrites gsx_metal_backend_devices during re-discovery/reset flows. */

    devices = MTLCopyAllDevices();
    if(devices == nil || devices.count == 0) {
        id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();

        if(default_device == nil) {
            gsx_metal_device_count = 0;
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }

        devices = @[ default_device ];
    }

    if(gsx_metal_backend_devices == NULL) {
        gsx_metal_backend_devices = (gsx_metal_backend_device *)calloc((size_t)devices.count, sizeof(*gsx_metal_backend_devices));
        if(gsx_metal_backend_devices == NULL) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal device storage");
        }
        gsx_metal_device_capacity = (int)devices.count;
    } else if(gsx_metal_device_capacity < (int)devices.count) {
        gsx_metal_backend_device *resized_devices = (gsx_metal_backend_device *)realloc(
            gsx_metal_backend_devices, (size_t)devices.count * sizeof(*gsx_metal_backend_devices));
        if(resized_devices == NULL) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to grow Metal device storage");
        }
        memset(
            resized_devices + gsx_metal_device_capacity,
            0,
            (size_t)((int)devices.count - gsx_metal_device_capacity) * sizeof(*gsx_metal_backend_devices)
        );
        gsx_metal_backend_devices = resized_devices;
        gsx_metal_device_capacity = (int)devices.count;
    }

    memset(gsx_metal_backend_devices, 0, (size_t)devices.count * sizeof(*gsx_metal_backend_devices));

    for(device_index = 0; device_index < devices.count; ++device_index) {
        gsx_metal_backend_device *device = &gsx_metal_backend_devices[registered_count];
        id<MTLDevice> mtl_device = devices[device_index];
        NSString *name = nil;
        const char *name_utf8 = NULL;
        uint64_t total_memory = 0;

        if(mtl_device == nil) {
            continue;
        }

        device->base.provider = &gsx_metal_backend_provider_singleton.base;
        device->base.info.backend_type = GSX_BACKEND_TYPE_METAL;
        device->base.info.backend_name = "metal";
        device->base.info.device_index = registered_count;

        name = [mtl_device name];
        name_utf8 = [name UTF8String];
        if(name_utf8 == NULL || name_utf8[0] == '\0') {
            name_utf8 = "metal-device";
        }
        strncpy(device->device_name, name_utf8, sizeof(device->device_name) - 1);
        device->device_name[sizeof(device->device_name) - 1] = '\0';
        device->base.info.name = device->device_name;

        if([mtl_device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            total_memory = (uint64_t)[mtl_device recommendedMaxWorkingSetSize];
        }
        device->base.info.total_memory_bytes = (gsx_size_t)total_memory;

        [mtl_device retain];
        device->mtl_device = mtl_device;

        error = gsx_builtin_registry_append_device(registry, &device->base);
        if(!gsx_error_is_success(error)) {
            gsx_metal_device_count = registered_count;
            return error;
        }
        registered_count += 1;
    }

    gsx_metal_device_count = registered_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend)
{
    gsx_metal_backend_device *metal_device = (gsx_metal_backend_device *)backend_device;
    gsx_metal_backend *metal_backend = NULL;
    id<MTLCommandQueue> major_queue = nil;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->options_size_bytes != 0 && desc->options == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "non-zero options_size_bytes requires a non-null options pointer");
    }
    if(desc->options != NULL || desc->options_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal backend does not expose backend-specific options yet");
    }
    if(out_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend must be non-null");
    }
    *out_backend = NULL;

    metal_backend = (gsx_metal_backend *)calloc(1, sizeof(*metal_backend));
    if(metal_backend == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal backend");
    }

    major_queue = [(id<MTLDevice>)metal_device->mtl_device newCommandQueue];
    if(major_queue == nil) {
        free(metal_backend);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "failed to create Metal command queue");
    }

    metal_backend->base.iface = &gsx_metal_backend_iface;
    metal_backend->base.provider = &gsx_metal_backend_provider_singleton.base;
    metal_backend->base.device = backend_device;
    metal_backend->base.live_buffer_count = 0;
    metal_backend->base.live_arena_count = 0;
    metal_backend->base.live_renderer_count = 0;
    metal_backend->base.live_loss_count = 0;
    metal_backend->base.live_optim_count = 0;
    metal_backend->base.live_adc_count = 0;

    [(id<MTLDevice>)metal_device->mtl_device retain];
    metal_backend->mtl_device = metal_device->mtl_device;
    metal_backend->major_command_queue = major_queue;
    metal_backend->capabilities.supported_data_types = GSX_DATA_TYPE_FLAG_F32 | GSX_DATA_TYPE_FLAG_I32;
    metal_backend->capabilities.supports_async_prefetch = true;

    gsx_metal_backend_init_buffer_type(metal_backend, &metal_backend->device_buffer_type, GSX_BACKEND_BUFFER_TYPE_DEVICE, "device", 256);
    gsx_metal_backend_init_buffer_type(metal_backend, &metal_backend->host_pinned_buffer_type, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, "host_pinned", 64);
    gsx_metal_backend_init_buffer_type(metal_backend, &metal_backend->unified_buffer_type, GSX_BACKEND_BUFFER_TYPE_UNIFIED, "unified", 64);

    *out_backend = &metal_backend->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_free(gsx_backend_t backend)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);

    if(backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend must be non-null");
    }
    if(metal_backend->base.live_buffer_count > 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend has live buffers");
    }
    if(metal_backend->base.live_arena_count > 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend has live arenas");
    }
    if(metal_backend->base.live_renderer_count > 0 || metal_backend->base.live_loss_count > 0
        || metal_backend->base.live_optim_count > 0 || metal_backend->base.live_adc_count > 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend has live renderers, losses, optimizers, or adcs");
    }

    if(metal_backend->major_command_queue != NULL) {
        [(id<MTLCommandQueue>)metal_backend->major_command_queue release];
        metal_backend->major_command_queue = NULL;
    }
    if(metal_backend->tensor_gather_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_gather_pipeline release];
        metal_backend->tensor_gather_pipeline = NULL;
    }
    if(metal_backend->tensor_exp_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_exp_pipeline release];
        metal_backend->tensor_exp_pipeline = NULL;
    }
    if(metal_backend->tensor_sigmoid_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_sigmoid_pipeline release];
        metal_backend->tensor_sigmoid_pipeline = NULL;
    }
    if(metal_backend->tensor_sigmoid_derivative_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_sigmoid_derivative_pipeline release];
        metal_backend->tensor_sigmoid_derivative_pipeline = NULL;
    }
    if(metal_backend->tensor_abs_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_abs_pipeline release];
        metal_backend->tensor_abs_pipeline = NULL;
    }
    if(metal_backend->tensor_sum_reduce_f32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_sum_reduce_f32_pipeline release];
        metal_backend->tensor_sum_reduce_f32_pipeline = NULL;
    }
    if(metal_backend->tensor_mean_reduce_f32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_mean_reduce_f32_pipeline release];
        metal_backend->tensor_mean_reduce_f32_pipeline = NULL;
    }
    if(metal_backend->tensor_max_reduce_f32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_max_reduce_f32_pipeline release];
        metal_backend->tensor_max_reduce_f32_pipeline = NULL;
    }
    if(metal_backend->tensor_mse_reduce_f32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_mse_reduce_f32_pipeline release];
        metal_backend->tensor_mse_reduce_f32_pipeline = NULL;
    }
    if(metal_backend->tensor_mae_reduce_f32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_mae_reduce_f32_pipeline release];
        metal_backend->tensor_mae_reduce_f32_pipeline = NULL;
    }
    if(metal_backend->tensor_clamp_f32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_clamp_f32_pipeline release];
        metal_backend->tensor_clamp_f32_pipeline = NULL;
    }
    if(metal_backend->tensor_clamp_i32_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->tensor_clamp_i32_pipeline release];
        metal_backend->tensor_clamp_i32_pipeline = NULL;
    }
    if(metal_backend->tensor_library != NULL) {
        [(id<MTLLibrary>)metal_backend->tensor_library release];
        metal_backend->tensor_library = NULL;
    }
    if(metal_backend->optim_adam_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->optim_adam_pipeline release];
        metal_backend->optim_adam_pipeline = NULL;
    }
    if(metal_backend->optim_library != NULL) {
        [(id<MTLLibrary>)metal_backend->optim_library release];
        metal_backend->optim_library = NULL;
    }
    if(metal_backend->loss_mse_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_mse_pipeline release];
        metal_backend->loss_mse_pipeline = NULL;
    }
    if(metal_backend->loss_l1_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_l1_pipeline release];
        metal_backend->loss_l1_pipeline = NULL;
    }
    if(metal_backend->loss_mse_backward_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_mse_backward_pipeline release];
        metal_backend->loss_mse_backward_pipeline = NULL;
    }
    if(metal_backend->loss_l1_backward_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_l1_backward_pipeline release];
        metal_backend->loss_l1_backward_pipeline = NULL;
    }
    if(metal_backend->loss_ssim_chw_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_ssim_chw_pipeline release];
        metal_backend->loss_ssim_chw_pipeline = NULL;
    }
    if(metal_backend->loss_ssim_hwc_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_ssim_hwc_pipeline release];
        metal_backend->loss_ssim_hwc_pipeline = NULL;
    }
    if(metal_backend->loss_ssim_backward_chw_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_ssim_backward_chw_pipeline release];
        metal_backend->loss_ssim_backward_chw_pipeline = NULL;
    }
    if(metal_backend->loss_ssim_backward_hwc_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->loss_ssim_backward_hwc_pipeline release];
        metal_backend->loss_ssim_backward_hwc_pipeline = NULL;
    }
    if(metal_backend->loss_library != NULL) {
        [(id<MTLLibrary>)metal_backend->loss_library release];
        metal_backend->loss_library = NULL;
    }
    if(metal_backend->render_compose_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->render_compose_pipeline release];
        metal_backend->render_compose_pipeline = NULL;
    }
    if(metal_backend->render_blend_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->render_blend_pipeline release];
        metal_backend->render_blend_pipeline = NULL;
    }
    if(metal_backend->render_blend_backward_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->render_blend_backward_pipeline release];
        metal_backend->render_blend_backward_pipeline = NULL;
    }
    if(metal_backend->render_create_instances_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->render_create_instances_pipeline release];
        metal_backend->render_create_instances_pipeline = NULL;
    }
    if(metal_backend->render_preprocess_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->render_preprocess_pipeline release];
        metal_backend->render_preprocess_pipeline = NULL;
    }
    if(metal_backend->render_preprocess_backward_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->render_preprocess_backward_pipeline release];
        metal_backend->render_preprocess_backward_pipeline = NULL;
    }
    if(metal_backend->render_library != NULL) {
        [(id<MTLLibrary>)metal_backend->render_library release];
        metal_backend->render_library = NULL;
    }
    if(metal_backend->sort_scatter_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->sort_scatter_pipeline release];
        metal_backend->sort_scatter_pipeline = NULL;
    }
    if(metal_backend->sort_scatter_offsets_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->sort_scatter_offsets_pipeline release];
        metal_backend->sort_scatter_offsets_pipeline = NULL;
    }
    if(metal_backend->sort_scan_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->sort_scan_pipeline release];
        metal_backend->sort_scan_pipeline = NULL;
    }
    if(metal_backend->sort_reduce_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->sort_reduce_pipeline release];
        metal_backend->sort_reduce_pipeline = NULL;
    }
    if(metal_backend->sort_histogram_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->sort_histogram_pipeline release];
        metal_backend->sort_histogram_pipeline = NULL;
    }
    if(metal_backend->sort_library != NULL) {
        [(id<MTLLibrary>)metal_backend->sort_library release];
        metal_backend->sort_library = NULL;
    }
    if(metal_backend->scan_add_offsets_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->scan_add_offsets_pipeline release];
        metal_backend->scan_add_offsets_pipeline = NULL;
    }
    if(metal_backend->scan_block_sums_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->scan_block_sums_pipeline release];
        metal_backend->scan_block_sums_pipeline = NULL;
    }
    if(metal_backend->scan_blocks_pipeline != NULL) {
        [(id<MTLComputePipelineState>)metal_backend->scan_blocks_pipeline release];
        metal_backend->scan_blocks_pipeline = NULL;
    }
    if(metal_backend->scan_library != NULL) {
        [(id<MTLLibrary>)metal_backend->scan_library release];
        metal_backend->scan_library = NULL;
    }
    if(metal_backend->mtl_device != NULL) {
        [(id<MTLDevice>)metal_backend->mtl_device release];
        metal_backend->mtl_device = NULL;
    }

    free(metal_backend);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->backend_type = GSX_BACKEND_TYPE_METAL;
    out_info->device = metal_backend->base.device;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);

    if(out_capabilities == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_capabilities must be non-null");
    }

    *out_capabilities = metal_backend->capabilities;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_get_major_stream(gsx_backend_t backend, void **out_stream)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);

    if(out_stream == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_stream must be non-null");
    }

    *out_stream = metal_backend->major_command_queue;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_major_stream_sync(gsx_backend_t backend)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);
    id<MTLCommandBuffer> command_buffer = nil;

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer for major stream synchronization");
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count)
{
    (void)backend;

    if(out_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_count must be non-null");
    }

    *out_count = 3;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);

    if(out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer_type must be non-null");
    }
    if(index < 0 || index >= 3) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer type index out of range");
    }

    if(index == 0) {
        *out_buffer_type = &metal_backend->device_buffer_type.base;
    } else if(index == 1) {
        *out_buffer_type = &metal_backend->host_pinned_buffer_type.base;
    } else {
        *out_buffer_type = &metal_backend->unified_buffer_type.base;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type)
{
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(backend);

    if(out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer_type must be non-null");
    }

    switch(type) {
    case GSX_BACKEND_BUFFER_TYPE_DEVICE:
        *out_buffer_type = &metal_backend->device_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
        *out_buffer_type = &metal_backend->host_pinned_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
        *out_buffer_type = &metal_backend->unified_buffer_type.base;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_BACKEND_BUFFER_TYPE_HOST:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "buffer type not supported by Metal backend");
    }
    return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "invalid buffer type class");
}

gsx_error gsx_metal_backend_provider_bootstrap(gsx_builtin_registry_state *registry)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(MTLCreateSystemDefaultDevice() == nil) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "No Metal devices available");
    }

    gsx_metal_backend_provider_singleton.base.iface = &gsx_metal_backend_provider_iface;
    gsx_metal_backend_provider_singleton.base.backend_type = GSX_BACKEND_TYPE_METAL;
    gsx_metal_backend_provider_singleton.base.backend_name = "metal";

    error = gsx_metal_backend_provider_singleton.base.iface->discover_devices(
        &gsx_metal_backend_provider_singleton.base,
        registry
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gsx_metal_device_count == 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "No Metal devices available");
    }

    return gsx_builtin_registry_append_provider(registry, &gsx_metal_backend_provider_singleton.base);
}
