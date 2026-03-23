#include "internal.h"
#include "objc-helpers.h"
#include "../bqueue.hpp"
#include "../gsx-data-impl.h"

#import <Metal/Metal.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

extern "C" {
extern const char gsx_metal_tensor_metallib_start[];
extern const char gsx_metal_tensor_metallib_end[];
}

namespace {

typedef struct gsx_metal_async_dl_image_params {
    uint32_t pixel_count;
} gsx_metal_async_dl_image_params;

struct AsyncRequest {
    gsx_size_t stable_sample_index = 0;
    gsx_size_t slot_index = 0;
};

struct SlotScratch {
    id<MTLBuffer> rgb_staging = nil;
    id<MTLBuffer> alpha_staging = nil;
    id<MTLBuffer> invdepth_staging = nil;
    gsx_size_t rgb_size_bytes = 0;
    gsx_size_t alpha_size_bytes = 0;
    gsx_size_t invdepth_size_bytes = 0;
};

struct gsx_metal_async_dl {
    gsx_async_dl base;
    gsx_backend_t backend = nullptr;
    gsx_async_dl_desc desc;
    id<MTLCommandQueue> helper_queue = nil;
    id<MTLComputePipelineState> rgb_pipeline = nil;
    id<MTLComputePipelineState> scalar_pipeline = nil;
    std::unique_ptr<BoundedBlockingQueue<AsyncRequest>> request_queue;
    std::unique_ptr<BoundedBlockingQueue<gsx_async_dl_ready_item>> ready_queue;
    std::thread worker;
    std::vector<SlotScratch> slot_scratch;
    std::atomic<gsx_size_t> inflight_count{ 0 };
    std::atomic<bool> has_error{ false };
    gsx_error_code error_code = GSX_ERROR_SUCCESS;
    std::string error_message;
    std::mutex error_mutex;
};

static gsx_error gsx_metal_async_dl_ensure_tensor_library(gsx_metal_async_dl *async_dl, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        reinterpret_cast<gsx_metal_backend *>(async_dl->backend),
        &reinterpret_cast<gsx_metal_backend *>(async_dl->backend)->tensor_library,
        gsx_metal_tensor_metallib_start,
        gsx_metal_tensor_metallib_end,
        "embedded Metal tensor metallib is empty",
        "failed to create dispatch data for embedded Metal tensor metallib",
        "failed to load embedded Metal tensor metallib",
        out_library);
}

static gsx_error gsx_metal_async_dl_create_pipeline(
    gsx_metal_async_dl *async_dl,
    const char *function_name,
    id<MTLComputePipelineState> *out_pipeline)
{
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> pipeline = nil;
    gsx_error error = gsx_metal_async_dl_ensure_tensor_library(async_dl, &library);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    function = [library newFunctionWithName:[NSString stringWithUTF8String:function_name]];
    if(function == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to look up Metal async dataloader kernel function");
    }
    pipeline = [(id<MTLDevice>)reinterpret_cast<gsx_metal_backend *>(async_dl->backend)->mtl_device newComputePipelineStateWithFunction:function error:NULL];
    [function release];
    if(pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to create Metal async dataloader compute pipeline");
    }
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_metal_async_dl_get_error(gsx_metal_async_dl *async_dl)
{
    std::lock_guard<std::mutex> lock(async_dl->error_mutex);
    if(!async_dl->has_error.load()) {
        return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
    }
    return gsx_make_error(async_dl->error_code, async_dl->error_message.c_str());
}

static void gsx_metal_async_dl_fail(gsx_metal_async_dl *async_dl, gsx_error error)
{
    {
        std::lock_guard<std::mutex> lock(async_dl->error_mutex);
        if(async_dl->has_error.load()) {
            return;
        }
        async_dl->has_error.store(true);
        async_dl->error_code = error.code;
        async_dl->error_message = error.message != nullptr ? error.message : "metal async dataloader worker failed";
    }
    if(async_dl->request_queue) {
        async_dl->request_queue->close();
    }
    if(async_dl->ready_queue) {
        async_dl->ready_queue->close();
    }
}

static gsx_error gsx_metal_async_dl_copy_slot(
    gsx_metal_async_dl *async_dl,
    const AsyncRequest &request,
    const gsx_dataset_cpu_sample *sample)
{
    gsx_dataloader_slot *slot = &async_dl->desc.slots[request.slot_index];
    SlotScratch *scratch = &async_dl->slot_scratch[request.slot_index];
    void *native_handle = nullptr;
    gsx_size_t offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_size_t pixel_count = 0;
    gsx_metal_async_dl_image_params params = { 0 };

    error = gsx_dataloader_compute_element_count(
        async_dl->desc.dataset_desc.width, async_dl->desc.dataset_desc.height, 1, &pixel_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    params.pixel_count = (uint32_t)pixel_count;
    memcpy([scratch->rgb_staging contents], sample->rgb_data, (size_t)scratch->rgb_size_bytes);

    command_buffer = [async_dl->helper_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal async dataloader command buffer");
    }
    encoder = [command_buffer computeCommandEncoder];
    if(encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal async dataloader compute encoder");
    }

    error = gsx_tensor_get_native_handle(slot->rgb_tensor, &native_handle, &offset_bytes);
    if(!gsx_error_is_success(error)) {
        [encoder endEncoding];
        return error;
    }
    [encoder setComputePipelineState:async_dl->rgb_pipeline];
    [encoder setBuffer:scratch->rgb_staging offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)native_handle offset:(NSUInteger)offset_bytes atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    gsx_metal_backend_dispatch_threads_1d(encoder, async_dl->rgb_pipeline, (NSUInteger)(pixel_count * 3u));

    if(async_dl->desc.dataset_desc.has_alpha) {
        memcpy([scratch->alpha_staging contents], sample->alpha_data, (size_t)scratch->alpha_size_bytes);
        error = gsx_tensor_get_native_handle(slot->alpha_tensor, &native_handle, &offset_bytes);
        if(!gsx_error_is_success(error)) {
            [encoder endEncoding];
            return error;
        }
        [encoder setComputePipelineState:async_dl->scalar_pipeline];
        [encoder setBuffer:scratch->alpha_staging offset:0 atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)native_handle offset:(NSUInteger)offset_bytes atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        gsx_metal_backend_dispatch_threads_1d(encoder, async_dl->scalar_pipeline, (NSUInteger)pixel_count);
    }

    if(async_dl->desc.dataset_desc.has_invdepth) {
        memcpy([scratch->invdepth_staging contents], sample->invdepth_data, (size_t)scratch->invdepth_size_bytes);
        error = gsx_tensor_get_native_handle(slot->invdepth_tensor, &native_handle, &offset_bytes);
        if(!gsx_error_is_success(error)) {
            [encoder endEncoding];
            return error;
        }
        [encoder setComputePipelineState:async_dl->scalar_pipeline];
        [encoder setBuffer:scratch->invdepth_staging offset:0 atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)native_handle offset:(NSUInteger)offset_bytes atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        gsx_metal_backend_dispatch_threads_1d(encoder, async_dl->scalar_pipeline, (NSUInteger)pixel_count);
    }

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_metal_async_dl_process_one(gsx_metal_async_dl *async_dl, const AsyncRequest &request)
{
    gsx_dataset_cpu_sample sample{};
    gsx_async_dl_ready_item ready_item{};
    gsx_error error = async_dl->desc.dataset_desc.get_sample(
        async_dl->desc.dataset_desc.object, request.stable_sample_index, &sample);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_validate_sample(&async_dl->desc.dataset_desc, &sample);
    if(!gsx_error_is_success(error)) {
        async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);
        return error;
    }
    if(request.slot_index >= async_dl->desc.slot_count) {
        async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader request slot index is out of range");
    }

    error = gsx_metal_async_dl_copy_slot(async_dl, request, &sample);
    if(!gsx_error_is_success(error)) {
        async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);
        return error;
    }

    ready_item.stable_sample_index = request.stable_sample_index;
    ready_item.slot_index = request.slot_index;
    ready_item.intrinsics = sample.intrinsics;
    ready_item.pose = sample.pose;
    ready_item.stable_sample_id = sample.stable_sample_id;
    ready_item.has_stable_sample_id = sample.has_stable_sample_id;
    async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);

    if(!async_dl->ready_queue->push(ready_item)) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader ready queue closed");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static void gsx_metal_async_dl_worker(gsx_metal_async_dl *async_dl)
{
    AsyncRequest request{};

    while(async_dl->request_queue->pop(request)) {
        const gsx_error error = gsx_metal_async_dl_process_one(async_dl, request);
        if(!gsx_error_is_success(error)) {
            gsx_metal_async_dl_fail(async_dl, error);
            return;
        }
    }
}

static gsx_error gsx_metal_async_dl_destroy_impl(gsx_async_dl_t async_dl_base)
{
    auto *async_dl = reinterpret_cast<gsx_metal_async_dl *>(async_dl_base);

    if(async_dl == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader must be non-null");
    }
    if(async_dl->request_queue) {
        async_dl->request_queue->close();
    }
    if(async_dl->ready_queue) {
        async_dl->ready_queue->close();
    }
    if(async_dl->worker.joinable()) {
        async_dl->worker.join();
    }
    for(SlotScratch &slot_scratch : async_dl->slot_scratch) {
        if(slot_scratch.rgb_staging != nil) {
            [slot_scratch.rgb_staging release];
            slot_scratch.rgb_staging = nil;
        }
        if(slot_scratch.alpha_staging != nil) {
            [slot_scratch.alpha_staging release];
            slot_scratch.alpha_staging = nil;
        }
        if(slot_scratch.invdepth_staging != nil) {
            [slot_scratch.invdepth_staging release];
            slot_scratch.invdepth_staging = nil;
        }
    }
    if(async_dl->rgb_pipeline != nil) {
        [async_dl->rgb_pipeline release];
        async_dl->rgb_pipeline = nil;
    }
    if(async_dl->scalar_pipeline != nil) {
        [async_dl->scalar_pipeline release];
        async_dl->scalar_pipeline = nil;
    }
    if(async_dl->helper_queue != nil) {
        [async_dl->helper_queue release];
        async_dl->helper_queue = nil;
    }
    if(async_dl->backend != nullptr) {
        if(async_dl->backend->live_async_dl_count == 0) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_async_dl_count underflow in async dataloader free");
        }
        async_dl->backend->live_async_dl_count -= 1;
    }
    delete async_dl;
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_metal_async_dl_submit_impl(gsx_async_dl_t async_dl_base, gsx_size_t stable_sample_index, gsx_size_t slot_index)
{
    auto *async_dl = reinterpret_cast<gsx_metal_async_dl *>(async_dl_base);
    AsyncRequest request{};
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    if(async_dl == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader must be non-null");
    }
    error = gsx_metal_async_dl_get_error(async_dl);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    request.stable_sample_index = stable_sample_index;
    request.slot_index = slot_index;
    if(!async_dl->request_queue->push(request)) {
        error = gsx_metal_async_dl_get_error(async_dl);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader request queue closed");
    }
    async_dl->inflight_count.fetch_add(1);
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_metal_async_dl_wait_impl(gsx_async_dl_t async_dl_base, gsx_async_dl_ready_item *out_item)
{
    auto *async_dl = reinterpret_cast<gsx_metal_async_dl *>(async_dl_base);
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    if(async_dl == nullptr || out_item == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader and out_item must be non-null");
    }
    if(async_dl->ready_queue->pop(*out_item)) {
        async_dl->inflight_count.fetch_sub(1);
        return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
    }

    error = gsx_metal_async_dl_get_error(async_dl);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader ready queue closed");
}

static gsx_error gsx_metal_async_dl_inflight_count_impl(gsx_async_dl_t async_dl_base, gsx_size_t *out_count)
{
    auto *async_dl = reinterpret_cast<gsx_metal_async_dl *>(async_dl_base);

    if(async_dl == nullptr || out_count == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader and out_count must be non-null");
    }
    *out_count = async_dl->inflight_count.load();
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static const gsx_async_dl_i gsx_metal_async_dl_iface = {
    gsx_metal_async_dl_destroy_impl,
    gsx_metal_async_dl_submit_impl,
    gsx_metal_async_dl_wait_impl,
    gsx_metal_async_dl_inflight_count_impl,
};

}  // namespace

extern "C" gsx_error gsx_metal_backend_create_async_dl(gsx_backend_t backend, const gsx_async_dl_desc *desc, gsx_async_dl_t *out_async_dl)
{
    gsx_metal_async_dl *async_dl = nullptr;
    gsx_size_t slot_index = 0;
    gsx_size_t scalar_size_bytes = 0;
    const char *rgb_kernel_name = nullptr;
    const char *scalar_kernel_name = nullptr;
    id<MTLDevice> device = nil;

    if(backend == nullptr || out_async_dl == nullptr || desc == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, out_async_dl, and desc must be non-null");
    }
    if(desc->slot_count == 0 || desc->slots == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader requires at least one slot");
    }
    *out_async_dl = nullptr;

    try {
        async_dl = new gsx_metal_async_dl();
        async_dl->base.iface = &gsx_metal_async_dl_iface;
        async_dl->backend = backend;
        async_dl->desc = *desc;
        async_dl->slot_scratch.resize(desc->slot_count);
        async_dl->request_queue = std::make_unique<BoundedBlockingQueue<AsyncRequest>>(desc->slot_count);
        async_dl->ready_queue = std::make_unique<BoundedBlockingQueue<gsx_async_dl_ready_item>>(desc->slot_count);
        device = (id<MTLDevice>)reinterpret_cast<gsx_metal_backend *>(desc->backend)->mtl_device;
        async_dl->helper_queue = [device newCommandQueue];
        if(async_dl->helper_queue == nil) {
            throw gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal async dataloader helper queue");
        }

        {
            gsx_error error = gsx_dataloader_compute_packed_bytes(
                desc->dataset_desc.image_data_type, desc->dataset_desc.width, desc->dataset_desc.height, 1, &scalar_size_bytes);

            if(!gsx_error_is_success(error)) {
                throw error;
            }
        }
        if(desc->dataset_desc.image_data_type == GSX_DATA_TYPE_U8) {
            rgb_kernel_name = "gsx_metal_async_dl_rgb_u8_hwc_to_chw_f32_kernel";
            scalar_kernel_name = "gsx_metal_async_dl_scalar_u8_hw_to_chw_f32_kernel";
        } else {
            rgb_kernel_name = "gsx_metal_async_dl_rgb_f32_hwc_to_chw_f32_kernel";
            scalar_kernel_name = "gsx_metal_async_dl_scalar_f32_hw_to_chw_f32_kernel";
        }
        {
            gsx_error error = gsx_metal_async_dl_create_pipeline(async_dl, rgb_kernel_name, &async_dl->rgb_pipeline);

            if(!gsx_error_is_success(error)) {
                throw error;
            }
        }
        {
            gsx_error error = gsx_metal_async_dl_create_pipeline(async_dl, scalar_kernel_name, &async_dl->scalar_pipeline);

            if(!gsx_error_is_success(error)) {
                throw error;
            }
        }

        for(slot_index = 0; slot_index < desc->slot_count; ++slot_index) {
            async_dl->slot_scratch[slot_index].rgb_size_bytes = scalar_size_bytes * 3u;
            async_dl->slot_scratch[slot_index].rgb_staging = [device
                newBufferWithLength:(NSUInteger)async_dl->slot_scratch[slot_index].rgb_size_bytes
                options:MTLResourceStorageModeShared];
            if(async_dl->slot_scratch[slot_index].rgb_staging == nil) {
                throw gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal async dataloader rgb staging buffer");
            }
            if(desc->dataset_desc.has_alpha) {
                async_dl->slot_scratch[slot_index].alpha_size_bytes = scalar_size_bytes;
                async_dl->slot_scratch[slot_index].alpha_staging = [device
                    newBufferWithLength:(NSUInteger)scalar_size_bytes
                    options:MTLResourceStorageModeShared];
                if(async_dl->slot_scratch[slot_index].alpha_staging == nil) {
                    throw gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal async dataloader alpha staging buffer");
                }
            }
            if(desc->dataset_desc.has_invdepth) {
                async_dl->slot_scratch[slot_index].invdepth_size_bytes = scalar_size_bytes;
                async_dl->slot_scratch[slot_index].invdepth_staging = [device
                    newBufferWithLength:(NSUInteger)scalar_size_bytes
                    options:MTLResourceStorageModeShared];
                if(async_dl->slot_scratch[slot_index].invdepth_staging == nil) {
                    throw gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal async dataloader inverse-depth staging buffer");
                }
            }
        }

        async_dl->worker = std::thread([async_dl]() {
            gsx_metal_async_dl_worker(async_dl);
        });
        backend->live_async_dl_count += 1;
    } catch(const gsx_error &error) {
        if(async_dl != nullptr) {
            for(SlotScratch &slot_scratch : async_dl->slot_scratch) {
                if(slot_scratch.rgb_staging != nil) {
                    [slot_scratch.rgb_staging release];
                }
                if(slot_scratch.alpha_staging != nil) {
                    [slot_scratch.alpha_staging release];
                }
                if(slot_scratch.invdepth_staging != nil) {
                    [slot_scratch.invdepth_staging release];
                }
            }
            if(async_dl->rgb_pipeline != nil) {
                [async_dl->rgb_pipeline release];
            }
            if(async_dl->scalar_pipeline != nil) {
                [async_dl->scalar_pipeline release];
            }
            if(async_dl->helper_queue != nil) {
                [async_dl->helper_queue release];
            }
        }
        delete async_dl;
        return error;
    } catch(const std::bad_alloc &) {
        if(async_dl != nullptr) {
            for(SlotScratch &slot_scratch : async_dl->slot_scratch) {
                if(slot_scratch.rgb_staging != nil) {
                    [slot_scratch.rgb_staging release];
                }
                if(slot_scratch.alpha_staging != nil) {
                    [slot_scratch.alpha_staging release];
                }
                if(slot_scratch.invdepth_staging != nil) {
                    [slot_scratch.invdepth_staging release];
                }
            }
            if(async_dl->rgb_pipeline != nil) {
                [async_dl->rgb_pipeline release];
            }
            if(async_dl->scalar_pipeline != nil) {
                [async_dl->scalar_pipeline release];
            }
            if(async_dl->helper_queue != nil) {
                [async_dl->helper_queue release];
            }
        }
        delete async_dl;
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal async dataloader");
    } catch(...) {
        if(async_dl != nullptr) {
            for(SlotScratch &slot_scratch : async_dl->slot_scratch) {
                if(slot_scratch.rgb_staging != nil) {
                    [slot_scratch.rgb_staging release];
                }
                if(slot_scratch.alpha_staging != nil) {
                    [slot_scratch.alpha_staging release];
                }
                if(slot_scratch.invdepth_staging != nil) {
                    [slot_scratch.invdepth_staging release];
                }
            }
            if(async_dl->rgb_pipeline != nil) {
                [async_dl->rgb_pipeline release];
            }
            if(async_dl->scalar_pipeline != nil) {
                [async_dl->scalar_pipeline release];
            }
            if(async_dl->helper_queue != nil) {
                [async_dl->helper_queue release];
            }
        }
        delete async_dl;
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to initialize metal async dataloader");
    }

    *out_async_dl = &async_dl->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}
