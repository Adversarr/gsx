#include "internal.h"
#include "../bqueue.hpp"
#include "../gsx-data-impl.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct AsyncRequest {
    gsx_size_t stable_sample_index = 0;
    gsx_size_t slot_index = 0;
};

struct SlotScratch {
    void *rgb_device_bytes = nullptr;
    void *alpha_device_bytes = nullptr;
    void *invdepth_device_bytes = nullptr;
    gsx_size_t rgb_size_bytes = 0;
    gsx_size_t alpha_size_bytes = 0;
    gsx_size_t invdepth_size_bytes = 0;
};

struct gsx_cuda_async_dl {
    gsx_async_dl base;
    gsx_backend_t backend = nullptr;
    gsx_async_dl_desc desc;
    cudaStream_t stream = nullptr;
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

__global__ static void gsx_cuda_async_dl_rgb_u8_hwc_to_chw_f32_kernel(
    const uint8_t *__restrict__ src_hwc,
    float *__restrict__ dst_chw,
    gsx_size_t pixel_count)
{
    gsx_size_t index = (gsx_size_t)blockIdx.x * (gsx_size_t)blockDim.x + (gsx_size_t)threadIdx.x;
    gsx_size_t total_count = pixel_count * 3u;

    if(index >= total_count) {
        return;
    }

    dst_chw[index] = (float)src_hwc[((index % pixel_count) * 3u) + (index / pixel_count)];
}

__global__ static void gsx_cuda_async_dl_rgb_f32_hwc_to_chw_f32_kernel(
    const float *__restrict__ src_hwc,
    float *__restrict__ dst_chw,
    gsx_size_t pixel_count)
{
    gsx_size_t index = (gsx_size_t)blockIdx.x * (gsx_size_t)blockDim.x + (gsx_size_t)threadIdx.x;
    gsx_size_t total_count = pixel_count * 3u;

    if(index >= total_count) {
        return;
    }

    dst_chw[index] = src_hwc[((index % pixel_count) * 3u) + (index / pixel_count)];
}

__global__ static void gsx_cuda_async_dl_scalar_u8_hw_to_chw_f32_kernel(
    const uint8_t *__restrict__ src_hw,
    float *__restrict__ dst_chw,
    gsx_size_t pixel_count)
{
    gsx_size_t index = (gsx_size_t)blockIdx.x * (gsx_size_t)blockDim.x + (gsx_size_t)threadIdx.x;

    if(index >= pixel_count) {
        return;
    }

    dst_chw[index] = (float)src_hw[index];
}

__global__ static void gsx_cuda_async_dl_scalar_f32_hw_to_chw_f32_kernel(
    const float *__restrict__ src_hw,
    float *__restrict__ dst_chw,
    gsx_size_t pixel_count)
{
    gsx_size_t index = (gsx_size_t)blockIdx.x * (gsx_size_t)blockDim.x + (gsx_size_t)threadIdx.x;

    if(index >= pixel_count) {
        return;
    }

    dst_chw[index] = src_hw[index];
}

static cudaError_t gsx_cuda_async_dl_launch_rgb_kernel(
    gsx_data_type src_type,
    const void *src_device_bytes,
    void *dst_device_bytes,
    gsx_size_t pixel_count,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = 0;
    const gsx_size_t total_count = pixel_count * 3u;

    if(total_count == 0) {
        return cudaSuccess;
    }
    grid_size = (int)((total_count + (gsx_size_t)block_size - 1u) / (gsx_size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    if(src_type == GSX_DATA_TYPE_U8) {
        gsx_cuda_async_dl_rgb_u8_hwc_to_chw_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const uint8_t *>(src_device_bytes), static_cast<float *>(dst_device_bytes), pixel_count);
    } else {
        gsx_cuda_async_dl_rgb_f32_hwc_to_chw_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float *>(src_device_bytes), static_cast<float *>(dst_device_bytes), pixel_count);
    }
    return cudaGetLastError();
}

static cudaError_t gsx_cuda_async_dl_launch_scalar_kernel(
    gsx_data_type src_type,
    const void *src_device_bytes,
    void *dst_device_bytes,
    gsx_size_t pixel_count,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = 0;

    if(pixel_count == 0) {
        return cudaSuccess;
    }
    grid_size = (int)((pixel_count + (gsx_size_t)block_size - 1u) / (gsx_size_t)block_size);
    if(grid_size > 65535) {
        grid_size = 65535;
    }

    if(src_type == GSX_DATA_TYPE_U8) {
        gsx_cuda_async_dl_scalar_u8_hw_to_chw_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const uint8_t *>(src_device_bytes), static_cast<float *>(dst_device_bytes), pixel_count);
    } else {
        gsx_cuda_async_dl_scalar_f32_hw_to_chw_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float *>(src_device_bytes), static_cast<float *>(dst_device_bytes), pixel_count);
    }
    return cudaGetLastError();
}

static gsx_error gsx_cuda_async_dl_get_error(gsx_cuda_async_dl *async_dl)
{
    std::lock_guard<std::mutex> lock(async_dl->error_mutex);
    if(!async_dl->has_error.load()) {
        return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
    }
    return gsx_make_error(async_dl->error_code, async_dl->error_message.c_str());
}

static void gsx_cuda_async_dl_fail(gsx_cuda_async_dl *async_dl, gsx_error error)
{
    {
        std::lock_guard<std::mutex> lock(async_dl->error_mutex);
        if(async_dl->has_error.load()) {
            return;
        }
        async_dl->has_error.store(true);
        async_dl->error_code = error.code;
        async_dl->error_message = error.message != nullptr ? error.message : "cuda async dataloader worker failed";
    }
    if(async_dl->request_queue) {
        async_dl->request_queue->close();
    }
    if(async_dl->ready_queue) {
        async_dl->ready_queue->close();
    }
}

static gsx_error gsx_cuda_async_dl_copy_slot(
    gsx_cuda_async_dl *async_dl,
    const AsyncRequest &request,
    const gsx_dataset_cpu_sample *sample)
{
    gsx_dataloader_slot *slot = &async_dl->desc.slots[request.slot_index];
    SlotScratch *scratch = &async_dl->slot_scratch[request.slot_index];
    cudaError_t cuda_error = cudaSuccess;
    void *native_handle = nullptr;
    gsx_size_t offset_bytes = 0;
    gsx_size_t pixel_count = 0;
    gsx_error error = gsx_dataloader_compute_element_count(
        async_dl->desc.dataset_desc.width, async_dl->desc.dataset_desc.height, 1, &pixel_count);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    cuda_error = cudaMemcpyAsync(
        scratch->rgb_device_bytes, sample->rgb_data, scratch->rgb_size_bytes, cudaMemcpyHostToDevice, async_dl->stream);
    if(cuda_error != cudaSuccess) {
        return gsx_cuda_make_error(cuda_error, "cuda async dataloader rgb staging upload failed");
    }
    error = gsx_tensor_get_native_handle(slot->rgb_tensor, &native_handle, &offset_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    cuda_error = gsx_cuda_async_dl_launch_rgb_kernel(
        async_dl->desc.dataset_desc.image_data_type,
        scratch->rgb_device_bytes,
        static_cast<unsigned char *>(native_handle) + offset_bytes,
        pixel_count,
        async_dl->stream);
    if(cuda_error != cudaSuccess) {
        return gsx_cuda_make_error(cuda_error, "cuda async dataloader rgb layout conversion failed");
    }

    if(async_dl->desc.dataset_desc.has_alpha) {
        cuda_error = cudaMemcpyAsync(
            scratch->alpha_device_bytes, sample->alpha_data, scratch->alpha_size_bytes, cudaMemcpyHostToDevice, async_dl->stream);
        if(cuda_error != cudaSuccess) {
            return gsx_cuda_make_error(cuda_error, "cuda async dataloader alpha staging upload failed");
        }
        error = gsx_tensor_get_native_handle(slot->alpha_tensor, &native_handle, &offset_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        cuda_error = gsx_cuda_async_dl_launch_scalar_kernel(
            async_dl->desc.dataset_desc.image_data_type,
            scratch->alpha_device_bytes,
            static_cast<unsigned char *>(native_handle) + offset_bytes,
            pixel_count,
            async_dl->stream);
        if(cuda_error != cudaSuccess) {
            return gsx_cuda_make_error(cuda_error, "cuda async dataloader alpha layout conversion failed");
        }
    }

    if(async_dl->desc.dataset_desc.has_invdepth) {
        cuda_error = cudaMemcpyAsync(
            scratch->invdepth_device_bytes,
            sample->invdepth_data,
            scratch->invdepth_size_bytes,
            cudaMemcpyHostToDevice,
            async_dl->stream);
        if(cuda_error != cudaSuccess) {
            return gsx_cuda_make_error(cuda_error, "cuda async dataloader inverse-depth staging upload failed");
        }
        error = gsx_tensor_get_native_handle(slot->invdepth_tensor, &native_handle, &offset_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        cuda_error = gsx_cuda_async_dl_launch_scalar_kernel(
            async_dl->desc.dataset_desc.image_data_type,
            scratch->invdepth_device_bytes,
            static_cast<unsigned char *>(native_handle) + offset_bytes,
            pixel_count,
            async_dl->stream);
        if(cuda_error != cudaSuccess) {
            return gsx_cuda_make_error(cuda_error, "cuda async dataloader inverse-depth layout conversion failed");
        }
    }

    return gsx_cuda_make_error(cudaStreamSynchronize(async_dl->stream), "cudaStreamSynchronize async dataloader upload failed");
}

static gsx_error gsx_cuda_async_dl_process_one(gsx_cuda_async_dl *async_dl, const AsyncRequest &request)
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

    error = gsx_cuda_async_dl_copy_slot(async_dl, request, &sample);
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

static void gsx_cuda_async_dl_worker(gsx_cuda_async_dl *async_dl)
{
    AsyncRequest request{};

    while(async_dl->request_queue->pop(request)) {
        const gsx_error error = gsx_cuda_async_dl_process_one(async_dl, request);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_async_dl_fail(async_dl, error);
            return;
        }
    }
}

static void gsx_cuda_async_dl_free_slot_scratch(gsx_cuda_async_dl *async_dl)
{
    for(SlotScratch &slot_scratch : async_dl->slot_scratch) {
        if(slot_scratch.rgb_device_bytes != nullptr) {
            (void)cudaFree(slot_scratch.rgb_device_bytes);
            slot_scratch.rgb_device_bytes = nullptr;
        }
        if(slot_scratch.alpha_device_bytes != nullptr) {
            (void)cudaFree(slot_scratch.alpha_device_bytes);
            slot_scratch.alpha_device_bytes = nullptr;
        }
        if(slot_scratch.invdepth_device_bytes != nullptr) {
            (void)cudaFree(slot_scratch.invdepth_device_bytes);
            slot_scratch.invdepth_device_bytes = nullptr;
        }
    }
}

static gsx_error gsx_cuda_async_dl_destroy_impl(gsx_async_dl_t async_dl_base)
{
    auto *async_dl = reinterpret_cast<gsx_cuda_async_dl *>(async_dl_base);

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
    gsx_cuda_async_dl_free_slot_scratch(async_dl);
    if(async_dl->stream != nullptr) {
        (void)cudaStreamDestroy(async_dl->stream);
        async_dl->stream = nullptr;
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

static gsx_error gsx_cuda_async_dl_submit_impl(gsx_async_dl_t async_dl_base, gsx_size_t stable_sample_index, gsx_size_t slot_index)
{
    auto *async_dl = reinterpret_cast<gsx_cuda_async_dl *>(async_dl_base);
    AsyncRequest request{};
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    if(async_dl == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader must be non-null");
    }
    error = gsx_cuda_async_dl_get_error(async_dl);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    request.stable_sample_index = stable_sample_index;
    request.slot_index = slot_index;
    if(!async_dl->request_queue->push(request)) {
        error = gsx_cuda_async_dl_get_error(async_dl);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader request queue closed");
    }
    async_dl->inflight_count.fetch_add(1);
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_cuda_async_dl_wait_impl(gsx_async_dl_t async_dl_base, gsx_async_dl_ready_item *out_item)
{
    auto *async_dl = reinterpret_cast<gsx_cuda_async_dl *>(async_dl_base);
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    if(async_dl == nullptr || out_item == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader and out_item must be non-null");
    }
    if(async_dl->ready_queue->pop(*out_item)) {
        async_dl->inflight_count.fetch_sub(1);
        return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
    }

    error = gsx_cuda_async_dl_get_error(async_dl);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader ready queue closed");
}

static gsx_error gsx_cuda_async_dl_inflight_count_impl(gsx_async_dl_t async_dl_base, gsx_size_t *out_count)
{
    auto *async_dl = reinterpret_cast<gsx_cuda_async_dl *>(async_dl_base);

    if(async_dl == nullptr || out_count == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader and out_count must be non-null");
    }
    *out_count = async_dl->inflight_count.load();
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static const gsx_async_dl_i gsx_cuda_async_dl_iface = {
    gsx_cuda_async_dl_destroy_impl,
    gsx_cuda_async_dl_submit_impl,
    gsx_cuda_async_dl_wait_impl,
    gsx_cuda_async_dl_inflight_count_impl,
};

}  // namespace

extern "C" gsx_error gsx_cuda_backend_create_async_dl(gsx_backend_t backend, const gsx_async_dl_desc *desc, gsx_async_dl_t *out_async_dl)
{
    gsx_cuda_async_dl *async_dl = nullptr;
    cudaError_t cuda_error = cudaSuccess;
    gsx_size_t slot_index = 0;
    gsx_size_t scalar_size_bytes = 0;

    if(backend == nullptr || out_async_dl == nullptr || desc == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, out_async_dl, and desc must be non-null");
    }
    if(desc->slot_count == 0 || desc->slots == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader requires at least one slot");
    }
    *out_async_dl = nullptr;

    try {
        async_dl = new gsx_cuda_async_dl();
        async_dl->base.iface = &gsx_cuda_async_dl_iface;
        async_dl->backend = backend;
        async_dl->desc = *desc;
        async_dl->slot_scratch.resize(desc->slot_count);
        async_dl->request_queue = std::make_unique<BoundedBlockingQueue<AsyncRequest>>(desc->slot_count);
        async_dl->ready_queue = std::make_unique<BoundedBlockingQueue<gsx_async_dl_ready_item>>(desc->slot_count);

        cuda_error = cudaSetDevice(reinterpret_cast<gsx_cuda_backend_device *>(desc->backend->device)->cuda_device_ordinal);
        if(cuda_error != cudaSuccess) {
            throw gsx_cuda_make_error(cuda_error, "cudaSetDevice failed for async dataloader");
        }
        cuda_error = cudaStreamCreateWithFlags(&async_dl->stream, cudaStreamNonBlocking);
        if(cuda_error != cudaSuccess) {
            throw gsx_cuda_make_error(cuda_error, "cudaStreamCreateWithFlags failed for async dataloader");
        }

        {
            gsx_error error = gsx_dataloader_compute_packed_bytes(
                desc->dataset_desc.image_data_type, desc->dataset_desc.width, desc->dataset_desc.height, 1, &scalar_size_bytes);

            if(!gsx_error_is_success(error)) {
                throw error;
            }
        }
        for(slot_index = 0; slot_index < desc->slot_count; ++slot_index) {
            async_dl->slot_scratch[slot_index].rgb_size_bytes = scalar_size_bytes * 3u;
            cuda_error = cudaMalloc(&async_dl->slot_scratch[slot_index].rgb_device_bytes, async_dl->slot_scratch[slot_index].rgb_size_bytes);
            if(cuda_error != cudaSuccess) {
                throw gsx_cuda_make_error(cuda_error, "cudaMalloc failed for async dataloader rgb staging");
            }
            if(desc->dataset_desc.has_alpha) {
                async_dl->slot_scratch[slot_index].alpha_size_bytes = scalar_size_bytes;
                cuda_error = cudaMalloc(&async_dl->slot_scratch[slot_index].alpha_device_bytes, scalar_size_bytes);
                if(cuda_error != cudaSuccess) {
                    throw gsx_cuda_make_error(cuda_error, "cudaMalloc failed for async dataloader alpha staging");
                }
            }
            if(desc->dataset_desc.has_invdepth) {
                async_dl->slot_scratch[slot_index].invdepth_size_bytes = scalar_size_bytes;
                cuda_error = cudaMalloc(&async_dl->slot_scratch[slot_index].invdepth_device_bytes, scalar_size_bytes);
                if(cuda_error != cudaSuccess) {
                    throw gsx_cuda_make_error(cuda_error, "cudaMalloc failed for async dataloader inverse-depth staging");
                }
            }
        }

        async_dl->worker = std::thread([async_dl]() {
            gsx_cuda_async_dl_worker(async_dl);
        });
        backend->live_async_dl_count += 1;
    } catch(const gsx_error &error) {
        if(async_dl != nullptr) {
            gsx_cuda_async_dl_free_slot_scratch(async_dl);
            if(async_dl->stream != nullptr) {
                (void)cudaStreamDestroy(async_dl->stream);
            }
        }
        delete async_dl;
        return error;
    } catch(const std::bad_alloc &) {
        if(async_dl != nullptr) {
            gsx_cuda_async_dl_free_slot_scratch(async_dl);
            if(async_dl->stream != nullptr) {
                (void)cudaStreamDestroy(async_dl->stream);
            }
        }
        delete async_dl;
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda async dataloader");
    } catch(...) {
        if(async_dl != nullptr) {
            gsx_cuda_async_dl_free_slot_scratch(async_dl);
            if(async_dl->stream != nullptr) {
                (void)cudaStreamDestroy(async_dl->stream);
            }
        }
        delete async_dl;
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to initialize cuda async dataloader");
    }

    *out_async_dl = &async_dl->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}
