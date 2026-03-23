#include "../bqueue.hpp"
#include "../gsx-data-impl.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

namespace {

struct AsyncRequest {
    gsx_size_t stable_sample_index = 0;
    gsx_size_t slot_index = 0;
};

struct gsx_cpu_async_dl {
    gsx_async_dl base;
    gsx_backend_t backend = nullptr;
    gsx_async_dl_desc desc;
    std::unique_ptr<BoundedBlockingQueue<AsyncRequest>> request_queue;
    std::unique_ptr<BoundedBlockingQueue<gsx_async_dl_ready_item>> ready_queue;
    std::thread worker;
    std::atomic<gsx_size_t> inflight_count{ 0 };
    std::atomic<bool> has_error{ false };
    gsx_error_code error_code = GSX_ERROR_SUCCESS;
    std::string error_message;
    std::mutex error_mutex;
};

static gsx_error gsx_cpu_async_dl_get_error(gsx_cpu_async_dl *async_dl)
{
    std::lock_guard<std::mutex> lock(async_dl->error_mutex);
    if(!async_dl->has_error.load()) {
        return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
    }
    return gsx_make_error(async_dl->error_code, async_dl->error_message.c_str());
}

static void gsx_cpu_async_dl_fail(gsx_cpu_async_dl *async_dl, gsx_error error)
{
    {
        std::lock_guard<std::mutex> lock(async_dl->error_mutex);
        if(async_dl->has_error.load()) {
            return;
        }
        async_dl->has_error.store(true);
        async_dl->error_code = error.code;
        async_dl->error_message = error.message != nullptr ? error.message : "async dataloader worker failed";
    }
    if(async_dl->request_queue) {
        async_dl->request_queue->close();
    }
    if(async_dl->ready_queue) {
        async_dl->ready_queue->close();
    }
}

static gsx_error gsx_cpu_async_dl_pack_tensor(
    const gsx_async_dl_desc *desc,
    const void *src_bytes,
    gsx_index_t channel_count,
    gsx_tensor_t tensor)
{
    void *native_handle = nullptr;
    gsx_size_t offset_bytes = 0;
    gsx_error error = gsx_tensor_get_native_handle(tensor, &native_handle, &offset_bytes);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(channel_count == 3) {
        return gsx_dataloader_pack_hwc_to_chw(
            src_bytes,
            desc->dataset_desc.image_data_type,
            desc->output_data_type,
            desc->dataset_desc.width,
            desc->dataset_desc.height,
            static_cast<unsigned char *>(native_handle) + offset_bytes);
    }
    return gsx_dataloader_pack_hw_to_chw(
        src_bytes,
        desc->dataset_desc.image_data_type,
        desc->output_data_type,
        desc->dataset_desc.width,
        desc->dataset_desc.height,
        static_cast<unsigned char *>(native_handle) + offset_bytes);
}

static gsx_error gsx_cpu_async_dl_process_one(gsx_cpu_async_dl *async_dl, const AsyncRequest &request)
{
    gsx_dataset_cpu_sample sample{};
    gsx_async_dl_ready_item ready_item{};
    gsx_dataloader_slot *slot = nullptr;
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

    slot = &async_dl->desc.slots[request.slot_index];
    error = gsx_cpu_async_dl_pack_tensor(&async_dl->desc, sample.rgb_data, 3, slot->rgb_tensor);
    if(!gsx_error_is_success(error)) {
        async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);
        return error;
    }
    if(async_dl->desc.dataset_desc.has_alpha) {
        error = gsx_cpu_async_dl_pack_tensor(&async_dl->desc, sample.alpha_data, 1, slot->alpha_tensor);
        if(!gsx_error_is_success(error)) {
            async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);
            return error;
        }
    }
    if(async_dl->desc.dataset_desc.has_invdepth) {
        error = gsx_cpu_async_dl_pack_tensor(&async_dl->desc, sample.invdepth_data, 1, slot->invdepth_tensor);
        if(!gsx_error_is_success(error)) {
            async_dl->desc.dataset_desc.release_sample(async_dl->desc.dataset_desc.object, &sample);
            return error;
        }
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

static void gsx_cpu_async_dl_worker(gsx_cpu_async_dl *async_dl)
{
    AsyncRequest request{};

    while(async_dl->request_queue->pop(request)) {
        const gsx_error error = gsx_cpu_async_dl_process_one(async_dl, request);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_async_dl_fail(async_dl, error);
            return;
        }
    }
}

static gsx_error gsx_cpu_async_dl_destroy_impl(gsx_async_dl_t async_dl_base)
{
    auto *async_dl = reinterpret_cast<gsx_cpu_async_dl *>(async_dl_base);

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
    if(async_dl->backend != nullptr) {
        if(async_dl->backend->live_async_dl_count == 0) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_async_dl_count underflow in async dataloader free");
        }
        async_dl->backend->live_async_dl_count -= 1;
    }
    delete async_dl;
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_cpu_async_dl_submit_impl(gsx_async_dl_t async_dl_base, gsx_size_t stable_sample_index, gsx_size_t slot_index)
{
    auto *async_dl = reinterpret_cast<gsx_cpu_async_dl *>(async_dl_base);
    AsyncRequest request{};
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    if(async_dl == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader must be non-null");
    }
    error = gsx_cpu_async_dl_get_error(async_dl);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    request.stable_sample_index = stable_sample_index;
    request.slot_index = slot_index;
    if(!async_dl->request_queue->push(request)) {
        error = gsx_cpu_async_dl_get_error(async_dl);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader request queue closed");
    }
    async_dl->inflight_count.fetch_add(1);
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_cpu_async_dl_wait_impl(gsx_async_dl_t async_dl_base, gsx_async_dl_ready_item *out_item)
{
    auto *async_dl = reinterpret_cast<gsx_cpu_async_dl *>(async_dl_base);
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    if(async_dl == nullptr || out_item == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader and out_item must be non-null");
    }
    if(async_dl->ready_queue->pop(*out_item)) {
        async_dl->inflight_count.fetch_sub(1);
        return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
    }

    error = gsx_cpu_async_dl_get_error(async_dl);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader ready queue closed");
}

static gsx_error gsx_cpu_async_dl_inflight_count_impl(gsx_async_dl_t async_dl_base, gsx_size_t *out_count)
{
    auto *async_dl = reinterpret_cast<gsx_cpu_async_dl *>(async_dl_base);

    if(async_dl == nullptr || out_count == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader and out_count must be non-null");
    }
    *out_count = async_dl->inflight_count.load();
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static const gsx_async_dl_i gsx_cpu_async_dl_iface = {
    gsx_cpu_async_dl_destroy_impl,
    gsx_cpu_async_dl_submit_impl,
    gsx_cpu_async_dl_wait_impl,
    gsx_cpu_async_dl_inflight_count_impl,
};

}  // namespace

extern "C" gsx_error gsx_cpu_backend_create_async_dl(gsx_backend_t backend, const gsx_async_dl_desc *desc, gsx_async_dl_t *out_async_dl)
{
    gsx_cpu_async_dl *async_dl = nullptr;

    if(backend == nullptr || out_async_dl == nullptr || desc == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, out_async_dl, and desc must be non-null");
    }
    if(desc->slot_count == 0 || desc->slots == nullptr) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "async dataloader requires at least one slot");
    }
    *out_async_dl = nullptr;

    try {
        async_dl = new gsx_cpu_async_dl();
        async_dl->base.iface = &gsx_cpu_async_dl_iface;
        async_dl->backend = backend;
        async_dl->desc = *desc;
        async_dl->request_queue = std::make_unique<BoundedBlockingQueue<AsyncRequest>>(desc->slot_count);
        async_dl->ready_queue = std::make_unique<BoundedBlockingQueue<gsx_async_dl_ready_item>>(desc->slot_count);
        async_dl->worker = std::thread([async_dl]() {
            gsx_cpu_async_dl_worker(async_dl);
        });
        backend->live_async_dl_count += 1;
    } catch(const std::bad_alloc &) {
        delete async_dl;
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu async dataloader");
    } catch(...) {
        delete async_dl;
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to initialize cpu async dataloader");
    }

    *out_async_dl = &async_dl->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}
