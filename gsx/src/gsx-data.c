#include "gsx-data-impl.h"
#include "pcg32.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct gsx_dataset {
    gsx_dataset_desc desc;
    gsx_size_t length;
    gsx_size_t live_dataloader_count;
};

struct gsx_dataloader {
    gsx_backend_t backend;
    gsx_dataset_t dataset;
    gsx_backend_buffer_type_t buffer_type;
    gsx_arena_t arena;
    gsx_dataloader_desc desc;
    gsx_size_t length;

    gsx_dataloader_slot *slots;
    gsx_size_t slot_count;
    gsx_size_t *free_slot_indices;
    gsx_size_t free_slot_count;
    gsx_size_t current_result_slot_index;
    bool has_current_result_slot;

    void *rgb_scratch;
    gsx_size_t rgb_scratch_capacity_bytes;
    void *alpha_scratch;
    gsx_size_t alpha_scratch_capacity_bytes;
    void *invdepth_scratch;
    gsx_size_t invdepth_scratch_capacity_bytes;

    gsx_size_t *permutation;
    gsx_size_t epoch_index;
    gsx_size_t next_output_ordinal;
    gsx_size_t next_submit_ordinal;
    gsx_dataloader_boundary_flags next_boundary_flags;
    gsx_pcg32 rng;
    gsx_pcg32 initial_rng;
    gsx_async_dl_t async_dl;
};

static gsx_error gsx_dataset_require_handle(gsx_dataset_t dataset)
{
    if(dataset == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset must be non-null");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_require_handle(gsx_dataloader_t dataloader)
{
    if(dataloader == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader must be non-null");
    }
    if(dataloader->backend == NULL || dataloader->dataset == NULL || dataloader->arena == NULL || dataloader->permutation == NULL || dataloader->slots == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "dataloader is detached from required owned state");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_ensure_scratch(
    void **scratch_bytes,
    gsx_size_t *scratch_capacity_bytes,
    gsx_size_t required_size_bytes)
{
    void *new_bytes = NULL;

    if(scratch_bytes == NULL || scratch_capacity_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scratch pointers must be non-null");
    }
    if(required_size_bytes > (gsx_size_t)SIZE_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scratch byte size exceeds platform allocation limits");
    }
    if(required_size_bytes <= *scratch_capacity_bytes) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    new_bytes = realloc(*scratch_bytes, (size_t)required_size_bytes);
    if(new_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate dataloader scratch storage");
    }

    *scratch_bytes = new_bytes;
    *scratch_capacity_bytes = required_size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_dataloader_rng_seed(gsx_size_t seed, gsx_pcg32 *out_rng)
{
    const uint64_t initstate = (uint64_t)seed;
    const uint64_t initseq = UINT64_C(0x9e3779b97f4a7c15) ^ (initstate << 1);

    pcg32_init(out_rng, initstate, initseq);
}

static uint64_t gsx_dataloader_rng_next_u64(gsx_pcg32 *rng)
{
    const uint64_t upper = (uint64_t)pcg32_next_uint(rng);
    const uint64_t lower = (uint64_t)pcg32_next_uint(rng);

    return (upper << 32u) | lower;
}

static gsx_size_t gsx_dataloader_rng_bounded(gsx_pcg32 *rng, gsx_size_t bound)
{
    const uint64_t bound_u64 = (uint64_t)bound;
    const uint64_t threshold = (UINT64_C(0) - bound_u64) % bound_u64;
    uint64_t value = 0;

    for(;;) {
        value = gsx_dataloader_rng_next_u64(rng);
        if(value >= threshold) {
            return (gsx_size_t)(value % bound_u64);
        }
    }
}

static void gsx_dataloader_fill_identity_permutation(gsx_dataloader_t dataloader)
{
    gsx_size_t index = 0;

    for(index = 0; index < dataloader->length; ++index) {
        dataloader->permutation[index] = index;
    }
}

static void gsx_dataloader_shuffle_permutation(gsx_dataloader_t dataloader, gsx_pcg32 *rng)
{
    gsx_size_t index = 0;

    gsx_dataloader_fill_identity_permutation(dataloader);
    if(dataloader->length <= 1) {
        return;
    }

    for(index = dataloader->length - 1; index > 0; --index) {
        gsx_size_t swap_index = gsx_dataloader_rng_bounded(rng, index + 1);
        gsx_size_t temp = dataloader->permutation[index];

        dataloader->permutation[index] = dataloader->permutation[swap_index];
        dataloader->permutation[swap_index] = temp;
    }
}

static gsx_error gsx_dataset_validate_desc(const gsx_dataset_desc *desc)
{
    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->get_length == NULL || desc->get_sample == NULL || desc->release_sample == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset callbacks must be non-null");
    }
    if(desc->width <= 0 || desc->height <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset width and height must be positive");
    }
    if(!gsx_dataloader_source_image_data_type_is_supported(desc->image_data_type)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "dataset supports only U8 and F32 host payloads");
    }
    if(!desc->has_rgb) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset must expose RGB payloads");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_validate_desc(gsx_backend_t backend, gsx_dataset_t dataset, const gsx_dataloader_desc *desc)
{
    gsx_backend_capabilities capabilities = { 0 };
    gsx_data_type_flags requested_type_flag = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || dataset == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, dataset, and desc must be non-null");
    }
    if(desc->enable_async_prefetch) {
        if(desc->prefetch_count == 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader prefetch_count must be positive when async prefetch is enabled");
        }
    } else if(desc->prefetch_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader prefetch_count must be zero when async prefetch is disabled");
    }
    if(!gsx_dataloader_output_image_data_type_is_supported(desc->image_data_type)) {
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "dataloader currently supports only F32 output image tensors; U8 and F16 conversion is not implemented yet");
    }
    if(!dataset->desc.has_rgb) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset must expose RGB payloads");
    }

    error = gsx_backend_get_capabilities(backend, &capabilities);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    requested_type_flag = ((gsx_data_type_flags)1u) << desc->image_data_type;
    if((capabilities.supported_data_types & requested_type_flag) == 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "requested dataloader image dtype is not supported by the backend");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_compute_required_capacity_bytes(
    gsx_backend_buffer_type_t buffer_type,
    gsx_dataset_t dataset,
    const gsx_dataloader_desc *desc,
    gsx_size_t slot_count,
    gsx_size_t *out_required_bytes)
{
    gsx_arena_t dry_run_arena = NULL;
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_tensor_t tensor = NULL;
    gsx_size_t required_bytes = 0;
    gsx_size_t slot_index = 0;
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || dataset == NULL || desc == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type, dataset, desc, and out_required_bytes must be non-null");
    }

    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(slot_index = 0; slot_index < slot_count; ++slot_index) {
        error = gsx_dataloader_make_tensor_desc(
            dry_run_arena,
            desc->image_data_type,
            GSX_STORAGE_FORMAT_CHW,
            dataset->desc.width,
            dataset->desc.height,
            3,
            &tensor_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_init(&tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_tensor_free(tensor);
        tensor = NULL;
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        if(dataset->desc.has_alpha) {
            error = gsx_dataloader_make_tensor_desc(
                dry_run_arena,
                desc->image_data_type,
                GSX_STORAGE_FORMAT_CHW,
                dataset->desc.width,
                dataset->desc.height,
                1,
                &tensor_desc);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_init(&tensor, &tensor_desc);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_free(tensor);
            tensor = NULL;
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        if(dataset->desc.has_invdepth) {
            error = gsx_dataloader_make_tensor_desc(
                dry_run_arena,
                desc->image_data_type,
                GSX_STORAGE_FORMAT_CHW,
                dataset->desc.width,
                dataset->desc.height,
                1,
                &tensor_desc);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_init(&tensor, &tensor_desc);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_free(tensor);
            tensor = NULL;
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, &required_bytes);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    *out_required_bytes = required_bytes;

cleanup:
    if(tensor != NULL) {
        (void)gsx_tensor_free(tensor);
    }
    if(dry_run_arena != NULL) {
        (void)gsx_arena_free(dry_run_arena);
    }
    return error;
}

static gsx_error gsx_dataloader_init_slots(gsx_dataloader_t dataloader)
{
    gsx_size_t slot_index = 0;
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dataloader == NULL || dataloader->slots == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader and slots must be non-null");
    }

    for(slot_index = 0; slot_index < dataloader->slot_count; ++slot_index) {
        gsx_dataloader_slot *slot = &dataloader->slots[slot_index];

        error = gsx_dataloader_make_tensor_desc(
            dataloader->arena,
            dataloader->desc.image_data_type,
            GSX_STORAGE_FORMAT_CHW,
            dataloader->dataset->desc.width,
            dataloader->dataset->desc.height,
            3,
            &tensor_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_tensor_init(&slot->rgb_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_dataloader_compute_packed_bytes(
            dataloader->desc.image_data_type,
            dataloader->dataset->desc.width,
            dataloader->dataset->desc.height,
            3,
            &slot->rgb_size_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }

        if(dataloader->dataset->desc.has_alpha) {
            error = gsx_dataloader_make_tensor_desc(
                dataloader->arena,
                dataloader->desc.image_data_type,
                GSX_STORAGE_FORMAT_CHW,
                dataloader->dataset->desc.width,
                dataloader->dataset->desc.height,
                1,
                &tensor_desc);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_tensor_init(&slot->alpha_tensor, &tensor_desc);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_dataloader_compute_packed_bytes(
                dataloader->desc.image_data_type,
                dataloader->dataset->desc.width,
                dataloader->dataset->desc.height,
                1,
                &slot->alpha_size_bytes);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }

        if(dataloader->dataset->desc.has_invdepth) {
            error = gsx_dataloader_make_tensor_desc(
                dataloader->arena,
                dataloader->desc.image_data_type,
                GSX_STORAGE_FORMAT_CHW,
                dataloader->dataset->desc.width,
                dataloader->dataset->desc.height,
                1,
                &tensor_desc);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_tensor_init(&slot->invdepth_tensor, &tensor_desc);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_dataloader_compute_packed_bytes(
                dataloader->desc.image_data_type,
                dataloader->dataset->desc.width,
                dataloader->dataset->desc.height,
                1,
                &slot->invdepth_size_bytes);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_destroy_async(gsx_dataloader_t dataloader)
{
    if(dataloader != NULL && dataloader->async_dl != NULL) {
        gsx_error error = dataloader->async_dl->iface->destroy(dataloader->async_dl);
        dataloader->async_dl = NULL;
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_init_async(gsx_dataloader_t dataloader)
{
    gsx_async_dl_desc async_desc;

    if(dataloader == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader must be non-null");
    }
    if(!dataloader->desc.enable_async_prefetch) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    memset(&async_desc, 0, sizeof(async_desc));
    async_desc.backend = dataloader->backend;
    async_desc.dataset_desc = dataloader->dataset->desc;
    async_desc.output_data_type = dataloader->desc.image_data_type;
    async_desc.slot_count = dataloader->slot_count;
    async_desc.slots = dataloader->slots;

    if(dataloader->backend->iface == NULL || dataloader->backend->iface->create_async_dl == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "async dataloader backend is unsupported");
    }
    return dataloader->backend->iface->create_async_dl(dataloader->backend, &async_desc, &dataloader->async_dl);
}

static void gsx_dataloader_reset_free_slots(gsx_dataloader_t dataloader)
{
    gsx_size_t slot_index = 0;

    dataloader->free_slot_count = dataloader->slot_count;
    dataloader->has_current_result_slot = false;
    dataloader->current_result_slot_index = 0;
    for(slot_index = 0; slot_index < dataloader->slot_count; ++slot_index) {
        dataloader->free_slot_indices[slot_index] = dataloader->slot_count - 1 - slot_index;
    }
}

static gsx_error gsx_dataloader_prime_async(gsx_dataloader_t dataloader)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dataloader == NULL || dataloader->async_dl == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader async helper must be non-null");
    }

    while(dataloader->free_slot_count > 0 && dataloader->next_submit_ordinal < dataloader->length) {
        const gsx_size_t slot_index = dataloader->free_slot_indices[dataloader->free_slot_count - 1];
        const gsx_size_t stable_sample_index = dataloader->permutation[dataloader->next_submit_ordinal];

        dataloader->free_slot_count -= 1;
        error = dataloader->async_dl->iface->submit(dataloader->async_dl, stable_sample_index, slot_index);
        if(!gsx_error_is_success(error)) {
            dataloader->free_slot_indices[dataloader->free_slot_count] = slot_index;
            dataloader->free_slot_count += 1;
            return error;
        }
        dataloader->next_submit_ordinal += 1;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_reset_iteration_state(gsx_dataloader_t dataloader)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    dataloader->rng = dataloader->initial_rng;
    dataloader->epoch_index = 0;
    dataloader->next_output_ordinal = 0;
    dataloader->next_submit_ordinal = 0;
    dataloader->next_boundary_flags = GSX_DATALOADER_BOUNDARY_NEW_EPOCH | GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION;

    if(dataloader->desc.shuffle_each_epoch) {
        gsx_dataloader_shuffle_permutation(dataloader, &dataloader->rng);
    } else {
        gsx_dataloader_fill_identity_permutation(dataloader);
    }

    gsx_dataloader_reset_free_slots(dataloader);

    error = gsx_dataloader_destroy_async(dataloader);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dataloader->desc.enable_async_prefetch) {
        error = gsx_dataloader_init_async(dataloader);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_dataloader_prime_async(dataloader);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_dataloader_advance_output_state(gsx_dataloader_t dataloader)
{
    dataloader->next_output_ordinal += 1;
    dataloader->next_boundary_flags = 0;
    if(dataloader->next_output_ordinal == dataloader->length) {
        dataloader->epoch_index += 1;
        dataloader->next_output_ordinal = 0;
        dataloader->next_submit_ordinal = 0;
        dataloader->next_boundary_flags = GSX_DATALOADER_BOUNDARY_NEW_EPOCH;
        if(dataloader->desc.shuffle_each_epoch) {
            gsx_dataloader_shuffle_permutation(dataloader, &dataloader->rng);
            dataloader->next_boundary_flags |= GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION;
        } else {
            gsx_dataloader_fill_identity_permutation(dataloader);
        }
        GSX_LOG_DEBUG("dataloader: epoch %zu complete\n", (size_t)dataloader->epoch_index);
    }
}

static gsx_error gsx_dataloader_upload_sync_image(
    gsx_dataloader_t dataloader,
    const void *src_bytes,
    gsx_index_t channel_count,
    gsx_tensor_t tensor,
    void **scratch_bytes,
    gsx_size_t *scratch_capacity_bytes)
{
    gsx_size_t required_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src_bytes == NULL || tensor == NULL || scratch_bytes == NULL || scratch_capacity_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sync upload inputs must be non-null");
    }

    error = gsx_dataloader_compute_packed_bytes(
        dataloader->desc.image_data_type,
        dataloader->dataset->desc.width,
        dataloader->dataset->desc.height,
        channel_count,
        &required_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_ensure_scratch(scratch_bytes, scratch_capacity_bytes, required_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(channel_count == 3) {
        error = gsx_dataloader_pack_hwc_to_chw(
            src_bytes,
            dataloader->dataset->desc.image_data_type,
            dataloader->desc.image_data_type,
            dataloader->dataset->desc.width,
            dataloader->dataset->desc.height,
            *scratch_bytes);
    } else {
        error = gsx_dataloader_pack_hw_to_chw(
            src_bytes,
            dataloader->dataset->desc.image_data_type,
            dataloader->desc.image_data_type,
            dataloader->dataset->desc.width,
            dataloader->dataset->desc.height,
            *scratch_bytes);
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_tensor_upload(tensor, *scratch_bytes, required_size_bytes);
}

static gsx_error gsx_dataloader_fill_sync_result(
    gsx_dataloader_t dataloader,
    gsx_size_t stable_sample_index,
    const gsx_dataset_cpu_sample *sample,
    gsx_dataloader_result *out_result)
{
    gsx_dataloader_slot *slot = &dataloader->slots[0];
    gsx_dataloader_boundary_flags boundary_flags = dataloader->next_boundary_flags;
    gsx_size_t epoch_index = dataloader->epoch_index;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_dataloader_upload_sync_image(
        dataloader,
        sample->rgb_data,
        3,
        slot->rgb_tensor,
        &dataloader->rgb_scratch,
        &dataloader->rgb_scratch_capacity_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(dataloader->dataset->desc.has_alpha) {
        error = gsx_dataloader_upload_sync_image(
            dataloader,
            sample->alpha_data,
            1,
            slot->alpha_tensor,
            &dataloader->alpha_scratch,
            &dataloader->alpha_scratch_capacity_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(dataloader->dataset->desc.has_invdepth) {
        error = gsx_dataloader_upload_sync_image(
            dataloader,
            sample->invdepth_data,
            1,
            slot->invdepth_tensor,
            &dataloader->invdepth_scratch,
            &dataloader->invdepth_scratch_capacity_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    memset(out_result, 0, sizeof(*out_result));
    out_result->intrinsics = sample->intrinsics;
    out_result->pose = sample->pose;
    out_result->rgb_image = slot->rgb_tensor;
    out_result->alpha_image = dataloader->dataset->desc.has_alpha ? slot->alpha_tensor : NULL;
    out_result->invdepth_image = dataloader->dataset->desc.has_invdepth ? slot->invdepth_tensor : NULL;
    out_result->stable_sample_index = stable_sample_index;
    out_result->stable_sample_id = sample->stable_sample_id;
    out_result->has_stable_sample_id = sample->has_stable_sample_id;
    out_result->epoch_index = epoch_index;
    out_result->boundary_flags = boundary_flags;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_dataloader_release_current_slot(gsx_dataloader_t dataloader)
{
    if(dataloader->desc.enable_async_prefetch && dataloader->has_current_result_slot) {
        dataloader->free_slot_indices[dataloader->free_slot_count] = dataloader->current_result_slot_index;
        dataloader->free_slot_count += 1;
    }
    dataloader->has_current_result_slot = false;
    dataloader->current_result_slot_index = 0;
}

static gsx_error gsx_dataloader_free_slots(gsx_dataloader_t dataloader)
{
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t slot_index = 0;

    if(dataloader == NULL || dataloader->slots == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(slot_index = 0; slot_index < dataloader->slot_count; ++slot_index) {
        gsx_dataloader_slot *slot = &dataloader->slots[slot_index];

        if(slot->rgb_tensor != NULL) {
            error = gsx_tensor_free(slot->rgb_tensor);
            if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
                first_error = error;
            }
            slot->rgb_tensor = NULL;
        }
        if(slot->alpha_tensor != NULL) {
            error = gsx_tensor_free(slot->alpha_tensor);
            if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
                first_error = error;
            }
            slot->alpha_tensor = NULL;
        }
        if(slot->invdepth_tensor != NULL) {
            error = gsx_tensor_free(slot->invdepth_tensor);
            if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
                first_error = error;
            }
            slot->invdepth_tensor = NULL;
        }
    }
    return first_error;
}

GSX_API gsx_error gsx_dataset_init(gsx_dataset_t *out_dataset, const gsx_dataset_desc *desc)
{
    struct gsx_dataset *dataset = NULL;
    gsx_size_t length = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_dataset == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_dataset must be non-null");
    }
    *out_dataset = NULL;

    error = gsx_dataset_validate_desc(desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = desc->get_length(desc->object, &length);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(length == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset length must be positive");
    }

    dataset = (struct gsx_dataset *)calloc(1, sizeof(*dataset));
    if(dataset == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate dataset");
    }

    dataset->desc = *desc;
    dataset->length = length;
    *out_dataset = dataset;
    GSX_LOG_INFO("dataset: created length=%zu geometry=%dx%d\n", (size_t)length, desc->width, desc->height);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_dataset_free(gsx_dataset_t dataset)
{
    gsx_error error = gsx_dataset_require_handle(dataset);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dataset->live_dataloader_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "dataset must outlive all dataloaders that borrow it");
    }

    free(dataset);
    GSX_LOG_DEBUG("dataset: freed\n");
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_dataset_get_info(gsx_dataset_t dataset, gsx_dataset_info *out_info)
{
    gsx_error error = gsx_dataset_require_handle(dataset);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }

    memset(out_info, 0, sizeof(*out_info));
    out_info->length = dataset->length;
    out_info->image_data_type = dataset->desc.image_data_type;
    out_info->width = dataset->desc.width;
    out_info->height = dataset->desc.height;
    out_info->has_rgb = dataset->desc.has_rgb;
    out_info->has_alpha = dataset->desc.has_alpha;
    out_info->has_invdepth = dataset->desc.has_invdepth;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_dataloader_init(
    gsx_dataloader_t *out_dataloader,
    gsx_backend_t backend,
    gsx_dataset_t dataset,
    const gsx_dataloader_desc *desc)
{
    struct gsx_dataloader *dataloader = NULL;
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_size_t required_bytes = 0;
    gsx_size_t permutation_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_dataloader == NULL || backend == NULL || dataset == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_dataloader, backend, dataset, and desc must be non-null");
    }
    *out_dataloader = NULL;

    error = gsx_dataloader_validate_desc(backend, dataset, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dataset->live_dataloader_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "the same dataset cannot be attached to multiple live dataloaders");
    }

    error = gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    dataloader = (struct gsx_dataloader *)calloc(1, sizeof(*dataloader));
    if(dataloader == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate dataloader");
    }

    dataloader->backend = backend;
    dataloader->dataset = dataset;
    dataloader->buffer_type = buffer_type;
    dataloader->desc = *desc;
    dataloader->length = dataset->length;
    dataloader->slot_count = desc->enable_async_prefetch ? desc->prefetch_count : 1;
    gsx_dataloader_rng_seed(desc->seed, &dataloader->initial_rng);

    error = gsx_dataloader_compute_required_capacity_bytes(
        dataloader->buffer_type, dataset, desc, dataloader->slot_count, &required_bytes);
    if(!gsx_error_is_success(error)) {
        goto fail;
    }

    dataloader->slots = (gsx_dataloader_slot *)calloc((size_t)dataloader->slot_count, sizeof(*dataloader->slots));
    dataloader->free_slot_indices = (gsx_size_t *)malloc((size_t)dataloader->slot_count * sizeof(*dataloader->free_slot_indices));
    if(dataloader->slots == NULL || dataloader->free_slot_indices == NULL) {
        error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate dataloader slot storage");
        goto fail;
    }

    if(gsx_size_mul_overflows(dataloader->length, (gsx_size_t)sizeof(*dataloader->permutation), &permutation_bytes)
        || permutation_bytes > (gsx_size_t)SIZE_MAX) {
        error = gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "dataloader permutation storage size overflows platform limits");
        goto fail;
    }
    dataloader->permutation = (gsx_size_t *)malloc((size_t)permutation_bytes);
    if(dataloader->permutation == NULL) {
        error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate dataloader permutation storage");
        goto fail;
    }

    arena_desc.initial_capacity_bytes = required_bytes;
    error = gsx_arena_init(&dataloader->arena, dataloader->buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        goto fail;
    }

    error = gsx_dataloader_init_slots(dataloader);
    if(!gsx_error_is_success(error)) {
        goto fail;
    }

    error = gsx_dataloader_reset_iteration_state(dataloader);
    if(!gsx_error_is_success(error)) {
        goto fail;
    }

    dataset->live_dataloader_count += 1;
    *out_dataloader = dataloader;
    GSX_LOG_INFO(
        "dataloader: created length=%zu output=%dx%d async=%d prefetch=%zu\n",
        (size_t)dataloader->length,
        dataset->desc.width,
        dataset->desc.height,
        desc->enable_async_prefetch ? 1 : 0,
        (size_t)desc->prefetch_count);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);

fail:
    (void)gsx_dataloader_destroy_async(dataloader);
    if(dataloader != NULL) {
        (void)gsx_dataloader_free_slots(dataloader);
        if(dataloader->arena != NULL) {
            (void)gsx_arena_free(dataloader->arena);
        }
        free(dataloader->rgb_scratch);
        free(dataloader->alpha_scratch);
        free(dataloader->invdepth_scratch);
        free(dataloader->free_slot_indices);
        free(dataloader->slots);
        free(dataloader->permutation);
        free(dataloader);
    }
    return error;
}

GSX_API gsx_error gsx_dataloader_free(gsx_dataloader_t dataloader)
{
    gsx_error error = gsx_dataloader_require_handle(dataloader);
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };

    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_dataloader_destroy_async(dataloader);
    if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
        first_error = error;
    }
    error = gsx_dataloader_free_slots(dataloader);
    if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
        first_error = error;
    }
    if(dataloader->arena != NULL) {
        error = gsx_arena_free(dataloader->arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }
    if(dataloader->dataset->live_dataloader_count != 0) {
        dataloader->dataset->live_dataloader_count -= 1;
    }

    free(dataloader->rgb_scratch);
    free(dataloader->alpha_scratch);
    free(dataloader->invdepth_scratch);
    free(dataloader->free_slot_indices);
    free(dataloader->slots);
    free(dataloader->permutation);
    free(dataloader);
    GSX_LOG_DEBUG("dataloader: freed\n");
    return first_error;
}

GSX_API gsx_error gsx_dataloader_reset(gsx_dataloader_t dataloader)
{
    gsx_error error = gsx_dataloader_require_handle(dataloader);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    gsx_dataloader_release_current_slot(dataloader);
    error = gsx_dataloader_reset_iteration_state(dataloader);
    if(gsx_error_is_success(error)) {
        GSX_LOG_DEBUG("dataloader: reset iteration state\n");
    }
    return error;
}

GSX_API gsx_error gsx_dataloader_get_info(gsx_dataloader_t dataloader, gsx_dataloader_info *out_info)
{
    gsx_error error = gsx_dataloader_require_handle(dataloader);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }

    memset(out_info, 0, sizeof(*out_info));
    out_info->length = dataloader->length;
    out_info->shuffle_each_epoch = dataloader->desc.shuffle_each_epoch;
    out_info->enable_async_prefetch = dataloader->desc.enable_async_prefetch;
    out_info->prefetch_count = dataloader->desc.prefetch_count;
    out_info->image_data_type = dataloader->desc.image_data_type;
    out_info->storage_format = GSX_STORAGE_FORMAT_CHW;
    out_info->output_width = dataloader->dataset->desc.width;
    out_info->output_height = dataloader->dataset->desc.height;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_dataloader_next_ex(gsx_dataloader_t dataloader, gsx_dataloader_result *out_result)
{
    gsx_error error = gsx_dataloader_require_handle(dataloader);

    if(out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(dataloader->desc.enable_async_prefetch) {
        gsx_async_dl_ready_item ready_item = { 0 };
        gsx_dataloader_boundary_flags boundary_flags = dataloader->next_boundary_flags;
        const gsx_size_t expected_sample_index = dataloader->permutation[dataloader->next_output_ordinal];
        const gsx_size_t epoch_index = dataloader->epoch_index;

        gsx_dataloader_release_current_slot(dataloader);
        error = gsx_dataloader_prime_async(dataloader);
        if(!gsx_error_is_success(error)) {
            return error;
        }

        error = dataloader->async_dl->iface->wait(dataloader->async_dl, &ready_item);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(ready_item.slot_index >= dataloader->slot_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader returned an out-of-range slot index");
        }
        if(ready_item.stable_sample_index != expected_sample_index) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "async dataloader returned a sample out of iterator order");
        }

        memset(out_result, 0, sizeof(*out_result));
        out_result->intrinsics = ready_item.intrinsics;
        out_result->pose = ready_item.pose;
        out_result->rgb_image = dataloader->slots[ready_item.slot_index].rgb_tensor;
        out_result->alpha_image = dataloader->dataset->desc.has_alpha ? dataloader->slots[ready_item.slot_index].alpha_tensor : NULL;
        out_result->invdepth_image = dataloader->dataset->desc.has_invdepth ? dataloader->slots[ready_item.slot_index].invdepth_tensor : NULL;
        out_result->stable_sample_index = ready_item.stable_sample_index;
        out_result->stable_sample_id = ready_item.stable_sample_id;
        out_result->has_stable_sample_id = ready_item.has_stable_sample_id;
        out_result->epoch_index = epoch_index;
        out_result->boundary_flags = boundary_flags;

        dataloader->has_current_result_slot = true;
        dataloader->current_result_slot_index = ready_item.slot_index;
        gsx_dataloader_advance_output_state(dataloader);
        return gsx_dataloader_prime_async(dataloader);
    }

    {
        gsx_dataset_cpu_sample sample = { 0 };
        const gsx_size_t stable_sample_index = dataloader->permutation[dataloader->next_output_ordinal];

        error = dataloader->dataset->desc.get_sample(dataloader->dataset->desc.object, stable_sample_index, &sample);
        if(!gsx_error_is_success(error)) {
            return error;
        }

        error = gsx_dataloader_validate_sample(&dataloader->dataset->desc, &sample);
        if(!gsx_error_is_success(error)) {
            dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
            return error;
        }

        error = gsx_dataloader_fill_sync_result(dataloader, stable_sample_index, &sample, out_result);
        dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
        if(!gsx_error_is_success(error)) {
            return error;
        }

        gsx_dataloader_advance_output_state(dataloader);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
}
