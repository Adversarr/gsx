#include "gsx-impl.h"
#include "pcg32.h"

#include <math.h>
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

    gsx_tensor_t rgb_tensor;
    gsx_tensor_t alpha_tensor;
    gsx_tensor_t invdepth_tensor;

    void *rgb_scratch;
    gsx_size_t rgb_scratch_capacity_bytes;
    void *alpha_scratch;
    gsx_size_t alpha_scratch_capacity_bytes;
    void *invdepth_scratch;
    gsx_size_t invdepth_scratch_capacity_bytes;

    gsx_size_t *permutation;
    gsx_size_t epoch_index;
    gsx_size_t next_sample_ordinal;
    gsx_dataloader_boundary_flags next_boundary_flags;
    pcg32 rng;
    pcg32 initial_rng;
};

static bool gsx_dataloader_storage_format_is_supported(gsx_storage_format storage_format)
{
    return storage_format == GSX_STORAGE_FORMAT_CHW || storage_format == GSX_STORAGE_FORMAT_HWC;
}

static bool gsx_dataloader_image_data_type_is_supported(gsx_data_type data_type)
{
    return data_type == GSX_DATA_TYPE_U8 || data_type == GSX_DATA_TYPE_F32;
}

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
    if(dataloader->backend == NULL || dataloader->dataset == NULL || dataloader->arena == NULL || dataloader->permutation == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "dataloader is detached from required owned state");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_compute_element_count(
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    gsx_size_t *out_element_count)
{
    gsx_size_t element_count = 0;

    if(out_element_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_element_count must be non-null");
    }
    if(width <= 0 || height <= 0 || channel_count <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image dimensions and channel count must be positive");
    }
    if(gsx_size_mul_overflows((gsx_size_t)width, (gsx_size_t)height, &element_count)
        || gsx_size_mul_overflows(element_count, (gsx_size_t)channel_count, &element_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "image element count overflows");
    }

    *out_element_count = element_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_compute_packed_bytes(
    gsx_data_type data_type,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    gsx_size_t *out_size_bytes)
{
    gsx_size_t element_count = 0;
    gsx_size_t element_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_size_bytes must be non-null");
    }

    error = gsx_data_type_get_size_bytes(data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_compute_element_count(width, height, channel_count, &element_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(gsx_size_mul_overflows(element_count, element_size_bytes, out_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "image byte size overflows");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_dataloader_compute_tensor_shape(
    gsx_storage_format storage_format,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    gsx_index_t *out_shape)
{
    memset(out_shape, 0, (size_t)GSX_TENSOR_MAX_DIM * sizeof(out_shape[0]));
    if(storage_format == GSX_STORAGE_FORMAT_CHW) {
        out_shape[0] = channel_count;
        out_shape[1] = height;
        out_shape[2] = width;
    } else {
        out_shape[0] = height;
        out_shape[1] = width;
        out_shape[2] = channel_count;
    }
}

static gsx_error gsx_dataloader_make_tensor_desc(
    gsx_arena_t arena,
    gsx_data_type data_type,
    gsx_storage_format storage_format,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    gsx_tensor_desc *out_desc)
{
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }

    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->rank = 3;
    out_desc->data_type = data_type;
    out_desc->storage_format = storage_format;
    out_desc->arena = arena;
    gsx_dataloader_compute_tensor_shape(storage_format, width, height, channel_count, out_desc->shape);
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

static void gsx_dataloader_rng_seed(gsx_size_t seed, pcg32 *out_rng)
{
    const uint64_t initstate = (uint64_t)seed;
    const uint64_t initseq = UINT64_C(0x9e3779b97f4a7c15) ^ (initstate << 1);

    pcg32_init(out_rng, initstate, initseq);
}

static uint64_t gsx_dataloader_rng_next_u64(pcg32 *rng)
{
    const uint64_t upper = (uint64_t)pcg32_next_uint(rng);
    const uint64_t lower = (uint64_t)pcg32_next_uint(rng);

    return (upper << 32u) | lower;
}

static gsx_size_t gsx_dataloader_rng_bounded(pcg32 *rng, gsx_size_t bound)
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

static void gsx_dataloader_shuffle_permutation(gsx_dataloader_t dataloader, pcg32 *rng)
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

static gsx_error gsx_dataloader_reset_iteration_state(gsx_dataloader_t dataloader)
{
    dataloader->rng = dataloader->initial_rng;
    dataloader->epoch_index = 0;
    dataloader->next_sample_ordinal = 0;
    dataloader->next_boundary_flags = GSX_DATALOADER_BOUNDARY_NEW_EPOCH | GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION;

    if(dataloader->desc.shuffle_each_epoch) {
        gsx_dataloader_shuffle_permutation(dataloader, &dataloader->rng);
    } else {
        gsx_dataloader_fill_identity_permutation(dataloader);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_validate_sync_config(const gsx_dataloader_desc *desc)
{
    if(desc->enable_async_prefetch) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "dataloader async prefetch is not implemented in v1");
    }
    if(desc->prefetch_count != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "dataloader prefetch_count must be zero in v1");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_validate_output_desc(gsx_backend_t backend, const gsx_dataloader_desc *desc)
{
    gsx_backend_capabilities capabilities = { 0 };
    gsx_data_type_flags requested_type_flag = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and desc must be non-null");
    }
    if(desc->output_width <= 0 || desc->output_height <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader output width and height must be positive");
    }
    if(!gsx_dataloader_storage_format_is_supported(desc->storage_format)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "dataloader supports only CHW and HWC output storage formats");
    }
    if(!gsx_dataloader_image_data_type_is_supported(desc->image_data_type)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "dataloader supports only U8 and F32 image tensor dtypes in v1");
    }
    if(desc->resize_policy != GSX_IMAGE_RESIZE_PIXEL_CENTER) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "dataloader resize policy is out of range");
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

static gsx_error gsx_dataloader_validate_cpu_image_view(
    const gsx_cpu_image_view *view,
    bool allow_absent,
    gsx_index_t expected_width,
    gsx_index_t expected_height,
    gsx_index_t expected_channel_count,
    const char *label)
{
    gsx_size_t element_size_bytes = 0;
    gsx_size_t min_row_stride_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image view must be non-null");
    }
    if(view->data == NULL) {
        if(allow_absent) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }
    if(view->width <= 0 || view->height <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }
    if(view->channel_count != expected_channel_count) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }
    if(expected_width > 0 && view->width != expected_width) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }
    if(expected_height > 0 && view->height != expected_height) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }
    if(!gsx_dataloader_image_data_type_is_supported(view->data_type)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, label);
    }

    error = gsx_data_type_get_size_bytes(view->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(gsx_size_mul_overflows((gsx_size_t)view->width, (gsx_size_t)view->channel_count, &min_row_stride_bytes)
        || gsx_size_mul_overflows(min_row_stride_bytes, element_size_bytes, &min_row_stride_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "image row stride overflows");
    }
    if(view->row_stride_bytes < min_row_stride_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_dataloader_validate_sample(const gsx_dataset_cpu_sample *sample)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(sample == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample must be non-null");
    }
    if(sample->pose.camera_id != sample->intrinsics.camera_id) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample pose.camera_id must match intrinsics.camera_id");
    }

    error = gsx_dataloader_validate_cpu_image_view(
        &sample->rgb, false, 0, 0, 3, "sample rgb view must be present, use a supported dtype, and have 3 channels");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(sample->intrinsics.width != sample->rgb.width || sample->intrinsics.height != sample->rgb.height) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample intrinsics width and height must match the rgb image geometry");
    }

    error = gsx_dataloader_validate_cpu_image_view(
        &sample->alpha,
        true,
        sample->rgb.width,
        sample->rgb.height,
        1,
        "sample alpha view must match rgb geometry, use a supported dtype, and have 1 channel");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_validate_cpu_image_view(
        &sample->invdepth,
        true,
        sample->rgb.width,
        sample->rgb.height,
        1,
        "sample inverse-depth view must match rgb geometry, use a supported dtype, and have 1 channel");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_dataloader_cpu_image_load_f32(const gsx_cpu_image_view *view, gsx_index_t x, gsx_index_t y, gsx_index_t channel_index)
{
    const uint8_t *row_bytes = (const uint8_t *)view->data + (size_t)((gsx_size_t)y * view->row_stride_bytes);

    if(view->data_type == GSX_DATA_TYPE_U8) {
        const uint8_t *pixel = row_bytes + (size_t)(((gsx_size_t)x * (gsx_size_t)view->channel_count) + (gsx_size_t)channel_index);

        return (float)(*pixel);
    }

    {
        /* Caller guarantees the pointer/stride is valid and float-decodable; invalid alignment or aliasing is undefined behavior. */
        const float *pixel = (const float *)(row_bytes + (size_t)((((gsx_size_t)x * (gsx_size_t)view->channel_count) + (gsx_size_t)channel_index) * sizeof(float)));

        return *pixel;
    }
}

static uint8_t gsx_dataloader_quantize_u8(float value)
{
    if(!isfinite((double)value)) {
        return value > 0.0f ? (uint8_t)255 : (uint8_t)0;
    }
    if(value < 0.0f) {
        value = 0.0f;
    } else if(value > 255.0f) {
        value = 255.0f;
    }
    return (uint8_t)(value + 0.5f);
}

static gsx_size_t gsx_dataloader_packed_offset(
    gsx_storage_format storage_format,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    gsx_index_t x,
    gsx_index_t y,
    gsx_index_t channel_index)
{
    if(storage_format == GSX_STORAGE_FORMAT_CHW) {
        return (((gsx_size_t)channel_index * (gsx_size_t)height) + (gsx_size_t)y) * (gsx_size_t)width + (gsx_size_t)x;
    }
    return (((gsx_size_t)y * (gsx_size_t)width) + (gsx_size_t)x) * (gsx_size_t)channel_count + (gsx_size_t)channel_index;
}

static void gsx_dataloader_store_output_value(
    void *dst_bytes,
    gsx_data_type data_type,
    gsx_storage_format storage_format,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    gsx_index_t x,
    gsx_index_t y,
    gsx_index_t channel_index,
    float value)
{
    const gsx_size_t offset = gsx_dataloader_packed_offset(storage_format, width, height, channel_count, x, y, channel_index);

    if(data_type == GSX_DATA_TYPE_U8) {
        ((uint8_t *)dst_bytes)[offset] = gsx_dataloader_quantize_u8(value);
    } else {
        ((float *)dst_bytes)[offset] = value;
    }
}

static float gsx_dataloader_sample_bilinear(const gsx_cpu_image_view *view, gsx_float_t src_x, gsx_float_t src_y, gsx_index_t channel_index)
{
    gsx_float_t clamped_x = src_x;
    gsx_float_t clamped_y = src_y;
    gsx_index_t x0 = 0;
    gsx_index_t y0 = 0;
    gsx_index_t x1 = 0;
    gsx_index_t y1 = 0;
    gsx_float_t tx = 0.0f;
    gsx_float_t ty = 0.0f;
    gsx_float_t top = 0.0f;
    gsx_float_t bottom = 0.0f;

    if(clamped_x < 0.0f) {
        clamped_x = 0.0f;
    } else if(clamped_x > (gsx_float_t)(view->width - 1)) {
        clamped_x = (gsx_float_t)(view->width - 1);
    }
    if(clamped_y < 0.0f) {
        clamped_y = 0.0f;
    } else if(clamped_y > (gsx_float_t)(view->height - 1)) {
        clamped_y = (gsx_float_t)(view->height - 1);
    }

    x0 = (gsx_index_t)floor((double)clamped_x);
    y0 = (gsx_index_t)floor((double)clamped_y);
    x1 = x0 < (view->width - 1) ? x0 + 1 : x0;
    y1 = y0 < (view->height - 1) ? y0 + 1 : y0;
    tx = clamped_x - (gsx_float_t)x0;
    ty = clamped_y - (gsx_float_t)y0;

    top = gsx_dataloader_cpu_image_load_f32(view, x0, y0, channel_index)
        + (gsx_dataloader_cpu_image_load_f32(view, x1, y0, channel_index) - gsx_dataloader_cpu_image_load_f32(view, x0, y0, channel_index)) * tx;
    bottom = gsx_dataloader_cpu_image_load_f32(view, x0, y1, channel_index)
        + (gsx_dataloader_cpu_image_load_f32(view, x1, y1, channel_index) - gsx_dataloader_cpu_image_load_f32(view, x0, y1, channel_index)) * tx;
    return top + (bottom - top) * ty;
}

static gsx_error gsx_dataloader_resize_and_pack(
    const gsx_cpu_image_view *view,
    gsx_data_type output_data_type,
    gsx_storage_format output_storage_format,
    gsx_index_t output_width,
    gsx_index_t output_height,
    void *dst_bytes)
{
    gsx_index_t y = 0;
    gsx_index_t x = 0;
    gsx_index_t channel_index = 0;
    gsx_float_t scale_x = 0.0f;
    gsx_float_t scale_y = 0.0f;

    if(view == NULL || dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "view and dst_bytes must be non-null");
    }

    scale_x = (gsx_float_t)view->width / (gsx_float_t)output_width;
    scale_y = (gsx_float_t)view->height / (gsx_float_t)output_height;

    for(y = 0; y < output_height; ++y) {
        gsx_float_t src_y = (((gsx_float_t)y + 0.5f) * scale_y) - 0.5f;

        for(x = 0; x < output_width; ++x) {
            gsx_float_t src_x = (((gsx_float_t)x + 0.5f) * scale_x) - 0.5f;

            for(channel_index = 0; channel_index < view->channel_count; ++channel_index) {
                const float value = gsx_dataloader_sample_bilinear(view, src_x, src_y, channel_index);

                gsx_dataloader_store_output_value(
                    dst_bytes,
                    output_data_type,
                    output_storage_format,
                    output_width,
                    output_height,
                    view->channel_count,
                    x,
                    y,
                    channel_index,
                    value);
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_camera_intrinsics gsx_dataloader_resize_intrinsics(
    const gsx_camera_intrinsics *intrinsics,
    gsx_index_t output_width,
    gsx_index_t output_height)
{
    gsx_camera_intrinsics resized = *intrinsics;
    const gsx_float_t scale_x = (gsx_float_t)output_width / (gsx_float_t)intrinsics->width;
    const gsx_float_t scale_y = (gsx_float_t)output_height / (gsx_float_t)intrinsics->height;

    resized.fx *= scale_x;
    resized.fy *= scale_y;
    resized.cx = ((intrinsics->cx + 0.5f) * scale_x) - 0.5f;
    resized.cy = ((intrinsics->cy + 0.5f) * scale_y) - 0.5f;
    resized.width = output_width;
    resized.height = output_height;
    return resized;
}

static gsx_error gsx_dataloader_release_result_tensors(gsx_dataloader_t dataloader)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };

    if(dataloader->rgb_tensor != NULL) {
        error = gsx_tensor_free(dataloader->rgb_tensor);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        dataloader->rgb_tensor = NULL;
    }
    if(dataloader->alpha_tensor != NULL) {
        error = gsx_tensor_free(dataloader->alpha_tensor);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        dataloader->alpha_tensor = NULL;
    }
    if(dataloader->invdepth_tensor != NULL) {
        error = gsx_tensor_free(dataloader->invdepth_tensor);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        dataloader->invdepth_tensor = NULL;
    }
    if(gsx_error_is_success(first_error)) {
        first_error = gsx_arena_reset(dataloader->arena);
    }

    return first_error;
}

static gsx_error gsx_dataloader_compute_required_capacity_bytes(
    gsx_backend_buffer_type_t buffer_type,
    const gsx_dataloader_desc *desc,
    gsx_size_t *out_required_bytes)
{
    gsx_arena_t dry_run_arena = NULL;
    gsx_tensor_t rgb_tensor = NULL;
    gsx_tensor_t alpha_tensor = NULL;
    gsx_tensor_t invdepth_tensor = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || desc == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type, desc, and out_required_bytes must be non-null");
    }

    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_dataloader_make_tensor_desc(
        dry_run_arena, desc->image_data_type, desc->storage_format, desc->output_width, desc->output_height, 3, &tensor_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_init(&rgb_tensor, &tensor_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_dataloader_make_tensor_desc(
        dry_run_arena, desc->image_data_type, desc->storage_format, desc->output_width, desc->output_height, 1, &tensor_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_init(&alpha_tensor, &tensor_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_init(&invdepth_tensor, &tensor_desc);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, &required_bytes);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    *out_required_bytes = required_bytes;

cleanup:
    if(rgb_tensor != NULL) {
        (void)gsx_tensor_free(rgb_tensor);
    }
    if(alpha_tensor != NULL) {
        (void)gsx_tensor_free(alpha_tensor);
    }
    if(invdepth_tensor != NULL) {
        (void)gsx_tensor_free(invdepth_tensor);
    }
    if(dry_run_arena != NULL) {
        (void)gsx_arena_free(dry_run_arena);
    }
    return error;
}

static gsx_error gsx_dataloader_ensure_result_tensor(
    gsx_dataloader_t dataloader,
    gsx_index_t channel_count,
    gsx_tensor_t *tensor_slot)
{
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor_slot == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_slot must be non-null");
    }
    if(*tensor_slot != NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_dataloader_make_tensor_desc(
        dataloader->arena,
        dataloader->desc.image_data_type,
        dataloader->desc.storage_format,
        dataloader->desc.output_width,
        dataloader->desc.output_height,
        channel_count,
        &tensor_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_tensor_init(tensor_slot, &tensor_desc);
}

static gsx_error gsx_dataloader_upload_image(
    gsx_dataloader_t dataloader,
    const gsx_cpu_image_view *view,
    gsx_tensor_t tensor,
    void **scratch_bytes,
    gsx_size_t *scratch_capacity_bytes)
{
    gsx_size_t required_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(view == NULL || tensor == NULL || scratch_bytes == NULL || scratch_capacity_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "upload_image inputs must be non-null");
    }

    error = gsx_dataloader_compute_packed_bytes(
        dataloader->desc.image_data_type,
        dataloader->desc.output_width,
        dataloader->desc.output_height,
        view->channel_count,
        &required_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_ensure_scratch(scratch_bytes, scratch_capacity_bytes, required_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_resize_and_pack(
        view, dataloader->desc.image_data_type, dataloader->desc.storage_format, dataloader->desc.output_width, dataloader->desc.output_height, *scratch_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_tensor_upload(tensor, *scratch_bytes, required_size_bytes);
}

GSX_API gsx_error gsx_dataset_init(gsx_dataset_t *out_dataset, const gsx_dataset_desc *desc)
{
    struct gsx_dataset *dataset = NULL;
    gsx_size_t length = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_dataset == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_dataset and desc must be non-null");
    }
    if(desc->get_length == NULL || desc->get_sample == NULL || desc->release_sample == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset callbacks must be non-null");
    }
    *out_dataset = NULL;

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
    GSX_LOG_INFO("dataset: created length=%zu\n", (size_t)length);
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

    out_info->length = dataset->length;
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

    error = gsx_dataloader_validate_sync_config(desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_validate_output_desc(backend, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dataset->live_dataloader_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "the same dataset cannot be attached to multiple live dataloaders in v1");
    }

    error = gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_dataloader_compute_required_capacity_bytes(buffer_type, desc, &required_bytes);
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
    gsx_dataloader_rng_seed(desc->seed, &dataloader->initial_rng);

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

    error = gsx_dataloader_reset_iteration_state(dataloader);
    if(!gsx_error_is_success(error)) {
        goto fail;
    }

    dataset->live_dataloader_count += 1;
    *out_dataloader = dataloader;
    GSX_LOG_INFO("dataloader: created length=%zu output=%dx%d\n", (size_t)dataloader->length, desc->output_width, desc->output_height);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);

fail:
    free(dataloader->rgb_scratch);
    free(dataloader->alpha_scratch);
    free(dataloader->invdepth_scratch);
    free(dataloader->permutation);
    if(dataloader->arena != NULL) {
        (void)gsx_arena_free(dataloader->arena);
    }
    free(dataloader);
    return error;
}

GSX_API gsx_error gsx_dataloader_free(gsx_dataloader_t dataloader)
{
    gsx_error error = gsx_dataloader_require_handle(dataloader);
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };

    if(!gsx_error_is_success(error)) {
        return error;
    }

    first_error = gsx_dataloader_release_result_tensors(dataloader);
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
    out_info->enable_async_prefetch = false;
    out_info->prefetch_count = 0;
    out_info->image_data_type = dataloader->desc.image_data_type;
    out_info->storage_format = dataloader->desc.storage_format;
    out_info->output_width = dataloader->desc.output_width;
    out_info->output_height = dataloader->desc.output_height;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_dataloader_set_output_shape(gsx_dataloader_t dataloader, gsx_index_t width, gsx_index_t height)
{
    gsx_dataloader_desc resized_desc;
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_dataloader_require_handle(dataloader);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(width <= 0 || height <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader output width and height must be positive");
    }
    if(width == dataloader->desc.output_width && height == dataloader->desc.output_height) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    resized_desc = dataloader->desc;
    resized_desc.output_width = width;
    resized_desc.output_height = height;
    error = gsx_dataloader_compute_required_capacity_bytes(dataloader->buffer_type, &resized_desc, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_dataloader_release_result_tensors(dataloader);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_arena_reserve(dataloader->arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    dataloader->desc = resized_desc;
    GSX_LOG_DEBUG("dataloader: output shape changed to %dx%d\n", width, height);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_dataloader_next_ex(gsx_dataloader_t dataloader, gsx_dataloader_result *out_result)
{
    gsx_dataset_cpu_sample sample = { 0 };
    gsx_size_t stable_sample_index = 0;
    gsx_dataloader_boundary_flags boundary_flags = 0;
    gsx_error error = gsx_dataloader_require_handle(dataloader);

    if(out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dataloader->next_sample_ordinal >= dataloader->length) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "dataloader next_sample_ordinal is out of range");
    }

    stable_sample_index = dataloader->permutation[dataloader->next_sample_ordinal];
    boundary_flags = dataloader->next_boundary_flags;

    error = dataloader->dataset->desc.get_sample(dataloader->dataset->desc.object, stable_sample_index, &sample);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_dataloader_validate_sample(&sample);
    if(!gsx_error_is_success(error)) {
        dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
        return error;
    }

    error = gsx_dataloader_ensure_result_tensor(dataloader, 3, &dataloader->rgb_tensor);
    if(!gsx_error_is_success(error)) {
        dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
        return error;
    }
    if(sample.alpha.data != NULL) {
        error = gsx_dataloader_ensure_result_tensor(dataloader, 1, &dataloader->alpha_tensor);
        if(!gsx_error_is_success(error)) {
            dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
            return error;
        }
    }
    if(sample.invdepth.data != NULL) {
        error = gsx_dataloader_ensure_result_tensor(dataloader, 1, &dataloader->invdepth_tensor);
        if(!gsx_error_is_success(error)) {
            dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
            return error;
        }
    }

    error = gsx_dataloader_upload_image(
        dataloader, &sample.rgb, dataloader->rgb_tensor, &dataloader->rgb_scratch, &dataloader->rgb_scratch_capacity_bytes);
    if(!gsx_error_is_success(error)) {
        dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
        return error;
    }
    if(sample.alpha.data != NULL) {
        error = gsx_dataloader_upload_image(
            dataloader, &sample.alpha, dataloader->alpha_tensor, &dataloader->alpha_scratch, &dataloader->alpha_scratch_capacity_bytes);
        if(!gsx_error_is_success(error)) {
            dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
            return error;
        }
    }
    if(sample.invdepth.data != NULL) {
        error = gsx_dataloader_upload_image(
            dataloader, &sample.invdepth, dataloader->invdepth_tensor, &dataloader->invdepth_scratch, &dataloader->invdepth_scratch_capacity_bytes);
        if(!gsx_error_is_success(error)) {
            dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);
            return error;
        }
    }

    memset(out_result, 0, sizeof(*out_result));
    out_result->intrinsics = gsx_dataloader_resize_intrinsics(&sample.intrinsics, dataloader->desc.output_width, dataloader->desc.output_height);
    out_result->pose = sample.pose;
    out_result->rgb_image = dataloader->rgb_tensor;
    out_result->alpha_image = sample.alpha.data != NULL ? dataloader->alpha_tensor : NULL;
    out_result->invdepth_image = sample.invdepth.data != NULL ? dataloader->invdepth_tensor : NULL;
    out_result->stable_sample_index = stable_sample_index;
    out_result->stable_sample_id = sample.stable_sample_id;
    out_result->has_stable_sample_id = sample.has_stable_sample_id;
    out_result->epoch_index = dataloader->epoch_index;
    out_result->boundary_flags = boundary_flags;

    dataloader->dataset->desc.release_sample(dataloader->dataset->desc.object, &sample);

    dataloader->next_sample_ordinal += 1;
    dataloader->next_boundary_flags = 0;
    if(dataloader->next_sample_ordinal == dataloader->length) {
        dataloader->epoch_index += 1;
        dataloader->next_sample_ordinal = 0;
        dataloader->next_boundary_flags = GSX_DATALOADER_BOUNDARY_NEW_EPOCH;
        if(dataloader->desc.shuffle_each_epoch) {
            gsx_dataloader_shuffle_permutation(dataloader, &dataloader->rng);
            dataloader->next_boundary_flags |= GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION;
        }
        GSX_LOG_DEBUG("dataloader: epoch %zu complete\n", (size_t)dataloader->epoch_index);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
