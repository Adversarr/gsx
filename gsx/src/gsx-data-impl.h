#ifndef GSX_DATA_IMPL_H
#define GSX_DATA_IMPL_H

#include "gsx-impl.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct gsx_dataloader_slot {
    gsx_tensor_t rgb_tensor;
    gsx_tensor_t alpha_tensor;
    gsx_tensor_t invdepth_tensor;
    gsx_size_t rgb_size_bytes;
    gsx_size_t alpha_size_bytes;
    gsx_size_t invdepth_size_bytes;
} gsx_dataloader_slot;

static inline bool gsx_dataloader_source_image_data_type_is_supported(gsx_data_type data_type)
{
    return data_type == GSX_DATA_TYPE_U8 || data_type == GSX_DATA_TYPE_F32;
}

static inline bool gsx_dataloader_output_image_data_type_is_supported(gsx_data_type data_type)
{
    return data_type == GSX_DATA_TYPE_F32;
}

static inline gsx_error gsx_dataloader_compute_element_count(
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

static inline gsx_error gsx_dataloader_compute_packed_bytes(
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

static inline void gsx_dataloader_compute_tensor_shape(
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

static inline gsx_error gsx_dataloader_make_tensor_desc(
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

static inline uint16_t gsx_dataloader_float_to_f16_bits(float value)
{
    union {
        float f32;
        uint32_t u32;
    } input = { 0.0f };
    uint32_t sign = 0;
    uint32_t exp = 0;
    uint32_t mantissa = 0;
    uint32_t out_exp = 0;
    uint32_t out_mantissa = 0;

    input.f32 = value;
    sign = (input.u32 >> 16u) & 0x8000u;
    exp = (input.u32 >> 23u) & 0xFFu;
    mantissa = input.u32 & 0x7FFFFFu;

    if(exp == 0xFFu) {
        if(mantissa != 0u) {
            return (uint16_t)(sign | 0x7E00u);
        }
        return (uint16_t)(sign | 0x7C00u);
    }
    if(exp > 142u) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if(exp < 113u) {
        if(exp < 103u) {
            return (uint16_t)sign;
        }
        mantissa |= 0x800000u;
        out_mantissa = mantissa >> (126u - exp);
        if((out_mantissa & 0x00001000u) != 0u) {
            out_mantissa += 0x00002000u;
        }
        return (uint16_t)(sign | (out_mantissa >> 13u));
    }

    out_exp = exp - 112u;
    out_mantissa = mantissa + 0x00001000u;
    if((out_mantissa & 0x00800000u) != 0u) {
        out_mantissa = 0u;
        out_exp += 1u;
    }
    if(out_exp >= 31u) {
        return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | (out_exp << 10u) | (out_mantissa >> 13u));
}

static inline float gsx_dataloader_read_source_value(
    const void *src_bytes,
    gsx_data_type src_type,
    gsx_size_t element_index)
{
    if(src_type == GSX_DATA_TYPE_U8) {
        return (float)((const uint8_t *)src_bytes)[element_index];
    }
    return ((const float *)src_bytes)[element_index];
}

static inline uint8_t gsx_dataloader_quantize_u8(float value)
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

static inline void gsx_dataloader_write_output_value(
    void *dst_bytes,
    gsx_data_type dst_type,
    gsx_size_t element_index,
    float value)
{
    if(dst_type == GSX_DATA_TYPE_U8) {
        ((uint8_t *)dst_bytes)[element_index] = gsx_dataloader_quantize_u8(value);
    } else if(dst_type == GSX_DATA_TYPE_F16) {
        ((uint16_t *)dst_bytes)[element_index] = gsx_dataloader_float_to_f16_bits(value);
    } else {
        ((float *)dst_bytes)[element_index] = value;
    }
}

static inline gsx_error gsx_dataloader_pack_hwc_to_chw(
    const void *src_bytes,
    gsx_data_type src_type,
    gsx_data_type dst_type,
    gsx_index_t width,
    gsx_index_t height,
    void *dst_bytes)
{
    gsx_index_t channel = 0;
    gsx_index_t y = 0;
    gsx_index_t x = 0;

    if(src_bytes == NULL || dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes and dst_bytes must be non-null");
    }

    for(channel = 0; channel < 3; ++channel) {
        for(y = 0; y < height; ++y) {
            for(x = 0; x < width; ++x) {
                const gsx_size_t src_index = (((gsx_size_t)y * (gsx_size_t)width) + (gsx_size_t)x) * 3u + (gsx_size_t)channel;
                const gsx_size_t dst_index = (((gsx_size_t)channel * (gsx_size_t)height) + (gsx_size_t)y) * (gsx_size_t)width + (gsx_size_t)x;
                const float value = gsx_dataloader_read_source_value(src_bytes, src_type, src_index);

                gsx_dataloader_write_output_value(dst_bytes, dst_type, dst_index, value);
            }
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_dataloader_pack_hw_to_chw(
    const void *src_bytes,
    gsx_data_type src_type,
    gsx_data_type dst_type,
    gsx_index_t width,
    gsx_index_t height,
    void *dst_bytes)
{
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src_bytes == NULL || dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes and dst_bytes must be non-null");
    }

    error = gsx_dataloader_compute_element_count(width, height, 1, &element_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(element_index = 0; element_index < element_count; ++element_index) {
        const float value = gsx_dataloader_read_source_value(src_bytes, src_type, element_index);
        gsx_dataloader_write_output_value(dst_bytes, dst_type, element_index, value);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_dataloader_validate_sample(
    const gsx_dataset_desc *dataset_desc,
    const gsx_dataset_cpu_sample *sample)
{
    if(dataset_desc == NULL || sample == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataset_desc and sample must be non-null");
    }
    if(sample->pose.camera_id != sample->intrinsics.camera_id) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample pose.camera_id must match intrinsics.camera_id");
    }
    if(sample->intrinsics.width != dataset_desc->width || sample->intrinsics.height != dataset_desc->height) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample intrinsics width and height must match the dataset descriptor");
    }
    if(dataset_desc->has_rgb && sample->rgb_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample rgb_data must be present when the dataset exposes RGB");
    }
    if(!dataset_desc->has_rgb && sample->rgb_data != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample rgb_data must be null when the dataset does not expose RGB");
    }
    if(dataset_desc->has_alpha && sample->alpha_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample alpha_data must be present when the dataset exposes alpha");
    }
    if(!dataset_desc->has_alpha && sample->alpha_data != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample alpha_data must be null when the dataset does not expose alpha");
    }
    if(dataset_desc->has_invdepth && sample->invdepth_data == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample invdepth_data must be present when the dataset exposes inverse-depth");
    }
    if(!dataset_desc->has_invdepth && sample->invdepth_data != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sample invdepth_data must be null when the dataset does not expose inverse-depth");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#endif /* GSX_DATA_IMPL_H */
