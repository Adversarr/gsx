#ifndef GSX_IMAGE_IMPL_H
#define GSX_IMAGE_IMPL_H

#include "../gsx-impl.h"

#include <math.h>
#include <string.h>

static inline float gsx_image_clamp_unit_float(float value)
{
    if(!isfinite((double)value)) {
        return value > 0.0f ? 1.0f : 0.0f;
    }
    if(value < 0.0f) {
        return 0.0f;
    }
    if(value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static inline uint8_t gsx_image_quantize_u8(float value)
{
    value = gsx_image_clamp_unit_float(value);
    return (uint8_t)(value * 255.0f + 0.5f);
}

static inline float gsx_image_dequantize_u8(uint8_t value)
{
    return (float)value / 255.0f;
}

static inline float gsx_image_linear_to_srgb(float value)
{
    value = gsx_image_clamp_unit_float(value);
    if(value <= 0.0031308f) {
        return 12.92f * value;
    }
    return 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
}

static inline float gsx_image_srgb_to_linear(float value)
{
    value = gsx_image_clamp_unit_float(value);
    if(value <= 0.04045f) {
        return value / 12.92f;
    }
    return powf((value + 0.055f) / 1.055f, 2.4f);
}

static inline gsx_error gsx_image_get_chw_hwc_dims(
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_storage_format storage_format,
    gsx_index_t *out_channels,
    gsx_index_t *out_height,
    gsx_index_t *out_width
)
{
    if(shape == NULL || out_channels == NULL || out_height == NULL || out_width == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image shape outputs must be non-null");
    }
    if(rank != 3) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image tensor rank must be 3");
    }

    switch(storage_format) {
    case GSX_STORAGE_FORMAT_CHW:
        *out_channels = shape[0];
        *out_height = shape[1];
        *out_width = shape[2];
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_STORAGE_FORMAT_HWC:
        *out_channels = shape[2];
        *out_height = shape[0];
        *out_width = shape[1];
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "image storage_format must be CHW or HWC");
    }
}

static inline bool gsx_image_same_extents(
    gsx_index_t src_rank,
    const gsx_index_t *src_shape,
    gsx_storage_format src_storage_format,
    gsx_index_t dst_rank,
    const gsx_index_t *dst_shape,
    gsx_storage_format dst_storage_format,
    gsx_index_t *out_channels,
    gsx_index_t *out_height,
    gsx_index_t *out_width
)
{
    gsx_index_t src_channels = 0;
    gsx_index_t src_height = 0;
    gsx_index_t src_width = 0;
    gsx_index_t dst_channels = 0;
    gsx_index_t dst_height = 0;
    gsx_index_t dst_width = 0;
    gsx_error error = gsx_image_get_chw_hwc_dims(src_rank, src_shape, src_storage_format, &src_channels, &src_height, &src_width);

    if(!gsx_error_is_success(error)) {
        return false;
    }
    error = gsx_image_get_chw_hwc_dims(dst_rank, dst_shape, dst_storage_format, &dst_channels, &dst_height, &dst_width);
    if(!gsx_error_is_success(error)) {
        return false;
    }
    if(src_channels != dst_channels || src_height != dst_height || src_width != dst_width) {
        return false;
    }
    if(out_channels != NULL) {
        *out_channels = src_channels;
    }
    if(out_height != NULL) {
        *out_height = src_height;
    }
    if(out_width != NULL) {
        *out_width = src_width;
    }
    return true;
}

static inline gsx_size_t gsx_image_offset_for_layout(
    gsx_storage_format storage_format,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    gsx_index_t channel,
    gsx_index_t y,
    gsx_index_t x
)
{
    if(storage_format == GSX_STORAGE_FORMAT_CHW) {
        return (gsx_size_t)(((channel * height) + y) * width + x);
    }
    return (gsx_size_t)(((y * width) + x) * channels + channel);
}

static inline gsx_error gsx_image_copy_storage_convert_bytes(
    void *dst_bytes,
    gsx_storage_format dst_storage_format,
    const void *src_bytes,
    gsx_storage_format src_storage_format,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    gsx_size_t element_size_bytes
)
{
    gsx_index_t channel = 0;
    gsx_index_t y = 0;
    gsx_index_t x = 0;
    const unsigned char *src = (const unsigned char *)src_bytes;
    unsigned char *dst = (unsigned char *)dst_bytes;

    if(dst_bytes == NULL || src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image storage conversion buffers must be non-null");
    }

    for(channel = 0; channel < channels; ++channel) {
        for(y = 0; y < height; ++y) {
            for(x = 0; x < width; ++x) {
                gsx_size_t src_index = gsx_image_offset_for_layout(src_storage_format, channels, height, width, channel, y, x);
                gsx_size_t dst_index = gsx_image_offset_for_layout(dst_storage_format, channels, height, width, channel, y, x);

                memcpy(dst + (size_t)(dst_index * element_size_bytes), src + (size_t)(src_index * element_size_bytes), (size_t)element_size_bytes);
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#endif /* GSX_IMAGE_IMPL_H */
