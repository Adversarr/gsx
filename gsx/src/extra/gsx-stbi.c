#include <gsx/extra/gsx-stbi.h>
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize2.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool gsx_image_is_supported_data_type(gsx_data_type data_type)
{
    return data_type == GSX_DATA_TYPE_U8 || data_type == GSX_DATA_TYPE_F32;
}

static bool gsx_image_is_supported_storage_format(gsx_storage_format storage_format)
{
    return storage_format == GSX_STORAGE_FORMAT_HWC || storage_format == GSX_STORAGE_FORMAT_CHW;
}

static bool gsx_checked_mul_size(gsx_size_t a, gsx_size_t b, gsx_size_t *out)
{
    if(out == NULL) {
        return false;
    }
    if(a != 0 && b > ((gsx_size_t)-1) / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool gsx_image_compute_total_elements(gsx_index_t width, gsx_index_t height, gsx_index_t channels, gsx_size_t *out_total)
{
    gsx_size_t total = 0;
    if(width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }
    if(!gsx_checked_mul_size((gsx_size_t)width, (gsx_size_t)height, &total)) {
        return false;
    }
    if(!gsx_checked_mul_size(total, (gsx_size_t)channels, &total)) {
        return false;
    }
    *out_total = total;
    return true;
}

static bool gsx_image_checked_index_to_int(gsx_index_t value, int *out)
{
    if(out == NULL || value <= 0 || value > INT_MAX) {
        return false;
    }
    *out = (int)value;
    return true;
}

static gsx_size_t gsx_image_element_size(gsx_data_type data_type)
{
    if(data_type == GSX_DATA_TYPE_F32) {
        return sizeof(float);
    }
    if(data_type == GSX_DATA_TYPE_U8) {
        return sizeof(uint8_t);
    }
    return 0;
}

static void gsx_image_hwc_to_chw(
    const uint8_t *src,
    uint8_t *dst,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_size_t element_size)
{
    for(gsx_index_t c = 0; c < channels; ++c) {
        for(gsx_index_t y = 0; y < height; ++y) {
            for(gsx_index_t x = 0; x < width; ++x) {
                const gsx_size_t src_index = (gsx_size_t)(((y * width) + x) * channels + c);
                const gsx_size_t dst_index = (gsx_size_t)(((c * height) + y) * width + x);
                memcpy(dst + dst_index * element_size, src + src_index * element_size, (size_t)element_size);
            }
        }
    }
}

static void gsx_image_chw_to_hwc(
    const uint8_t *src,
    uint8_t *dst,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_size_t element_size)
{
    for(gsx_index_t y = 0; y < height; ++y) {
        for(gsx_index_t x = 0; x < width; ++x) {
            for(gsx_index_t c = 0; c < channels; ++c) {
                const gsx_size_t src_index = (gsx_size_t)(((c * height) + y) * width + x);
                const gsx_size_t dst_index = (gsx_size_t)(((y * width) + x) * channels + c);
                memcpy(dst + dst_index * element_size, src + src_index * element_size, (size_t)element_size);
            }
        }
    }
}

static void gsx_image_hwc_to_chw_float(
    const float *src,
    float *dst,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels)
{
    for(gsx_index_t c = 0; c < channels; ++c) {
        for(gsx_index_t y = 0; y < height; ++y) {
            for(gsx_index_t x = 0; x < width; ++x) {
                const gsx_size_t src_index = (gsx_size_t)(((y * width) + x) * channels + c);
                const gsx_size_t dst_index = (gsx_size_t)(((c * height) + y) * width + x);
                dst[dst_index] = src[src_index];
            }
        }
    }
}

static void gsx_image_chw_to_hwc_float(
    const float *src,
    float *dst,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels)
{
    for(gsx_index_t y = 0; y < height; ++y) {
        for(gsx_index_t x = 0; x < width; ++x) {
            for(gsx_index_t c = 0; c < channels; ++c) {
                const gsx_size_t src_index = (gsx_size_t)(((c * height) + y) * width + x);
                const gsx_size_t dst_index = (gsx_size_t)(((y * width) + x) * channels + c);
                dst[dst_index] = src[src_index];
            }
        }
    }
}

static uint8_t gsx_image_quantize_f32_to_u8(float value)
{
    if(!isfinite((double)value)) {
        return value > 0.0f ? (uint8_t)255 : (uint8_t)0;
    }
    if(value < 0.0f) {
        value = 0.0f;
    } else if(value > 1.0f) {
        value = 1.0f;
    }
    return (uint8_t)(value * 255.0f + 0.5f);
}

static gsx_error gsx_image_make_hwc_u8_buffer(
    const void *pixels,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_data_type input_data_type,
    gsx_storage_format input_storage_format,
    uint8_t *out_hwc_u8)
{
    gsx_size_t total = 0;
    if(!gsx_image_compute_total_elements(width, height, channels, &total)) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "invalid dimensions or channels"};
    }

    if(input_data_type == GSX_DATA_TYPE_U8) {
        if(input_storage_format == GSX_STORAGE_FORMAT_HWC) {
            memcpy(out_hwc_u8, pixels, (size_t)total);
            return (gsx_error){GSX_ERROR_SUCCESS, NULL};
        }
        gsx_image_chw_to_hwc((const uint8_t *)pixels, out_hwc_u8, width, height, channels, sizeof(uint8_t));
        return (gsx_error){GSX_ERROR_SUCCESS, NULL};
    }

    if(input_data_type == GSX_DATA_TYPE_F32) {
        const float *src = (const float *)pixels;
        if(input_storage_format == GSX_STORAGE_FORMAT_HWC) {
            for(gsx_size_t i = 0; i < total; ++i) {
                out_hwc_u8[i] = gsx_image_quantize_f32_to_u8(src[i]);
            }
            return (gsx_error){GSX_ERROR_SUCCESS, NULL};
        }

        for(gsx_index_t y = 0; y < height; ++y) {
            for(gsx_index_t x = 0; x < width; ++x) {
                for(gsx_index_t c = 0; c < channels; ++c) {
                    const gsx_size_t src_index = (gsx_size_t)(((c * height) + y) * width + x);
                    const gsx_size_t dst_index = (gsx_size_t)(((y * width) + x) * channels + c);
                    out_hwc_u8[dst_index] = gsx_image_quantize_f32_to_u8(src[src_index]);
                }
            }
        }
        return (gsx_error){GSX_ERROR_SUCCESS, NULL};
    }

    return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "unsupported input data type"};
}

gsx_error gsx_image_load(
    gsx_image *out,
    const char *path,
    gsx_index_t desired_channels,
    gsx_data_type output_data_type,
    gsx_storage_format output_storage_format)
{
    if(out == NULL || path == NULL) {
        return (gsx_error){GSX_ERROR_INVALID_ARGUMENT, "null output or path"};
    }
    if(out->pixels != NULL) {
        return (gsx_error){GSX_ERROR_INVALID_STATE, "out->pixels must be NULL"};
    }
    if(desired_channels < 1 || desired_channels > 4) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "desired_channels must be 1-4"};
    }
    if(!gsx_image_is_supported_data_type(output_data_type)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "output data type must be U8 or F32"};
    }
    if(!gsx_image_is_supported_storage_format(output_storage_format)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "output storage format must be HWC or CHW"};
    }

    {
        int requested_channels = (int)desired_channels;
        int width = 0;
        int height = 0;
        int channels = 0;
        gsx_size_t total = 0;
        gsx_size_t element_size = gsx_image_element_size(output_data_type);
        gsx_size_t total_bytes = 0;
        void *decoded = NULL;
        void *converted = NULL;

        if(element_size == 0) {
            return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "unsupported output data type"};
        }

        decoded = (void *)stbi_load(path, &width, &height, &channels, requested_channels);
        if(decoded == NULL) {
            return (gsx_error){GSX_ERROR_IO, "failed to load image"};
        }

        if(!gsx_image_compute_total_elements((gsx_index_t)width, (gsx_index_t)height, desired_channels, &total)) {
            stbi_image_free(decoded);
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "image dimensions overflow"};
        }
        if(!gsx_checked_mul_size(total, element_size, &total_bytes)) {
            stbi_image_free(decoded);
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "image byte size overflow"};
        }

        if(total_bytes > (gsx_size_t)SIZE_MAX) {
            stbi_image_free(decoded);
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "image byte size exceeds platform limits"};
        }
        converted = malloc((size_t)total_bytes);
        if(converted == NULL) {
            stbi_image_free(decoded);
            return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate image output"};
        }

        if(output_data_type == GSX_DATA_TYPE_F32) {
            const uint8_t *src_u8 = (const uint8_t *)decoded;
            float *dst_f32 = (float *)converted;
            if(output_storage_format == GSX_STORAGE_FORMAT_HWC) {
                for(gsx_size_t i = 0; i < total; ++i) {
                    dst_f32[i] = (float)src_u8[i] / 255.0f;
                }
            } else {
                for(gsx_index_t c = 0; c < desired_channels; ++c) {
                    for(gsx_index_t y = 0; y < height; ++y) {
                        for(gsx_index_t x = 0; x < width; ++x) {
                            const gsx_size_t src_index = (gsx_size_t)(((y * width) + x) * desired_channels + c);
                            const gsx_size_t dst_index = (gsx_size_t)(((c * height) + y) * width + x);
                            dst_f32[dst_index] = (float)src_u8[src_index] / 255.0f;
                        }
                    }
                }
            }
        } else {
            if(output_storage_format == GSX_STORAGE_FORMAT_HWC) {
                memcpy(converted, decoded, (size_t)total_bytes);
            } else {
                gsx_image_hwc_to_chw(
                    (const uint8_t *)decoded,
                    (uint8_t *)converted,
                    (gsx_index_t)width,
                    (gsx_index_t)height,
                    desired_channels,
                    element_size);
            }
        }
        stbi_image_free(decoded);
        out->pixels = converted;

        out->width = (gsx_index_t)width;
        out->height = (gsx_index_t)height;
        out->channels = desired_channels;
        out->data_type = output_data_type;
        out->storage_format = output_storage_format;
        return (gsx_error){GSX_ERROR_SUCCESS, NULL};
    }
}

gsx_error gsx_image_free(gsx_image *img)
{
    if(img == NULL) {
        return (gsx_error){GSX_ERROR_INVALID_ARGUMENT, "null image"};
    }
    free(img->pixels);
    img->pixels = NULL;
    img->width = 0;
    img->height = 0;
    img->channels = 0;
    img->data_type = GSX_DATA_TYPE_U8;
    img->storage_format = GSX_STORAGE_FORMAT_HWC;
    return (gsx_error){GSX_ERROR_SUCCESS, NULL};
}

gsx_error gsx_image_write_png(
    const char *path,
    const void *pixels,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_data_type input_data_type,
    gsx_storage_format input_storage_format)
{
    gsx_size_t total = 0;
    uint8_t *buffer = NULL;
    int width_i = 0;
    int height_i = 0;
    int channels_i = 0;
    gsx_error error = {GSX_ERROR_SUCCESS, NULL};
    int ok = 0;
    int stride_bytes = 0;

    if(path == NULL || pixels == NULL) {
        return (gsx_error){GSX_ERROR_INVALID_ARGUMENT, "null path or pixels"};
    }
    if(width <= 0 || height <= 0 || channels < 1 || channels > 4) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "invalid dimensions or channels"};
    }
    if(!gsx_image_is_supported_data_type(input_data_type)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "input data type must be U8 or F32"};
    }
    if(!gsx_image_is_supported_storage_format(input_storage_format)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "input storage format must be HWC or CHW"};
    }
    if(!gsx_image_checked_index_to_int(width, &width_i) ||
       !gsx_image_checked_index_to_int(height, &height_i) ||
       !gsx_image_checked_index_to_int(channels, &channels_i)) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "dimensions exceed encoder limits"};
    }
    if(!gsx_image_compute_total_elements(width, height, channels, &total)) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "invalid dimensions or channels"};
    }
    if(total > (gsx_size_t)SIZE_MAX) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "image byte size exceeds platform limits"};
    }
    if((gsx_size_t)width_i > (gsx_size_t)INT_MAX / (gsx_size_t)channels_i) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "row stride exceeds encoder limits"};
    }
    stride_bytes = width_i * channels_i;

    buffer = (uint8_t *)malloc((size_t)total);
    if(buffer == NULL) {
        return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate conversion buffer"};
    }

    error = gsx_image_make_hwc_u8_buffer(
        pixels,
        width,
        height,
        channels,
        input_data_type,
        input_storage_format,
        buffer);
    if(error.code != GSX_ERROR_SUCCESS) {
        free(buffer);
        return error;
    }

    ok = stbi_write_png(path, width_i, height_i, channels_i, buffer, stride_bytes);
    free(buffer);

    if(!ok) {
        return (gsx_error){GSX_ERROR_IO, "failed to write png"};
    }
    return (gsx_error){GSX_ERROR_SUCCESS, NULL};
}

gsx_error gsx_image_write_jpg(
    const char *path,
    const void *pixels,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_data_type input_data_type,
    gsx_storage_format input_storage_format,
    gsx_index_t quality)
{
    gsx_size_t total = 0;
    uint8_t *buffer = NULL;
    int width_i = 0;
    int height_i = 0;
    int channels_i = 0;
    gsx_error error = {GSX_ERROR_SUCCESS, NULL};
    int ok = 0;

    if(path == NULL || pixels == NULL) {
        return (gsx_error){GSX_ERROR_INVALID_ARGUMENT, "null path or pixels"};
    }
    if(width <= 0 || height <= 0 || channels < 1 || channels > 4) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "invalid dimensions or channels"};
    }
    if(quality < 1 || quality > 100) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "quality must be 1-100"};
    }
    if(!gsx_image_is_supported_data_type(input_data_type)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "input data type must be U8 or F32"};
    }
    if(!gsx_image_is_supported_storage_format(input_storage_format)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "input storage format must be HWC or CHW"};
    }
    if(!gsx_image_checked_index_to_int(width, &width_i) ||
       !gsx_image_checked_index_to_int(height, &height_i) ||
       !gsx_image_checked_index_to_int(channels, &channels_i)) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "dimensions exceed encoder limits"};
    }
    if(!gsx_image_compute_total_elements(width, height, channels, &total)) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "invalid dimensions or channels"};
    }
    if(total > (gsx_size_t)SIZE_MAX) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "image byte size exceeds platform limits"};
    }

    buffer = (uint8_t *)malloc((size_t)total);
    if(buffer == NULL) {
        return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate conversion buffer"};
    }

    error = gsx_image_make_hwc_u8_buffer(
        pixels,
        width,
        height,
        channels,
        input_data_type,
        input_storage_format,
        buffer);
    if(error.code != GSX_ERROR_SUCCESS) {
        free(buffer);
        return error;
    }

    ok = stbi_write_jpg(path, width_i, height_i, channels_i, buffer, (int)quality);
    free(buffer);

    if(!ok) {
        return (gsx_error){GSX_ERROR_IO, "failed to write jpg"};
    }
    return (gsx_error){GSX_ERROR_SUCCESS, NULL};
}

static stbir_pixel_layout gsx_image_channels_to_pixel_layout(gsx_index_t channels)
{
    if(channels == 1) {
        return STBIR_1CHANNEL;
    }
    if(channels == 2) {
        return STBIR_2CHANNEL;
    }
    if(channels == 3) {
        return STBIR_RGB;
    }
    return STBIR_4CHANNEL;
}

gsx_error gsx_image_resize(
    gsx_image *out,
    const gsx_image *input,
    gsx_index_t output_width,
    gsx_index_t output_height)
{
    if(out == NULL || input == NULL) {
        return (gsx_error){GSX_ERROR_INVALID_ARGUMENT, "null output or input"};
    }
    if(out->pixels != NULL) {
        return (gsx_error){GSX_ERROR_INVALID_STATE, "out->pixels must be NULL"};
    }
    if(output_width <= 0 || output_height <= 0) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "output dimensions must be positive"};
    }
    if(!gsx_image_is_supported_data_type(input->data_type)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "input data type must be U8 or F32"};
    }
    if(!gsx_image_is_supported_storage_format(input->storage_format)) {
        return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "input storage format must be HWC or CHW"};
    }
    if(input->channels < 1 || input->channels > 4) {
        return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "channels must be 1-4"};
    }
    if(input->pixels == NULL) {
        return (gsx_error){GSX_ERROR_INVALID_STATE, "input->pixels is NULL"};
    }

    {
        int input_width_i = 0;
        int input_height_i = 0;
        int output_width_i = 0;
        int output_height_i = 0;
        int channels_i = 0;
        gsx_size_t input_total = 0;
        gsx_size_t output_total = 0;
        gsx_size_t input_element_size = 0;
        gsx_size_t output_element_size = 0;
        gsx_size_t input_bytes = 0;
        gsx_size_t output_bytes = 0;
        void *output_pixels = NULL;
        int input_stride = 0;
        int output_stride = 0;

        if(!gsx_image_checked_index_to_int(input->width, &input_width_i) ||
           !gsx_image_checked_index_to_int(input->height, &input_height_i) ||
           !gsx_image_checked_index_to_int(output_width, &output_width_i) ||
           !gsx_image_checked_index_to_int(output_height, &output_height_i) ||
           !gsx_image_checked_index_to_int(input->channels, &channels_i)) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "dimensions exceed resize limits"};
        }

        if(!gsx_image_compute_total_elements(input->width, input->height, input->channels, &input_total)) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "input dimensions overflow"};
        }
        if(!gsx_image_compute_total_elements(output_width, output_height, input->channels, &output_total)) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "output dimensions overflow"};
        }

        input_element_size = gsx_image_element_size(input->data_type);
        output_element_size = gsx_image_element_size(input->data_type);
        if(input_element_size == 0 || output_element_size == 0) {
            return (gsx_error){GSX_ERROR_NOT_SUPPORTED, "unsupported data type"};
        }

        if((gsx_size_t)input_width_i > (gsx_size_t)INT_MAX / (gsx_size_t)channels_i ||
           (gsx_size_t)input_width_i * (gsx_size_t)channels_i > (gsx_size_t)INT_MAX / input_element_size) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "input stride exceeds resize limits"};
        }
        if((gsx_size_t)output_width_i > (gsx_size_t)INT_MAX / (gsx_size_t)channels_i ||
           (gsx_size_t)output_width_i * (gsx_size_t)channels_i > (gsx_size_t)INT_MAX / output_element_size) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "output stride exceeds resize limits"};
        }

        if(!gsx_checked_mul_size(input_total, input_element_size, &input_bytes) ||
           !gsx_checked_mul_size(output_total, output_element_size, &output_bytes)) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "byte size overflow"};
        }

        if(input_bytes > (gsx_size_t)SIZE_MAX || output_bytes > (gsx_size_t)SIZE_MAX) {
            return (gsx_error){GSX_ERROR_OUT_OF_RANGE, "byte size exceeds platform limits"};
        }

        output_pixels = malloc((size_t)output_bytes);
        if(output_pixels == NULL) {
            return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate output"};
        }

        input_stride = input_width_i * channels_i * (int)input_element_size;
        output_stride = output_width_i * channels_i * (int)output_element_size;

        if(input->data_type == GSX_DATA_TYPE_U8) {
            const uint8_t *src = (const uint8_t *)input->pixels;
            uint8_t *dst = (uint8_t *)output_pixels;
            uint8_t *result = NULL;

            if(input->storage_format == GSX_STORAGE_FORMAT_CHW) {
                uint8_t *tmp_hwc = (uint8_t *)malloc((size_t)input_bytes);
                if(tmp_hwc == NULL) {
                    free(output_pixels);
                    return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate temp buffer"};
                }
                gsx_image_chw_to_hwc(src, tmp_hwc, input->width, input->height, input->channels, input_element_size);
                result = stbir_resize_uint8_linear(
                    tmp_hwc, input_width_i, input_height_i, input_stride,
                    dst, output_width_i, output_height_i, output_stride,
                    gsx_image_channels_to_pixel_layout(input->channels));
                free(tmp_hwc);
            } else {
                result = stbir_resize_uint8_linear(
                    src, input_width_i, input_height_i, input_stride,
                    dst, output_width_i, output_height_i, output_stride,
                    gsx_image_channels_to_pixel_layout(input->channels));
            }

            if(result == NULL) {
                free(output_pixels);
                return (gsx_error){GSX_ERROR_UNKNOWN, "resize failed"};
            }

            if(input->storage_format == GSX_STORAGE_FORMAT_CHW) {
                uint8_t *tmp_chw = (uint8_t *)malloc((size_t)output_bytes);
                if(tmp_chw == NULL) {
                    free(output_pixels);
                    return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate temp buffer"};
                }
                gsx_image_hwc_to_chw(dst, tmp_chw, output_width, output_height, input->channels, output_element_size);
                free(output_pixels);
                output_pixels = tmp_chw;
            }
        } else {
            const float *src = (const float *)input->pixels;
            float *dst = (float *)output_pixels;
            float *result = NULL;

            if(input->storage_format == GSX_STORAGE_FORMAT_CHW) {
                float *tmp_hwc = (float *)malloc((size_t)input_bytes);
                if(tmp_hwc == NULL) {
                    free(output_pixels);
                    return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate temp buffer"};
                }
                gsx_image_chw_to_hwc_float(src, tmp_hwc, input->width, input->height, input->channels);
                result = stbir_resize_float_linear(
                    tmp_hwc, input_width_i, input_height_i, input_stride,
                    dst, output_width_i, output_height_i, output_stride,
                    gsx_image_channels_to_pixel_layout(input->channels));
                free(tmp_hwc);
            } else {
                result = stbir_resize_float_linear(
                    src, input_width_i, input_height_i, input_stride,
                    dst, output_width_i, output_height_i, output_stride,
                    gsx_image_channels_to_pixel_layout(input->channels));
            }

            if(result == NULL) {
                free(output_pixels);
                return (gsx_error){GSX_ERROR_UNKNOWN, "resize failed"};
            }

            if(input->storage_format == GSX_STORAGE_FORMAT_CHW) {
                float *tmp_chw = (float *)malloc((size_t)output_bytes);
                if(tmp_chw == NULL) {
                    free(output_pixels);
                    return (gsx_error){GSX_ERROR_OUT_OF_MEMORY, "failed to allocate temp buffer"};
                }
                gsx_image_hwc_to_chw_float(dst, tmp_chw, output_width, output_height, input->channels);
                free(output_pixels);
                output_pixels = tmp_chw;
            }
        }

        out->pixels = output_pixels;
        out->width = output_width;
        out->height = output_height;
        out->channels = input->channels;
        out->data_type = input->data_type;
        out->storage_format = input->storage_format;
        return (gsx_error){GSX_ERROR_SUCCESS, NULL};
    }
}
