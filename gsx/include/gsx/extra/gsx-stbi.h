#ifndef GSX_STBI_H
#define GSX_STBI_H

#include <gsx/gsx-core.h>

GSX_EXTERN_C_BEGIN

typedef struct gsx_image {
    void *pixels;
    gsx_index_t width;
    gsx_index_t height;
    gsx_index_t channels;
    gsx_data_type data_type;
    gsx_storage_format storage_format;
} gsx_image;

GSX_API gsx_error gsx_image_load(
    gsx_image *out,
    const char *path,
    gsx_index_t desired_channels,
    gsx_data_type output_data_type,
    gsx_storage_format output_storage_format);
GSX_API gsx_error gsx_image_free(gsx_image *img);
/*
 * F32 write input uses a fixed conversion policy in this version:
 * values are interpreted in [0,1], clamped to [0,1], then quantized to uint8
 * via nearest rounding using x * 255 + 0.5.
 */
GSX_API gsx_error gsx_image_write_png(
    const char *path,
    const void *pixels,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_data_type input_data_type,
    gsx_storage_format input_storage_format);
GSX_API gsx_error gsx_image_write_jpg(
    const char *path,
    const void *pixels,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    gsx_data_type input_data_type,
    gsx_storage_format input_storage_format,
    gsx_index_t quality);

GSX_API gsx_error gsx_image_resize(
    gsx_image *out,
    const gsx_image *input,
    gsx_index_t output_width,
    gsx_index_t output_height);

GSX_EXTERN_C_END

#endif /* GSX_STBI_H */
