#ifndef GSX_IMAGE_H
#define GSX_IMAGE_H

#include "gsx/gsx-core.h"

GSX_EXTERN_C_BEGIN

// NOTE: These functions are not intended to be general purpose tensor
// operations. They are intended to provide efficient image operations, and may
// have specific requirements on the input and output tensors. These functions
// may be implemented using custom kernels that are optimized for image
// processing, and may not support all possible tensor shapes, data types, and
// storage formats. The caller is responsible for ensuring that the input and
// output tensors meet the requirements of each function, and for handling any
// errors that may be returned.

typedef enum gsx_image_colorspace {
    GSX_IMAGE_COLOR_SPACE_LINEAR = 0,
    GSX_IMAGE_COLOR_SPACE_SRGB = 1
} gsx_image_colorspace;

// Convert an RGB image between linear and sRGB color spaces. The input and
// output tensors must use the same shape, data type, buffer_type, and backend.
// Only float32 tensors with exactly 3 channels are currently supported.
GSX_API gsx_error gsx_tensor_image_convert_colorspace(
    gsx_tensor_t dst,
    gsx_image_colorspace dst_colorspace,
    gsx_tensor_t src,
    gsx_image_colorspace src_colorspace);

// Convert an image with same dtype and buffer_type from one storage format to
// another (CHW <-> HWC). The input and output tensors must have the same
// logical image extents, data type, buffer_type, and backend.
GSX_API gsx_error gsx_tensor_image_convert_storage_format(gsx_tensor_t dst, gsx_tensor_t src);

// Convert an image between float32 and uint8. The input and output tensors must
// have the same shape, storage_format, buffer_type, and backend. Float32 values
// are interpreted in [0, 1] and uint8 values are interpreted in [0, 255].
GSX_API gsx_error gsx_tensor_image_convert_data_type(gsx_tensor_t dst, gsx_tensor_t src);

GSX_EXTERN_C_END

#endif
