#include "gsx/extra/gsx-image.h"

#include "../gsx-impl.h"
#include "gsx-image-impl.h"

static gsx_error gsx_tensor_image_require_accessible(gsx_tensor_t tensor)
{
    if(tensor == NULL || tensor->arena == NULL || tensor->arena->buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must be non-null and have valid storage");
    }
    if(tensor->arena->dry_run) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor storage is unavailable in dry-run mode");
    }
    if(tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor backing buffer is unavailable");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_image_validate_shape_storage(gsx_tensor_t tensor)
{
    gsx_size_t element_size_bytes = 0;
    gsx_size_t expected_size_bytes = 0;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must be non-null");
    }
    error = gsx_data_type_get_size_bytes(tensor->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    expected_size_bytes = element_size_bytes;
    for(dim = 0; dim < tensor->rank; ++dim) {
        if(gsx_size_mul_overflows(expected_size_bytes, (gsx_size_t)tensor->shape[dim], &expected_size_bytes)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor size overflows");
        }
    }
    if(expected_size_bytes != tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor shape/storage metadata is inconsistent");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_backend_tensor_view gsx_tensor_image_make_backend_view(gsx_tensor_t tensor)
{
    gsx_backend_tensor_view tensor_view = { 0 };

    tensor_view.buffer = tensor->backing_buffer;
    tensor_view.offset_bytes = tensor->offset_bytes;
    tensor_view.size_bytes = tensor->size_bytes;
    tensor_view.effective_alignment_bytes = tensor->effective_alignment_bytes;
    tensor_view.data_type = tensor->data_type;
    return tensor_view;
}

static gsx_error gsx_tensor_image_require_same_backend(gsx_tensor_t lhs, gsx_tensor_t rhs)
{
    if(lhs == NULL || rhs == NULL || lhs->arena == NULL || rhs->arena == NULL || lhs->arena->buffer_type == NULL
        || rhs->arena->buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensors must be non-null and have valid arenas");
    }
    if(lhs->arena->buffer_type->backend != rhs->arena->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensors must belong to the same backend");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_image_validate_extents(gsx_tensor_t dst, gsx_tensor_t src, bool require_same_storage_format)
{
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;

    if(src == NULL || dst == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must be non-null");
    }
    if(require_same_storage_format && src->storage_format != dst->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst storage_format must match");
    }
    if(!gsx_image_same_extents(
            src->rank,
            src->shape,
            src->storage_format,
            dst->rank,
            dst->shape,
            dst->storage_format,
            &channels,
            &height,
            &width)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must describe the same logical image extents");
    }
    if(channels <= 0 || height <= 0 || width <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image extents must be positive");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_image_convert_colorspace(
    gsx_tensor_t dst,
    gsx_image_colorspace dst_colorspace,
    gsx_tensor_t src,
    gsx_image_colorspace src_colorspace
)
{
    gsx_backend_tensor_view src_view = { 0 };
    gsx_backend_tensor_view dst_view = { 0 };
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src == NULL || dst == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must be non-null");
    }
    if((src_colorspace != GSX_IMAGE_COLOR_SPACE_LINEAR && src_colorspace != GSX_IMAGE_COLOR_SPACE_SRGB)
        || (dst_colorspace != GSX_IMAGE_COLOR_SPACE_LINEAR && dst_colorspace != GSX_IMAGE_COLOR_SPACE_SRGB)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image colorspace must be linear or sRGB");
    }

    error = gsx_tensor_image_require_accessible(src);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_require_accessible(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_require_same_backend(src, dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(src->backing_buffer->buffer_type != dst->backing_buffer->buffer_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must use the same buffer_type");
    }
    if(src->data_type != dst->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must use the same data_type");
    }
    if(src->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "colorspace conversion currently only supports float32 images");
    }
    if(src->storage_format != dst->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst storage_format must match");
    }
    error = gsx_tensor_image_validate_extents(dst, src, true);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_image_same_extents(
            src->rank,
            src->shape,
            src->storage_format,
            dst->rank,
            dst->shape,
            dst->storage_format,
            &channels,
            &height,
            &width)
        || channels != 3) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "colorspace conversion requires a 3-channel RGB image");
    }
    error = gsx_tensor_image_validate_shape_storage(src);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_validate_shape_storage(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dst->backing_buffer->iface->image_convert_colorspace == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend image colorspace conversion is not available");
    }

    if(src_colorspace == dst_colorspace) {
        if(src == dst) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return gsx_tensor_copy(dst, src);
    }

    src_view = gsx_tensor_image_make_backend_view(src);
    dst_view = gsx_tensor_image_make_backend_view(dst);
    return dst->backing_buffer->iface->image_convert_colorspace(
        dst->backing_buffer,
        &src_view,
        src->storage_format,
        src->rank,
        src->shape,
        src_colorspace,
        &dst_view,
        dst_colorspace);
}

GSX_API gsx_error gsx_tensor_image_convert_storage_format(gsx_tensor_t dst, gsx_tensor_t src)
{
    gsx_backend_tensor_view src_view = { 0 };
    gsx_backend_tensor_view dst_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src == NULL || dst == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must be non-null");
    }

    error = gsx_tensor_image_require_accessible(src);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_require_accessible(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_require_same_backend(src, dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(src->backing_buffer->buffer_type != dst->backing_buffer->buffer_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must use the same buffer_type");
    }
    if(src->data_type != dst->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must use the same data_type");
    }
    if(src->storage_format == dst->storage_format) {
        if(src == dst) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return gsx_tensor_copy(dst, src);
    }
    error = gsx_tensor_image_validate_extents(dst, src, false);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_validate_shape_storage(src);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_validate_shape_storage(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dst->backing_buffer->iface->image_convert_storage_format == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend image storage conversion is not available");
    }

    src_view = gsx_tensor_image_make_backend_view(src);
    dst_view = gsx_tensor_image_make_backend_view(dst);
    return dst->backing_buffer->iface->image_convert_storage_format(
        dst->backing_buffer,
        &src_view,
        src->rank,
        src->shape,
        src->storage_format,
        &dst_view,
        dst->rank,
        dst->shape,
        dst->storage_format);
}

GSX_API gsx_error gsx_tensor_image_convert_data_type(gsx_tensor_t dst, gsx_tensor_t src)
{
    gsx_backend_tensor_view src_view = { 0 };
    gsx_backend_tensor_view dst_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src == NULL || dst == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must be non-null");
    }

    error = gsx_tensor_image_require_accessible(src);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_require_accessible(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_require_same_backend(src, dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(src->backing_buffer->buffer_type != dst->backing_buffer->buffer_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must use the same buffer_type");
    }
    if(src->storage_format != dst->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst storage_format must match");
    }
    error = gsx_tensor_image_validate_extents(dst, src, true);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!((src->data_type == GSX_DATA_TYPE_F32 && dst->data_type == GSX_DATA_TYPE_U8)
            || (src->data_type == GSX_DATA_TYPE_U8 && dst->data_type == GSX_DATA_TYPE_F32))) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "image data type conversion only supports float32 and uint8");
    }
    error = gsx_tensor_image_validate_shape_storage(src);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_image_validate_shape_storage(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(dst->backing_buffer->iface->image_convert_data_type == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend image data type conversion is not available");
    }

    src_view = gsx_tensor_image_make_backend_view(src);
    dst_view = gsx_tensor_image_make_backend_view(dst);
    return dst->backing_buffer->iface->image_convert_data_type(
        dst->backing_buffer,
        &src_view,
        src->storage_format,
        src->rank,
        src->shape,
        &dst_view);
}
