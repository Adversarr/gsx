#ifndef GSX_TENSOR_HELPERS_H
#define GSX_TENSOR_HELPERS_H

#include "gsx-impl.h"

#include <string.h>

static inline gsx_error gsx_tensor_require_accessible_storage(gsx_tensor_t tensor, const char *null_message, const char *storage_message)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, null_message);
    }
    if(tensor->arena == NULL || tensor->arena->dry_run || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, storage_message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_tensor_validate_bound_to_backend(
    gsx_backend_t backend,
    gsx_tensor_t tensor,
    bool allow_null,
    const char *null_message,
    const char *storage_message,
    const char *backend_message)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL) {
        if(allow_null) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, null_message);
    }

    error = gsx_tensor_require_accessible_storage(tensor, null_message, storage_message);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor->backing_buffer->buffer_type == NULL || tensor->backing_buffer->buffer_type->backend != backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, backend_message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_tensor_validate_match(gsx_tensor_t lhs, gsx_tensor_t rhs, const char *message)
{
    gsx_index_t dim = 0;

    if(lhs->rank != rhs->rank
        || lhs->data_type != rhs->data_type
        || lhs->storage_format != rhs->storage_format
        || lhs->size_bytes != rhs->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    for(dim = 0; dim < lhs->rank; ++dim) {
        if(lhs->shape[dim] != rhs->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_tensor_validate_layout_match(gsx_tensor_t reference, gsx_tensor_t tensor, const char *message)
{
    gsx_index_t dim = 0;

    if(reference->rank != tensor->rank || reference->storage_format != tensor->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    for(dim = 0; dim < reference->rank; ++dim) {
        if(reference->shape[dim] != tensor->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline bool gsx_tensor_overlaps(gsx_tensor_t lhs, gsx_tensor_t rhs)
{
    gsx_size_t lhs_end_bytes = 0;
    gsx_size_t rhs_end_bytes = 0;

    if(lhs == NULL || rhs == NULL || lhs->backing_buffer == NULL || rhs->backing_buffer == NULL) {
        return false;
    }
    if(lhs->backing_buffer != rhs->backing_buffer) {
        return false;
    }
    if(gsx_size_add_overflows(lhs->offset_bytes, lhs->size_bytes, &lhs_end_bytes)
        || gsx_size_add_overflows(rhs->offset_bytes, rhs->size_bytes, &rhs_end_bytes)) {
        return true;
    }

    return lhs->offset_bytes < rhs_end_bytes && rhs->offset_bytes < lhs_end_bytes;
}

static inline gsx_error gsx_tensor_validate_no_alias(gsx_tensor_t lhs, gsx_tensor_t rhs, const char *message)
{
    if(gsx_tensor_overlaps(lhs, rhs)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_tensor_validate_no_alias_list(gsx_tensor_t *tensors, gsx_size_t tensor_count, const char *message)
{
    gsx_size_t left = 0;
    gsx_size_t right = 0;

    for(left = 0; left < tensor_count; ++left) {
        if(tensors[left] == NULL) {
            continue;
        }
        for(right = left + 1; right < tensor_count; ++right) {
            if(tensors[right] == NULL) {
                continue;
            }
            if(gsx_tensor_overlaps(tensors[left], tensors[right])) {
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline void gsx_tensor_fill_backend_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view)
{
    out_view->buffer = tensor->backing_buffer;
    out_view->offset_bytes = tensor->offset_bytes;
    out_view->size_bytes = tensor->size_bytes;
    out_view->effective_alignment_bytes = tensor->effective_alignment_bytes;
    out_view->data_type = tensor->data_type;
}

static inline gsx_error gsx_tensor_get_element_count(gsx_tensor_t tensor, gsx_size_t *out_element_count)
{
    gsx_size_t element_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL || out_element_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and out_element_count must be non-null");
    }

    error = gsx_data_type_get_size_bytes(tensor->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(element_size_bytes == 0 || tensor->size_bytes % element_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor byte size is inconsistent with its data type");
    }

    *out_element_count = tensor->size_bytes / element_size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_tensor_require_positive_rank1(gsx_tensor_t tensor, const char *message)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    if(tensor->rank != 1 || tensor->shape[0] <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_tensor_get_positive_leading_extent(gsx_tensor_t tensor, gsx_size_t *out_leading_extent, const char *message)
{
    if(tensor == NULL || out_leading_extent == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    if(tensor->rank <= 0 || tensor->shape[0] <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, message);
    }

    *out_leading_extent = (gsx_size_t)tensor->shape[0];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline bool gsx_tensor_extract_image_layout(
    gsx_tensor_t tensor,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_channels,
    gsx_size_t *out_height,
    gsx_size_t *out_width)
{
    gsx_size_t outer_count = 1;
    gsx_index_t axis = 0;
    gsx_size_t channels = 0;
    gsx_size_t height = 0;
    gsx_size_t width = 0;

    if(tensor == NULL || out_outer_count == NULL || out_channels == NULL || out_height == NULL || out_width == NULL) {
        return false;
    }
    if(tensor->rank < 3) {
        return false;
    }
    for(axis = 0; axis < tensor->rank - 3; ++axis) {
        gsx_size_t dim_extent = (gsx_size_t)tensor->shape[axis];
        gsx_size_t next_outer_count = 0;

        if(dim_extent == 0 || gsx_size_mul_overflows(outer_count, dim_extent, &next_outer_count)) {
            return false;
        }
        outer_count = next_outer_count;
    }
    if(tensor->storage_format == GSX_STORAGE_FORMAT_HWC) {
        height = (gsx_size_t)tensor->shape[tensor->rank - 3];
        width = (gsx_size_t)tensor->shape[tensor->rank - 2];
        channels = (gsx_size_t)tensor->shape[tensor->rank - 1];
    } else {
        channels = (gsx_size_t)tensor->shape[tensor->rank - 3];
        height = (gsx_size_t)tensor->shape[tensor->rank - 2];
        width = (gsx_size_t)tensor->shape[tensor->rank - 1];
    }
    if(channels == 0 || height == 0 || width == 0) {
        return false;
    }

    *out_outer_count = outer_count;
    *out_channels = channels;
    *out_height = height;
    *out_width = width;
    return true;
}

static inline float gsx_loss_scale_grad(
    const gsx_loss *loss, gsx_size_t element_count, gsx_float_t scale)
{
    if(loss->grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN) {
        return scale / (float)element_count;
    }

    return scale;
}

static inline gsx_error gsx_tensor_init_desc_like_f32(gsx_tensor_t reference, gsx_arena_t arena, gsx_tensor_desc *out_desc)
{
    if(reference == NULL || arena == NULL || out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reference, arena, and out_desc must be non-null");
    }

    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->rank = reference->rank;
    memcpy(out_desc->shape, reference->shape, sizeof(out_desc->shape));
    out_desc->requested_alignment_bytes = reference->requested_alignment_bytes;
    out_desc->data_type = GSX_DATA_TYPE_F32;
    out_desc->storage_format = reference->storage_format;
    out_desc->arena = arena;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline void gsx_tensor_dispose_handles(gsx_tensor_t *tensors, gsx_index_t count)
{
    gsx_index_t index = 0;

    if(tensors == NULL) {
        return;
    }
    for(index = 0; index < count; ++index) {
        if(tensors[index] != NULL) {
            (void)gsx_tensor_free(tensors[index]);
            tensors[index] = NULL;
        }
    }
}

static inline gsx_error gsx_tensor_free_handles(gsx_tensor_t *tensors, gsx_index_t count)
{
    gsx_index_t index = 0;
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };

    if(tensors == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    for(index = 0; index < count; ++index) {
        if(tensors[index] != NULL) {
            gsx_error error = gsx_tensor_free(tensors[index]);

            if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
                first_error = error;
            }
            if(gsx_error_is_success(error)) {
                tensors[index] = NULL;
            }
        }
    }

    return first_error;
}

static inline gsx_error gsx_tensor_download_bytes(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_backend_tensor_view tensor_view = { 0 };

    if(tensor == NULL || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and tensor storage must be non-null");
    }
    if(byte_count > tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "byte_count exceeds tensor capacity");
    }
    if(byte_count != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero byte_count");
    }

    gsx_tensor_fill_backend_view(tensor, &tensor_view);
    return tensor->backing_buffer->iface->get_tensor(tensor->backing_buffer, &tensor_view, dst_bytes, 0, byte_count);
}

#endif /* GSX_TENSOR_HELPERS_H */
