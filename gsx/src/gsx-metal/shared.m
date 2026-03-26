#include "internal.h"

#import <Metal/Metal.h>

#include <string.h>

gsx_metal_backend_provider gsx_metal_backend_provider_singleton = { 0 };
gsx_metal_backend_device *gsx_metal_backend_devices = NULL;
int gsx_metal_device_count = 0;
int gsx_metal_device_capacity = 0;

const gsx_backend_provider_i gsx_metal_backend_provider_iface = {
    gsx_metal_backend_provider_discover_devices,
    gsx_metal_backend_provider_create_backend
};

const gsx_backend_i gsx_metal_backend_iface = {
    gsx_metal_backend_free,
    gsx_metal_backend_get_info,
    gsx_metal_backend_get_capabilities,
    gsx_metal_backend_get_major_stream,
    gsx_metal_backend_major_stream_sync,
    gsx_metal_backend_count_buffer_types,
    gsx_metal_backend_get_buffer_type,
    gsx_metal_backend_find_buffer_type,
    gsx_metal_backend_query_unary_reduce_workspace_size,
    gsx_metal_backend_query_binary_reduce_workspace_size,
    gsx_metal_backend_create_renderer,
    gsx_metal_backend_create_loss,
    gsx_metal_backend_create_optim,
    gsx_metal_backend_create_adc,
    gsx_metal_backend_create_async_dl
};

const gsx_backend_buffer_type_i gsx_metal_backend_buffer_type_iface = {
    gsx_metal_backend_buffer_type_get_info,
    gsx_metal_backend_buffer_type_get_alloc_size,
    gsx_metal_backend_buffer_type_init_buffer
};

const gsx_backend_buffer_i gsx_metal_backend_buffer_iface = {
    gsx_metal_backend_buffer_free,
    gsx_metal_backend_buffer_get_info,
    gsx_metal_backend_buffer_get_native_handle,
    gsx_metal_backend_buffer_upload,
    gsx_metal_backend_buffer_download,
    gsx_metal_backend_buffer_set_zero,
    gsx_metal_backend_buffer_memset_tensor,
    gsx_metal_backend_buffer_set_tensor,
    gsx_metal_backend_buffer_get_tensor,
    gsx_metal_backend_buffer_copy_tensor,
    gsx_metal_backend_buffer_fill_tensor,
    gsx_metal_backend_buffer_fill_rand_tensor,
    gsx_metal_backend_buffer_fill_randn_tensor,
    gsx_metal_backend_buffer_fill_randint_tensor,
    gsx_metal_backend_buffer_multinomial_tensor,
    gsx_metal_backend_buffer_check_finite_tensor,
    gsx_metal_backend_buffer_gather_tensor,
    gsx_metal_backend_buffer_unary_tensor,
    gsx_metal_backend_buffer_unary_tensor_inplace,
    gsx_metal_backend_buffer_unary_reduce_tensor,
    gsx_metal_backend_buffer_binary_reduce_tensor,
    gsx_metal_backend_buffer_clamp_inplace_tensor,
    gsx_metal_backend_buffer_image_convert_colorspace,
    gsx_metal_backend_buffer_image_convert_storage_format,
    gsx_metal_backend_buffer_image_convert_data_type
};

gsx_error gsx_metal_backend_query_unary_reduce_workspace_size(
    gsx_backend_t backend,
    gsx_backend_buffer_type_class workspace_buffer_type,
    gsx_data_type data_type,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
)
{
    gsx_backend_tensor_view x_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_size_t outer_count = 0;
    gsx_size_t reduce_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    (void)backend;
    (void)workspace_buffer_type;

    if(x_shape == NULL || out_shape == NULL || out_workspace_size_bytes == NULL || out_workspace_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce workspace query inputs must be non-null");
    }
    if(data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal unary_reduce only supports float32 tensors");
    }
    switch(op) {
    case GSX_IMPL_UNARY_REDUCE_OP_SUM:
    case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
    case GSX_IMPL_UNARY_REDUCE_OP_MAX:
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary_reduce op");
    }

    x_view.data_type = data_type;
    out_view.data_type = data_type;
    x_view.size_bytes = 0;
    out_view.size_bytes = 0;
    error = gsx_metal_backend_reduce_validate_shape_contract(
        &x_view,
        &out_view,
        x_rank,
        x_shape,
        out_rank,
        out_shape,
        start_axis,
        &outer_count,
        &reduce_count
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(outer_count > UINT32_MAX || reduce_count > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce launch parameters exceed Metal kernel limits");
    }

    *out_workspace_size_bytes = 0;
    *out_workspace_alignment_bytes = 0;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_query_binary_reduce_workspace_size(
    gsx_backend_t backend,
    gsx_backend_buffer_type_class workspace_buffer_type,
    gsx_data_type data_type,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
)
{
    gsx_backend_tensor_view lhs_view = { 0 };
    gsx_backend_tensor_view rhs_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_size_t outer_count_lhs = 0;
    gsx_size_t reduce_count_lhs = 0;
    gsx_size_t outer_count_rhs = 0;
    gsx_size_t reduce_count_rhs = 0;
    gsx_size_t lhs_elements = 1;
    gsx_size_t rhs_elements = 1;
    gsx_size_t out_elements = 1;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    (void)backend;
    (void)workspace_buffer_type;

    if(lhs_shape == NULL || rhs_shape == NULL || out_shape == NULL || out_workspace_size_bytes == NULL
        || out_workspace_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary reduce workspace query inputs must be non-null");
    }
    if(data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal binary_reduce only supports float32 tensors");
    }
    switch(op) {
    case GSX_IMPL_BINARY_REDUCE_OP_MSE:
    case GSX_IMPL_BINARY_REDUCE_OP_MAE:
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown binary_reduce op");
    }
    if(rhs_rank != lhs_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs_rank and rhs_rank must match");
    }
    for(dim = 0; dim < lhs_rank; ++dim) {
        if(lhs_shape[dim] != rhs_shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs and rhs shape must match");
        }
        if(gsx_size_mul_overflows(lhs_elements, (gsx_size_t)lhs_shape[dim], &lhs_elements)
            || gsx_size_mul_overflows(rhs_elements, (gsx_size_t)rhs_shape[dim], &rhs_elements)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "binary reduce element count overflows");
        }
    }
    for(dim = 0; dim < out_rank; ++dim) {
        if(gsx_size_mul_overflows(out_elements, (gsx_size_t)out_shape[dim], &out_elements)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "out element count overflows");
        }
    }

    lhs_view.data_type = data_type;
    rhs_view.data_type = data_type;
    out_view.data_type = data_type;
    if(gsx_size_mul_overflows(lhs_elements, sizeof(float), &lhs_view.size_bytes)
        || gsx_size_mul_overflows(rhs_elements, sizeof(float), &rhs_view.size_bytes)
        || gsx_size_mul_overflows(out_elements, sizeof(float), &out_view.size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce tensor byte size overflows");
    }
    error = gsx_metal_backend_reduce_validate_shape_contract(
        &lhs_view,
        &out_view,
        lhs_rank,
        lhs_shape,
        out_rank,
        out_shape,
        start_axis,
        &outer_count_lhs,
        &reduce_count_lhs
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_reduce_validate_shape_contract(
        &rhs_view,
        &out_view,
        rhs_rank,
        rhs_shape,
        out_rank,
        out_shape,
        start_axis,
        &outer_count_rhs,
        &reduce_count_rhs
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(outer_count_lhs != outer_count_rhs || reduce_count_lhs != reduce_count_rhs) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs and rhs reduce metadata must match");
    }
    if(outer_count_lhs > UINT32_MAX || reduce_count_lhs > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce launch parameters exceed Metal kernel limits");
    }

    *out_workspace_size_bytes = 0;
    *out_workspace_alignment_bytes = 0;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_metal_backend *gsx_metal_backend_from_base(gsx_backend_t backend)
{
    return (gsx_metal_backend *)backend;
}

gsx_metal_backend_buffer_type *gsx_metal_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type)
{
    return (gsx_metal_backend_buffer_type *)buffer_type;
}

gsx_metal_backend_buffer *gsx_metal_backend_buffer_from_base(gsx_backend_buffer_t buffer)
{
    return (gsx_metal_backend_buffer *)buffer;
}

gsx_backend_buffer_type_class gsx_metal_backend_buffer_get_type_class(gsx_backend_buffer_t buffer)
{
    return gsx_metal_backend_buffer_type_from_base(buffer->buffer_type)->info.type;
}

void *gsx_metal_backend_buffer_get_host_bytes(gsx_backend_buffer_t buffer)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);

    if(metal_buffer == NULL || metal_buffer->mtl_buffer == NULL || gsx_metal_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        return NULL;
    }
    return [(id<MTLBuffer>)metal_buffer->mtl_buffer contents];
}

void gsx_metal_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes)
{
    gsx_size_t offset_bytes = 0;

    for(offset_bytes = 0; offset_bytes < total_bytes; offset_bytes += value_size_bytes) {
        memcpy((unsigned char *)dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
    }
}

bool gsx_metal_backend_f16_is_finite(uint16_t value)
{
    return ((value >> 10) & 0x1FU) != 0x1FU;
}


gsx_error gsx_metal_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count)
{
    gsx_size_t end_offset = 0;

    if(gsx_size_add_overflows(offset_bytes, byte_count, &end_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer range overflow");
    }
    if(end_offset > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "buffer range exceeds buffer size");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_reduce_validate_shape_contract(
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_reduce_count
)
{
    gsx_index_t dim = 0;
    gsx_size_t outer_count = 1;
    gsx_size_t reduce_count = 1;
    gsx_size_t x_element_count = 1;
    gsx_size_t out_element_count = 1;
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_out_bytes = 0;

    if(x_view == NULL || out_view == NULL || x_shape == NULL || out_shape == NULL || out_outer_count == NULL
        || out_reduce_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce shape inputs must be non-null");
    }
    if(start_axis < 0 || start_axis >= x_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "start_axis must be in range [0, x_rank)");
    }
    if(out_rank != start_axis + 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_rank must equal start_axis + 1");
    }
    if(out_shape[start_axis] != 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out reduced axis extent must be 1");
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(x_shape[dim] != out_shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out prefix shape must match x prefix shape");
        }
    }
    for(dim = 0; dim < x_rank; ++dim) {
        if(gsx_size_mul_overflows(x_element_count, (gsx_size_t)x_shape[dim], &x_element_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "x element count overflows");
        }
    }
    for(dim = 0; dim < out_rank; ++dim) {
        if(gsx_size_mul_overflows(out_element_count, (gsx_size_t)out_shape[dim], &out_element_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "out element count overflows");
        }
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(gsx_size_mul_overflows(outer_count, (gsx_size_t)x_shape[dim], &outer_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "outer_count overflows");
        }
    }
    for(dim = start_axis; dim < x_rank; ++dim) {
        if(gsx_size_mul_overflows(reduce_count, (gsx_size_t)x_shape[dim], &reduce_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce_count overflows");
        }
    }
    if(gsx_size_mul_overflows(x_element_count, sizeof(float), &expected_x_bytes)
        || gsx_size_mul_overflows(out_element_count, sizeof(float), &expected_out_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce tensor byte size overflows");
    }
    if((x_view->size_bytes != 0 || out_view->size_bytes != 0)
        && (expected_x_bytes != x_view->size_bytes || expected_out_bytes != out_view->size_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match provided reduce shape metadata");
    }
    *out_outer_count = outer_count;
    *out_reduce_count = reduce_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_metal_backend_init_buffer_type(
    gsx_metal_backend *metal_backend,
    gsx_metal_backend_buffer_type *buffer_type,
    gsx_backend_buffer_type_class type,
    const char *name,
    gsx_size_t alignment_bytes
)
{
    buffer_type->base.iface = &gsx_metal_backend_buffer_type_iface;
    buffer_type->base.backend = &metal_backend->base;
    buffer_type->base.live_arena_count = 0;
    buffer_type->info.backend = &metal_backend->base;
    buffer_type->info.type = type;
    buffer_type->info.name = name;
    buffer_type->info.alignment_bytes = alignment_bytes;
    buffer_type->info.max_allocation_size_bytes = 0;
}

gsx_error gsx_metal_backend_tensor_view_validate(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view)
{
    gsx_size_t tensor_end_bytes = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }
    if(gsx_size_add_overflows(tensor_view->offset_bytes, tensor_view->size_bytes, &tensor_end_bytes) || tensor_end_bytes > buffer->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor view exceeds backing buffer");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_tensor_view_check_range(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_size_t offset_bytes,
    gsx_size_t byte_count
)
{
    gsx_size_t tensor_end_bytes = 0;
    gsx_size_t absolute_offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(offset_bytes > tensor_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange offset is out of range");
    }
    if(gsx_size_add_overflows(offset_bytes, byte_count, &tensor_end_bytes) || tensor_end_bytes > tensor_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange exceeds tensor size");
    }
    if(gsx_size_add_overflows(tensor_view->offset_bytes, offset_bytes, &absolute_offset_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor subrange absolute offset overflows");
    }

    return gsx_metal_backend_buffer_check_range(buffer, absolute_offset_bytes, byte_count);
}
