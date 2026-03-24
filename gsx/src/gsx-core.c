#include "gsx-impl.h"

#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_tensor gsx_tensor;

static gsx_id_t gsx_core_reset_epoch_seed = 1;

static gsx_id_t gsx_core_take_reset_epoch(void)
{
    gsx_id_t epoch = gsx_core_reset_epoch_seed;

    if(gsx_core_reset_epoch_seed == UINT64_MAX) {
        gsx_core_reset_epoch_seed = 1;
    } else {
        gsx_core_reset_epoch_seed += 1;
    }
    return epoch;
}

static bool gsx_storage_format_is_valid(gsx_storage_format storage_format)
{
    switch(storage_format) {
    case GSX_STORAGE_FORMAT_CHW:
    case GSX_STORAGE_FORMAT_HWC:
    case GSX_STORAGE_FORMAT_TILED_CHW:
        return true;
    }

    return false;
}

static gsx_data_type_flags gsx_data_type_to_flag(gsx_data_type data_type)
{
    switch(data_type) {
    case GSX_DATA_TYPE_F32:
        return GSX_DATA_TYPE_FLAG_F32;
    case GSX_DATA_TYPE_F16:
        return GSX_DATA_TYPE_FLAG_F16;
    case GSX_DATA_TYPE_U8:
        return GSX_DATA_TYPE_FLAG_U8;
    case GSX_DATA_TYPE_I32:
        return GSX_DATA_TYPE_FLAG_I32;
    case GSX_DATA_TYPE_U32:
        return GSX_DATA_TYPE_FLAG_U32;
    }

    return 0;
}

static gsx_error gsx_arena_validate_alignment(gsx_size_t alignment_bytes)
{
    if(alignment_bytes != 0 && !gsx_is_power_of_two(alignment_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "alignment must be zero or a power of two");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_arena_get_effective_alignment(
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t requested_alignment_bytes,
    gsx_size_t *out_effective_alignment_bytes
)
{
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_size_t effective_alignment_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_effective_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_effective_alignment_bytes must be non-null");
    }

    error = gsx_arena_validate_alignment(requested_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_backend_buffer_type_get_info(buffer_type, &buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    effective_alignment_bytes = buffer_type_info.alignment_bytes;
    if(requested_alignment_bytes > effective_alignment_bytes) {
        effective_alignment_bytes = requested_alignment_bytes;
    }

    *out_effective_alignment_bytes = effective_alignment_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_arena_round_capacity(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_capacity_bytes, gsx_size_t *out_capacity_bytes)
{
    if(out_capacity_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_capacity_bytes must be non-null");
    }

    return gsx_backend_buffer_type_get_alloc_size(buffer_type, requested_capacity_bytes, out_capacity_bytes);
}

static void gsx_arena_link_tensor(gsx_arena_t arena, gsx_tensor *tensor)
{
    tensor->prev_active = arena->active_tail;
    tensor->next_active = NULL;
    if(arena->active_tail != NULL) {
        arena->active_tail->next_active = tensor;
    } else {
        arena->active_head = tensor;
    }
    arena->active_tail = tensor;
    arena->active_tensor_count += 1;
}

static void gsx_arena_unlink_tensor(gsx_arena_t arena, gsx_tensor *tensor)
{
    if(tensor->prev_active != NULL) {
        tensor->prev_active->next_active = tensor->next_active;
    } else if(arena->active_head == tensor) {
        arena->active_head = tensor->next_active;
    }

    if(tensor->next_active != NULL) {
        tensor->next_active->prev_active = tensor->prev_active;
    } else if(arena->active_tail == tensor) {
        arena->active_tail = tensor->prev_active;
    }

    tensor->prev_active = NULL;
    tensor->next_active = NULL;
    if(arena->active_tensor_count != 0) {
        arena->active_tensor_count -= 1;
    }
}

static void gsx_tensor_drop_live_bytes(gsx_tensor *tensor)
{
    if(tensor->arena->used_bytes >= tensor->alloc_span_bytes) {
        tensor->arena->used_bytes -= tensor->alloc_span_bytes;
    } else {
        tensor->arena->used_bytes = 0;
    }
}

static gsx_error gsx_arena_retain_backend_lifetime(gsx_arena_t arena)
{
    if(arena == NULL || arena->buffer_type == NULL || arena->buffer_type->backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena backend lifetime inputs must be non-null");
    }

    arena->buffer_type->backend->live_arena_count += 1;
    arena->buffer_type->live_arena_count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_arena_release_backend_lifetime(gsx_arena_t arena)
{
    if(arena == NULL || arena->buffer_type == NULL || arena->buffer_type->backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena backend lifetime inputs must be non-null");
    }
    if(arena->buffer_type->backend->live_arena_count == 0 || arena->buffer_type->live_arena_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "arena lifetime counters underflow during arena free");
    }

    arena->buffer_type->backend->live_arena_count -= 1;
    arena->buffer_type->live_arena_count -= 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_validate_shape(const gsx_tensor_desc *desc)
{
    gsx_index_t dim = 0;

    if(desc->rank <= 0 || desc->rank > GSX_TENSOR_MAX_DIM) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor rank must be in range [1, GSX_TENSOR_MAX_DIM]");
    }
    for(dim = 0; dim < desc->rank; ++dim) {
        if(desc->shape[dim] <= 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor shape entries must be positive");
        }
    }
    if(!gsx_storage_format_is_valid(desc->storage_format)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor storage format is invalid");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_validate_dtype(gsx_arena_t arena, gsx_data_type data_type)
{
    gsx_data_type_flags data_type_flag = gsx_data_type_to_flag(data_type);

    if(data_type_flag == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor data type is invalid");
    }

    (void)arena;

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_validate_count(gsx_index_t tensor_count)
{
    if(tensor_count < 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_count must be non-negative");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_compute_size_bytes(const gsx_tensor_desc *desc, gsx_size_t element_size_bytes, gsx_size_t *out_size_bytes)
{
    gsx_size_t size_bytes = element_size_bytes;
    gsx_index_t dim = 0;

    if(out_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_size_bytes must be non-null");
    }

    for(dim = 0; dim < desc->rank; ++dim) {
        if(gsx_size_mul_overflows(size_bytes, (gsx_size_t)desc->shape[dim], &size_bytes)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor size overflows");
        }
    }

    *out_size_bytes = size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_validate_shape_storage_consistency(gsx_tensor_t tensor)
{
    gsx_tensor_desc desc = { 0 };
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

    desc.rank = tensor->rank;
    for(dim = 0; dim < tensor->rank; ++dim) {
        desc.shape[dim] = tensor->shape[dim];
    }

    error = gsx_tensor_compute_size_bytes(&desc, element_size_bytes, &expected_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(expected_size_bytes != tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor shape/storage metadata is inconsistent");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_arena_compute_allocation(
    gsx_arena_t arena,
    gsx_size_t requested_alignment_bytes,
    gsx_size_t size_bytes,
    gsx_size_t *out_effective_alignment_bytes,
    gsx_size_t *out_alloc_start_bytes,
    gsx_size_t *out_alloc_end_bytes,
    gsx_size_t *out_alloc_span_bytes
)
{
    gsx_size_t effective_alignment_bytes = arena->effective_alignment_bytes;
    gsx_size_t alloc_start_bytes = arena->cursor_bytes;
    gsx_size_t alloc_end_bytes = arena->cursor_bytes;

    if(out_effective_alignment_bytes == NULL || out_alloc_start_bytes == NULL || out_alloc_end_bytes == NULL || out_alloc_span_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "allocation outputs must be non-null");
    }

    if(requested_alignment_bytes != 0 && requested_alignment_bytes > effective_alignment_bytes) {
        effective_alignment_bytes = requested_alignment_bytes;
    }

    if(size_bytes == 0) {
        *out_effective_alignment_bytes = effective_alignment_bytes;
        *out_alloc_start_bytes = arena->cursor_bytes;
        *out_alloc_end_bytes = arena->cursor_bytes;
        *out_alloc_span_bytes = 0;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gsx_round_up_overflows(arena->cursor_bytes, effective_alignment_bytes, &alloc_start_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor allocation start overflows");
    }
    if(gsx_size_add_overflows(alloc_start_bytes, size_bytes, &alloc_end_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor allocation end overflows");
    }

    *out_effective_alignment_bytes = effective_alignment_bytes;
    *out_alloc_start_bytes = alloc_start_bytes;
    *out_alloc_end_bytes = alloc_end_bytes;
    *out_alloc_span_bytes = alloc_end_bytes - arena->cursor_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_arena_reserve_internal(gsx_arena_t arena, gsx_size_t capacity_bytes)
{
    gsx_size_t rounded_capacity_bytes = 0;
    gsx_backend_buffer_t new_buffer = NULL;
    gsx_backend_buffer_desc buffer_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_arena_round_capacity(arena->buffer_type, capacity_bytes, &rounded_capacity_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(rounded_capacity_bytes <= arena->capacity_bytes) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(arena->dry_run) {
        arena->capacity_bytes = rounded_capacity_bytes;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    buffer_desc.buffer_type = arena->buffer_type;
    buffer_desc.size_bytes = rounded_capacity_bytes;
    buffer_desc.alignment_bytes = arena->effective_alignment_bytes;
    error = gsx_backend_buffer_init(&new_buffer, &buffer_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(arena->backing_buffer != NULL) {
        error = gsx_backend_buffer_free(arena->backing_buffer);
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(new_buffer);
            return error;
        }
    }

    arena->backing_buffer = new_buffer;
    arena->capacity_bytes = rounded_capacity_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_require_handle(gsx_tensor_t tensor)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must be non-null");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_require_live_storage(gsx_tensor_t tensor)
{
    return gsx_tensor_require_handle(tensor);
}

static gsx_error gsx_tensor_require_accessible_storage(gsx_tensor_t tensor)
{
    gsx_error error = gsx_tensor_require_live_storage(tensor);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor->arena->dry_run) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor storage is unavailable in dry-run mode");
    }
    if(tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor backing buffer is unavailable");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_backend_tensor_view gsx_tensor_make_backend_view(gsx_tensor_t tensor)
{
    gsx_backend_tensor_view tensor_view = { 0 };

    tensor_view.buffer = tensor->backing_buffer;
    tensor_view.offset_bytes = tensor->offset_bytes;
    tensor_view.size_bytes = tensor->size_bytes;
    tensor_view.effective_alignment_bytes = tensor->effective_alignment_bytes;
    tensor_view.data_type = tensor->data_type;
    return tensor_view;
}

static gsx_error gsx_tensor_set_bytes(gsx_tensor_t tensor, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_backend_tensor_view tensor_view = gsx_tensor_make_backend_view(tensor);

    return tensor->backing_buffer->iface->set_tensor(tensor->backing_buffer, &tensor_view, src_bytes, 0, byte_count);
}

static gsx_error gsx_tensor_get_bytes(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_backend_tensor_view tensor_view = gsx_tensor_make_backend_view(tensor);

    return tensor->backing_buffer->iface->get_tensor(tensor->backing_buffer, &tensor_view, dst_bytes, 0, byte_count);
}

static bool gsx_tensors_are_compatible(gsx_tensor_t lhs, gsx_tensor_t rhs)
{
    gsx_index_t dim = 0;

    if(lhs->rank != rhs->rank || lhs->data_type != rhs->data_type || lhs->storage_format != rhs->storage_format || lhs->size_bytes != rhs->size_bytes) {
        return false;
    }
    for(dim = 0; dim < lhs->rank; ++dim) {
        if(lhs->shape[dim] != rhs->shape[dim]) {
            return false;
        }
    }
    return true;
}

static gsx_error gsx_tensors_require_same_backend(gsx_tensor_t lhs, gsx_tensor_t rhs)
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

static gsx_error gsx_scalar_bounds_validate_order(gsx_data_type data_type, const void *min_value, const void *max_value)
{
    switch(data_type) {
    case GSX_DATA_TYPE_F32:
        if(*(const float *)min_value > *(const float *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_U8:
        if(*(const uint8_t *)min_value > *(const uint8_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_U32:
        if(*(const uint32_t *)min_value > *(const uint32_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_I32:
        if(*(const int32_t *)min_value > *(const int32_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_F16:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp min/max validation does not support f16");
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor data type is invalid");
    }
}

GSX_API gsx_error gsx_arena_init(gsx_arena_t *out_arena, gsx_backend_buffer_type_t buffer_type, const gsx_arena_desc *desc)
{
    gsx_arena_desc resolved_desc = { 0 };
    struct gsx_arena *arena = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_arena == NULL || buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_arena and buffer_type must be non-null");
    }
    *out_arena = NULL;

    if(desc != NULL) {
        resolved_desc = *desc;
    }

    error = gsx_arena_validate_alignment(resolved_desc.requested_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    arena = (struct gsx_arena *)calloc(1, sizeof(*arena));
    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate arena");
    }

    arena->buffer_type = buffer_type;
    arena->requested_alignment_bytes = resolved_desc.requested_alignment_bytes;
    arena->dry_run = resolved_desc.dry_run;
    arena->reset_epoch = gsx_core_take_reset_epoch();

    error = gsx_arena_get_effective_alignment(buffer_type, resolved_desc.requested_alignment_bytes, &arena->effective_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        free(arena);
        return error;
    }

    error = gsx_arena_reserve_internal(arena, resolved_desc.initial_capacity_bytes);
    if(!gsx_error_is_success(error)) {
        free(arena);
        return error;
    }

    error = gsx_arena_retain_backend_lifetime(arena);
    if(!gsx_error_is_success(error)) {
        if(arena->backing_buffer != NULL) {
            (void)gsx_backend_buffer_free(arena->backing_buffer);
        }
        free(arena);
        return error;
    }

    *out_arena = arena;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_free(gsx_arena_t arena)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena must be non-null");
    }
    if(arena->tensor_handle_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot free arena while tensor handles still exist");
    }

    if(arena->backing_buffer != NULL) {
        error = gsx_backend_buffer_free(arena->backing_buffer);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    error = gsx_arena_release_backend_lifetime(arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    free(arena);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_get_backend(gsx_arena_t arena, gsx_backend_t *out_backend)
{
    if(arena == NULL || out_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_backend must be non-null");
    }

    *out_backend = arena->buffer_type->backend;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_get_buffer_type(gsx_arena_t arena, gsx_backend_buffer_type_t *out_buffer_type)
{
    if(arena == NULL || out_buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_buffer_type must be non-null");
    }

    *out_buffer_type = arena->buffer_type;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_get_info(gsx_arena_t arena, gsx_arena_info *out_info)
{
    if(arena == NULL || out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_info must be non-null");
    }

    out_info->capacity_bytes = arena->capacity_bytes;
    out_info->used_bytes = arena->used_bytes;
    out_info->peak_bytes = arena->peak_bytes;
    out_info->effective_alignment_bytes = arena->effective_alignment_bytes;
    out_info->active_tensor_count = arena->active_tensor_count;
    out_info->dry_run = arena->dry_run;
    out_info->buffer_type = arena->buffer_type;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_reserve(gsx_arena_t arena, gsx_size_t capacity_bytes)
{
    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena must be non-null");
    }
    if(arena->active_tensor_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot reserve arena capacity while live tensors still exist");
    }

    return gsx_arena_reserve_internal(arena, capacity_bytes);
}

GSX_API gsx_error gsx_arena_reset(gsx_arena_t arena)
{
    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena must be non-null");
    }
    if(arena->active_tensor_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot reset arena while live tensors still exist");
    }

    arena->cursor_bytes = 0;
    arena->used_bytes = 0;
    arena->peak_bytes = 0;
    arena->required_bytes = 0;
    arena->reset_epoch = gsx_core_take_reset_epoch();
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_get_mark(gsx_arena_t arena, gsx_arena_mark *out_mark)
{
    if(arena == NULL || out_mark == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_mark must be non-null");
    }

    out_mark->offset_bytes = arena->cursor_bytes;
    out_mark->reset_epoch = arena->reset_epoch;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_rewind(gsx_arena_t arena, gsx_arena_mark mark)
{
    gsx_tensor *tensor = NULL;

    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena must be non-null");
    }
    if(mark.reset_epoch != arena->reset_epoch) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena mark is stale");
    }
    if(mark.offset_bytes > arena->cursor_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena mark is beyond the current cursor");
    }

    tensor = arena->active_head;
    while(tensor != NULL) {
        if(tensor->alloc_end_bytes > mark.offset_bytes) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot rewind arena while live tensors would be invalidated");
        }
        tensor = tensor->next_active;
    }

    arena->cursor_bytes = mark.offset_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_get_required_bytes(gsx_arena_t arena, gsx_size_t *out_required_bytes)
{
    if(arena == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_required_bytes must be non-null");
    }

    *out_required_bytes = arena->required_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_arena_plan_required_bytes(
    gsx_backend_buffer_type_t buffer_type,
    const gsx_arena_desc *arena_desc,
    gsx_arena_plan_callback plan_callback,
    void *user_data,
    gsx_size_t *out_required_bytes)
{
    gsx_arena_desc resolved_desc = { 0 };
    gsx_arena_t dry_run_arena = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error cleanup_error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || plan_callback == NULL || out_required_bytes == NULL) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT,
            "buffer_type, plan_callback, and out_required_bytes must be non-null");
    }

    *out_required_bytes = 0;
    if(arena_desc != NULL) {
        resolved_desc = *arena_desc;
    }
    resolved_desc.dry_run = true;

    error = gsx_arena_init(&dry_run_arena, buffer_type, &resolved_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = plan_callback(dry_run_arena, user_data);
    if(gsx_error_is_success(error)) {
        error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);
    }

    cleanup_error = gsx_arena_free(dry_run_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return cleanup_error;
}

GSX_API gsx_error gsx_tensor_init(gsx_tensor_t *out_tensor, const gsx_tensor_desc *desc)
{
    gsx_tensor *tensor = NULL;
    gsx_size_t element_size_bytes = 0;
    gsx_size_t size_bytes = 0;
    gsx_size_t effective_alignment_bytes = 0;
    gsx_size_t alloc_start_bytes = 0;
    gsx_size_t alloc_end_bytes = 0;
    gsx_size_t alloc_span_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_tensor == NULL || desc == NULL || desc->arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor, desc, and desc->arena must be non-null");
    }
    *out_tensor = NULL;

    error = gsx_arena_validate_alignment(desc->requested_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape(desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_dtype(desc->arena, desc->data_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_data_type_get_size_bytes(desc->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_compute_size_bytes(desc, element_size_bytes, &size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_compute_allocation(
        desc->arena,
        desc->requested_alignment_bytes,
        size_bytes,
        &effective_alignment_bytes,
        &alloc_start_bytes,
        &alloc_end_bytes,
        &alloc_span_bytes
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(alloc_end_bytes > desc->arena->required_bytes) {
        desc->arena->required_bytes = alloc_end_bytes;
    }

    if(alloc_end_bytes > desc->arena->capacity_bytes) {
        bool can_grow_liveness = desc->arena->dry_run || desc->arena->active_tensor_count == 0;

        if(can_grow_liveness) {
            error = gsx_arena_reserve_internal(desc->arena, alloc_end_bytes);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        } else {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "arena capacity is insufficient for tensor allocation");
        }
    }

    tensor = (gsx_tensor *)calloc(1, sizeof(*tensor));
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate tensor handle");
    }

    tensor->arena = desc->arena;
    tensor->backing_buffer = desc->arena->backing_buffer;
    tensor->offset_bytes = alloc_start_bytes;
    tensor->size_bytes = size_bytes;
    tensor->alloc_span_bytes = alloc_span_bytes;
    tensor->requested_alignment_bytes = desc->requested_alignment_bytes;
    tensor->effective_alignment_bytes = effective_alignment_bytes;
    tensor->alloc_start_bytes = alloc_start_bytes;
    tensor->alloc_end_bytes = alloc_end_bytes;
    tensor->rank = desc->rank;
    memset(tensor->shape, 0, sizeof(tensor->shape));
    memcpy(tensor->shape, desc->shape, (size_t)desc->rank * sizeof(tensor->shape[0]));
    tensor->data_type = desc->data_type;
    tensor->storage_format = desc->storage_format;

    desc->arena->cursor_bytes = alloc_end_bytes;
    desc->arena->used_bytes += alloc_span_bytes;
    if(desc->arena->used_bytes > desc->arena->peak_bytes) {
        desc->arena->peak_bytes = desc->arena->used_bytes;
    }
    desc->arena->tensor_handle_count += 1;
    gsx_arena_link_tensor(desc->arena, tensor);

    *out_tensor = tensor;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_init_many(gsx_tensor_t *out_tensors, gsx_arena_t arena, const gsx_tensor_desc *descs, gsx_index_t tensor_count)
{
    gsx_index_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_tensor_validate_count(tensor_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(out_tensors == NULL || arena == NULL || descs == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensors, arena, and descs must be non-null");
    }

    for(index = 0; index < tensor_count; ++index) {
        out_tensors[index] = NULL;
    }

    for(index = 0; index < tensor_count; ++index) {
        gsx_tensor_desc resolved_desc = descs[index];
        resolved_desc.arena = arena;
        error = gsx_tensor_init(&out_tensors[index], &resolved_desc);
        if(!gsx_error_is_success(error)) {
            gsx_index_t rollback = 0;
            for(rollback = index; rollback > 0; --rollback) {
                if(out_tensors[rollback - 1] != NULL) {
                    (void)gsx_tensor_free(out_tensors[rollback - 1]);
                    out_tensors[rollback - 1] = NULL;
                }
            }
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_free(gsx_tensor_t tensor)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must be non-null");
    }
    if(tensor->arena->tensor_handle_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "arena tensor_handle_count underflow in tensor free");
    }

    gsx_arena_unlink_tensor(tensor->arena, tensor);
    gsx_tensor_drop_live_bytes(tensor);
    tensor->arena->tensor_handle_count -= 1;
    free(tensor);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_free_many(gsx_tensor_t *tensors, gsx_index_t tensor_count)
{
    gsx_index_t index = 0;
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_tensor_validate_count(tensor_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(tensors == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensors must be non-null");
    }

    for(index = tensor_count; index > 0; --index) {
        gsx_tensor_t tensor = tensors[index - 1];
        if(tensor == NULL) {
            continue;
        }
        error = gsx_tensor_free(tensor);
        if(gsx_error_is_success(error)) {
            tensors[index - 1] = NULL;
            continue;
        }
        if(gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }

    return first_error;
}

typedef struct gsx_tensor_plan_required_bytes_payload {
    const gsx_tensor_desc *descs;
    gsx_index_t tensor_count;
} gsx_tensor_plan_required_bytes_payload;

static gsx_error gsx_tensor_plan_required_bytes_callback(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_tensor_plan_required_bytes_payload *payload = (gsx_tensor_plan_required_bytes_payload *)user_data;
    gsx_tensor_t *planned_tensors = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || payload == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dry_run_arena and payload must be non-null");
    }
    if(payload->tensor_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    planned_tensors = (gsx_tensor_t *)calloc((size_t)payload->tensor_count, sizeof(*planned_tensors));
    if(planned_tensors == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate tensor plan handle array");
    }

    error = gsx_tensor_init_many(planned_tensors, dry_run_arena, payload->descs, payload->tensor_count);
    if(gsx_error_is_success(error)) {
        error = gsx_tensor_free_many(planned_tensors, payload->tensor_count);
    }

    free(planned_tensors);
    return error;
}

GSX_API gsx_error gsx_tensor_plan_required_bytes(
    gsx_backend_buffer_type_t buffer_type,
    const gsx_arena_desc *arena_desc,
    const gsx_tensor_desc *descs,
    gsx_index_t tensor_count,
    gsx_size_t *out_required_bytes)
{
    gsx_tensor_plan_required_bytes_payload payload = { 0 };
    gsx_error error = gsx_tensor_validate_count(tensor_count);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_required_bytes must be non-null");
    }
    if(tensor_count != 0 && descs == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "descs must be non-null when tensor_count is positive");
    }

    payload.descs = descs;
    payload.tensor_count = tensor_count;
    return gsx_arena_plan_required_bytes(
        buffer_type,
        arena_desc,
        gsx_tensor_plan_required_bytes_callback,
        &payload,
        out_required_bytes);
}

GSX_API gsx_error gsx_tensor_get_desc(gsx_tensor_t tensor, gsx_tensor_desc *out_desc)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }
    error = gsx_tensor_require_live_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    out_desc->rank = tensor->rank;
    memcpy(out_desc->shape, tensor->shape, sizeof(out_desc->shape));
    out_desc->requested_alignment_bytes = tensor->requested_alignment_bytes;
    out_desc->data_type = tensor->data_type;
    out_desc->storage_format = tensor->storage_format;
    out_desc->arena = tensor->arena;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_get_info(gsx_tensor_t tensor, gsx_tensor_info *out_info)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }
    error = gsx_tensor_require_live_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    out_info->rank = tensor->rank;
    memcpy(out_info->shape, tensor->shape, sizeof(out_info->shape));
    out_info->size_bytes = tensor->size_bytes;
    out_info->effective_alignment_bytes = tensor->effective_alignment_bytes;
    out_info->data_type = tensor->data_type;
    out_info->storage_format = tensor->storage_format;
    out_info->arena = tensor->arena;
    out_info->buffer_type = tensor->arena->buffer_type;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_get_size_bytes(gsx_tensor_t tensor, gsx_size_t *out_size_bytes)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_size_bytes must be non-null");
    }
    error = gsx_tensor_require_live_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_size_bytes = tensor->size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_tensor_get_native_handle(gsx_tensor_t tensor, void **out_handle, gsx_size_t *out_offset_bytes)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_handle == NULL || out_offset_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_handle and out_offset_bytes must be non-null");
    }

    *out_handle = NULL;
    *out_offset_bytes = 0;
    error = gsx_tensor_require_accessible_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_offset_bytes = tensor->offset_bytes;
    return gsx_backend_buffer_get_native_handle(tensor->backing_buffer, out_handle);
}

GSX_API gsx_error gsx_tensor_upload(gsx_tensor_t tensor, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_error error = gsx_tensor_require_accessible_storage(tensor);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count > tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "byte_count exceeds tensor capacity");
    }
    if(byte_count != 0 && src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null when byte_count is non-zero");
    }

    return gsx_tensor_set_bytes(tensor, src_bytes, byte_count);
}

GSX_API gsx_error gsx_tensor_download(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_error error = gsx_tensor_require_accessible_storage(tensor);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count > tensor->size_bytes) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "byte_count exceeds tensor capacity");
    }
    if(byte_count != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when byte_count is non-zero");
    }

    return gsx_tensor_get_bytes(tensor, dst_bytes, byte_count);
}

GSX_API gsx_error gsx_tensor_set_zero(gsx_tensor_t tensor)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_error error = gsx_tensor_require_accessible_storage(tensor);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    tensor_view = gsx_tensor_make_backend_view(tensor);
    return tensor->backing_buffer->iface->memset_tensor(tensor->backing_buffer, &tensor_view, 0, 0, tensor->size_bytes);
}

GSX_API gsx_error gsx_tensor_copy(gsx_tensor_t src, gsx_tensor_t dst)
{
    gsx_backend_tensor_view src_view = { 0 };
    gsx_backend_tensor_view dst_view = { 0 };
    gsx_error error = gsx_tensor_require_accessible_storage(src);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_require_accessible_storage(dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(src, dst);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_tensors_are_compatible(src, dst)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "source and destination tensors are incompatible");
    }

    src_view = gsx_tensor_make_backend_view(src);
    dst_view = gsx_tensor_make_backend_view(dst);
    return dst->backing_buffer->iface->copy_tensor(dst->backing_buffer, &src_view, &dst_view);
}

GSX_API gsx_error gsx_tensor_fill(gsx_tensor_t tensor, const void *value_bytes, gsx_size_t value_size_bytes)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_size_t element_size_bytes = 0;
    gsx_error error = gsx_tensor_require_accessible_storage(tensor);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_data_type_get_size_bytes(tensor->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(value_size_bytes != element_size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_size_bytes must match the tensor element size");
    }
    if(tensor->size_bytes != 0 && value_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_bytes must be non-null when the tensor is non-empty");
    }

    tensor_view = gsx_tensor_make_backend_view(tensor);
    return tensor->backing_buffer->iface->fill_tensor(tensor->backing_buffer, &tensor_view, value_bytes, value_size_bytes);
}

GSX_API gsx_error gsx_tensor_check_finite(gsx_tensor_t tensor, bool *out_is_finite)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_is_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_is_finite must be non-null");
    }
    *out_is_finite = true;

    error = gsx_tensor_require_accessible_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    tensor_view = gsx_tensor_make_backend_view(tensor);
    return tensor->backing_buffer->iface->check_finite_tensor(tensor->backing_buffer, &tensor_view, out_is_finite);
}

GSX_API gsx_error gsx_tensor_gather(gsx_tensor_t x, gsx_tensor_t index, gsx_tensor_t out)
{
    gsx_backend_tensor_view x_view = { 0 };
    gsx_backend_tensor_view index_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_index_t dim = 0;

    if(x == NULL || index == NULL || out == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x, index, and out must be non-null");
    }

    error = gsx_tensor_require_accessible_storage(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_require_accessible_storage(index);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_require_accessible_storage(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(x, index);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(x, out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(index->rank != 1 || index->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index must be a rank-1 int32 tensor");
    }
    if(x->rank != out->rank || x->rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must have the same rank and rank must be at least 1");
    }
    if(index->shape[0] != out->shape[0]) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index length must match out leading dimension");
    }
    if(x->data_type != out->data_type || x->storage_format != out->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must use the same data_type and storage_format");
    }
    if(x == out) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must not alias");
    }
    for(dim = 1; dim < x->rank; ++dim) {
        if(x->shape[dim] != out->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out trailing dimensions must match");
        }
    }
    error = gsx_tensor_validate_shape_storage_consistency(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(index);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    x_view = gsx_tensor_make_backend_view(x);
    index_view = gsx_tensor_make_backend_view(index);
    out_view = gsx_tensor_make_backend_view(out);
    return out->backing_buffer->iface->gather_tensor(
        out->backing_buffer, &x_view, &index_view, &out_view, x->rank, x->shape, out->rank, out->shape);
}

GSX_API gsx_error gsx_tensor_resize(gsx_tensor_t x, gsx_tensor_t out)
{
    gsx_backend_tensor_view x_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_index_t dim = 0;
    gsx_size_t element_size_bytes = 0;
    gsx_size_t row_bytes = 0;
    gsx_size_t rows_to_copy = 0;
    gsx_size_t copy_size_bytes = 0;

    if(x == NULL || out == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must be non-null");
    }

    error = gsx_tensor_require_accessible_storage(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_require_accessible_storage(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(x, out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(x->rank != out->rank || x->rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must have the same rank and rank must be at least 1");
    }
    if(x->data_type != out->data_type || x->storage_format != out->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must use the same data_type and storage_format");
    }
    for(dim = 1; dim < x->rank; ++dim) {
        if(x->shape[dim] != out->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out trailing dimensions must match");
        }
    }

    if(x == out) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_data_type_get_size_bytes(x->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    row_bytes = element_size_bytes;
    for(dim = 1; dim < x->rank; ++dim) {
        if(gsx_size_mul_overflows(row_bytes, (gsx_size_t)x->shape[dim], &row_bytes)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "resize row byte size overflows");
        }
    }
    rows_to_copy = x->shape[0] < out->shape[0] ? (gsx_size_t)x->shape[0] : (gsx_size_t)out->shape[0];
    if(gsx_size_mul_overflows(rows_to_copy, row_bytes, &copy_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "resize copy byte size overflows");
    }

    error = gsx_tensor_set_zero(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(copy_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(copy_size_bytes == x->size_bytes && copy_size_bytes == out->size_bytes) {
        return gsx_tensor_copy(x, out);
    }

    x_view = gsx_tensor_make_backend_view(x);
    out_view = gsx_tensor_make_backend_view(out);
    x_view.size_bytes = copy_size_bytes;
    out_view.size_bytes = copy_size_bytes;
    return out->backing_buffer->iface->copy_tensor(out->backing_buffer, &x_view, &out_view);
}

static const char *gsx_tensor_unary_op_name(gsx_impl_unary_op op)
{
    switch(op) {
    case GSX_IMPL_UNARY_OP_EXP:
        return "exp";
    case GSX_IMPL_UNARY_OP_SIGMOID:
        return "sigmoid";
    case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE:
        return "sigmoid_derivative";
    case GSX_IMPL_UNARY_OP_ABS:
        return "abs";
    default:
        return "unknown_unary_op";
    }
}

static gsx_error gsx_tensor_dispatch_unary_outplace(
    gsx_tensor_t x,
    gsx_tensor_t out,
    gsx_impl_unary_op op
)
{
    gsx_backend_tensor_view x_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(x == NULL || out == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must be non-null");
    }

    error = gsx_tensor_require_accessible_storage(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_require_accessible_storage(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(x, out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_tensors_are_compatible(x, out)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must be shape-compatible");
    }
    error = gsx_tensor_validate_shape_storage_consistency(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out->backing_buffer->iface->unary_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend unary_tensor is not available");
    }

    x_view = gsx_tensor_make_backend_view(x);
    out_view = gsx_tensor_make_backend_view(out);
    error = out->backing_buffer->iface->unary_tensor(out->backing_buffer, &x_view, &out_view, x->rank, x->shape, op);
    if(error.code == GSX_ERROR_NOT_SUPPORTED && error.message == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, gsx_tensor_unary_op_name(op));
    }
    return error;
}

static gsx_error gsx_tensor_dispatch_unary_inplace(
    gsx_tensor_t x,
    gsx_impl_unary_op op
)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(x == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x must be non-null");
    }

    error = gsx_tensor_require_accessible_storage(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(x->backing_buffer->iface->unary_tensor_inplace == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend unary_tensor_inplace is not available");
    }

    tensor_view = gsx_tensor_make_backend_view(x);
    error = x->backing_buffer->iface->unary_tensor_inplace(x->backing_buffer, &tensor_view, op);
    if(error.code == GSX_ERROR_NOT_SUPPORTED && error.message == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, gsx_tensor_unary_op_name(op));
    }
    return error;
}

GSX_API gsx_error gsx_tensor_exp(gsx_tensor_t x, gsx_tensor_t out)
{
    return gsx_tensor_dispatch_unary_outplace(x, out, GSX_IMPL_UNARY_OP_EXP);
}

GSX_API gsx_error gsx_tensor_sigmoid(gsx_tensor_t x, gsx_tensor_t out)
{
    return gsx_tensor_dispatch_unary_outplace(x, out, GSX_IMPL_UNARY_OP_SIGMOID);
}

GSX_API gsx_error gsx_tensor_sigmoid_derivative(gsx_tensor_t x, gsx_tensor_t out)
{
    return gsx_tensor_dispatch_unary_outplace(x, out, GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE);
}

GSX_API gsx_error gsx_tensor_abs(gsx_tensor_t x, gsx_tensor_t out)
{
    return gsx_tensor_dispatch_unary_outplace(x, out, GSX_IMPL_UNARY_OP_ABS);
}

GSX_API gsx_error gsx_tensor_exp_inplace(gsx_tensor_t x)
{
    return gsx_tensor_dispatch_unary_inplace(x, GSX_IMPL_UNARY_OP_EXP);
}

GSX_API gsx_error gsx_tensor_sigmoid_inplace(gsx_tensor_t x)
{
    return gsx_tensor_dispatch_unary_inplace(x, GSX_IMPL_UNARY_OP_SIGMOID);
}

GSX_API gsx_error gsx_tensor_sigmoid_derivative_inplace(gsx_tensor_t x)
{
    return gsx_tensor_dispatch_unary_inplace(x, GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE);
}

GSX_API gsx_error gsx_tensor_abs_inplace(gsx_tensor_t x)
{
    return gsx_tensor_dispatch_unary_inplace(x, GSX_IMPL_UNARY_OP_ABS);
}

GSX_API gsx_error gsx_tensor_clamp_inplace(gsx_tensor_t x, void *min_value, void *max_value)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(x == NULL || min_value == NULL || max_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x, min_value, and max_value must be non-null");
    }

    error = gsx_tensor_require_accessible_storage(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_scalar_bounds_validate_order(x->data_type, min_value, max_value);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(x->backing_buffer->iface->clamp_inplace_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend clamp_inplace_tensor is not available");
    }

    tensor_view = gsx_tensor_make_backend_view(x);
    return x->backing_buffer->iface->clamp_inplace_tensor(x->backing_buffer, &tensor_view, min_value, max_value);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

static const char *gsx_tensor_unary_reduce_op_name(gsx_impl_unary_reduce_op op)
{
    switch(op) {
    case GSX_IMPL_UNARY_REDUCE_OP_SUM:
        return "sum";
    case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
        return "mean";
    case GSX_IMPL_UNARY_REDUCE_OP_MAX:
        return "max";
    default:
        return "unknown_unary_reduce_op";
    }
}

static const char *gsx_tensor_binary_reduce_op_name(gsx_impl_binary_reduce_op op)
{
    switch(op) {
    case GSX_IMPL_BINARY_REDUCE_OP_MSE:
        return "mse";
    case GSX_IMPL_BINARY_REDUCE_OP_MAE:
        return "mae";
    default:
        return "unknown_binary_reduce_op";
    }
}

static gsx_error gsx_tensor_reduce_validate_backend_with_workspace(gsx_arena_t arena, gsx_tensor_t lhs, gsx_tensor_t rhs)
{
    gsx_backend_t workspace_backend = NULL;
    gsx_backend_t lhs_backend = NULL;
    gsx_backend_t rhs_backend = NULL;

    if(arena == NULL || arena->buffer_type == NULL || arena->buffer_type->backend == NULL || lhs == NULL || rhs == NULL
        || lhs->arena == NULL || lhs->arena->buffer_type == NULL || rhs->arena == NULL || rhs->arena->buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "workspace and tensors must be non-null");
    }

    workspace_backend = arena->buffer_type->backend;
    lhs_backend = lhs->arena->buffer_type->backend;
    rhs_backend = rhs->arena->buffer_type->backend;
    if(workspace_backend != lhs_backend || workspace_backend != rhs_backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "workspace arena and tensors must belong to the same backend");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_unary_reduce_validate_shape(gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis)
{
    gsx_index_t dim = 0;

    if(tensor_in == NULL || tensor_out == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_in and tensor_out must be non-null");
    }
    if(start_axis < 0 || start_axis >= tensor_in->rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "start_axis must be in range [0, tensor_in->rank)");
    }
    if(tensor_out->rank != start_axis + 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_out rank must equal start_axis + 1");
    }
    if(tensor_in->data_type != tensor_out->data_type || tensor_in->storage_format != tensor_out->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_in and tensor_out must have the same data_type and storage_format");
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(tensor_in->shape[dim] != tensor_out->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_out prefix shape must match tensor_in before start_axis");
        }
    }
    if(tensor_out->shape[start_axis] != 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_out reduced axis extent must be 1");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_reduce_query_unary_workspace_size(
    gsx_arena_t arena,
    gsx_tensor_t tensor_in,
    gsx_tensor_t tensor_out,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
)
{
    gsx_backend_buffer_type_info workspace_buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || tensor_in == NULL || tensor_out == NULL || out_workspace_size_bytes == NULL
        || out_workspace_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce workspace query inputs must be non-null");
    }
    if(arena->buffer_type == NULL || arena->buffer_type->backend == NULL || arena->buffer_type->backend->iface == NULL
        || arena->buffer_type->backend->iface->query_unary_reduce_workspace_size == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend unary reduce workspace query is not available");
    }
    error = gsx_backend_buffer_type_get_info(arena->buffer_type, &workspace_buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return arena->buffer_type->backend->iface->query_unary_reduce_workspace_size(
        arena->buffer_type->backend,
        workspace_buffer_type_info.type,
        tensor_in->data_type,
        tensor_in->rank,
        tensor_in->shape,
        tensor_out->rank,
        tensor_out->shape,
        start_axis,
        op,
        out_workspace_size_bytes,
        out_workspace_alignment_bytes
    );
}

static gsx_error gsx_tensor_reduce_query_binary_workspace_size(
    gsx_arena_t arena,
    gsx_tensor_t lhs,
    gsx_tensor_t rhs,
    gsx_tensor_t out,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
)
{
    gsx_backend_buffer_type_info workspace_buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || lhs == NULL || rhs == NULL || out == NULL || out_workspace_size_bytes == NULL
        || out_workspace_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary reduce workspace query inputs must be non-null");
    }
    if(arena->buffer_type == NULL || arena->buffer_type->backend == NULL || arena->buffer_type->backend->iface == NULL
        || arena->buffer_type->backend->iface->query_binary_reduce_workspace_size == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend binary reduce workspace query is not available");
    }
    error = gsx_backend_buffer_type_get_info(arena->buffer_type, &workspace_buffer_type_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return arena->buffer_type->backend->iface->query_binary_reduce_workspace_size(
        arena->buffer_type->backend,
        workspace_buffer_type_info.type,
        lhs->data_type,
        lhs->rank,
        lhs->shape,
        rhs->rank,
        rhs->shape,
        out->rank,
        out->shape,
        start_axis,
        op,
        out_workspace_size_bytes,
        out_workspace_alignment_bytes
    );
}

static gsx_error gsx_tensor_reduce_plan_workspace_dry_run(
    gsx_arena_t arena,
    gsx_size_t workspace_size_bytes,
    gsx_size_t workspace_alignment_bytes
)
{
    gsx_size_t effective_alignment_bytes = 0;
    gsx_size_t alloc_start_bytes = 0;
    gsx_size_t alloc_end_bytes = 0;
    gsx_size_t alloc_span_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena must be non-null");
    }
    error = gsx_arena_validate_alignment(workspace_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_arena_compute_allocation(
        arena,
        workspace_alignment_bytes,
        workspace_size_bytes,
        &effective_alignment_bytes,
        &alloc_start_bytes,
        &alloc_end_bytes,
        &alloc_span_bytes
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(alloc_end_bytes > arena->required_bytes) {
        arena->required_bytes = alloc_end_bytes;
    }
    if(alloc_end_bytes > arena->capacity_bytes) {
        bool can_grow_liveness = arena->dry_run || arena->active_tensor_count == 0;

        if(can_grow_liveness) {
            error = gsx_arena_reserve_internal(arena, alloc_end_bytes);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        } else {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "arena capacity is insufficient for tensor allocation");
        }
    }
    (void)effective_alignment_bytes;
    (void)alloc_start_bytes;
    (void)alloc_span_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_tensor_reduce_make_workspace(
    gsx_arena_t arena,
    gsx_size_t workspace_size_bytes,
    gsx_size_t workspace_alignment_bytes,
    gsx_tensor_t *out_workspace
)
{
    gsx_tensor_desc workspace_desc = { 0 };
    gsx_size_t workspace_element_count = 0;

    if(arena == NULL || out_workspace == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_workspace must be non-null");
    }
    if(workspace_alignment_bytes != 0 && !gsx_is_power_of_two(workspace_alignment_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce workspace alignment must be a power of two");
    }
    *out_workspace = NULL;
    if(workspace_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    workspace_element_count = (workspace_size_bytes + sizeof(float) - 1) / sizeof(float);
    if(workspace_element_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce workspace size exceeds rank-1 tensor extent limit");
    }

    workspace_desc.arena = arena;
    workspace_desc.rank = 1;
    workspace_desc.shape[0] = (gsx_index_t)workspace_element_count;
    workspace_desc.data_type = GSX_DATA_TYPE_F32;
    workspace_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    workspace_desc.requested_alignment_bytes = workspace_alignment_bytes;
    return gsx_tensor_init(out_workspace, &workspace_desc);
}

static gsx_error gsx_tensor_dispatch_unary_reduce(
    gsx_arena_t arena,
    gsx_tensor_t tensor_in,
    gsx_tensor_t tensor_out,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
)
{
    gsx_tensor_t workspace = NULL;
    gsx_arena_mark workspace_mark = { 0 };
    gsx_backend_tensor_view in_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_backend_tensor_view workspace_view = { 0 };
    gsx_size_t workspace_size_bytes = 0;
    gsx_size_t workspace_alignment_bytes = 0;
    bool has_workspace_mark = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error cleanup_error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || tensor_in == NULL || tensor_out == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena, tensor_in, and tensor_out must be non-null");
    }
    error = gsx_tensor_reduce_validate_backend_with_workspace(arena, tensor_in, tensor_out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_unary_reduce_validate_shape(tensor_in, tensor_out, start_axis);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(tensor_in);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(tensor_out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_reduce_query_unary_workspace_size(
        arena, tensor_in, tensor_out, start_axis, op, &workspace_size_bytes, &workspace_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        if(error.code == GSX_ERROR_NOT_SUPPORTED && error.message == NULL) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, gsx_tensor_unary_reduce_op_name(op));
        }
        return error;
    }
    if(arena->dry_run) {
        return gsx_tensor_reduce_plan_workspace_dry_run(arena, workspace_size_bytes, workspace_alignment_bytes);
    }
    if(workspace_size_bytes != 0) {
        error = gsx_arena_get_mark(arena, &workspace_mark);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        has_workspace_mark = true;
    }
    error = gsx_tensor_require_accessible_storage(tensor_in);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_require_accessible_storage(tensor_out);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(tensor_out->backing_buffer->iface->unary_reduce_tensor == NULL) {
        error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend unary_reduce_tensor is not available");
        goto cleanup;
    }

    error = gsx_tensor_reduce_make_workspace(arena, workspace_size_bytes, workspace_alignment_bytes, &workspace);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(workspace != NULL) {
        error = gsx_tensor_require_accessible_storage(workspace);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }

    in_view = gsx_tensor_make_backend_view(tensor_in);
    out_view = gsx_tensor_make_backend_view(tensor_out);
    if(workspace != NULL) {
        workspace_view = gsx_tensor_make_backend_view(workspace);
    }
    error = tensor_out->backing_buffer->iface->unary_reduce_tensor(
        tensor_out->backing_buffer,
        &in_view,
        &out_view,
        &workspace_view,
        tensor_in->rank,
        tensor_in->shape,
        tensor_out->rank,
        tensor_out->shape,
        start_axis,
        op
    );
    if(!gsx_error_is_success(error)) {
        if(error.code == GSX_ERROR_NOT_SUPPORTED && error.message == NULL) {
            error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, gsx_tensor_unary_reduce_op_name(op));
        }
        goto cleanup;
    }
cleanup:
    if(workspace != NULL) {
        cleanup_error = gsx_tensor_free(workspace);
        if(gsx_error_is_success(error) && !gsx_error_is_success(cleanup_error)) {
            error = cleanup_error;
        }
    }
    if(has_workspace_mark) {
        cleanup_error = gsx_arena_rewind(arena, workspace_mark);
        if(gsx_error_is_success(error) && !gsx_error_is_success(cleanup_error)) {
            error = cleanup_error;
        }
    }
    return error;
}

static gsx_error gsx_tensor_dispatch_binary_reduce(
    gsx_arena_t arena,
    gsx_tensor_t lhs,
    gsx_tensor_t rhs,
    gsx_tensor_t out,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
)
{
    gsx_tensor_t workspace = NULL;
    gsx_arena_mark workspace_mark = { 0 };
    gsx_backend_tensor_view lhs_view = { 0 };
    gsx_backend_tensor_view rhs_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_backend_tensor_view workspace_view = { 0 };
    gsx_size_t workspace_size_bytes = 0;
    gsx_size_t workspace_alignment_bytes = 0;
    bool has_workspace_mark = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error cleanup_error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || lhs == NULL || rhs == NULL || out == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena, lhs, rhs, and out must be non-null");
    }
    error = gsx_tensor_reduce_validate_backend_with_workspace(arena, lhs, rhs);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(lhs, out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_tensors_are_compatible(lhs, rhs)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs and rhs must be shape-compatible");
    }
    error = gsx_tensor_unary_reduce_validate_shape(lhs, out, start_axis);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(lhs);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(rhs);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_validate_shape_storage_consistency(out);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_reduce_query_binary_workspace_size(
        arena, lhs, rhs, out, start_axis, op, &workspace_size_bytes, &workspace_alignment_bytes);
    if(!gsx_error_is_success(error)) {
        if(error.code == GSX_ERROR_NOT_SUPPORTED && error.message == NULL) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, gsx_tensor_binary_reduce_op_name(op));
        }
        return error;
    }
    if(arena->dry_run) {
        return gsx_tensor_reduce_plan_workspace_dry_run(arena, workspace_size_bytes, workspace_alignment_bytes);
    }
    if(workspace_size_bytes != 0) {
        error = gsx_arena_get_mark(arena, &workspace_mark);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        has_workspace_mark = true;
    }
    error = gsx_tensor_require_accessible_storage(lhs);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_require_accessible_storage(rhs);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_tensor_require_accessible_storage(out);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(out->backing_buffer->iface->binary_reduce_tensor == NULL) {
        error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend binary_reduce_tensor is not available");
        goto cleanup;
    }

    error = gsx_tensor_reduce_make_workspace(arena, workspace_size_bytes, workspace_alignment_bytes, &workspace);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    if(workspace != NULL) {
        error = gsx_tensor_require_accessible_storage(workspace);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
    }

    lhs_view = gsx_tensor_make_backend_view(lhs);
    rhs_view = gsx_tensor_make_backend_view(rhs);
    out_view = gsx_tensor_make_backend_view(out);
    if(workspace != NULL) {
        workspace_view = gsx_tensor_make_backend_view(workspace);
    }
    error = out->backing_buffer->iface->binary_reduce_tensor(
        out->backing_buffer,
        &lhs_view,
        &rhs_view,
        &out_view,
        &workspace_view,
        lhs->rank,
        lhs->shape,
        rhs->rank,
        rhs->shape,
        out->rank,
        out->shape,
        start_axis,
        op
    );
    if(!gsx_error_is_success(error)) {
        if(error.code == GSX_ERROR_NOT_SUPPORTED && error.message == NULL) {
            error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, gsx_tensor_binary_reduce_op_name(op));
        }
        goto cleanup;
    }
cleanup:
    if(workspace != NULL) {
        cleanup_error = gsx_tensor_free(workspace);
        if(gsx_error_is_success(error) && !gsx_error_is_success(cleanup_error)) {
            error = cleanup_error;
        }
    }
    if(has_workspace_mark) {
        cleanup_error = gsx_arena_rewind(arena, workspace_mark);
        if(gsx_error_is_success(error) && !gsx_error_is_success(cleanup_error)) {
            error = cleanup_error;
        }
    }
    return error;
}

GSX_API gsx_error gsx_tensor_sum(gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis)
{
    return gsx_tensor_dispatch_unary_reduce(arena, tensor_in, tensor_out, start_axis, GSX_IMPL_UNARY_REDUCE_OP_SUM);
}

GSX_API gsx_error gsx_tensor_mean(gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis)
{
    return gsx_tensor_dispatch_unary_reduce(arena, tensor_in, tensor_out, start_axis, GSX_IMPL_UNARY_REDUCE_OP_MEAN);
}

GSX_API gsx_error gsx_tensor_max(gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis)
{
    return gsx_tensor_dispatch_unary_reduce(arena, tensor_in, tensor_out, start_axis, GSX_IMPL_UNARY_REDUCE_OP_MAX);
}

GSX_API gsx_error gsx_tensor_mse(gsx_arena_t arena, gsx_tensor_t pred, gsx_tensor_t target, gsx_tensor_t out, gsx_index_t start_axis)
{
    return gsx_tensor_dispatch_binary_reduce(arena, pred, target, out, start_axis, GSX_IMPL_BINARY_REDUCE_OP_MSE);
}

GSX_API gsx_error gsx_tensor_mae(gsx_arena_t arena, gsx_tensor_t pred, gsx_tensor_t target, gsx_tensor_t out, gsx_index_t start_axis)
{
    return gsx_tensor_dispatch_binary_reduce(arena, pred, target, out, start_axis, GSX_IMPL_BINARY_REDUCE_OP_MAE);
}

#define GSX_GS_FIELD_COUNT ((gsx_size_t)(GSX_GS_FIELD_METRICS_ACC + 1))
#define GSX_GS_PARAM_FIELD_COUNT ((gsx_size_t)8)
#define GSX_GS_GRAD_FIELD_BEGIN ((gsx_gs_field)GSX_GS_FIELD_GRAD_MEAN3D)
#define GSX_GS_GRAD_FIELD_END ((gsx_gs_field)GSX_GS_FIELD_GRAD_SH3)

struct gsx_gs {
    gsx_arena_t arena;
    gsx_backend_t backend;
    gsx_size_t count;
    gsx_gs_aux_flags aux_flags;
    gsx_tensor_t fields[GSX_GS_FIELD_COUNT];
};

static gsx_gs_aux_flags gsx_gs_aux_known_flags(void)
{
    return GSX_GS_AUX_VISIBLE_COUNTER | GSX_GS_AUX_MAX_SCREEN_RADIUS | GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_ABSGRAD_ACC
        | GSX_GS_AUX_METRICS_ACC | GSX_GS_AUX_SH1 | GSX_GS_AUX_SH2 | GSX_GS_AUX_SH3;
}

static bool gsx_gs_field_is_valid(gsx_gs_field field)
{
    return field >= GSX_GS_FIELD_MEAN3D && field <= GSX_GS_FIELD_METRICS_ACC;
}

static gsx_gs_aux_flags gsx_gs_field_to_aux_flag(gsx_gs_field field)
{
    switch(field) {
    case GSX_GS_FIELD_SH1:
    case GSX_GS_FIELD_GRAD_SH1:
        return GSX_GS_AUX_SH1;
    case GSX_GS_FIELD_SH2:
    case GSX_GS_FIELD_GRAD_SH2:
        return GSX_GS_AUX_SH2;
    case GSX_GS_FIELD_SH3:
    case GSX_GS_FIELD_GRAD_SH3:
        return GSX_GS_AUX_SH3;
    case GSX_GS_FIELD_VISIBLE_COUNTER:
        return GSX_GS_AUX_VISIBLE_COUNTER;
    case GSX_GS_FIELD_MAX_SCREEN_RADIUS:
        return GSX_GS_AUX_MAX_SCREEN_RADIUS;
    case GSX_GS_FIELD_GRAD_ACC:
        return GSX_GS_AUX_GRAD_ACC;
    case GSX_GS_FIELD_ABSGRAD_ACC:
        return GSX_GS_AUX_ABSGRAD_ACC;
    case GSX_GS_FIELD_METRICS_ACC:
        return GSX_GS_AUX_METRICS_ACC;
    default:
        return GSX_GS_AUX_NONE;
    }
}

static bool gsx_gs_field_is_aux_controlled(gsx_gs_field field)
{
    return gsx_gs_field_to_aux_flag(field) != GSX_GS_AUX_NONE;
}

static gsx_index_t gsx_gs_field_rank(gsx_gs_field field)
{
    switch(field) {
    case GSX_GS_FIELD_ROTATION:
    case GSX_GS_FIELD_GRAD_ROTATION:
        return 2;
    case GSX_GS_FIELD_MEAN3D:
    case GSX_GS_FIELD_LOGSCALE:
    case GSX_GS_FIELD_SH0:
    case GSX_GS_FIELD_GRAD_MEAN3D:
    case GSX_GS_FIELD_GRAD_LOGSCALE:
    case GSX_GS_FIELD_GRAD_SH0:
        return 2;
    case GSX_GS_FIELD_SH1:
    case GSX_GS_FIELD_GRAD_SH1:
    case GSX_GS_FIELD_SH2:
    case GSX_GS_FIELD_GRAD_SH2:
    case GSX_GS_FIELD_SH3:
    case GSX_GS_FIELD_GRAD_SH3:
        return 3;
    default:
        return 1;
    }
}

static gsx_index_t gsx_gs_field_dim1(gsx_gs_field field)
{
    switch(field) {
    case GSX_GS_FIELD_ROTATION:
    case GSX_GS_FIELD_GRAD_ROTATION:
        return 4;
    case GSX_GS_FIELD_MEAN3D:
    case GSX_GS_FIELD_LOGSCALE:
    case GSX_GS_FIELD_SH0:
    case GSX_GS_FIELD_GRAD_MEAN3D:
    case GSX_GS_FIELD_GRAD_LOGSCALE:
    case GSX_GS_FIELD_GRAD_SH0:
        return 3;
    case GSX_GS_FIELD_SH1:
    case GSX_GS_FIELD_GRAD_SH1:
        return 3;
    case GSX_GS_FIELD_SH2:
    case GSX_GS_FIELD_GRAD_SH2:
        return 5;
    case GSX_GS_FIELD_SH3:
    case GSX_GS_FIELD_GRAD_SH3:
        return 7;
    default:
        return 1;
    }
}

static gsx_index_t gsx_gs_field_dim2(gsx_gs_field field)
{
    switch(field) {
    case GSX_GS_FIELD_SH1:
    case GSX_GS_FIELD_GRAD_SH1:
    case GSX_GS_FIELD_SH2:
    case GSX_GS_FIELD_GRAD_SH2:
    case GSX_GS_FIELD_SH3:
    case GSX_GS_FIELD_GRAD_SH3:
        return 3;
    default:
        return 1;
    }
}

static gsx_error gsx_gs_validate_count(gsx_size_t count)
{
    if(count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gaussian count exceeds supported tensor index range");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_require_handle(gsx_gs_t gs)
{
    if(gs == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs must be non-null");
    }
    if(gs->arena == NULL || gs->backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs handle is not initialized");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_validate_aux_flags(gsx_gs_aux_flags aux_flags)
{
    if((aux_flags & ~gsx_gs_aux_known_flags()) != 0u) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "aux_flags contains unsupported bits");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_gs_field_enabled_by_aux_flags(gsx_gs_aux_flags aux_flags, gsx_gs_field field)
{
    if(!gsx_gs_field_is_aux_controlled(field)) {
        return true;
    }
    return (aux_flags & gsx_gs_field_to_aux_flag(field)) != 0u;
}

static void gsx_gs_collect_enabled_field_mask(gsx_gs_aux_flags aux_flags, bool *out_mask)
{
    gsx_size_t field_index = 0;

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        out_mask[field_index] = gsx_gs_field_enabled_by_aux_flags(aux_flags, (gsx_gs_field)field_index);
    }
}

static void gsx_gs_collect_live_field_mask(gsx_gs_t gs, bool *out_mask)
{
    gsx_size_t field_index = 0;

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        out_mask[field_index] = gs->fields[field_index] != NULL;
    }
}

static gsx_error gsx_gs_make_tensor_desc_for_arena(
    gsx_arena_t arena,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_tensor_desc *out_desc
)
{
    gsx_index_t rank = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and out_desc must be non-null");
    }
    error = gsx_gs_validate_count(count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gaussian count must be non-zero for tensor allocation");
    }

    memset(out_desc, 0, sizeof(*out_desc));
    rank = gsx_gs_field_rank(field);
    out_desc->arena = arena;
    out_desc->rank = rank;
    out_desc->shape[0] = (gsx_index_t)count;
    if(rank >= 2) {
        out_desc->shape[1] = gsx_gs_field_dim1(field);
    }
    if(rank >= 3) {
        out_desc->shape[2] = gsx_gs_field_dim2(field);
    }
    out_desc->data_type = GSX_DATA_TYPE_F32;
    out_desc->storage_format = GSX_STORAGE_FORMAT_CHW;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_make_tensor_desc(gsx_gs_t gs, gsx_gs_field field, gsx_size_t count, gsx_tensor_desc *out_desc)
{
    return gsx_gs_make_tensor_desc_for_arena(gs->arena, field, count, out_desc);
}

static gsx_error gsx_gs_verify_tensor_contract(gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t tensor)
{
    gsx_tensor_desc expected_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must be non-null");
    }
    if(tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must reference accessible storage");
    }
    if(tensor->backing_buffer->buffer_type->backend != gs->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor backend must match gs backend");
    }
    error = gsx_gs_make_tensor_desc(gs, field, gs->count, &expected_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor->rank != expected_desc.rank || tensor->data_type != expected_desc.data_type
        || tensor->storage_format != expected_desc.storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor rank/data_type/storage_format does not match gs field contract");
    }
    if(memcmp(tensor->shape, expected_desc.shape, sizeof(expected_desc.shape)) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor shape does not match gs field contract");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_gs_release_field_range(gsx_tensor_t *fields, gsx_size_t start, gsx_size_t end)
{
    gsx_size_t index = 0;

    for(index = start; index < end; ++index) {
        if(fields[index] != NULL) {
            gsx_tensor_free(fields[index]);
            fields[index] = NULL;
        }
    }
}

static void gsx_gs_free_new_fields(gsx_tensor_t *new_fields)
{
    gsx_size_t index = 0;
    for(index = 0; index < GSX_GS_FIELD_COUNT; ++index) {
        if(new_fields[index] != NULL) {
            gsx_tensor_free(new_fields[index]);
            new_fields[index] = NULL;
        }
    }
}

typedef enum gsx_gs_rebuild_mode {
    GSX_GS_REBUILD_MODE_COPY = 0,
    GSX_GS_REBUILD_MODE_GATHER = 1
} gsx_gs_rebuild_mode;

static void gsx_gs_cleanup_rebuild(gsx_arena_t *arena, gsx_tensor_t *fields)
{
    if(fields != NULL) {
        gsx_gs_free_new_fields(fields);
    }
    if(arena != NULL && *arena != NULL) {
        (void)gsx_arena_free(*arena);
        *arena = NULL;
    }
}

static gsx_error gsx_gs_plan_exact_layout_bytes(
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t requested_alignment_bytes,
    const bool *field_mask,
    gsx_size_t target_count,
    gsx_size_t *out_required_bytes
)
{
    gsx_arena_desc sizing_desc = { 0 };
    gsx_tensor_desc tensor_descs[GSX_GS_FIELD_COUNT] = { 0 };
    gsx_index_t tensor_count = 0;
    gsx_size_t field_index = 0;

    if(buffer_type == NULL || field_mask == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type, field_mask, and out_required_bytes must be non-null");
    }
    *out_required_bytes = 0;
    if(target_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(!field_mask[field_index]) {
            continue;
        }
        tensor_descs[tensor_count].rank = gsx_gs_field_rank((gsx_gs_field)field_index);
        tensor_descs[tensor_count].shape[0] = (gsx_index_t)target_count;
        if(tensor_descs[tensor_count].rank >= 2) {
            tensor_descs[tensor_count].shape[1] = gsx_gs_field_dim1((gsx_gs_field)field_index);
        }
        if(tensor_descs[tensor_count].rank >= 3) {
            tensor_descs[tensor_count].shape[2] = gsx_gs_field_dim2((gsx_gs_field)field_index);
        }
        tensor_descs[tensor_count].data_type = GSX_DATA_TYPE_F32;
        tensor_descs[tensor_count].storage_format = GSX_STORAGE_FORMAT_CHW;
        tensor_count += 1;
    }
    if(tensor_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    sizing_desc.requested_alignment_bytes = requested_alignment_bytes;
    sizing_desc.dry_run = true;
    return gsx_tensor_plan_required_bytes(buffer_type, &sizing_desc, tensor_descs, tensor_count, out_required_bytes);
}

static gsx_error gsx_gs_init_exact_arena(
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t requested_alignment_bytes,
    bool dry_run,
    gsx_size_t required_bytes,
    gsx_arena_t *out_arena
)
{
    gsx_arena_desc arena_desc = { 0 };

    if(buffer_type == NULL || out_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_arena must be non-null");
    }

    arena_desc.initial_capacity_bytes = dry_run ? 0 : required_bytes;
    arena_desc.requested_alignment_bytes = requested_alignment_bytes;
    arena_desc.dry_run = dry_run;
    return gsx_arena_init(out_arena, buffer_type, &arena_desc);
}

static gsx_error gsx_gs_allocate_field_set(
    gsx_arena_t arena,
    const bool *field_mask,
    gsx_size_t target_count,
    gsx_tensor_t *out_fields
)
{
    gsx_size_t field_index = 0;

    if(arena == NULL || field_mask == NULL || out_fields == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena, field_mask, and out_fields must be non-null");
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        out_fields[field_index] = NULL;
    }
    if(target_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        gsx_tensor_desc desc = { 0 };
        gsx_error error = { GSX_ERROR_SUCCESS, NULL };

        if(!field_mask[field_index]) {
            continue;
        }
        error = gsx_gs_make_tensor_desc_for_arena(arena, (gsx_gs_field)field_index, target_count, &desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(out_fields);
            return error;
        }
        error = gsx_tensor_init(&out_fields[field_index], &desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(out_fields);
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_zero_field_set_if_needed(gsx_arena_t arena, gsx_tensor_t *fields)
{
    gsx_size_t field_index = 0;

    if(arena == NULL || fields == NULL || arena->dry_run) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(fields[field_index] == NULL) {
            continue;
        }
        gsx_error error = gsx_tensor_set_zero(fields[field_index]);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_populate_rebuilt_fields(
    gsx_gs_t gs,
    gsx_tensor_t *new_fields,
    const bool *field_mask,
    gsx_size_t target_count,
    gsx_tensor_t index,
    gsx_gs_rebuild_mode rebuild_mode
)
{
    gsx_size_t field_index = 0;

    if(gs == NULL || new_fields == NULL || field_mask == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs, new_fields, and field_mask must be non-null");
    }
    if(gs->arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs arena must be live during rebuild");
    }
    if(gs->arena->dry_run) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        gsx_tensor_t src = gs->fields[field_index];
        gsx_tensor_t dst = new_fields[field_index];
        gsx_error error = { GSX_ERROR_SUCCESS, NULL };

        if(!field_mask[field_index] || dst == NULL) {
            continue;
        }
        if(src == NULL) {
            error = gsx_tensor_set_zero(dst);
        } else if(rebuild_mode == GSX_GS_REBUILD_MODE_GATHER) {
            error = gsx_tensor_gather(src, index, dst);
        } else if(target_count == gs->count) {
            error = gsx_tensor_copy(src, dst);
        } else {
            error = gsx_tensor_resize(src, dst);
        }
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_commit_rebuilt_storage(
    gsx_gs_t gs,
    gsx_arena_t *new_arena,
    gsx_tensor_t *new_fields,
    gsx_size_t target_count,
    gsx_gs_aux_flags target_aux_flags
)
{
    gsx_arena_t old_arena = NULL;
    gsx_tensor_t old_fields[GSX_GS_FIELD_COUNT] = { NULL };
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t field_index = 0;

    if(gs == NULL || new_arena == NULL || *new_arena == NULL || new_fields == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs, new_arena, and new_fields must be valid");
    }

    old_arena = gs->arena;
    memcpy(old_fields, gs->fields, sizeof(old_fields));
    memset(gs->fields, 0, sizeof(gs->fields));
    memcpy(gs->fields, new_fields, sizeof(gs->fields));
    memset(new_fields, 0, sizeof(gs->fields));
    gs->arena = *new_arena;
    *new_arena = NULL;
    gs->backend = gs->arena->buffer_type->backend;
    gs->count = target_count;
    gs->aux_flags = target_aux_flags;

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(old_fields[field_index] == NULL) {
            continue;
        }
        gsx_error error = gsx_tensor_free(old_fields[field_index]);
        if(gsx_error_is_success(first_error) && !gsx_error_is_success(error)) {
            first_error = error;
        }
    }
    if(old_arena != NULL) {
        gsx_error error = gsx_arena_free(old_arena);
        if(gsx_error_is_success(first_error) && !gsx_error_is_success(error)) {
            first_error = error;
        }
    }

    return first_error;
}

static gsx_error gsx_gs_rebuild_storage(
    gsx_gs_t gs,
    const bool *target_field_mask,
    gsx_size_t target_count,
    gsx_gs_aux_flags target_aux_flags,
    gsx_tensor_t index,
    gsx_gs_rebuild_mode rebuild_mode
)
{
    gsx_arena_t new_arena = NULL;
    gsx_tensor_t new_fields[GSX_GS_FIELD_COUNT] = { NULL };
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(gs == NULL || gs->arena == NULL || target_field_mask == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs and target_field_mask must be valid");
    }

    error = gsx_gs_plan_exact_layout_bytes(
        gs->arena->buffer_type, gs->arena->effective_alignment_bytes, target_field_mask, target_count, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_init_exact_arena(
        gs->arena->buffer_type, gs->arena->effective_alignment_bytes, gs->arena->dry_run, required_bytes, &new_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_allocate_field_set(new_arena, target_field_mask, target_count, new_fields);
    if(!gsx_error_is_success(error)) {
        gsx_gs_cleanup_rebuild(&new_arena, new_fields);
        return error;
    }
    error = gsx_gs_populate_rebuilt_fields(gs, new_fields, target_field_mask, target_count, index, rebuild_mode);
    if(!gsx_error_is_success(error)) {
        gsx_gs_cleanup_rebuild(&new_arena, new_fields);
        return error;
    }

    return gsx_gs_commit_rebuilt_storage(gs, &new_arena, new_fields, target_count, target_aux_flags);
}

GSX_API gsx_error gsx_gs_init(gsx_gs_t *out_gs, const gsx_gs_desc *desc)
{
    gsx_gs_t gs = NULL;
    bool enabled_fields[GSX_GS_FIELD_COUNT] = { false };
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_gs == NULL || desc == NULL || desc->buffer_type == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_gs, desc, and desc->buffer_type must be non-null");
    }
    *out_gs = NULL;

    error = gsx_gs_validate_aux_flags(desc->aux_flags);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_validate_count(desc->count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(desc->buffer_type->backend == NULL || desc->buffer_type->backend->provider == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "descriptor buffer_type backend is not initialized");
    }
    // if(desc->buffer_type->backend->provider->backend_type != GSX_BACKEND_TYPE_CPU) {
    //     return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "gs runtime currently supports only cpu backend");
    // }

    gs = (gsx_gs_t)calloc(1, sizeof(*gs));
    if(gs == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate gs handle");
    }
    gsx_gs_collect_enabled_field_mask(desc->aux_flags, enabled_fields);
    error = gsx_gs_plan_exact_layout_bytes(
        desc->buffer_type, desc->arena_desc.requested_alignment_bytes, enabled_fields, desc->count, &required_bytes);
    if(!gsx_error_is_success(error)) {
        free(gs);
        return error;
    }
    error = gsx_gs_init_exact_arena(
        desc->buffer_type, desc->arena_desc.requested_alignment_bytes, desc->arena_desc.dry_run, required_bytes, &gs->arena);
    if(!gsx_error_is_success(error)) {
        free(gs);
        return error;
    }
    gs->backend = gs->arena->buffer_type->backend;
    gs->count = desc->count;
    gs->aux_flags = desc->aux_flags;

    error = gsx_gs_allocate_field_set(gs->arena, enabled_fields, gs->count, gs->fields);
    if(!gsx_error_is_success(error)) {
        (void)gsx_arena_free(gs->arena);
        free(gs);
        return error;
    }
    error = gsx_gs_zero_field_set_if_needed(gs->arena, gs->fields);
    if(!gsx_error_is_success(error)) {
        gsx_gs_release_field_range(gs->fields, 0, GSX_GS_FIELD_COUNT);
        (void)gsx_arena_free(gs->arena);
        free(gs);
        return error;
    }

    *out_gs = gs;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_free(gsx_gs_t gs)
{
    gsx_size_t field_index = 0;
    gsx_error error = gsx_gs_require_handle(gs);
    gsx_error step_error = { GSX_ERROR_SUCCESS, NULL };

    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(gs->fields[field_index] != NULL) {
            step_error = gsx_tensor_free(gs->fields[field_index]);
            if(gsx_error_is_success(error) && !gsx_error_is_success(step_error)) {
                error = step_error;
            }
            if(gsx_error_is_success(step_error)) {
                gs->fields[field_index] = NULL;
            }
        }
    }

    if(gs->arena != NULL) {
        step_error = gsx_arena_free(gs->arena);
        if(gsx_error_is_success(error) && !gsx_error_is_success(step_error)) {
            error = step_error;
        }
        if(gsx_error_is_success(step_error)) {
            gs->arena = NULL;
        }
    }

    gs->backend = NULL;
    free(gs);
    return error;
}

GSX_API gsx_error gsx_gs_get_info(gsx_gs_t gs, gsx_gs_info *out_info)
{
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->arena = gs->arena;
    out_info->count = gs->count;
    out_info->aux_flags = gs->aux_flags;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_get_field(gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t *out_tensor)
{
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor must be non-null");
    }
    *out_tensor = NULL;
    if(!gsx_gs_field_is_valid(field)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gs field is out of range");
    }
    if(gs->fields[field] == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field storage is not currently enabled");
    }

    *out_tensor = gs->fields[field];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_zero_gradients(gsx_gs_t gs)
{
    gsx_gs_field field = GSX_GS_GRAD_FIELD_BEGIN;
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(field = GSX_GS_GRAD_FIELD_BEGIN; field <= GSX_GS_GRAD_FIELD_END; ++field) {
        if(gs->fields[field] == NULL) {
            continue;
        }
        error = gsx_tensor_set_zero(gs->fields[field]);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_set_field(gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t tensor)
{
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_gs_field_is_valid(field)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gs field is out of range");
    }
    if(gs->fields[field] == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs field storage is not currently enabled");
    }
    error = gsx_gs_verify_tensor_contract(gs, field, tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_tensor_copy(tensor, gs->fields[field]);
}

GSX_API gsx_error gsx_gs_set_aux_enabled(gsx_gs_t gs, gsx_gs_aux_flags aux_flags, bool enabled)
{
    bool target_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_gs_aux_flags target_aux_flags = 0;
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_validate_aux_flags(aux_flags);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(aux_flags == GSX_GS_AUX_NONE) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    target_aux_flags = enabled ? (gs->aux_flags | aux_flags) : (gs->aux_flags & ~aux_flags);
    if(target_aux_flags == gs->aux_flags) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gs->count == 0) {
        gs->aux_flags = target_aux_flags;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_gs_collect_enabled_field_mask(target_aux_flags, target_field_mask);
    return gsx_gs_rebuild_storage(gs, target_field_mask, gs->count, target_aux_flags, NULL, GSX_GS_REBUILD_MODE_COPY);
}

GSX_API gsx_error gsx_gs_zero_aux_tensors(gsx_gs_t gs, gsx_gs_aux_flags aux_flags)
{
    gsx_size_t field_index = 0;
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_validate_aux_flags(aux_flags);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(aux_flags == GSX_GS_AUX_NONE) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        gsx_gs_field field = (gsx_gs_field)field_index;
        gsx_gs_aux_flags field_flag = gsx_gs_field_to_aux_flag(field);

        if(field_flag == GSX_GS_AUX_NONE || (aux_flags & field_flag) == 0u || gs->fields[field] == NULL) {
            continue;
        }
        error = gsx_tensor_set_zero(gs->fields[field]);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_permute(gsx_gs_t gs, gsx_tensor_t permutation)
{
    bool live_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(permutation == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "permutation must be non-null");
    }
    if(gs->count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot permute an empty gaussian set");
    }
    error = gsx_tensor_require_accessible_storage(permutation);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensors_require_same_backend(gs->fields[GSX_GS_FIELD_MEAN3D], permutation);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(permutation->rank != 1 || permutation->shape[0] != (gsx_index_t)gs->count || permutation->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "permutation tensor must be int32 rank-1 with length equal to gs count");
    }

    gsx_gs_collect_live_field_mask(gs, live_field_mask);
    return gsx_gs_rebuild_storage(
        gs, live_field_mask, gs->count, gs->aux_flags, permutation, GSX_GS_REBUILD_MODE_GATHER);
}

GSX_API gsx_error gsx_gs_gather(gsx_gs_t gs, gsx_tensor_t index)
{
    bool live_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_size_t selected_count = 0;
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(index == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index must be non-null");
    }
    if(gs->count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "cannot gather from an empty gaussian set");
    }
    if(index->rank != 1 || index->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index must be int32 rank-1");
    }
    if(index->shape[0] <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index length must be positive");
    }
    selected_count = (gsx_size_t)index->shape[0];
    error = gsx_gs_validate_count(selected_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    gsx_gs_collect_live_field_mask(gs, live_field_mask);
    return gsx_gs_rebuild_storage(gs, live_field_mask, selected_count, gs->aux_flags, index, GSX_GS_REBUILD_MODE_GATHER);
}

GSX_API gsx_error gsx_gs_resize(gsx_gs_t gs, gsx_size_t new_count)
{
    bool enabled_field_mask[GSX_GS_FIELD_COUNT] = { false };
    bool live_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_gs_validate_count(new_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(new_count == gs->count) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(new_count == 0) {
        memset(enabled_field_mask, 0, sizeof(enabled_field_mask));
        return gsx_gs_rebuild_storage(gs, enabled_field_mask, 0, gs->aux_flags, NULL, GSX_GS_REBUILD_MODE_COPY);
    }

    if(gs->count == 0) {
        gsx_gs_collect_enabled_field_mask(gs->aux_flags, enabled_field_mask);
        return gsx_gs_rebuild_storage(gs, enabled_field_mask, new_count, gs->aux_flags, NULL, GSX_GS_REBUILD_MODE_COPY);
    }

    gsx_gs_collect_live_field_mask(gs, live_field_mask);
    return gsx_gs_rebuild_storage(gs, live_field_mask, new_count, gs->aux_flags, NULL, GSX_GS_REBUILD_MODE_COPY);
}

GSX_API gsx_error gsx_gs_check_finite(gsx_gs_t gs, gsx_gs_finite_check_result *out_result)
{
    gsx_size_t field_index = 0;
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_result == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_result must be non-null");
    }

    out_result->is_finite = true;
    out_result->first_non_finite_field = GSX_GS_FIELD_MEAN3D;
    out_result->first_non_finite_flat_index = 0;
    out_result->non_finite_count = 0;

    for(field_index = 0; field_index < GSX_GS_PARAM_FIELD_COUNT; ++field_index) {
        gsx_gs_field field = (gsx_gs_field)field_index;
        gsx_tensor_t tensor = gs->fields[field];
        bool field_is_finite = true;

        if(tensor == NULL) {
            continue;
        }
        if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "parameter field has incompatible storage");
        }
        error = gsx_tensor_check_finite(tensor, &field_is_finite);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(!field_is_finite) {
            if(out_result->is_finite) {
                out_result->first_non_finite_field = field;
                out_result->first_non_finite_flat_index = 0;
            }
            out_result->is_finite = false;
            out_result->non_finite_count += 1;
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_float_t gsx_expf(gsx_float_t x)
{
    return expf(x);
}

GSX_API gsx_float_t gsx_sigmoid(gsx_float_t x)
{
    return 1.0f / (1.0f + expf(-x));
}

GSX_API gsx_float_t gsx_sigmoid_derivative(gsx_float_t x)
{
    gsx_float_t s = gsx_sigmoid(x);
    return s * (1.0f - s);
}

GSX_API gsx_float_t gsx_logit(gsx_float_t x)
{
    return logf(x / (1.0f - x));
}

GSX_API gsx_float_t gsx_logf(gsx_float_t x)
{
    return logf(x);
}

struct gsx_logger_state {
    gsx_log_callback log_callback;
    void * log_callback_user_data;
};

static struct gsx_logger_state g_logger_state = {gsx_log_callback_default, NULL};

static void gsx_log_internal_v(enum gsx_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

void gsx_log_internal(enum gsx_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    gsx_log_internal_v(level, format, args);
    va_end(args);
}

void gsx_log_callback_default(enum gsx_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void gsx_set_log_callback(gsx_log_callback callback, void * user_data) {
    g_logger_state.log_callback = callback;
    g_logger_state.log_callback_user_data = user_data;
}


#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
