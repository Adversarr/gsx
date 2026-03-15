#include "gsx-impl.h"

#include <math.h>
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
    case GSX_DATA_TYPE_BF16:
        return GSX_DATA_TYPE_FLAG_BF16;
    case GSX_DATA_TYPE_U8:
        return GSX_DATA_TYPE_FLAG_U8;
    case GSX_DATA_TYPE_I8:
        return GSX_DATA_TYPE_FLAG_I8;
    case GSX_DATA_TYPE_U16:
        return GSX_DATA_TYPE_FLAG_U16;
    case GSX_DATA_TYPE_I16:
        return GSX_DATA_TYPE_FLAG_I16;
    case GSX_DATA_TYPE_I32:
        return GSX_DATA_TYPE_FLAG_I32;
    case GSX_DATA_TYPE_U32:
        return GSX_DATA_TYPE_FLAG_U32;
    case GSX_DATA_TYPE_U64:
        return GSX_DATA_TYPE_FLAG_U64;
    case GSX_DATA_TYPE_I64:
        return GSX_DATA_TYPE_FLAG_I64;
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
    gsx_backend_capabilities capabilities = { 0 };
    gsx_data_type_flags data_type_flag = gsx_data_type_to_flag(data_type);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(data_type_flag == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor data type is invalid");
    }

    error = gsx_backend_get_capabilities(arena->buffer_type->backend, &capabilities);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if((capabilities.supported_data_types & data_type_flag) == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor data type is not supported by the backend");
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
    if(lhs->backing_buffer->buffer_type->backend != rhs->backing_buffer->buffer_type->backend) {
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
    case GSX_DATA_TYPE_I8:
        if(*(const int8_t *)min_value > *(const int8_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_U16:
        if(*(const uint16_t *)min_value > *(const uint16_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_I16:
        if(*(const int16_t *)min_value > *(const int16_t *)max_value) {
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
    case GSX_DATA_TYPE_U64:
        if(*(const uint64_t *)min_value > *(const uint64_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_I64:
        if(*(const int64_t *)min_value > *(const int64_t *)max_value) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "min_value must be less than or equal to max_value");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_F16:
    case GSX_DATA_TYPE_BF16:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp min/max validation does not support f16 or bf16");
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
    if(resolved_desc.growth_mode != GSX_ARENA_GROWTH_MODE_FIXED
        && resolved_desc.growth_mode != GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena growth mode is invalid");
    }

    arena = (struct gsx_arena *)calloc(1, sizeof(*arena));
    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate arena");
    }

    arena->buffer_type = buffer_type;
    arena->requested_alignment_bytes = resolved_desc.requested_alignment_bytes;
    arena->growth_mode = resolved_desc.growth_mode;
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
    out_info->growth_mode = arena->growth_mode;
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
        bool can_grow_mode = desc->arena->growth_mode == GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
        bool can_grow_liveness = desc->arena->dry_run || desc->arena->active_tensor_count == 0;

        if(can_grow_mode && can_grow_liveness) {
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

GSX_API gsx_error gsx_tensor_exp(gsx_tensor_t x, gsx_tensor_t out)
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

    x_view = gsx_tensor_make_backend_view(x);
    out_view = gsx_tensor_make_backend_view(out);
    return out->backing_buffer->iface->exp_tensor(out->backing_buffer, &x_view, &out_view, x->rank, x->shape);
}

GSX_API gsx_error gsx_tensor_clamp_inplace(gsx_arena_t arena, gsx_tensor_t x, void *min_value, void *max_value)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL || x == NULL || min_value == NULL || max_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena, x, min_value, and max_value must be non-null");
    }

    error = gsx_tensor_require_accessible_storage(x);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(arena->buffer_type == NULL || arena->buffer_type->backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena backend must be available");
    }
    if(x->backing_buffer->buffer_type->backend != arena->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena and x must belong to the same backend");
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

#define GSX_STUB_TENSOR_WORKSPACE_FN(name, signature) \
    GSX_API gsx_error name signature \
    { \
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, #name " is not implemented in this round"); \
    }

GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_sigmoid, (gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_sigmoid_grad, (gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_abs, (gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_exp_inplace, (gsx_arena_t arena, gsx_tensor_t x))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_sigmoid_inplace, (gsx_arena_t arena, gsx_tensor_t x))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_sigmoid_grad_inplace, (gsx_arena_t arena, gsx_tensor_t x))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_abs_inplace, (gsx_arena_t arena, gsx_tensor_t x))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_sum, (gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_mean, (gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_max, (gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_mse, (gsx_arena_t arena, gsx_tensor_t pred, gsx_tensor_t target, gsx_tensor_t out))
GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_mae, (gsx_arena_t arena, gsx_tensor_t pred, gsx_tensor_t target, gsx_tensor_t out))

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

static gsx_error gsx_gs_ensure_field_allocated(gsx_gs_t gs, gsx_gs_field field, gsx_size_t count, gsx_tensor_t *out_created)
{
    gsx_tensor_desc desc = { 0 };
    gsx_tensor_t tensor = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_created != NULL) {
        *out_created = NULL;
    }
    if(gs->fields[field] != NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_gs_make_tensor_desc(gs, field, count, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_init(&tensor, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(tensor);
    if(!gsx_error_is_success(error)) {
        gsx_tensor_free(tensor);
        return error;
    }
    gs->fields[field] = tensor;
    if(out_created != NULL) {
        *out_created = tensor;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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

static gsx_error gsx_gs_release_field(gsx_gs_t gs, gsx_gs_field field)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(gs->fields[field] == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    error = gsx_tensor_free(gs->fields[field]);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    gs->fields[field] = NULL;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_gs_clone_field_with_index_tensor(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_tensor_t index,
    gsx_tensor_t *out_tensor
)
{
    gsx_tensor_t src = gs->fields[field];
    gsx_tensor_desc dst_desc = { 0 };
    gsx_tensor_t dst = NULL;
    gsx_size_t selected_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor must be non-null");
    }
    *out_tensor = NULL;
    if(src == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "source field tensor is missing");
    }
    if(index == NULL || index->rank != 1 || index->shape[0] <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index must be rank-1 with positive length");
    }
    selected_count = (gsx_size_t)index->shape[0];
    error = gsx_gs_make_tensor_desc(gs, field, selected_count, &dst_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_init(&dst, &dst_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_tensor_gather(src, index, dst);
    if(!gsx_error_is_success(error)) {
        gsx_tensor_free(dst);
        return error;
    }

    *out_tensor = dst;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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

static void gsx_gs_cleanup_sizing_work(gsx_arena_t *arena, gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        (void)gsx_tensor_free(*tensor);
        *tensor = NULL;
    }
    if(arena != NULL && *arena != NULL) {
        (void)gsx_arena_free(*arena);
        *arena = NULL;
    }
}

static gsx_error gsx_gs_compute_required_bytes_for_layout(
    gsx_gs_t gs,
    const bool *field_mask,
    gsx_size_t target_count,
    gsx_size_t *out_required_bytes
)
{
    gsx_arena_t dry_run_arena = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_tensor_t temp_tensor = NULL;
    gsx_size_t field_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(field_mask == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "field_mask and out_required_bytes must be non-null");
    }

    arena_desc.requested_alignment_bytes = gs->arena->effective_alignment_bytes;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, gs->arena->buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    dry_run_arena->cursor_bytes = gs->arena->cursor_bytes;
    dry_run_arena->required_bytes = gs->arena->required_bytes;
    dry_run_arena->capacity_bytes = gs->arena->capacity_bytes;

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(!field_mask[field_index]) {
            continue;
        }
        error = gsx_gs_make_tensor_desc_for_arena(dry_run_arena, (gsx_gs_field)field_index, target_count, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_gs_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);
    gsx_gs_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
    return error;
}

static gsx_error gsx_gs_prepare_capacity_for_layout(gsx_gs_t gs, const bool *field_mask, gsx_size_t target_count)
{
    gsx_size_t required_bytes = 0;
    gsx_size_t field_index = 0;

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(field_mask[field_index]) {
            break;
        }
    }
    if(field_index == GSX_GS_FIELD_COUNT) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_error error = gsx_gs_compute_required_bytes_for_layout(gs, field_mask, target_count, &required_bytes);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(required_bytes <= gs->arena->capacity_bytes) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gs->arena->growth_mode == GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND && gs->arena->active_tensor_count == 0) {
        return gsx_arena_reserve(gs->arena, required_bytes);
    }
    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "arena capacity is insufficient for gs layout allocation");
}

GSX_API gsx_error gsx_gs_init(gsx_gs_t *out_gs, const gsx_gs_desc *desc)
{
    gsx_gs_t gs = NULL;
    bool enabled_fields[GSX_GS_FIELD_COUNT] = { false };
    gsx_size_t field_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_gs == NULL || desc == NULL || desc->arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_gs, desc, and desc->arena must be non-null");
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
    if(desc->arena->buffer_type == NULL || desc->arena->buffer_type->backend == NULL
        || desc->arena->buffer_type->backend->provider == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "descriptor arena backend is not initialized");
    }
    if(desc->arena->buffer_type->backend->provider->backend_type != GSX_BACKEND_TYPE_CPU) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "gs runtime currently supports only cpu backend");
    }

    gs = (gsx_gs_t)calloc(1, sizeof(*gs));
    if(gs == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate gs handle");
    }
    gs->arena = desc->arena;
    gs->backend = desc->arena->buffer_type->backend;
    gs->count = desc->count;
    gs->aux_flags = desc->aux_flags;

    if(gs->count != 0) {
        gsx_gs_collect_enabled_field_mask(gs->aux_flags, enabled_fields);
        error = gsx_gs_prepare_capacity_for_layout(gs, enabled_fields, gs->count);
        if(!gsx_error_is_success(error)) {
            free(gs);
            return error;
        }
        for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
            gsx_gs_field field = (gsx_gs_field)field_index;
            if(!enabled_fields[field_index]) {
                continue;
            }
            error = gsx_gs_ensure_field_allocated(gs, field, gs->count, NULL);
            if(!gsx_error_is_success(error)) {
                gsx_gs_release_field_range(gs->fields, 0, GSX_GS_FIELD_COUNT);
                free(gs);
                return error;
            }
        }
    }

    *out_gs = gs;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_free(gsx_gs_t gs)
{
    gsx_size_t field_index = 0;
    gsx_error error = gsx_gs_require_handle(gs);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(gs->fields[field_index] != NULL) {
            error = gsx_tensor_free(gs->fields[field_index]);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            gs->fields[field_index] = NULL;
        }
    }
    free(gs);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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
    gsx_tensor_t created[GSX_GS_FIELD_COUNT] = { NULL };
    bool field_mask[GSX_GS_FIELD_COUNT] = { false };
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

    if(enabled) {
        if(gs->count == 0) {
            gs->aux_flags |= aux_flags;
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
            gsx_gs_field field = (gsx_gs_field)field_index;
            gsx_gs_aux_flags field_flag = gsx_gs_field_to_aux_flag(field);
            if(field_flag == GSX_GS_AUX_NONE || (aux_flags & field_flag) == 0u || gs->fields[field_index] != NULL) {
                continue;
            }
            field_mask[field_index] = true;
        }
        error = gsx_gs_prepare_capacity_for_layout(gs, field_mask, gs->count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
            gsx_gs_field field = (gsx_gs_field)field_index;
            gsx_gs_aux_flags field_flag = gsx_gs_field_to_aux_flag(field);

            if(field_flag == GSX_GS_AUX_NONE || (aux_flags & field_flag) == 0u) {
                continue;
            }
            error = gsx_gs_ensure_field_allocated(gs, field, gs->count, &created[field_index]);
            if(!gsx_error_is_success(error)) {
                gsx_size_t rollback_index = 0;
                for(rollback_index = 0; rollback_index < GSX_GS_FIELD_COUNT; ++rollback_index) {
                    if(created[rollback_index] != NULL && gs->fields[rollback_index] == created[rollback_index]) {
                        gsx_tensor_free(created[rollback_index]);
                        gs->fields[rollback_index] = NULL;
                    }
                }
                return error;
            }
        }
        gs->aux_flags |= aux_flags;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        gsx_gs_field field = (gsx_gs_field)field_index;
        gsx_gs_aux_flags field_flag = gsx_gs_field_to_aux_flag(field);

        if(field_flag == GSX_GS_AUX_NONE || (aux_flags & field_flag) == 0u) {
            continue;
        }
        error = gsx_gs_release_field(gs, field);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    gs->aux_flags &= ~aux_flags;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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
    gsx_tensor_t new_fields[GSX_GS_FIELD_COUNT] = { NULL };
    bool live_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_size_t field_index = 0;
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
    error = gsx_gs_prepare_capacity_for_layout(gs, live_field_mask, gs->count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(gs->fields[field_index] == NULL) {
            continue;
        }
        error = gsx_gs_clone_field_with_index_tensor(gs, (gsx_gs_field)field_index, permutation, &new_fields[field_index]);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(new_fields[field_index] == NULL) {
            continue;
        }
        error = gsx_tensor_free(gs->fields[field_index]);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        gs->fields[field_index] = new_fields[field_index];
        new_fields[field_index] = NULL;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_gather(gsx_gs_t gs, gsx_tensor_t index)
{
    gsx_tensor_t new_fields[GSX_GS_FIELD_COUNT] = { NULL };
    bool live_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_size_t selected_count = 0;
    gsx_size_t field_index = 0;
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
    error = gsx_gs_prepare_capacity_for_layout(gs, live_field_mask, selected_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        gsx_tensor_desc field_desc = { 0 };

        if(gs->fields[field_index] == NULL) {
            continue;
        }
        error = gsx_gs_make_tensor_desc(gs, (gsx_gs_field)field_index, selected_count, &field_desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        error = gsx_tensor_init(&new_fields[field_index], &field_desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        error = gsx_tensor_gather(gs->fields[field_index], index, new_fields[field_index]);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(new_fields[field_index] == NULL) {
            continue;
        }
        error = gsx_tensor_free(gs->fields[field_index]);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        gs->fields[field_index] = new_fields[field_index];
        new_fields[field_index] = NULL;
    }
    gs->count = selected_count;

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_gs_resize(gsx_gs_t gs, gsx_size_t new_count)
{
    gsx_tensor_t new_fields[GSX_GS_FIELD_COUNT] = { NULL };
    bool enabled_field_mask[GSX_GS_FIELD_COUNT] = { false };
    bool live_field_mask[GSX_GS_FIELD_COUNT] = { false };
    gsx_size_t field_index = 0;
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
        for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
            if(gs->fields[field_index] == NULL) {
                continue;
            }
            error = gsx_tensor_free(gs->fields[field_index]);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            gs->fields[field_index] = NULL;
        }
        gs->count = 0;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gs->count == 0) {
        gsx_gs_collect_enabled_field_mask(gs->aux_flags, enabled_field_mask);
        error = gsx_gs_prepare_capacity_for_layout(gs, enabled_field_mask, new_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        gs->count = new_count;
        for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
            gsx_gs_field field = (gsx_gs_field)field_index;
            if(!enabled_field_mask[field_index]) {
                continue;
            }
            error = gsx_gs_ensure_field_allocated(gs, field, gs->count, NULL);
            if(!gsx_error_is_success(error)) {
                gsx_size_t rollback = 0;
                for(rollback = 0; rollback < GSX_GS_FIELD_COUNT; ++rollback) {
                    if(gs->fields[rollback] != NULL) {
                        gsx_tensor_free(gs->fields[rollback]);
                        gs->fields[rollback] = NULL;
                    }
                }
                gs->count = 0;
                return error;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_gs_collect_live_field_mask(gs, live_field_mask);
    error = gsx_gs_prepare_capacity_for_layout(gs, live_field_mask, new_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        gsx_tensor_desc field_desc = { 0 };

        if(gs->fields[field_index] == NULL) {
            continue;
        }
        error = gsx_gs_make_tensor_desc(gs, (gsx_gs_field)field_index, new_count, &field_desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        error = gsx_tensor_init(&new_fields[field_index], &field_desc);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        error = gsx_tensor_resize(gs->fields[field_index], new_fields[field_index]);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
    }

    for(field_index = 0; field_index < GSX_GS_FIELD_COUNT; ++field_index) {
        if(new_fields[field_index] == NULL) {
            continue;
        }
        error = gsx_tensor_free(gs->fields[field_index]);
        if(!gsx_error_is_success(error)) {
            gsx_gs_free_new_fields(new_fields);
            return error;
        }
        gs->fields[field_index] = new_fields[field_index];
        new_fields[field_index] = NULL;
    }
    gs->count = new_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
