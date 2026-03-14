#include "gsx-impl.h"

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

static gsx_error gsx_data_type_get_size_bytes(gsx_data_type data_type, gsx_size_t *out_size_bytes)
{
    if(out_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_size_bytes must be non-null");
    }

    switch(data_type) {
    case GSX_DATA_TYPE_F32:
    case GSX_DATA_TYPE_I32:
    case GSX_DATA_TYPE_U32:
        *out_size_bytes = 4;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_F16:
    case GSX_DATA_TYPE_BF16:
    case GSX_DATA_TYPE_U16:
    case GSX_DATA_TYPE_I16:
        *out_size_bytes = 2;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_U8:
    case GSX_DATA_TYPE_I8:
        *out_size_bytes = 1;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_U64:
    case GSX_DATA_TYPE_I64:
        *out_size_bytes = 8;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "data type is unsupported");
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

#define GSX_STUB_TENSOR_FN(name, signature) \
    GSX_API gsx_error name signature \
    { \
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, #name " is not implemented in this round"); \
    }

GSX_STUB_TENSOR_WORKSPACE_FN(gsx_tensor_exp, (gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out))
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

GSX_STUB_TENSOR_FN(gsx_gs_init, (gsx_gs_t *out_gs, const gsx_gs_desc *desc))
GSX_STUB_TENSOR_FN(gsx_gs_free, (gsx_gs_t gs))
GSX_STUB_TENSOR_FN(gsx_gs_get_info, (gsx_gs_t gs, gsx_gs_info *out_info))
GSX_STUB_TENSOR_FN(gsx_gs_get_field, (gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t *out_tensor))
GSX_STUB_TENSOR_FN(gsx_gs_zero_gradients, (gsx_gs_t gs))
GSX_STUB_TENSOR_FN(gsx_gs_set_field, (gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t tensor))
GSX_STUB_TENSOR_FN(gsx_gs_clamp_opacity, (gsx_gs_t gs, gsx_float_t min_value, gsx_float_t max_value))
GSX_STUB_TENSOR_FN(gsx_gs_set_aux_enabled, (gsx_gs_t gs, gsx_gs_aux_flags aux_flags, bool enabled))
GSX_STUB_TENSOR_FN(gsx_gs_zero_aux_tensors, (gsx_gs_t gs, gsx_gs_aux_flags aux_flags))
GSX_STUB_TENSOR_FN(gsx_gs_permute, (gsx_gs_t gs, gsx_tensor_t permutation))
GSX_STUB_TENSOR_FN(gsx_gs_prune, (gsx_gs_t gs, gsx_tensor_t keep_mask))
GSX_STUB_TENSOR_FN(gsx_gs_grow, (gsx_gs_t gs, gsx_size_t growth_count))
GSX_STUB_TENSOR_FN(gsx_gs_check_finite, (gsx_gs_t gs, gsx_gs_finite_check_result *out_result))

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
