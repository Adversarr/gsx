#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Metal optimizer implementation.
 * Adam step uses a fused Metal compute kernel.
 * Structural mutations (permute/gather/resize) migrate state tensors through
 * scratch storage before committing the new layout.
 */

typedef struct gsx_metal_optim {
    struct gsx_optim base;
    gsx_size_t *step_counts;
    gsx_tensor_t *first_moments;
    gsx_tensor_t *second_moments;
    gsx_tensor_t *scratch_first_moments;
    gsx_tensor_t *scratch_second_moments;
    gsx_arena_t state_arena;
    gsx_arena_t scratch_arena;
} gsx_metal_optim;

static gsx_error gsx_metal_optim_destroy(gsx_optim_t optim);
static gsx_error gsx_metal_optim_step_selected(gsx_optim_t optim, const bool *selected);
static gsx_error gsx_metal_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation);
static gsx_error gsx_metal_optim_gather(gsx_optim_t optim, gsx_tensor_t indices);
static gsx_error gsx_metal_optim_resize(gsx_optim_t optim, gsx_size_t new_count);
static gsx_error gsx_metal_optim_reset_all(gsx_optim_t optim);
static gsx_error gsx_metal_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index);

static const gsx_optim_i gsx_metal_optim_iface = {
    gsx_metal_optim_destroy,
    gsx_metal_optim_step_selected,
    gsx_metal_optim_permute,
    gsx_metal_optim_gather,
    gsx_metal_optim_resize,
    gsx_metal_optim_reset_all,
    gsx_metal_optim_reset_by_index
};

static bool gsx_metal_optim_buffer_is_device(gsx_backend_buffer_t buffer)
{
    return buffer != NULL && gsx_metal_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static gsx_error gsx_metal_optim_make_state_tensor_desc(gsx_tensor_t parameter, gsx_arena_t arena, gsx_tensor_desc *out_desc)
{
    if(parameter == NULL || arena == NULL || out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "parameter, arena, and out_desc must be non-null");
    }

    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->rank = parameter->rank;
    memcpy(out_desc->shape, parameter->shape, sizeof(out_desc->shape));
    out_desc->requested_alignment_bytes = parameter->requested_alignment_bytes;
    out_desc->data_type = GSX_DATA_TYPE_F32;
    out_desc->storage_format = parameter->storage_format;
    out_desc->arena = arena;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_optim_dispose_tensor_handles(gsx_tensor_t *tensors, gsx_index_t count)
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

static gsx_error gsx_metal_optim_free_tensor_handles(gsx_tensor_t *tensors, gsx_index_t count)
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

static gsx_error gsx_metal_optim_release_scratch_contents(gsx_metal_optim *cpu_optim)
{
    gsx_error error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);

    if(cpu_optim->scratch_first_moments != NULL) {
        error = gsx_metal_optim_free_tensor_handles(cpu_optim->scratch_first_moments, cpu_optim->base.param_group_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cpu_optim->scratch_second_moments != NULL) {
        error = gsx_metal_optim_free_tensor_handles(cpu_optim->scratch_second_moments, cpu_optim->base.param_group_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cpu_optim->scratch_arena != NULL) {
        error = gsx_arena_reset(cpu_optim->scratch_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_sync_backend(const gsx_metal_optim *cpu_optim)
{
    return gsx_backend_major_stream_sync(cpu_optim->base.backend);
}

static gsx_error gsx_metal_optim_validate_control_tensor(
    const gsx_metal_optim *cpu_optim,
    gsx_tensor_t tensor,
    gsx_data_type data_type,
    gsx_size_t expected_count,
    const char *tensor_name
)
{
    if(tensor == NULL || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer control tensors must be live and accessible");
    }
    if(tensor->backing_buffer->buffer_type->backend != cpu_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer control tensors must belong to the optimizer backend");
    }
    if(tensor->data_type != data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, tensor_name);
    }
    if(tensor->rank != 1 || (gsx_size_t)tensor->shape[0] != expected_count) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer control tensor has an incompatible rank or leading extent");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_download_tensor_bytes(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count)
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

    tensor_view.buffer = tensor->backing_buffer;
    tensor_view.offset_bytes = tensor->offset_bytes;
    tensor_view.size_bytes = tensor->size_bytes;
    tensor_view.effective_alignment_bytes = tensor->effective_alignment_bytes;
    tensor_view.data_type = tensor->data_type;
    return tensor->backing_buffer->iface->get_tensor(tensor->backing_buffer, &tensor_view, dst_bytes, 0, byte_count);
}

// TODO: may be reuse gsx_tensor_gather function.
static gsx_error gsx_metal_optim_gather_tensor(gsx_tensor_t src, gsx_tensor_t indices, gsx_tensor_t dst)
{
    gsx_backend_tensor_view src_view = { 0 };
    gsx_backend_tensor_view indices_view = { 0 };
    gsx_backend_tensor_view dst_view = { 0 };

    if(src == NULL || indices == NULL || dst == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src, indices, and dst must be non-null");
    }
    if(src->backing_buffer == NULL || indices->backing_buffer == NULL || dst->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer gather tensors must remain live and accessible");
    }

    src_view.buffer = src->backing_buffer;
    src_view.offset_bytes = src->offset_bytes;
    src_view.size_bytes = src->size_bytes;
    src_view.effective_alignment_bytes = src->effective_alignment_bytes;
    src_view.data_type = src->data_type;

    indices_view.buffer = indices->backing_buffer;
    indices_view.offset_bytes = indices->offset_bytes;
    indices_view.size_bytes = indices->size_bytes;
    indices_view.effective_alignment_bytes = indices->effective_alignment_bytes;
    indices_view.data_type = indices->data_type;

    dst_view.buffer = dst->backing_buffer;
    dst_view.offset_bytes = dst->offset_bytes;
    dst_view.size_bytes = dst->size_bytes;
    dst_view.effective_alignment_bytes = dst->effective_alignment_bytes;
    dst_view.data_type = dst->data_type;

    return dst->backing_buffer->iface->gather_tensor(
        dst->backing_buffer,
        &src_view,
        &indices_view,
        &dst_view,
        src->rank,
        src->shape,
        dst->rank,
        dst->shape
    );
}

static void gsx_metal_optim_destroy_incomplete(gsx_metal_optim *cpu_optim)
{
    if(cpu_optim == NULL) {
        return;
    }

    gsx_metal_optim_dispose_tensor_handles(cpu_optim->first_moments, cpu_optim->base.param_group_count);
    gsx_metal_optim_dispose_tensor_handles(cpu_optim->second_moments, cpu_optim->base.param_group_count);
    gsx_metal_optim_dispose_tensor_handles(cpu_optim->scratch_first_moments, cpu_optim->base.param_group_count);
    gsx_metal_optim_dispose_tensor_handles(cpu_optim->scratch_second_moments, cpu_optim->base.param_group_count);
    if(cpu_optim->state_arena != NULL) {
        (void)gsx_arena_free(cpu_optim->state_arena);
    }
    if(cpu_optim->scratch_arena != NULL) {
        (void)gsx_arena_free(cpu_optim->scratch_arena);
    }
    free(cpu_optim->step_counts);
    free(cpu_optim->first_moments);
    free(cpu_optim->second_moments);
    free(cpu_optim->scratch_first_moments);
    free(cpu_optim->scratch_second_moments);
    gsx_optim_base_deinit(&cpu_optim->base);
    free(cpu_optim);
}

static gsx_error gsx_metal_optim_validate_group_parameter_gradient(const gsx_metal_optim *cpu_optim, gsx_index_t index)
{
    const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[index];
    gsx_tensor_t parameter = param_group->parameter;
    gsx_tensor_t gradient = param_group->gradient;
    gsx_index_t dim = 0;

    if(parameter == NULL || gradient == NULL || parameter->backing_buffer == NULL || gradient->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer parameter and gradient tensors must remain live and accessible");
    }
    if(parameter->backing_buffer->buffer_type->backend != cpu_optim->base.backend
        || gradient->backing_buffer->buffer_type->backend != cpu_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer tensors no longer belong to the owning backend");
    }
    if(!gsx_metal_optim_buffer_is_device(parameter->backing_buffer) || !gsx_metal_optim_buffer_is_device(gradient->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer requires device-backed parameter and gradient tensors");
    }
    if(parameter->rank != gradient->rank
        || parameter->data_type != gradient->data_type
        || parameter->storage_format != gradient->storage_format
        || parameter->size_bytes != gradient->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer parameter and gradient tensors are no longer compatible");
    }
    if(parameter->data_type != GSX_DATA_TYPE_F32 || parameter->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer tensors must remain float32");
    }
    for(dim = 0; dim < parameter->rank; ++dim) {
        if(parameter->shape[dim] != gradient->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer parameter and gradient tensor shapes no longer match");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_validate_group_state_exact(const gsx_metal_optim *cpu_optim, gsx_index_t index)
{
    const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[index];
    gsx_tensor_t first_moment = cpu_optim->first_moments[index];
    gsx_tensor_t second_moment = cpu_optim->second_moments[index];
    gsx_index_t dim = 0;
    gsx_error error = gsx_metal_optim_validate_group_parameter_gradient(cpu_optim, index);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(first_moment == NULL || second_moment == NULL || first_moment->backing_buffer == NULL || second_moment->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors must remain live and accessible");
    }
    if(first_moment->backing_buffer->buffer_type->backend != cpu_optim->base.backend
        || second_moment->backing_buffer->buffer_type->backend != cpu_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer belong to the optimizer backend");
    }
    if(!gsx_metal_optim_buffer_is_device(first_moment->backing_buffer) || !gsx_metal_optim_buffer_is_device(second_moment->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer state tensors must remain device-backed");
    }
    if(first_moment->data_type != GSX_DATA_TYPE_F32
        || second_moment->data_type != GSX_DATA_TYPE_F32
        || first_moment->size_bytes != param_group->parameter->size_bytes
        || second_moment->size_bytes != param_group->parameter->size_bytes
        || first_moment->rank != param_group->parameter->rank
        || second_moment->rank != param_group->parameter->rank
        || first_moment->storage_format != param_group->parameter->storage_format
        || second_moment->storage_format != param_group->parameter->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer match the parameter tensor layout");
    }
    for(dim = 0; dim < param_group->parameter->rank; ++dim) {
        if(first_moment->shape[dim] != param_group->parameter->shape[dim]
            || second_moment->shape[dim] != param_group->parameter->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer match the parameter tensor shape");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_compute_row_bytes(gsx_tensor_t tensor, gsx_size_t *out_row_bytes)
{
    gsx_size_t leading_extent = 0;

    if(tensor == NULL || out_row_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and out_row_bytes must be non-null");
    }
    if(tensor->rank <= 0 || tensor->shape[0] <= 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer tensors must keep a positive leading extent");
    }

    leading_extent = (gsx_size_t)tensor->shape[0];
    if(tensor->size_bytes % leading_extent != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer tensor size must be divisible by the leading extent");
    }

    *out_row_bytes = tensor->size_bytes / leading_extent;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_validate_group_state_transition(
    const gsx_metal_optim *cpu_optim,
    gsx_index_t index,
    gsx_size_t *out_old_count,
    gsx_size_t *out_new_count,
    gsx_size_t *out_old_row_bytes,
    gsx_size_t *out_new_row_bytes
)
{
    const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[index];
    gsx_tensor_t first_moment = cpu_optim->first_moments[index];
    gsx_tensor_t second_moment = cpu_optim->second_moments[index];
    gsx_index_t dim = 0;
    gsx_error error = gsx_metal_optim_validate_group_parameter_gradient(cpu_optim, index);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(first_moment == NULL || second_moment == NULL || first_moment->backing_buffer == NULL || second_moment->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors must remain live and accessible");
    }
    if(first_moment->backing_buffer->buffer_type->backend != cpu_optim->base.backend
        || second_moment->backing_buffer->buffer_type->backend != cpu_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer belong to the optimizer backend");
    }
    if(!gsx_metal_optim_buffer_is_device(first_moment->backing_buffer) || !gsx_metal_optim_buffer_is_device(second_moment->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer state tensors must remain device-backed");
    }
    if(first_moment->rank != second_moment->rank
        || first_moment->data_type != second_moment->data_type
        || first_moment->storage_format != second_moment->storage_format
        || first_moment->size_bytes != second_moment->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer moment tensors must match each other");
    }
    for(dim = 0; dim < first_moment->rank; ++dim) {
        if(first_moment->shape[dim] != second_moment->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer moment tensor shapes must stay aligned");
        }
    }
    if(first_moment->data_type != GSX_DATA_TYPE_F32
        || first_moment->rank != param_group->parameter->rank
        || first_moment->storage_format != param_group->parameter->storage_format) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors must stay float32 and follow the parameter layout");
    }
    for(dim = 1; dim < param_group->parameter->rank; ++dim) {
        if(first_moment->shape[dim] != param_group->parameter->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors may only differ on the leading axis");
        }
    }

    error = gsx_metal_optim_compute_row_bytes(first_moment, out_old_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_optim_compute_row_bytes(param_group->parameter, out_new_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(*out_old_row_bytes != *out_new_row_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer match the parameter row layout");
    }

    *out_old_count = (gsx_size_t)first_moment->shape[0];
    *out_new_count = (gsx_size_t)param_group->parameter->shape[0];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_optim_cleanup_sizing_work(gsx_arena_t *arena, gsx_tensor_t *tensor)
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

static gsx_error gsx_metal_optim_compute_required_bytes(const gsx_metal_optim *cpu_optim, gsx_size_t *out_required_bytes)
{
    gsx_arena_t dry_run_arena = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_tensor_t temp_tensor = NULL;
    gsx_index_t index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_required_bytes must be non-null");
    }

    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, cpu_optim->base.state_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(index = 0; index < cpu_optim->base.param_group_count; ++index) {
        error = gsx_metal_optim_make_state_tensor_desc(cpu_optim->base.param_groups[index].parameter, dry_run_arena, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);
    gsx_metal_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
    return error;
}

static gsx_error gsx_metal_optim_allocate_state_tensors_on_arena(
    const gsx_metal_optim *cpu_optim,
    gsx_arena_t arena,
    gsx_tensor_t *first_moments,
    gsx_tensor_t *second_moments,
    bool zero_init
)
{
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_index_t index = 0;

    for(index = 0; index < cpu_optim->base.param_group_count; ++index) {
        gsx_error error = gsx_metal_optim_make_state_tensor_desc(cpu_optim->base.param_groups[index].parameter, arena, &tensor_desc);

        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
            gsx_metal_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
            return error;
        }
        error = gsx_tensor_init(&first_moments[index], &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
            gsx_metal_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
            return error;
        }
        error = gsx_tensor_init(&second_moments[index], &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_metal_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
            gsx_metal_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
            return error;
        }
        if(zero_init) {
            error = gsx_tensor_set_zero(first_moments[index]);
            if(!gsx_error_is_success(error)) {
                gsx_metal_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
                gsx_metal_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
                return error;
            }
            error = gsx_tensor_set_zero(second_moments[index]);
            if(!gsx_error_is_success(error)) {
                gsx_metal_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
                gsx_metal_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
                return error;
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_init_arena(gsx_backend_buffer_type_t buffer_type, gsx_size_t required_bytes, gsx_arena_t *out_arena)
{
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_arena must be non-null");
    }

    error = gsx_arena_init(out_arena, buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reserve(*out_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        (void)gsx_arena_free(*out_arena);
        *out_arena = NULL;
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_allocate_arrays(gsx_metal_optim *cpu_optim)
{
    gsx_index_t count = cpu_optim->base.param_group_count;

    if(count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cpu_optim->step_counts = (gsx_size_t *)calloc((size_t)count, sizeof(*cpu_optim->step_counts));
    cpu_optim->first_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cpu_optim->first_moments));
    cpu_optim->second_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cpu_optim->second_moments));
    cpu_optim->scratch_first_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cpu_optim->scratch_first_moments));
    cpu_optim->scratch_second_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cpu_optim->scratch_second_moments));
    if(cpu_optim->step_counts == NULL || cpu_optim->first_moments == NULL || cpu_optim->second_moments == NULL
        || cpu_optim->scratch_first_moments == NULL || cpu_optim->scratch_second_moments == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal optimizer metadata arrays");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_init_state_storage(gsx_metal_optim *cpu_optim)
{
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_metal_optim_compute_required_bytes(cpu_optim, &required_bytes);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_optim_init_arena(cpu_optim->base.state_buffer_type, required_bytes, &cpu_optim->state_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_optim_allocate_state_tensors_on_arena(
        cpu_optim,
        cpu_optim->state_arena,
        cpu_optim->first_moments,
        cpu_optim->second_moments,
        true
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_optim_init_arena(cpu_optim->base.state_buffer_type, required_bytes, &cpu_optim->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_prepare_scratch_for_target_layout(gsx_metal_optim *cpu_optim)
{
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_metal_optim_release_scratch_contents(cpu_optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_optim_compute_required_bytes(cpu_optim, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reserve(cpu_optim->scratch_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_optim_allocate_state_tensors_on_arena(
        cpu_optim,
        cpu_optim->scratch_arena,
        cpu_optim->scratch_first_moments,
        cpu_optim->scratch_second_moments,
        false
    );
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_optim_release_scratch_contents(cpu_optim);
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_optim_commit_scratch(gsx_metal_optim *cpu_optim)
{
    gsx_arena_t arena = cpu_optim->state_arena;
    gsx_tensor_t *tensors = cpu_optim->first_moments;

    cpu_optim->state_arena = cpu_optim->scratch_arena;
    cpu_optim->scratch_arena = arena;

    cpu_optim->first_moments = cpu_optim->scratch_first_moments;
    cpu_optim->scratch_first_moments = tensors;

    tensors = cpu_optim->second_moments;
    cpu_optim->second_moments = cpu_optim->scratch_second_moments;
    cpu_optim->scratch_second_moments = tensors;
}

static gsx_error gsx_metal_optim_fail_after_prepare(gsx_metal_optim *cpu_optim, gsx_error error)
{
    if(!gsx_error_is_success(error)) {
        (void)gsx_metal_optim_release_scratch_contents(cpu_optim);
    }
    return error;
}

gsx_error gsx_metal_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim)
{
    gsx_metal_optim *cpu_optim = NULL;
    gsx_index_t group_index = 0;
    gsx_backend_buffer_type_info state_buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_optim == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_optim and desc must be non-null");
    }

    for(group_index = 0; group_index < desc->param_group_count; ++group_index) {
        const gsx_optim_param_group_desc *param_group = &desc->param_groups[group_index];

        if(param_group->parameter == NULL || param_group->gradient == NULL
            || param_group->parameter->backing_buffer == NULL || param_group->gradient->backing_buffer == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer parameter and gradient tensors must be live and accessible");
        }
        if(!gsx_metal_optim_buffer_is_device(param_group->parameter->backing_buffer)
            || !gsx_metal_optim_buffer_is_device(param_group->gradient->backing_buffer)) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer requires device-backed parameter and gradient tensors");
        }
    }

    *out_optim = NULL;
    cpu_optim = (gsx_metal_optim *)calloc(1, sizeof(*cpu_optim));
    if(cpu_optim == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal optimizer");
    }

    error = gsx_optim_base_init(&cpu_optim->base, &gsx_metal_optim_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        gsx_metal_optim_destroy_incomplete(cpu_optim);
        return error;
    }
    error = gsx_backend_buffer_type_get_info(cpu_optim->base.state_buffer_type, &state_buffer_type_info);
    if(!gsx_error_is_success(error)) {
        gsx_metal_optim_destroy_incomplete(cpu_optim);
        return error;
    }
    if(state_buffer_type_info.type != GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        gsx_metal_optim_destroy_incomplete(cpu_optim);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer requires device-backed state tensors");
    }
    error = gsx_metal_optim_allocate_arrays(cpu_optim);
    if(!gsx_error_is_success(error)) {
        gsx_metal_optim_destroy_incomplete(cpu_optim);
        return error;
    }
    error = gsx_metal_optim_init_state_storage(cpu_optim);
    if(!gsx_error_is_success(error)) {
        gsx_metal_optim_destroy_incomplete(cpu_optim);
        return error;
    }

    *out_optim = &cpu_optim->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_destroy(gsx_optim_t optim)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(optim == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim must be non-null");
    }

    error = gsx_metal_optim_free_tensor_handles(cpu_optim->first_moments, cpu_optim->base.param_group_count);
    if(!gsx_error_is_success(error)) {
        first_error = error;
    }
    error = gsx_metal_optim_free_tensor_handles(cpu_optim->second_moments, cpu_optim->base.param_group_count);
    if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
        first_error = error;
    }
    error = gsx_metal_optim_release_scratch_contents(cpu_optim);
    if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
        first_error = error;
    }

    if(cpu_optim->state_arena != NULL) {
        error = gsx_arena_free(cpu_optim->state_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }
    if(cpu_optim->scratch_arena != NULL) {
        error = gsx_arena_free(cpu_optim->scratch_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }

    free(cpu_optim->step_counts);
    free(cpu_optim->first_moments);
    free(cpu_optim->second_moments);
    free(cpu_optim->scratch_first_moments);
    free(cpu_optim->scratch_second_moments);
    gsx_optim_base_deinit(&cpu_optim->base);
    free(cpu_optim);
    return first_error;
}

static gsx_error gsx_metal_optim_step_selected(gsx_optim_t optim, const bool *selected)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    gsx_index_t group_index = 0;

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[group_index];
        gsx_metal_adam_step_params kernel_params = { 0 };
        gsx_size_t next_step_count = 0;
        gsx_size_t element_count = 0;
        double beta1_correction = 0.0;
        double beta2_correction = 0.0;
        gsx_error error = { GSX_ERROR_SUCCESS, NULL };

        if(selected != NULL && !selected[group_index]) {
            continue;
        }

        error = gsx_metal_optim_validate_group_state_exact(cpu_optim, group_index);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(cpu_optim->step_counts[group_index] == UINT64_MAX) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer Adam step counter overflowed");
        }

        next_step_count = cpu_optim->step_counts[group_index] + 1;
        beta1_correction = 1.0 - pow((double)param_group->beta1, (double)next_step_count);
        beta2_correction = 1.0 - pow((double)param_group->beta2, (double)next_step_count);
        if(beta1_correction <= 0.0 || beta2_correction <= 0.0) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer Adam bias correction became invalid");
        }

        element_count = param_group->parameter->size_bytes / sizeof(float);
        if(element_count > UINT32_MAX) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer tensor is too large for Metal Adam kernel dispatch");
        }

        kernel_params.learning_rate = cpu_optim->base.learning_rates[group_index];
        kernel_params.beta1 = param_group->beta1;
        kernel_params.beta2 = param_group->beta2;
        kernel_params.epsilon = param_group->epsilon;
        kernel_params.weight_decay = param_group->weight_decay;
        kernel_params.max_grad = param_group->max_grad;
        kernel_params.inv_beta1_correction = (float)(1.0 / beta1_correction);
        kernel_params.inv_beta2_correction = (float)(1.0 / beta2_correction);
        kernel_params.element_count = (uint32_t)element_count;

        error = gsx_metal_backend_dispatch_adam_step(
            cpu_optim->base.backend,
            param_group->parameter,
            param_group->gradient,
            cpu_optim->first_moments[group_index],
            cpu_optim->second_moments[group_index],
            &kernel_params
        );
        if(!gsx_error_is_success(error)) {
            return error;
        }

        cpu_optim->step_counts[group_index] = next_step_count;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_reset_all(gsx_optim_t optim)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    gsx_index_t index = 0;

    for(index = 0; index < cpu_optim->base.param_group_count; ++index) {
        gsx_error error = gsx_metal_optim_reset_by_index(optim, index);

        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(index < 0 || index >= cpu_optim->base.param_group_count) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer param-group index is out of range");
    }
    if(cpu_optim->first_moments[index] == NULL || cpu_optim->second_moments[index] == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors are unavailable");
    }

    error = gsx_tensor_set_zero(cpu_optim->first_moments[index]);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cpu_optim->second_moments[index]);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cpu_optim->step_counts[index] = 0;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    bool *seen = NULL;
    int32_t *permutation_values = NULL;
    gsx_size_t expected_count = 0;
    gsx_index_t group_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        error = gsx_metal_optim_validate_group_state_exact(cpu_optim, group_index);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(group_index == 0) {
            expected_count = (gsx_size_t)cpu_optim->first_moments[group_index]->shape[0];
        } else if((gsx_size_t)cpu_optim->first_moments[group_index]->shape[0] != expected_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extent for permutation");
        }
    }

    error = gsx_metal_optim_validate_control_tensor(
        cpu_optim,
        permutation,
        GSX_DATA_TYPE_I32,
        expected_count,
        "optimizer permutation tensor must be rank-1 int32"
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }

    permutation_values = (int32_t *)calloc((size_t)expected_count, sizeof(*permutation_values));
    seen = (bool *)calloc((size_t)expected_count, sizeof(*seen));
    if(permutation_values == NULL || seen == NULL) {
        free(permutation_values);
        free(seen);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer permutation validation storage");
    }
    error = gsx_metal_optim_download_tensor_bytes(permutation, permutation_values, expected_count * sizeof(*permutation_values));
    if(!gsx_error_is_success(error)) {
        free(permutation_values);
        free(seen);
        return error;
    }
    error = gsx_metal_optim_sync_backend(cpu_optim);
    if(!gsx_error_is_success(error)) {
        free(permutation_values);
        free(seen);
        return error;
    }

    for(group_index = 0; group_index < (gsx_index_t)expected_count; ++group_index) {
        gsx_size_t src_index = 0;

        if(permutation_values[group_index] < 0) {
            free(permutation_values);
            free(seen);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer permutation entries must be non-negative");
        }
        src_index = (gsx_size_t)permutation_values[group_index];
        if(src_index >= expected_count) {
            free(permutation_values);
            free(seen);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer permutation entries must be in range");
        }
        if(seen[src_index]) {
            free(permutation_values);
            free(seen);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer permutation entries must be unique");
        }
        seen[src_index] = true;
    }
    free(seen);
    seen = NULL;
    free(permutation_values);
    permutation_values = NULL;

    error = gsx_metal_optim_prepare_scratch_for_target_layout(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        error = gsx_metal_optim_gather_tensor(
            cpu_optim->first_moments[group_index],
            permutation,
            cpu_optim->scratch_first_moments[group_index]
        );
        if(!gsx_error_is_success(error)) {
            return gsx_metal_optim_fail_after_prepare(cpu_optim, error);
        }
        error = gsx_metal_optim_gather_tensor(
            cpu_optim->second_moments[group_index],
            permutation,
            cpu_optim->scratch_second_moments[group_index]
        );
        if(!gsx_error_is_success(error)) {
            return gsx_metal_optim_fail_after_prepare(cpu_optim, error);
        }
    }

    gsx_metal_optim_commit_scratch(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_gather(gsx_optim_t optim, gsx_tensor_t indices)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    int32_t *index_values = NULL;
    gsx_size_t old_count = 0;
    gsx_size_t new_count = 0;
    gsx_index_t group_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        gsx_size_t group_old_count = 0;
        gsx_size_t group_new_count = 0;
        gsx_size_t old_row_bytes = 0;
        gsx_size_t new_row_bytes = 0;

        error = gsx_metal_optim_validate_group_state_transition(
            cpu_optim,
            group_index,
            &group_old_count,
            &group_new_count,
            &old_row_bytes,
            &new_row_bytes
        );
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(group_index == 0) {
            old_count = group_old_count;
            new_count = group_new_count;
        } else if(group_old_count != old_count || group_new_count != new_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extents for gather");
        }
    }

    error = gsx_metal_optim_validate_control_tensor(
        cpu_optim,
        indices,
        GSX_DATA_TYPE_I32,
        new_count,
        "optimizer indices tensor must be rank-1 int32"
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }

    index_values = (int32_t *)calloc((size_t)new_count, sizeof(*index_values));
    if(new_count != 0 && index_values == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer gather validation storage");
    }
    error = gsx_metal_optim_download_tensor_bytes(indices, index_values, new_count * sizeof(*index_values));
    if(!gsx_error_is_success(error)) {
        free(index_values);
        return error;
    }
    error = gsx_metal_optim_sync_backend(cpu_optim);
    if(!gsx_error_is_success(error)) {
        free(index_values);
        return error;
    }

    for(group_index = 0; group_index < (gsx_index_t)new_count; ++group_index) {
        if(index_values[group_index] < 0 || (gsx_size_t)index_values[group_index] >= old_count) {
            free(index_values);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer gather indices must be in range");
        }
    }
    free(index_values);
    index_values = NULL;

    error = gsx_metal_optim_prepare_scratch_for_target_layout(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        error = gsx_metal_optim_gather_tensor(
            cpu_optim->first_moments[group_index],
            indices,
            cpu_optim->scratch_first_moments[group_index]
        );
        if(!gsx_error_is_success(error)) {
            return gsx_metal_optim_fail_after_prepare(cpu_optim, error);
        }
        error = gsx_metal_optim_gather_tensor(
            cpu_optim->second_moments[group_index],
            indices,
            cpu_optim->scratch_second_moments[group_index]
        );
        if(!gsx_error_is_success(error)) {
            return gsx_metal_optim_fail_after_prepare(cpu_optim, error);
        }
    }

    gsx_metal_optim_commit_scratch(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_optim_resize(gsx_optim_t optim, gsx_size_t new_count)
{
    gsx_metal_optim *cpu_optim = (gsx_metal_optim *)optim;
    gsx_size_t old_count = 0;
    gsx_index_t group_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        gsx_size_t group_old_count = 0;
        gsx_size_t group_new_count = 0;
        gsx_size_t old_row_bytes = 0;
        gsx_size_t new_row_bytes = 0;

        error = gsx_metal_optim_validate_group_state_transition(
            cpu_optim,
            group_index,
            &group_old_count,
            &group_new_count,
            &old_row_bytes,
            &new_row_bytes
        );
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(group_new_count != new_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer resize expects parameter tensors to already expose the target leading extent");
        }
        if(group_index == 0) {
            old_count = group_old_count;
        } else if(group_old_count != old_count || group_new_count != new_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extents for resize");
        }
    }

    error = gsx_metal_optim_prepare_scratch_for_target_layout(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        error = gsx_tensor_resize(cpu_optim->first_moments[group_index], cpu_optim->scratch_first_moments[group_index]);
        if(!gsx_error_is_success(error)) {
            return gsx_metal_optim_fail_after_prepare(cpu_optim, error);
        }
        error = gsx_tensor_resize(cpu_optim->second_moments[group_index], cpu_optim->scratch_second_moments[group_index]);
        if(!gsx_error_is_success(error)) {
            return gsx_metal_optim_fail_after_prepare(cpu_optim, error);
        }
    }

    gsx_metal_optim_commit_scratch(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
