#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* TODO: decouple optimizer tensor/state access from backend-private buffer internals via a backend-agnostic access layer. */

typedef struct gsx_cpu_optim {
    struct gsx_optim base;
    gsx_size_t *step_counts;
    gsx_tensor_t *first_moments;
    gsx_tensor_t *second_moments;
    gsx_tensor_t *scratch_first_moments;
    gsx_tensor_t *scratch_second_moments;
    gsx_arena_t state_arena;
    gsx_arena_t scratch_arena;
} gsx_cpu_optim;

static gsx_error gsx_cpu_optim_destroy(gsx_optim_t optim);
static gsx_error gsx_cpu_optim_step_selected(gsx_optim_t optim, const bool *selected);
static gsx_error gsx_cpu_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation);
static gsx_error gsx_cpu_optim_prune(gsx_optim_t optim, gsx_tensor_t keep_mask);
static gsx_error gsx_cpu_optim_grow(gsx_optim_t optim, gsx_size_t growth_count);
static gsx_error gsx_cpu_optim_reset_all(gsx_optim_t optim);
static gsx_error gsx_cpu_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index);

static const gsx_optim_i gsx_cpu_optim_iface = {
    gsx_cpu_optim_destroy,
    gsx_cpu_optim_step_selected,
    gsx_cpu_optim_permute,
    gsx_cpu_optim_prune,
    gsx_cpu_optim_grow,
    gsx_cpu_optim_reset_all,
    gsx_cpu_optim_reset_by_index
};

static unsigned char *gsx_cpu_optim_tensor_data_bytes(gsx_tensor_t tensor)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)tensor->backing_buffer;

    return (unsigned char *)cpu_buffer->data + (size_t)tensor->offset_bytes;
}

static float *gsx_cpu_optim_tensor_data_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cpu_optim_tensor_data_bytes(tensor);
}

static gsx_error gsx_cpu_optim_make_state_tensor_desc(gsx_tensor_t parameter, gsx_arena_t arena, gsx_tensor_desc *out_desc)
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

static void gsx_cpu_optim_dispose_tensor_handles(gsx_tensor_t *tensors, gsx_index_t count)
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

static gsx_error gsx_cpu_optim_free_tensor_handles(gsx_tensor_t *tensors, gsx_index_t count)
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

static gsx_error gsx_cpu_optim_release_scratch_contents(gsx_cpu_optim *cpu_optim)
{
    gsx_error error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);

    if(cpu_optim->scratch_first_moments != NULL) {
        error = gsx_cpu_optim_free_tensor_handles(cpu_optim->scratch_first_moments, cpu_optim->base.param_group_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cpu_optim->scratch_second_moments != NULL) {
        error = gsx_cpu_optim_free_tensor_handles(cpu_optim->scratch_second_moments, cpu_optim->base.param_group_count);
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

static void gsx_cpu_optim_destroy_incomplete(gsx_cpu_optim *cpu_optim)
{
    if(cpu_optim == NULL) {
        return;
    }

    gsx_cpu_optim_dispose_tensor_handles(cpu_optim->first_moments, cpu_optim->base.param_group_count);
    gsx_cpu_optim_dispose_tensor_handles(cpu_optim->second_moments, cpu_optim->base.param_group_count);
    gsx_cpu_optim_dispose_tensor_handles(cpu_optim->scratch_first_moments, cpu_optim->base.param_group_count);
    gsx_cpu_optim_dispose_tensor_handles(cpu_optim->scratch_second_moments, cpu_optim->base.param_group_count);
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

static gsx_error gsx_cpu_optim_validate_group_parameter_gradient(const gsx_cpu_optim *cpu_optim, gsx_index_t index)
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

static gsx_error gsx_cpu_optim_validate_group_state_exact(const gsx_cpu_optim *cpu_optim, gsx_index_t index)
{
    const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[index];
    gsx_tensor_t first_moment = cpu_optim->first_moments[index];
    gsx_tensor_t second_moment = cpu_optim->second_moments[index];
    gsx_index_t dim = 0;
    gsx_error error = gsx_cpu_optim_validate_group_parameter_gradient(cpu_optim, index);

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

static void gsx_cpu_optim_cleanup_sizing_work(gsx_arena_t *arena, gsx_tensor_t *tensor)
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

static gsx_error gsx_cpu_optim_compute_required_bytes(const gsx_cpu_optim *cpu_optim, gsx_size_t *out_required_bytes)
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

    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    error = gsx_arena_init(&dry_run_arena, cpu_optim->base.state_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(index = 0; index < cpu_optim->base.param_group_count; ++index) {
        error = gsx_cpu_optim_make_state_tensor_desc(cpu_optim->base.param_groups[index].parameter, dry_run_arena, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);
    gsx_cpu_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
    return error;
}

static gsx_error gsx_cpu_optim_allocate_state_tensors_on_arena(
    const gsx_cpu_optim *cpu_optim,
    gsx_arena_t arena,
    gsx_tensor_t *first_moments,
    gsx_tensor_t *second_moments,
    bool zero_init
)
{
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_index_t index = 0;

    for(index = 0; index < cpu_optim->base.param_group_count; ++index) {
        gsx_error error = gsx_cpu_optim_make_state_tensor_desc(cpu_optim->base.param_groups[index].parameter, arena, &tensor_desc);

        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
            gsx_cpu_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
            return error;
        }
        error = gsx_tensor_init(&first_moments[index], &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
            gsx_cpu_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
            return error;
        }
        error = gsx_tensor_init(&second_moments[index], &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
            gsx_cpu_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
            return error;
        }
        if(zero_init) {
            error = gsx_tensor_set_zero(first_moments[index]);
            if(!gsx_error_is_success(error)) {
                gsx_cpu_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
                gsx_cpu_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
                return error;
            }
            error = gsx_tensor_set_zero(second_moments[index]);
            if(!gsx_error_is_success(error)) {
                gsx_cpu_optim_dispose_tensor_handles(first_moments, cpu_optim->base.param_group_count);
                gsx_cpu_optim_dispose_tensor_handles(second_moments, cpu_optim->base.param_group_count);
                return error;
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_init_arena(gsx_backend_buffer_type_t buffer_type, gsx_size_t required_bytes, gsx_arena_t *out_arena)
{
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_arena must be non-null");
    }

    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
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

static gsx_error gsx_cpu_optim_allocate_arrays(gsx_cpu_optim *cpu_optim)
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
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu optimizer metadata arrays");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_init_state_storage(gsx_cpu_optim *cpu_optim)
{
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_cpu_optim_compute_required_bytes(cpu_optim, &required_bytes);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_init_arena(cpu_optim->base.state_buffer_type, required_bytes, &cpu_optim->state_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_allocate_state_tensors_on_arena(
        cpu_optim,
        cpu_optim->state_arena,
        cpu_optim->first_moments,
        cpu_optim->second_moments,
        true
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_init_arena(cpu_optim->base.state_buffer_type, required_bytes, &cpu_optim->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_prepare_scratch_for_target_layout(gsx_cpu_optim *cpu_optim)
{
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_cpu_optim_release_scratch_contents(cpu_optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_compute_required_bytes(cpu_optim, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reserve(cpu_optim->scratch_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_allocate_state_tensors_on_arena(
        cpu_optim,
        cpu_optim->scratch_arena,
        cpu_optim->scratch_first_moments,
        cpu_optim->scratch_second_moments,
        false
    );
    if(!gsx_error_is_success(error)) {
        (void)gsx_cpu_optim_release_scratch_contents(cpu_optim);
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cpu_optim_commit_scratch(gsx_cpu_optim *cpu_optim)
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

static gsx_error gsx_cpu_optim_fail_after_prepare(gsx_cpu_optim *cpu_optim, gsx_error error)
{
    if(!gsx_error_is_success(error)) {
        (void)gsx_cpu_optim_release_scratch_contents(cpu_optim);
    }
    return error;
}

static gsx_error gsx_cpu_optim_compute_row_bytes(gsx_tensor_t tensor, gsx_size_t *out_row_bytes)
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

static gsx_error gsx_cpu_optim_validate_control_tensor(
    const gsx_cpu_optim *cpu_optim,
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

static gsx_error gsx_cpu_optim_validate_group_state_transition(
    const gsx_cpu_optim *cpu_optim,
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
    gsx_error error = gsx_cpu_optim_validate_group_parameter_gradient(cpu_optim, index);

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

    error = gsx_cpu_optim_compute_row_bytes(first_moment, out_old_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_compute_row_bytes(param_group->parameter, out_new_row_bytes);
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

static void gsx_cpu_optim_apply_group_clip(const gsx_cpu_optim *cpu_optim, gsx_index_t index)
{
    const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[index];
    float *gradient_values = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    double norm_sq = 0.0;
    double grad_norm = 0.0;
    float scale = 1.0f;

    if(param_group->max_grad_norm <= 0.0f) {
        return;
    }

    gradient_values = gsx_cpu_optim_tensor_data_f32(param_group->gradient);
    element_count = param_group->gradient->size_bytes / sizeof(float);
    for(element_index = 0; element_index < element_count; ++element_index) {
        norm_sq += (double)gradient_values[element_index] * (double)gradient_values[element_index];
    }

    grad_norm = sqrt(norm_sq);
    if(grad_norm == 0.0 || grad_norm <= (double)param_group->max_grad_norm) {
        return;
    }

    scale = (float)((double)param_group->max_grad_norm / grad_norm);
    for(element_index = 0; element_index < element_count; ++element_index) {
        gradient_values[element_index] *= scale;
    }
}

gsx_error gsx_cpu_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim)
{
    gsx_cpu_optim *cpu_optim = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_optim == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_optim and desc must be non-null");
    }

    *out_optim = NULL;
    cpu_optim = (gsx_cpu_optim *)calloc(1, sizeof(*cpu_optim));
    if(cpu_optim == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu optimizer");
    }

    error = gsx_optim_base_init(&cpu_optim->base, &gsx_cpu_optim_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_optim_destroy_incomplete(cpu_optim);
        return error;
    }
    error = gsx_cpu_optim_allocate_arrays(cpu_optim);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_optim_destroy_incomplete(cpu_optim);
        return error;
    }
    error = gsx_cpu_optim_init_state_storage(cpu_optim);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_optim_destroy_incomplete(cpu_optim);
        return error;
    }

    *out_optim = &cpu_optim->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_destroy(gsx_optim_t optim)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(optim == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim must be non-null");
    }

    error = gsx_cpu_optim_free_tensor_handles(cpu_optim->first_moments, cpu_optim->base.param_group_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_free_tensor_handles(cpu_optim->second_moments, cpu_optim->base.param_group_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_optim_release_scratch_contents(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(cpu_optim->state_arena != NULL) {
        error = gsx_arena_free(cpu_optim->state_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cpu_optim->scratch_arena != NULL) {
        error = gsx_arena_free(cpu_optim->scratch_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    free(cpu_optim->step_counts);
    free(cpu_optim->first_moments);
    free(cpu_optim->second_moments);
    free(cpu_optim->scratch_first_moments);
    free(cpu_optim->scratch_second_moments);
    gsx_optim_base_deinit(&cpu_optim->base);
    free(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_step_selected(gsx_optim_t optim, const bool *selected)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
    gsx_index_t group_index = 0;

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        const gsx_optim_param_group_desc *param_group = &cpu_optim->base.param_groups[group_index];
        float *parameter_values = NULL;
        float *gradient_values = NULL;
        float *first_moment_values = NULL;
        float *second_moment_values = NULL;
        gsx_size_t element_count = 0;
        gsx_size_t element_index = 0;
        double beta1_correction = 0.0;
        double beta2_correction = 0.0;
        gsx_error error = { GSX_ERROR_SUCCESS, NULL };

        if(selected != NULL && !selected[group_index]) {
            continue;
        }

        error = gsx_cpu_optim_validate_group_state_exact(cpu_optim, group_index);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(cpu_optim->step_counts[group_index] == UINT64_MAX) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer Adam step counter overflowed");
        }

        gsx_cpu_optim_apply_group_clip(cpu_optim, group_index);
        cpu_optim->step_counts[group_index] += 1;
        beta1_correction = 1.0 - pow((double)param_group->beta1, (double)cpu_optim->step_counts[group_index]);
        beta2_correction = 1.0 - pow((double)param_group->beta2, (double)cpu_optim->step_counts[group_index]);
        parameter_values = gsx_cpu_optim_tensor_data_f32(param_group->parameter);
        gradient_values = gsx_cpu_optim_tensor_data_f32(param_group->gradient);
        first_moment_values = gsx_cpu_optim_tensor_data_f32(cpu_optim->first_moments[group_index]);
        second_moment_values = gsx_cpu_optim_tensor_data_f32(cpu_optim->second_moments[group_index]);
        element_count = param_group->parameter->size_bytes / sizeof(float);

        for(element_index = 0; element_index < element_count; ++element_index) {
            float gradient_value = gradient_values[element_index];
            float first_moment_value =
                param_group->beta1 * first_moment_values[element_index] + (1.0f - param_group->beta1) * gradient_value;
            float second_moment_value =
                param_group->beta2 * second_moment_values[element_index] + (1.0f - param_group->beta2) * gradient_value * gradient_value;
            float first_moment_hat = (float)((double)first_moment_value / beta1_correction);
            float second_moment_hat = (float)((double)second_moment_value / beta2_correction);
            float parameter_value = parameter_values[element_index];

            first_moment_values[element_index] = first_moment_value;
            second_moment_values[element_index] = second_moment_value;
            if(param_group->weight_decay > 0.0f) {
                parameter_value -= cpu_optim->base.learning_rates[group_index] * param_group->weight_decay * parameter_value;
            }
            parameter_value -=
                cpu_optim->base.learning_rates[group_index]
                * (first_moment_hat / (sqrtf(second_moment_hat) + param_group->epsilon));
            parameter_values[element_index] = parameter_value;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_reset_all(gsx_optim_t optim)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
    gsx_index_t index = 0;

    for(index = 0; index < cpu_optim->base.param_group_count; ++index) {
        gsx_error error = gsx_cpu_optim_reset_by_index(optim, index);

        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
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

static gsx_error gsx_cpu_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
    bool *seen = NULL;
    gsx_size_t expected_count = 0;
    gsx_index_t group_index = 0;
    const int32_t *permutation_values = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        error = gsx_cpu_optim_validate_group_state_exact(cpu_optim, group_index);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(group_index == 0) {
            expected_count = (gsx_size_t)cpu_optim->first_moments[group_index]->shape[0];
        } else if((gsx_size_t)cpu_optim->first_moments[group_index]->shape[0] != expected_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extent for permutation");
        }
    }

    error = gsx_cpu_optim_validate_control_tensor(
        cpu_optim,
        permutation,
        GSX_DATA_TYPE_I32,
        expected_count,
        "optimizer permutation tensor must be rank-1 int32"
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }

    seen = (bool *)calloc((size_t)expected_count, sizeof(*seen));
    if(seen == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer permutation validation bitmap");
    }
    permutation_values = (const int32_t *)gsx_cpu_optim_tensor_data_bytes(permutation);
    for(group_index = 0; group_index < (gsx_index_t)expected_count; ++group_index) {
        gsx_size_t src_index = 0;

        if(permutation_values[group_index] < 0) {
            free(seen);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer permutation entries must be non-negative");
        }
        src_index = (gsx_size_t)permutation_values[group_index];
        if(src_index >= expected_count) {
            free(seen);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer permutation entries must be in range");
        }
        if(seen[src_index]) {
            free(seen);
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer permutation entries must be unique");
        }
        seen[src_index] = true;
    }
    free(seen);

    error = gsx_cpu_optim_prepare_scratch_for_target_layout(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        gsx_size_t row_bytes = 0;
        gsx_size_t row_index = 0;
        unsigned char *scratch_first_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->scratch_first_moments[group_index]);
        unsigned char *scratch_second_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->scratch_second_moments[group_index]);
        const unsigned char *old_first_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->first_moments[group_index]);
        const unsigned char *old_second_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->second_moments[group_index]);

        error = gsx_cpu_optim_compute_row_bytes(cpu_optim->first_moments[group_index], &row_bytes);
        if(!gsx_error_is_success(error)) {
            return gsx_cpu_optim_fail_after_prepare(cpu_optim, error);
        }
        for(row_index = 0; row_index < expected_count; ++row_index) {
            gsx_size_t src_index = (gsx_size_t)permutation_values[row_index];

            memcpy(
                scratch_first_bytes + (size_t)(row_index * row_bytes),
                old_first_bytes + (size_t)(src_index * row_bytes),
                (size_t)row_bytes
            );
            memcpy(
                scratch_second_bytes + (size_t)(row_index * row_bytes),
                old_second_bytes + (size_t)(src_index * row_bytes),
                (size_t)row_bytes
            );
        }
    }

    gsx_cpu_optim_commit_scratch(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_prune(gsx_optim_t optim, gsx_tensor_t keep_mask)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
    const uint8_t *keep_values = NULL;
    gsx_size_t old_count = 0;
    gsx_size_t new_count = 0;
    gsx_size_t kept_count = 0;
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

        error = gsx_cpu_optim_validate_group_state_transition(
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
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extents for prune");
        }
    }

    error = gsx_cpu_optim_validate_control_tensor(
        cpu_optim,
        keep_mask,
        GSX_DATA_TYPE_U8,
        old_count,
        "optimizer keep_mask tensor must be rank-1 uint8"
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }

    keep_values = (const uint8_t *)gsx_cpu_optim_tensor_data_bytes(keep_mask);
    for(group_index = 0; group_index < (gsx_index_t)old_count; ++group_index) {
        if(keep_values[group_index] != 0) {
            kept_count += 1;
        }
    }
    if(kept_count != new_count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer keep_mask survivor count does not match the current parameter layout");
    }

    error = gsx_cpu_optim_prepare_scratch_for_target_layout(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        gsx_size_t row_bytes = 0;
        gsx_size_t src_row = 0;
        gsx_size_t dst_row = 0;
        unsigned char *scratch_first_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->scratch_first_moments[group_index]);
        unsigned char *scratch_second_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->scratch_second_moments[group_index]);
        const unsigned char *old_first_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->first_moments[group_index]);
        const unsigned char *old_second_bytes = gsx_cpu_optim_tensor_data_bytes(cpu_optim->second_moments[group_index]);

        error = gsx_cpu_optim_compute_row_bytes(cpu_optim->first_moments[group_index], &row_bytes);
        if(!gsx_error_is_success(error)) {
            return gsx_cpu_optim_fail_after_prepare(cpu_optim, error);
        }
        for(src_row = 0; src_row < old_count; ++src_row) {
            if(keep_values[src_row] == 0) {
                continue;
            }
            memcpy(
                scratch_first_bytes + (size_t)(dst_row * row_bytes),
                old_first_bytes + (size_t)(src_row * row_bytes),
                (size_t)row_bytes
            );
            memcpy(
                scratch_second_bytes + (size_t)(dst_row * row_bytes),
                old_second_bytes + (size_t)(src_row * row_bytes),
                (size_t)row_bytes
            );
            dst_row += 1;
        }
    }

    gsx_cpu_optim_commit_scratch(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_optim_grow(gsx_optim_t optim, gsx_size_t growth_count)
{
    gsx_cpu_optim *cpu_optim = (gsx_cpu_optim *)optim;
    gsx_size_t old_count = 0;
    gsx_size_t new_count = 0;
    gsx_index_t group_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cpu_optim->base.param_group_count == 0 || growth_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        gsx_size_t group_old_count = 0;
        gsx_size_t group_new_count = 0;
        gsx_size_t expected_new_count = 0;
        gsx_size_t old_row_bytes = 0;
        gsx_size_t new_row_bytes = 0;

        error = gsx_cpu_optim_validate_group_state_transition(
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
        if(gsx_size_add_overflows(group_old_count, growth_count, &expected_new_count) || group_new_count != expected_new_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer grow expects parameter tensors to already expose the grown leading extent");
        }
        if(group_index == 0) {
            old_count = group_old_count;
            new_count = group_new_count;
        } else if(group_old_count != old_count || group_new_count != new_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extents for grow");
        }
    }

    error = gsx_cpu_optim_prepare_scratch_for_target_layout(cpu_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cpu_optim->base.param_group_count; ++group_index) {
        gsx_size_t copy_bytes = cpu_optim->first_moments[group_index]->size_bytes;

        error = gsx_tensor_set_zero(cpu_optim->scratch_first_moments[group_index]);
        if(!gsx_error_is_success(error)) {
            return gsx_cpu_optim_fail_after_prepare(cpu_optim, error);
        }
        error = gsx_tensor_set_zero(cpu_optim->scratch_second_moments[group_index]);
        if(!gsx_error_is_success(error)) {
            return gsx_cpu_optim_fail_after_prepare(cpu_optim, error);
        }

        memcpy(
            gsx_cpu_optim_tensor_data_bytes(cpu_optim->scratch_first_moments[group_index]),
            gsx_cpu_optim_tensor_data_bytes(cpu_optim->first_moments[group_index]),
            (size_t)copy_bytes
        );
        memcpy(
            gsx_cpu_optim_tensor_data_bytes(cpu_optim->scratch_second_moments[group_index]),
            gsx_cpu_optim_tensor_data_bytes(cpu_optim->second_moments[group_index]),
            (size_t)copy_bytes
        );
    }

    gsx_cpu_optim_commit_scratch(cpu_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
