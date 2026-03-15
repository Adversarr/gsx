#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Metal optimizer implementation.
 * Adam step uses a fused Metal compute kernel.
 * Structural mutations (permute/prune/grow) are not supported.
 */

typedef struct gsx_metal_optim {
    struct gsx_optim base;
    gsx_size_t *step_counts;
    gsx_tensor_t *first_moments;
    gsx_tensor_t *second_moments;
    gsx_arena_t state_arena;
} gsx_metal_optim;

static gsx_error gsx_metal_optim_destroy(gsx_optim_t optim);
static gsx_error gsx_metal_optim_step_selected(gsx_optim_t optim, const bool *selected);
static gsx_error gsx_metal_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation);
static gsx_error gsx_metal_optim_prune(gsx_optim_t optim, gsx_tensor_t keep_mask);
static gsx_error gsx_metal_optim_grow(gsx_optim_t optim, gsx_size_t growth_count);
static gsx_error gsx_metal_optim_reset_all(gsx_optim_t optim);
static gsx_error gsx_metal_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index);

static const gsx_optim_i gsx_metal_optim_iface = {
    gsx_metal_optim_destroy,
    gsx_metal_optim_step_selected,
    gsx_metal_optim_permute,
    gsx_metal_optim_prune,
    gsx_metal_optim_grow,
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

static void gsx_metal_optim_destroy_incomplete(gsx_metal_optim *cpu_optim)
{
    if(cpu_optim == NULL) {
        return;
    }

    gsx_metal_optim_dispose_tensor_handles(cpu_optim->first_moments, cpu_optim->base.param_group_count);
    gsx_metal_optim_dispose_tensor_handles(cpu_optim->second_moments, cpu_optim->base.param_group_count);
    if(cpu_optim->state_arena != NULL) {
        (void)gsx_arena_free(cpu_optim->state_arena);
    }
    free(cpu_optim->step_counts);
    free(cpu_optim->first_moments);
    free(cpu_optim->second_moments);
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

    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
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

static gsx_error gsx_metal_optim_allocate_arrays(gsx_metal_optim *cpu_optim)
{
    gsx_index_t count = cpu_optim->base.param_group_count;

    if(count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cpu_optim->step_counts = (gsx_size_t *)calloc((size_t)count, sizeof(*cpu_optim->step_counts));
    cpu_optim->first_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cpu_optim->first_moments));
    cpu_optim->second_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cpu_optim->second_moments));
    if(cpu_optim->step_counts == NULL || cpu_optim->first_moments == NULL || cpu_optim->second_moments == NULL) {
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

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
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

    if(cpu_optim->state_arena != NULL) {
        error = gsx_arena_free(cpu_optim->state_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
    }

    free(cpu_optim->step_counts);
    free(cpu_optim->first_moments);
    free(cpu_optim->second_moments);
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
    (void)optim;
    (void)permutation;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer does not support permute");
}

static gsx_error gsx_metal_optim_prune(gsx_optim_t optim, gsx_tensor_t keep_mask)
{
    (void)optim;
    (void)keep_mask;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer does not support prune");
}

static gsx_error gsx_metal_optim_grow(gsx_optim_t optim, gsx_size_t growth_count)
{
    (void)optim;
    (void)growth_count;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal optimizer does not support grow");
}
