#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_cuda_optim {
    struct gsx_optim base;
    gsx_size_t *step_counts;
    gsx_tensor_t *first_moments;
    gsx_tensor_t *second_moments;
    gsx_tensor_t *scratch_first_moments;
    gsx_tensor_t *scratch_second_moments;
    gsx_arena_t state_arena;
    gsx_arena_t scratch_arena;
} gsx_cuda_optim;

static gsx_error gsx_cuda_optim_destroy(gsx_optim_t optim);
static gsx_error gsx_cuda_optim_step_selected(gsx_optim_t optim, const bool *selected);
static gsx_error gsx_cuda_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation);
static gsx_error gsx_cuda_optim_gather(gsx_optim_t optim, gsx_tensor_t indices);
static gsx_error gsx_cuda_optim_resize(gsx_optim_t optim, gsx_size_t new_count);
static gsx_error gsx_cuda_optim_reset_all(gsx_optim_t optim);
static gsx_error gsx_cuda_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index);

static const gsx_optim_i gsx_cuda_optim_iface = {
    gsx_cuda_optim_destroy,
    gsx_cuda_optim_step_selected,
    gsx_cuda_optim_permute,
    gsx_cuda_optim_gather,
    gsx_cuda_optim_resize,
    gsx_cuda_optim_reset_all,
    gsx_cuda_optim_reset_by_index
};

static bool gsx_cuda_optim_buffer_is_device(gsx_backend_buffer_t buffer)
{
    return buffer != NULL && gsx_cuda_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static unsigned char *gsx_cuda_optim_tensor_device_bytes(gsx_tensor_t tensor)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(tensor->backing_buffer);

    return (unsigned char *)cuda_buffer->ptr + (size_t)tensor->offset_bytes;
}

static float *gsx_cuda_optim_tensor_device_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cuda_optim_tensor_device_bytes(tensor);
}

static gsx_error gsx_cuda_optim_sync_backend(const gsx_cuda_optim *cuda_optim)
{
    void *stream = NULL;
    gsx_error error = gsx_backend_get_major_stream(cuda_optim->base.backend, &stream);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_cuda_make_error(cudaStreamSynchronize((cudaStream_t)stream), "cudaStreamSynchronize failed");
}

static gsx_error gsx_cuda_optim_make_state_tensor_desc(gsx_tensor_t parameter, gsx_arena_t arena, gsx_tensor_desc *out_desc)
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

static void gsx_cuda_optim_dispose_tensor_handles(gsx_tensor_t *tensors, gsx_index_t count)
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

static gsx_error gsx_cuda_optim_free_tensor_handles(gsx_tensor_t *tensors, gsx_index_t count)
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

static gsx_error gsx_cuda_optim_release_scratch_contents(gsx_cuda_optim *cuda_optim)
{
    gsx_error error = gsx_make_error(GSX_ERROR_SUCCESS, NULL);

    if(cuda_optim->scratch_first_moments != NULL) {
        error = gsx_cuda_optim_free_tensor_handles(cuda_optim->scratch_first_moments, cuda_optim->base.param_group_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cuda_optim->scratch_second_moments != NULL) {
        error = gsx_cuda_optim_free_tensor_handles(cuda_optim->scratch_second_moments, cuda_optim->base.param_group_count);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cuda_optim->scratch_arena != NULL) {
        error = gsx_arena_reset(cuda_optim->scratch_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cuda_optim_destroy_incomplete(gsx_cuda_optim *cuda_optim)
{
    if(cuda_optim == NULL) {
        return;
    }

    gsx_cuda_optim_dispose_tensor_handles(cuda_optim->first_moments, cuda_optim->base.param_group_count);
    gsx_cuda_optim_dispose_tensor_handles(cuda_optim->second_moments, cuda_optim->base.param_group_count);
    gsx_cuda_optim_dispose_tensor_handles(cuda_optim->scratch_first_moments, cuda_optim->base.param_group_count);
    gsx_cuda_optim_dispose_tensor_handles(cuda_optim->scratch_second_moments, cuda_optim->base.param_group_count);
    if(cuda_optim->state_arena != NULL) {
        (void)gsx_arena_free(cuda_optim->state_arena);
    }
    if(cuda_optim->scratch_arena != NULL) {
        (void)gsx_arena_free(cuda_optim->scratch_arena);
    }
    free(cuda_optim->step_counts);
    free(cuda_optim->first_moments);
    free(cuda_optim->second_moments);
    free(cuda_optim->scratch_first_moments);
    free(cuda_optim->scratch_second_moments);
    gsx_optim_base_deinit(&cuda_optim->base);
    free(cuda_optim);
}

static gsx_error gsx_cuda_optim_validate_group_parameter_gradient(const gsx_cuda_optim *cuda_optim, gsx_index_t index)
{
    const gsx_optim_param_group_desc *param_group = &cuda_optim->base.param_groups[index];
    gsx_tensor_t parameter = param_group->parameter;
    gsx_tensor_t gradient = param_group->gradient;
    gsx_index_t dim = 0;

    if(parameter == NULL || gradient == NULL || parameter->backing_buffer == NULL || gradient->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer parameter and gradient tensors must remain live and accessible");
    }
    if(parameter->backing_buffer->buffer_type->backend != cuda_optim->base.backend
        || gradient->backing_buffer->buffer_type->backend != cuda_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer tensors no longer belong to the owning backend");
    }
    if(!gsx_cuda_optim_buffer_is_device(parameter->backing_buffer) || !gsx_cuda_optim_buffer_is_device(gradient->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda optimizer requires device-backed parameter and gradient tensors");
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

static gsx_error gsx_cuda_optim_validate_group_state_exact(const gsx_cuda_optim *cuda_optim, gsx_index_t index)
{
    const gsx_optim_param_group_desc *param_group = &cuda_optim->base.param_groups[index];
    gsx_tensor_t first_moment = cuda_optim->first_moments[index];
    gsx_tensor_t second_moment = cuda_optim->second_moments[index];
    gsx_index_t dim = 0;
    gsx_error error = gsx_cuda_optim_validate_group_parameter_gradient(cuda_optim, index);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(first_moment == NULL || second_moment == NULL || first_moment->backing_buffer == NULL || second_moment->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors must remain live and accessible");
    }
    if(first_moment->backing_buffer->buffer_type->backend != cuda_optim->base.backend
        || second_moment->backing_buffer->buffer_type->backend != cuda_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer belong to the optimizer backend");
    }
    if(!gsx_cuda_optim_buffer_is_device(first_moment->backing_buffer) || !gsx_cuda_optim_buffer_is_device(second_moment->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda optimizer state tensors must remain device-backed");
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

static void gsx_cuda_optim_cleanup_sizing_work(gsx_arena_t *arena, gsx_tensor_t *tensor)
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

static gsx_error gsx_cuda_optim_compute_required_bytes(const gsx_cuda_optim *cuda_optim, gsx_size_t *out_required_bytes)
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
    error = gsx_arena_init(&dry_run_arena, cuda_optim->base.state_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(index = 0; index < cuda_optim->base.param_group_count; ++index) {
        error = gsx_cuda_optim_make_state_tensor_desc(cuda_optim->base.param_groups[index].parameter, dry_run_arena, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_init(&temp_tensor, &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
        error = gsx_tensor_free(temp_tensor);
        temp_tensor = NULL;
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
            return error;
        }
    }

    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);
    gsx_cuda_optim_cleanup_sizing_work(&dry_run_arena, &temp_tensor);
    return error;
}

static gsx_error gsx_cuda_optim_allocate_state_tensors_on_arena(
    const gsx_cuda_optim *cuda_optim,
    gsx_arena_t arena,
    gsx_tensor_t *first_moments,
    gsx_tensor_t *second_moments,
    bool zero_init
)
{
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_index_t index = 0;

    for(index = 0; index < cuda_optim->base.param_group_count; ++index) {
        gsx_error error = gsx_cuda_optim_make_state_tensor_desc(cuda_optim->base.param_groups[index].parameter, arena, &tensor_desc);

        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_dispose_tensor_handles(first_moments, cuda_optim->base.param_group_count);
            gsx_cuda_optim_dispose_tensor_handles(second_moments, cuda_optim->base.param_group_count);
            return error;
        }
        error = gsx_tensor_init(&first_moments[index], &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_dispose_tensor_handles(first_moments, cuda_optim->base.param_group_count);
            gsx_cuda_optim_dispose_tensor_handles(second_moments, cuda_optim->base.param_group_count);
            return error;
        }
        error = gsx_tensor_init(&second_moments[index], &tensor_desc);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_optim_dispose_tensor_handles(first_moments, cuda_optim->base.param_group_count);
            gsx_cuda_optim_dispose_tensor_handles(second_moments, cuda_optim->base.param_group_count);
            return error;
        }
        if(zero_init) {
            error = gsx_tensor_set_zero(first_moments[index]);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_optim_dispose_tensor_handles(first_moments, cuda_optim->base.param_group_count);
                gsx_cuda_optim_dispose_tensor_handles(second_moments, cuda_optim->base.param_group_count);
                return error;
            }
            error = gsx_tensor_set_zero(second_moments[index]);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_optim_dispose_tensor_handles(first_moments, cuda_optim->base.param_group_count);
                gsx_cuda_optim_dispose_tensor_handles(second_moments, cuda_optim->base.param_group_count);
                return error;
            }
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_init_arena(gsx_backend_buffer_type_t buffer_type, gsx_size_t required_bytes, gsx_arena_t *out_arena)
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

static gsx_error gsx_cuda_optim_allocate_arrays(gsx_cuda_optim *cuda_optim)
{
    gsx_index_t count = cuda_optim->base.param_group_count;

    if(count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_optim->step_counts = (gsx_size_t *)calloc((size_t)count, sizeof(*cuda_optim->step_counts));
    cuda_optim->first_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cuda_optim->first_moments));
    cuda_optim->second_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cuda_optim->second_moments));
    cuda_optim->scratch_first_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cuda_optim->scratch_first_moments));
    cuda_optim->scratch_second_moments = (gsx_tensor_t *)calloc((size_t)count, sizeof(*cuda_optim->scratch_second_moments));
    if(cuda_optim->step_counts == NULL || cuda_optim->first_moments == NULL || cuda_optim->second_moments == NULL
        || cuda_optim->scratch_first_moments == NULL || cuda_optim->scratch_second_moments == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda optimizer metadata arrays");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_init_state_storage(gsx_cuda_optim *cuda_optim)
{
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_cuda_optim_compute_required_bytes(cuda_optim, &required_bytes);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_init_arena(cuda_optim->base.state_buffer_type, required_bytes, &cuda_optim->state_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_allocate_state_tensors_on_arena(
        cuda_optim,
        cuda_optim->state_arena,
        cuda_optim->first_moments,
        cuda_optim->second_moments,
        true
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_init_arena(cuda_optim->base.state_buffer_type, required_bytes, &cuda_optim->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_prepare_scratch_for_target_layout(gsx_cuda_optim *cuda_optim)
{
    gsx_size_t required_bytes = 0;
    gsx_error error = gsx_cuda_optim_release_scratch_contents(cuda_optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_compute_required_bytes(cuda_optim, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reserve(cuda_optim->scratch_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_allocate_state_tensors_on_arena(
        cuda_optim,
        cuda_optim->scratch_arena,
        cuda_optim->scratch_first_moments,
        cuda_optim->scratch_second_moments,
        false
    );
    if(!gsx_error_is_success(error)) {
        (void)gsx_cuda_optim_release_scratch_contents(cuda_optim);
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cuda_optim_commit_scratch(gsx_cuda_optim *cuda_optim)
{
    gsx_arena_t arena = cuda_optim->state_arena;
    gsx_tensor_t *tensors = cuda_optim->first_moments;

    cuda_optim->state_arena = cuda_optim->scratch_arena;
    cuda_optim->scratch_arena = arena;

    cuda_optim->first_moments = cuda_optim->scratch_first_moments;
    cuda_optim->scratch_first_moments = tensors;

    tensors = cuda_optim->second_moments;
    cuda_optim->second_moments = cuda_optim->scratch_second_moments;
    cuda_optim->scratch_second_moments = tensors;
}

static gsx_error gsx_cuda_optim_fail_after_prepare(gsx_cuda_optim *cuda_optim, gsx_error error)
{
    if(!gsx_error_is_success(error)) {
        (void)gsx_cuda_optim_release_scratch_contents(cuda_optim);
    }
    return error;
}

static gsx_error gsx_cuda_optim_compute_row_bytes(gsx_tensor_t tensor, gsx_size_t *out_row_bytes)
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

static gsx_error gsx_cuda_optim_validate_control_tensor(
    const gsx_cuda_optim *cuda_optim,
    gsx_tensor_t tensor,
    gsx_data_type data_type,
    gsx_size_t expected_count,
    const char *tensor_name
)
{
    if(tensor == NULL || tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer control tensors must be live and accessible");
    }
    if(tensor->backing_buffer->buffer_type->backend != cuda_optim->base.backend) {
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

static gsx_error gsx_cuda_optim_download_tensor_bytes(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count)
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

static gsx_error gsx_cuda_optim_validate_group_state_transition(
    const gsx_cuda_optim *cuda_optim,
    gsx_index_t index,
    gsx_size_t *out_old_count,
    gsx_size_t *out_new_count,
    gsx_size_t *out_old_row_bytes,
    gsx_size_t *out_new_row_bytes
)
{
    const gsx_optim_param_group_desc *param_group = &cuda_optim->base.param_groups[index];
    gsx_tensor_t first_moment = cuda_optim->first_moments[index];
    gsx_tensor_t second_moment = cuda_optim->second_moments[index];
    gsx_index_t dim = 0;
    gsx_error error = gsx_cuda_optim_validate_group_parameter_gradient(cuda_optim, index);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(first_moment == NULL || second_moment == NULL || first_moment->backing_buffer == NULL || second_moment->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors must remain live and accessible");
    }
    if(first_moment->backing_buffer->buffer_type->backend != cuda_optim->base.backend
        || second_moment->backing_buffer->buffer_type->backend != cuda_optim->base.backend) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors no longer belong to the optimizer backend");
    }
    if(!gsx_cuda_optim_buffer_is_device(first_moment->backing_buffer) || !gsx_cuda_optim_buffer_is_device(second_moment->backing_buffer)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda optimizer state tensors must remain device-backed");
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

    error = gsx_cuda_optim_compute_row_bytes(first_moment, out_old_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_compute_row_bytes(param_group->parameter, out_new_row_bytes);
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

static gsx_error gsx_cuda_optim_upload_temp_index_buffer(
    const gsx_cuda_optim *cuda_optim,
    const int32_t *indices,
    gsx_size_t count,
    gsx_backend_buffer_t *out_buffer
)
{
    gsx_backend_buffer_desc buffer_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer must be non-null");
    }
    *out_buffer = NULL;

    if(count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(indices == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "indices must be non-null for non-zero count");
    }

    buffer_desc.buffer_type = cuda_optim->base.state_buffer_type;
    buffer_desc.size_bytes = count * sizeof(int32_t);
    buffer_desc.alignment_bytes = sizeof(int32_t);
    error = gsx_backend_buffer_init(out_buffer, &buffer_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_backend_buffer_upload(*out_buffer, 0, indices, buffer_desc.size_bytes);
    if(!gsx_error_is_success(error)) {
        (void)gsx_backend_buffer_free(*out_buffer);
        *out_buffer = NULL;
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_copy_prefix_bytes(gsx_tensor_t src, gsx_tensor_t dst, gsx_size_t copy_bytes)
{
    gsx_backend_tensor_view src_view = { 0 };
    gsx_backend_tensor_view dst_view = { 0 };

    if(src == NULL || dst == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src and dst must be non-null");
    }
    if(copy_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    src_view.buffer = src->backing_buffer;
    src_view.offset_bytes = src->offset_bytes;
    src_view.size_bytes = copy_bytes;
    src_view.effective_alignment_bytes = src->effective_alignment_bytes;
    src_view.data_type = src->data_type;

    dst_view.buffer = dst->backing_buffer;
    dst_view.offset_bytes = dst->offset_bytes;
    dst_view.size_bytes = copy_bytes;
    dst_view.effective_alignment_bytes = dst->effective_alignment_bytes;
    dst_view.data_type = dst->data_type;

    return dst->backing_buffer->iface->copy_tensor(dst->backing_buffer, &src_view, &dst_view);
}

static gsx_error gsx_cuda_optim_validate_init_requirements(const gsx_optim_desc *desc)
{
    gsx_index_t index = 0;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }

    for(index = 0; index < desc->param_group_count; ++index) {
        const gsx_optim_param_group_desc *param_group = &desc->param_groups[index];

        if(param_group->parameter == NULL || param_group->gradient == NULL
            || param_group->parameter->backing_buffer == NULL || param_group->gradient->backing_buffer == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda optimizer requires live parameter and gradient tensors");
        }
        if(!gsx_cuda_optim_buffer_is_device(param_group->parameter->backing_buffer)
            || !gsx_cuda_optim_buffer_is_device(param_group->gradient->backing_buffer)) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda optimizer requires device-backed parameter and gradient tensors");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim)
{
    gsx_cuda_optim *cuda_optim = NULL;
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_optim == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_optim and desc must be non-null");
    }

    *out_optim = NULL;
    error = gsx_cuda_optim_validate_init_requirements(desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cuda_optim = (gsx_cuda_optim *)calloc(1, sizeof(*cuda_optim));
    if(cuda_optim == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda optimizer");
    }

    error = gsx_optim_base_init(&cuda_optim->base, &gsx_cuda_optim_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_optim_destroy_incomplete(cuda_optim);
        return error;
    }
    error = gsx_backend_buffer_type_get_info(cuda_optim->base.state_buffer_type, &buffer_type_info);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_optim_destroy_incomplete(cuda_optim);
        return error;
    }
    if(buffer_type_info.type != GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        gsx_cuda_optim_destroy_incomplete(cuda_optim);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda optimizer state_buffer_type must be the device buffer type");
    }
    error = gsx_cuda_optim_allocate_arrays(cuda_optim);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_optim_destroy_incomplete(cuda_optim);
        return error;
    }
    error = gsx_cuda_optim_init_state_storage(cuda_optim);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_optim_destroy_incomplete(cuda_optim);
        return error;
    }
    *out_optim = &cuda_optim->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_destroy(gsx_optim_t optim)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(optim == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim must be non-null");
    }

    error = gsx_cuda_optim_free_tensor_handles(cuda_optim->first_moments, cuda_optim->base.param_group_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_free_tensor_handles(cuda_optim->second_moments, cuda_optim->base.param_group_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_optim_release_scratch_contents(cuda_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(cuda_optim->state_arena != NULL) {
        error = gsx_arena_free(cuda_optim->state_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(cuda_optim->scratch_arena != NULL) {
        error = gsx_arena_free(cuda_optim->scratch_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    free(cuda_optim->step_counts);
    free(cuda_optim->first_moments);
    free(cuda_optim->second_moments);
    free(cuda_optim->scratch_first_moments);
    free(cuda_optim->scratch_second_moments);
    gsx_optim_base_deinit(&cuda_optim->base);
    free(cuda_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_step_selected(gsx_optim_t optim, const bool *selected)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    void *stream = NULL;
    gsx_error error = gsx_backend_get_major_stream(cuda_optim->base.backend, &stream);
    gsx_index_t group_index = 0;

    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        const gsx_optim_param_group_desc *param_group = &cuda_optim->base.param_groups[group_index];
        gsx_size_t element_count = 0;
        double beta1_correction = 0.0;
        double beta2_correction = 0.0;

        if(selected != NULL && !selected[group_index]) {
            continue;
        }

        error = gsx_cuda_optim_validate_group_state_exact(cuda_optim, group_index);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(cuda_optim->step_counts[group_index] == UINT64_MAX) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer Adam step counter overflowed");
        }

        element_count = param_group->parameter->size_bytes / sizeof(float);
        cuda_optim->step_counts[group_index] += 1;
        beta1_correction = 1.0 - pow((double)param_group->beta1, (double)cuda_optim->step_counts[group_index]);
        beta2_correction = 1.0 - pow((double)param_group->beta2, (double)cuda_optim->step_counts[group_index]);

        gsx_cuda_adam_step_f32_kernel_launch(
            gsx_cuda_optim_tensor_device_f32(param_group->parameter),
            gsx_cuda_optim_tensor_device_f32(param_group->gradient),
            gsx_cuda_optim_tensor_device_f32(cuda_optim->first_moments[group_index]),
            gsx_cuda_optim_tensor_device_f32(cuda_optim->second_moments[group_index]),
            element_count,
            param_group->beta1,
            param_group->beta2,
            cuda_optim->base.learning_rates[group_index],
            param_group->weight_decay,
            param_group->epsilon,
            param_group->max_grad,
            1.0 / beta1_correction,
            1.0 / beta2_correction,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cudaGetLastError(), "cuda optimizer adam kernel launch failed");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_reset_all(gsx_optim_t optim)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    gsx_index_t index = 0;

    for(index = 0; index < cuda_optim->base.param_group_count; ++index) {
        gsx_error error = gsx_cuda_optim_reset_by_index(optim, index);

        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_reset_by_index(gsx_optim_t optim, gsx_index_t index)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(index < 0 || index >= cuda_optim->base.param_group_count) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer param-group index is out of range");
    }
    if(cuda_optim->first_moments[index] == NULL || cuda_optim->second_moments[index] == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer state tensors are unavailable");
    }

    error = gsx_tensor_set_zero(cuda_optim->first_moments[index]);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_optim->second_moments[index]);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cuda_optim->step_counts[index] = 0;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    bool *seen = NULL;
    int32_t *permutation_values = NULL;
    gsx_backend_buffer_t permutation_index_buffer = NULL;
    gsx_size_t expected_count = 0;
    gsx_index_t group_index = 0;
    void *stream = NULL;
    cudaError_t cuda_error = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_backend_get_major_stream(cuda_optim->base.backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        error = gsx_cuda_optim_validate_group_state_exact(cuda_optim, group_index);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(group_index == 0) {
            expected_count = (gsx_size_t)cuda_optim->first_moments[group_index]->shape[0];
        } else if((gsx_size_t)cuda_optim->first_moments[group_index]->shape[0] != expected_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer param groups must share the same leading extent for permutation");
        }
    }

    error = gsx_cuda_optim_validate_control_tensor(
        cuda_optim,
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
    error = gsx_cuda_optim_download_tensor_bytes(permutation, permutation_values, expected_count * sizeof(*permutation_values));
    if(!gsx_error_is_success(error)) {
        free(permutation_values);
        free(seen);
        return error;
    }
    error = gsx_cuda_optim_sync_backend(cuda_optim);
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

    error = gsx_cuda_optim_upload_temp_index_buffer(cuda_optim, permutation_values, expected_count, &permutation_index_buffer);
    free(permutation_values);
    permutation_values = NULL;
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_cuda_optim_prepare_scratch_for_target_layout(cuda_optim);
    if(!gsx_error_is_success(error)) {
        (void)gsx_backend_buffer_free(permutation_index_buffer);
        return error;
    }

    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        gsx_size_t row_bytes = 0;

        error = gsx_cuda_optim_compute_row_bytes(cuda_optim->first_moments[group_index], &row_bytes);
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(permutation_index_buffer);
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }

        cuda_error = gsx_cuda_gather_rows_kernel_launch(
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->first_moments[group_index]),
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->scratch_first_moments[group_index]),
            row_bytes,
            expected_count,
            (const int32_t *)gsx_cuda_backend_buffer_from_base(permutation_index_buffer)->ptr,
            expected_count,
            NULL,   // TODO: check out of range.
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda optimizer permutation kernel launch failed");
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(permutation_index_buffer);
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }

        cuda_error = gsx_cuda_gather_rows_kernel_launch(
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->second_moments[group_index]),
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->scratch_second_moments[group_index]),
            row_bytes,
            expected_count,
            (const int32_t *)gsx_cuda_backend_buffer_from_base(permutation_index_buffer)->ptr,
            expected_count,
            NULL,   // TODO: check out of range.
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda optimizer permutation kernel launch failed");
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(permutation_index_buffer);
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
    }

    error = gsx_backend_buffer_free(permutation_index_buffer);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
    }

    gsx_cuda_optim_commit_scratch(cuda_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_gather(gsx_optim_t optim, gsx_tensor_t indices)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    int32_t *index_values = NULL;
    gsx_backend_buffer_t index_buffer = NULL;
    gsx_size_t old_count = 0;
    gsx_size_t new_count = 0;
    gsx_index_t group_index = 0;
    void *stream = NULL;
    cudaError_t cuda_error = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_backend_get_major_stream(cuda_optim->base.backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        gsx_size_t group_old_count = 0;
        gsx_size_t group_new_count = 0;
        gsx_size_t old_row_bytes = 0;
        gsx_size_t new_row_bytes = 0;

        error = gsx_cuda_optim_validate_group_state_transition(
            cuda_optim,
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

    error = gsx_cuda_optim_validate_control_tensor(
        cuda_optim,
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
    error = gsx_cuda_optim_download_tensor_bytes(indices, index_values, new_count * sizeof(*index_values));
    if(!gsx_error_is_success(error)) {
        free(index_values);
        return error;
    }
    error = gsx_cuda_optim_sync_backend(cuda_optim);
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

    error = gsx_cuda_optim_upload_temp_index_buffer(cuda_optim, index_values, new_count, &index_buffer);
    free(index_values);
    index_values = NULL;
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_cuda_optim_prepare_scratch_for_target_layout(cuda_optim);
    if(!gsx_error_is_success(error)) {
        (void)gsx_backend_buffer_free(index_buffer);
        return error;
    }
    
    // TODO: may be reuse gsx_tensor_gather function.
    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        gsx_size_t row_bytes = 0;

        error = gsx_cuda_optim_compute_row_bytes(cuda_optim->first_moments[group_index], &row_bytes);
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(index_buffer);
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }

        cuda_error = gsx_cuda_gather_rows_kernel_launch(
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->first_moments[group_index]),
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->scratch_first_moments[group_index]),
            row_bytes,
            new_count,
            (const int32_t *)gsx_cuda_backend_buffer_from_base(index_buffer)->ptr,
            old_count,
            NULL,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda optimizer gather kernel launch failed");
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(index_buffer);
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }

        cuda_error = gsx_cuda_gather_rows_kernel_launch(
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->second_moments[group_index]),
            gsx_cuda_optim_tensor_device_bytes(cuda_optim->scratch_second_moments[group_index]),
            row_bytes,
            new_count,
            (const int32_t *)gsx_cuda_backend_buffer_from_base(index_buffer)->ptr,
            old_count,
            NULL,
            (cudaStream_t)stream
        );
        error = gsx_cuda_make_error(cuda_error, "cuda optimizer gather kernel launch failed");
        if(!gsx_error_is_success(error)) {
            (void)gsx_backend_buffer_free(index_buffer);
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
    }

    error = gsx_backend_buffer_free(index_buffer);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
    }

    gsx_cuda_optim_commit_scratch(cuda_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_optim_resize(gsx_optim_t optim, gsx_size_t new_count)
{
    gsx_cuda_optim *cuda_optim = (gsx_cuda_optim *)optim;
    gsx_size_t old_count = 0;
    gsx_index_t group_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_optim->base.param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        gsx_size_t group_old_count = 0;
        gsx_size_t group_new_count = 0;
        gsx_size_t old_row_bytes = 0;
        gsx_size_t new_row_bytes = 0;

        error = gsx_cuda_optim_validate_group_state_transition(
            cuda_optim,
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

    error = gsx_cuda_optim_prepare_scratch_for_target_layout(cuda_optim);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    // TODO: may be reuse gsx_tensor_resize function.
    for(group_index = 0; group_index < cuda_optim->base.param_group_count; ++group_index) {
        gsx_size_t row_bytes = 0;
        gsx_size_t copy_rows = old_count < new_count ? old_count : new_count;
        gsx_size_t copy_bytes = 0;

        error = gsx_cuda_optim_compute_row_bytes(cuda_optim->first_moments[group_index], &row_bytes);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
        copy_bytes = row_bytes * copy_rows;

        error = gsx_tensor_set_zero(cuda_optim->scratch_first_moments[group_index]);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
        error = gsx_tensor_set_zero(cuda_optim->scratch_second_moments[group_index]);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
        error = gsx_cuda_optim_copy_prefix_bytes(cuda_optim->first_moments[group_index], cuda_optim->scratch_first_moments[group_index], copy_bytes);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
        error = gsx_cuda_optim_copy_prefix_bytes(cuda_optim->second_moments[group_index], cuda_optim->scratch_second_moments[group_index], copy_bytes);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_optim_fail_after_prepare(cuda_optim, error);
        }
    }

    gsx_cuda_optim_commit_scratch(cuda_optim);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
