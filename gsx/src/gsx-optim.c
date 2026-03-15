#include "gsx-impl.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Shared optimizer layer — responsibilities and contracts:
 *
 * This file owns backend-agnostic dispatch, parameter-group registry, learning-rate
 * metadata, and input validation. It does not contain optimizer math, tensor data
 * access primitives, or compute scheduling decisions. Those are entirely owned by
 * each backend's create_optim implementation (e.g. gsx-cpu/optim.c).
 *
 * Stream contract: backends must schedule all optimizer work — step, reset, and
 * structural mutations — on the backend's caller-visible major stream. No
 * synchronization is inserted here; stream ordering at training-loop boundaries is
 * the caller's responsibility.
 *
 * Numerical parity: different backends may produce non-identical float32 results due
 * to differing FMA behavior, instruction ordering, or hardware precision
 * characteristics. Callers must use tolerance-based comparisons when comparing
 * outputs across backends; bitwise equality is not guaranteed.
 *
 * Mutation failure semantics: structural mutations (permute, gather, resize) may
 * partially apply their state changes before a failure is detected and returned.
 * On any failure the optimizer handle remains valid: subsequent info queries,
 * learning-rate get/set, reset, and free calls must succeed or return well-defined
 * errors. The caller is responsible for keeping parameter and gradient tensors
 * consistent with the post-failure optimizer state.
 */

bool gsx_optim_algorithm_is_valid(gsx_optim_algorithm algorithm)
{
    return algorithm == GSX_OPTIM_ALGORITHM_ADAM;
}

bool gsx_optim_param_role_is_valid(gsx_optim_param_role role)
{
    switch(role) {
    case GSX_OPTIM_PARAM_ROLE_MEAN3D:
    case GSX_OPTIM_PARAM_ROLE_LOGSCALE:
    case GSX_OPTIM_PARAM_ROLE_ROTATION:
    case GSX_OPTIM_PARAM_ROLE_OPACITY:
    case GSX_OPTIM_PARAM_ROLE_SH0:
    case GSX_OPTIM_PARAM_ROLE_SH1:
    case GSX_OPTIM_PARAM_ROLE_SH2:
    case GSX_OPTIM_PARAM_ROLE_SH3:
    case GSX_OPTIM_PARAM_ROLE_CUSTOM:
        return true;
    }

    return false;
}

bool gsx_optim_param_role_is_builtin(gsx_optim_param_role role)
{
    return role >= GSX_OPTIM_PARAM_ROLE_MEAN3D && role <= GSX_OPTIM_PARAM_ROLE_SH3;
}

bool gsx_optim_float_is_finite(gsx_float_t value)
{
    return isfinite((double)value) != 0;
}

static gsx_error gsx_optim_validate_param_group_numeric_fields(const gsx_optim_param_group_desc *param_group)
{
    if(!gsx_optim_float_is_finite(param_group->learning_rate) || param_group->learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer learning_rate must be finite and non-negative");
    }
    if(!gsx_optim_float_is_finite(param_group->beta1) || param_group->beta1 < 0.0f || param_group->beta1 >= 1.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer beta1 must be finite and in [0, 1)");
    }
    if(!gsx_optim_float_is_finite(param_group->beta2) || param_group->beta2 < 0.0f || param_group->beta2 >= 1.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer beta2 must be finite and in [0, 1)");
    }
    if(!gsx_optim_float_is_finite(param_group->weight_decay) || param_group->weight_decay < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer weight_decay must be finite and non-negative");
    }
    if(!gsx_optim_float_is_finite(param_group->epsilon) || param_group->epsilon <= 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer epsilon must be finite and strictly positive");
    }
    if(!gsx_optim_float_is_finite(param_group->max_grad)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer max_grad must be finite");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_optim_validate_param_group_tensors(const gsx_optim_param_group_desc *param_group, gsx_backend_t backend)
{
    gsx_tensor_t parameter = param_group->parameter;
    gsx_tensor_t gradient = param_group->gradient;
    gsx_index_t dim = 0;

    if(parameter == NULL || gradient == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer parameter and gradient tensors must be non-null");
    }
    if(parameter->backing_buffer == NULL || gradient->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer tensors must reference accessible storage");
    }
    if(parameter->backing_buffer->buffer_type->backend != backend || gradient->backing_buffer->buffer_type->backend != backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer tensors must belong to the requested backend");
    }
    if(parameter->rank != gradient->rank
        || parameter->data_type != gradient->data_type
        || parameter->storage_format != gradient->storage_format
        || parameter->size_bytes != gradient->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer parameter and gradient tensors must be compatible");
    }
    if(parameter->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer parameter and gradient tensors must use float32");
    }
    for(dim = 0; dim < parameter->rank; ++dim) {
        if(parameter->shape[dim] != gradient->shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer parameter and gradient tensor shapes must match");
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_optim_validate_param_groups(const gsx_optim_desc *desc, gsx_backend_t backend)
{
    bool seen_builtin_roles[GSX_OPTIM_BUILTIN_ROLE_COUNT] = { false };
    gsx_index_t index = 0;

    if(desc->param_group_count < 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer param_group_count must be non-negative");
    }
    if(desc->param_group_count != 0 && desc->param_groups == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "non-zero optimizer param_group_count requires a non-null param_groups pointer");
    }

    for(index = 0; index < desc->param_group_count; ++index) {
        const gsx_optim_param_group_desc *param_group = &desc->param_groups[index];
        gsx_error error = { GSX_ERROR_SUCCESS, NULL };

        if(!gsx_optim_param_role_is_valid(param_group->role)) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer param-group role is invalid");
        }
        if(gsx_optim_param_role_is_builtin(param_group->role)) {
            if(seen_builtin_roles[param_group->role]) {
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer built-in param-group roles must be unique");
            }
            seen_builtin_roles[param_group->role] = true;
        }

        error = gsx_optim_validate_param_group_numeric_fields(param_group);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_optim_validate_param_group_tensors(param_group, backend);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_optim_validate_desc(gsx_backend_t backend, const gsx_optim_desc *desc)
{
    if(backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and desc must be non-null");
    }
    if(!gsx_optim_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer algorithm is out of range");
    }
    if(desc->state_buffer_type != NULL && desc->state_buffer_type->backend != backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer state_buffer_type must belong to the requested backend");
    }

    return gsx_optim_validate_param_groups(desc, backend);
}

static gsx_error gsx_optim_require_handle(gsx_optim_t optim)
{
    if(optim == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim must be non-null");
    }
    if(optim->iface == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "optimizer implementation is missing an interface");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_optim_validate_index(const gsx_optim *optim, gsx_index_t index)
{
    if(index < 0 || index >= optim->param_group_count) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer param-group index is out of range");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_optim_base_init(
    gsx_optim *optim,
    const gsx_optim_i *iface,
    gsx_backend_t backend,
    const gsx_optim_desc *desc
)
{
    gsx_index_t index = 0;

    if(optim == NULL || iface == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim, iface, backend, and desc must be non-null");
    }

    memset(optim, 0, sizeof(*optim));
    optim->iface = iface;
    optim->backend = backend;
    optim->backend->live_optim_count += 1;
    optim->algorithm = desc->algorithm;
    optim->param_group_count = desc->param_group_count;
    for(index = 0; index < GSX_OPTIM_BUILTIN_ROLE_COUNT; ++index) {
        optim->role_to_index[index] = -1;
    }

    if(desc->state_buffer_type != NULL) {
        optim->state_buffer_type = desc->state_buffer_type;
    } else {
        gsx_error error = gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &optim->state_buffer_type);

        if(!gsx_error_is_success(error)) {
            gsx_optim_base_deinit(optim);
            return error;
        }
    }

    if(desc->param_group_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    optim->param_groups = (gsx_optim_param_group_desc *)calloc((size_t)desc->param_group_count, sizeof(*optim->param_groups));
    if(optim->param_groups == NULL) {
        gsx_optim_base_deinit(optim);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer param-group storage");
    }
    optim->learning_rates = (gsx_float_t *)calloc((size_t)desc->param_group_count, sizeof(*optim->learning_rates));
    if(optim->learning_rates == NULL) {
        gsx_optim_base_deinit(optim);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer learning-rate storage");
    }

    for(index = 0; index < desc->param_group_count; ++index) {
        optim->param_groups[index] = desc->param_groups[index];
        optim->learning_rates[index] = desc->param_groups[index].learning_rate;
        if(gsx_optim_param_role_is_builtin(desc->param_groups[index].role)) {
            optim->role_to_index[desc->param_groups[index].role] = index;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

void gsx_optim_base_deinit(gsx_optim *optim)
{
    if(optim == NULL) {
        return;
    }

    if(optim->backend != NULL && optim->backend->live_optim_count != 0) {
        optim->backend->live_optim_count -= 1;
    }
    free(optim->param_groups);
    free(optim->learning_rates);
    optim->param_groups = NULL;
    optim->learning_rates = NULL;
    optim->state_buffer_type = NULL;
    optim->backend = NULL;
    optim->iface = NULL;
    optim->algorithm = 0;
    optim->param_group_count = 0;
}

gsx_error gsx_optim_lookup_role_index(const gsx_optim *optim, gsx_optim_param_role role, gsx_index_t *out_index)
{
    if(optim == NULL || out_index == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim and out_index must be non-null");
    }
    if(role == GSX_OPTIM_PARAM_ROLE_CUSTOM) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "role-based optimizer lookup does not accept GSX_OPTIM_PARAM_ROLE_CUSTOM");
    }
    if(!gsx_optim_param_role_is_builtin(role)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer role is out of range");
    }
    if(optim->role_to_index[role] < 0) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "optimizer does not contain the requested built-in role");
    }

    *out_index = optim->role_to_index[role];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_optim_copy_param_group_desc(const gsx_optim *optim, gsx_index_t index, gsx_optim_param_group_desc *out_desc)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(optim == NULL || out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim and out_desc must be non-null");
    }

    error = gsx_optim_validate_index(optim, index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_desc = optim->param_groups[index];
    out_desc->learning_rate = optim->learning_rates[index];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_optim_select_param_groups(const gsx_optim *optim, const gsx_optim_step_request *request, bool *selected)
{
    const gsx_optim_param_role_flags known_role_flags =
        GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D | GSX_OPTIM_PARAM_ROLE_FLAG_LOGSCALE | GSX_OPTIM_PARAM_ROLE_FLAG_ROTATION
        | GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY | GSX_OPTIM_PARAM_ROLE_FLAG_SH0 | GSX_OPTIM_PARAM_ROLE_FLAG_SH1
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH2 | GSX_OPTIM_PARAM_ROLE_FLAG_SH3;
    gsx_index_t index = 0;

    if(optim == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optim must be non-null");
    }
    if(optim->param_group_count != 0 && selected == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "selected must be non-null when the optimizer has param groups");
    }
    if(request == NULL || request->force_all) {
        for(index = 0; index < optim->param_group_count; ++index) {
            selected[index] = true;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if((request->role_flags & ~known_role_flags) != 0u) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer step request role_flags contains unsupported bits");
    }
    if(request->param_group_index_count < 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "optimizer step request param_group_index_count must be non-negative");
    }
    if(request->param_group_index_count != 0 && request->param_group_indices == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "non-zero param_group_index_count requires a non-null param_group_indices pointer");
    }

    for(index = 0; index < GSX_OPTIM_BUILTIN_ROLE_COUNT; ++index) {
        if((request->role_flags & (1u << index)) != 0u && optim->role_to_index[index] >= 0) {
            selected[optim->role_to_index[index]] = true;
        }
    }
    for(index = 0; index < request->param_group_index_count; ++index) {
        gsx_index_t param_group_index = request->param_group_indices[index];
        gsx_error error = gsx_optim_validate_index(optim, param_group_index);

        if(!gsx_error_is_success(error)) {
            return error;
        }
        selected[param_group_index] = true;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_optim_init(gsx_optim_t *out_optim, gsx_backend_t backend, const gsx_optim_desc *desc)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_optim == NULL || backend == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_optim, backend, and desc must be non-null");
    }

    *out_optim = NULL;
    if(backend->iface == NULL || backend->iface->create_optim == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "backend does not implement optimizer creation");
    }

    error = gsx_optim_validate_desc(backend, desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return backend->iface->create_optim(backend, desc, out_optim);
}

GSX_API gsx_error gsx_optim_free(gsx_optim_t optim)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(optim->iface->destroy == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer destroy is not implemented");
    }

    return optim->iface->destroy(optim);
}

GSX_API gsx_error gsx_optim_get_info(gsx_optim_t optim, gsx_optim_info *out_info)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->algorithm = optim->algorithm;
    out_info->param_group_count = optim->param_group_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_optim_get_param_group_desc_by_index(gsx_optim_t optim, gsx_index_t index, gsx_optim_param_group_desc *out_desc)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_optim_copy_param_group_desc(optim, index, out_desc);
}

GSX_API gsx_error gsx_optim_get_param_group_desc_by_role(gsx_optim_t optim, gsx_optim_param_role role, gsx_optim_param_group_desc *out_desc)
{
    gsx_index_t index = 0;
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }

    error = gsx_optim_lookup_role_index(optim, role, &index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_optim_copy_param_group_desc(optim, index, out_desc);
}

GSX_API gsx_error gsx_optim_step(gsx_optim_t optim, const gsx_optim_step_request *request)
{
    bool *selected = NULL;
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(optim->iface->step_selected == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer step is not implemented");
    }

    if(optim->param_group_count != 0) {
        selected = (bool *)calloc((size_t)optim->param_group_count, sizeof(*selected));
        if(selected == NULL) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer step selection bitmap");
        }
    }

    error = gsx_optim_select_param_groups(optim, request, selected);
    if(gsx_error_is_success(error)) {
        error = optim->iface->step_selected(optim, selected);
    }

    free(selected);
    return error;
}

GSX_API gsx_error gsx_optim_get_learning_rate_by_index(gsx_optim_t optim, gsx_index_t index, gsx_float_t *out_learning_rate)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_learning_rate == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_learning_rate must be non-null");
    }

    error = gsx_optim_validate_index(optim, index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_learning_rate = optim->learning_rates[index];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_optim_get_learning_rate_by_role(gsx_optim_t optim, gsx_optim_param_role role, gsx_float_t *out_learning_rate)
{
    gsx_index_t index = 0;
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_learning_rate == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_learning_rate must be non-null");
    }

    error = gsx_optim_lookup_role_index(optim, role, &index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_learning_rate = optim->learning_rates[index];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_optim_set_learning_rate_by_index(gsx_optim_t optim, gsx_index_t index, gsx_float_t learning_rate)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_optim_float_is_finite(learning_rate) || learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "learning_rate must be finite and non-negative");
    }

    error = gsx_optim_validate_index(optim, index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    optim->learning_rates[index] = learning_rate;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_optim_set_learning_rate_by_role(gsx_optim_t optim, gsx_optim_param_role role, gsx_float_t learning_rate)
{
    gsx_index_t index = 0;
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_optim_float_is_finite(learning_rate) || learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "learning_rate must be finite and non-negative");
    }

    error = gsx_optim_lookup_role_index(optim, role, &index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    optim->learning_rates[index] = learning_rate;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(permutation == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "permutation must be non-null");
    }
    if(optim->iface->permute == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer permute is not implemented");
    }

    return optim->iface->permute(optim, permutation);
}

GSX_API gsx_error gsx_optim_gather(gsx_optim_t optim, gsx_tensor_t indices)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(indices == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "indices must be non-null");
    }
    if(optim->iface->gather == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer gather is not implemented");
    }

    return optim->iface->gather(optim, indices);
}

GSX_API gsx_error gsx_optim_resize(gsx_optim_t optim, gsx_size_t new_count)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(optim->iface->resize == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer resize is not implemented");
    }

    return optim->iface->resize(optim, new_count);
}

GSX_API gsx_error gsx_optim_reset(gsx_optim_t optim)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(optim->iface->reset_all == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer reset is not implemented");
    }

    return optim->iface->reset_all(optim);
}

GSX_API gsx_error gsx_optim_reset_param_group_by_index(gsx_optim_t optim, gsx_index_t index)
{
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(optim->iface->reset_by_index == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer per-group reset is not implemented");
    }

    error = gsx_optim_validate_index(optim, index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return optim->iface->reset_by_index(optim, index);
}

GSX_API gsx_error gsx_optim_reset_param_group_by_role(gsx_optim_t optim, gsx_optim_param_role role)
{
    gsx_index_t index = 0;
    gsx_error error = gsx_optim_require_handle(optim);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(optim->iface->reset_by_index == NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "optimizer per-group reset is not implemented");
    }

    error = gsx_optim_lookup_role_index(optim, role, &index);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return optim->iface->reset_by_index(optim, index);
}
