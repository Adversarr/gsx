#include "gsx-impl.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct gsx_scheduler {
    gsx_scheduler_desc desc;
    gsx_scheduler_state state;
};

static bool gsx_scheduler_algorithm_is_valid(gsx_scheduler_algorithm algorithm)
{
    return algorithm == GSX_SCHEDULER_ALGORITHM_CONSTANT || algorithm == GSX_SCHEDULER_ALGORITHM_DELAYED_EXPONENTIAL;
}

static bool gsx_scheduler_float_is_finite(gsx_float_t value)
{
    return isfinite((double)value) != 0;
}

static gsx_error gsx_scheduler_require_handle(gsx_scheduler_t scheduler)
{
    if(scheduler == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scheduler must be non-null");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_scheduler_validate_desc(const gsx_scheduler_desc *desc)
{
    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(!gsx_scheduler_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scheduler algorithm is out of range");
    }
    if(!gsx_scheduler_float_is_finite(desc->initial_learning_rate) || desc->initial_learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scheduler initial_learning_rate must be finite and non-negative");
    }
    if(!gsx_scheduler_float_is_finite(desc->final_learning_rate) || desc->final_learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scheduler final_learning_rate must be finite and non-negative");
    }
    if(!gsx_scheduler_float_is_finite(desc->delay_multiplier) || desc->delay_multiplier < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scheduler delay_multiplier must be finite and non-negative");
    }
    if(desc->decay_begin_step > desc->decay_end_step) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scheduler decay_begin_step must be less than or equal to decay_end_step");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_scheduler_validate_state(const gsx_scheduler_state *state)
{
    if(state == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "state must be non-null");
    }
    if(!gsx_scheduler_float_is_finite(state->current_learning_rate) || state->current_learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "scheduler current_learning_rate must be finite and non-negative");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static double gsx_scheduler_clamp01(double value)
{
    if(value < 0.0) {
        return 0.0;
    }
    if(value > 1.0) {
        return 1.0;
    }
    return value;
}

static double gsx_scheduler_lerp(double from, double to, double t)
{
    return from + (to - from) * t;
}

static double gsx_scheduler_compute_base_exponential_lerp(const gsx_scheduler_desc *desc, gsx_size_t global_step)
{
    double t = 0.0;
    double base_lr = 0.0;
    double initial_lr = (double)desc->initial_learning_rate;
    double final_lr = (double)desc->final_learning_rate;

    if(desc->decay_end_step > desc->decay_begin_step) {
        t = ((double)global_step - (double)desc->decay_begin_step) / ((double)desc->decay_end_step - (double)desc->decay_begin_step);
    } else if(global_step >= desc->decay_end_step) {
        t = 1.0;
    }
    t = gsx_scheduler_clamp01(t);

    if(initial_lr == 0.0 && final_lr == 0.0) {
        return 0.0;
    }
    if(initial_lr > 0.0 && final_lr > 0.0) {
        return exp((1.0 - t) * log(initial_lr) + t * log(final_lr));
    }

    base_lr = gsx_scheduler_lerp(initial_lr, final_lr, t);
    if(base_lr < 0.0) {
        return 0.0;
    }
    return base_lr;
}

static double gsx_scheduler_compute_delay_rate(const gsx_scheduler_desc *desc, gsx_size_t global_step)
{
    const double pi = 3.14159265358979323846;
    double ratio = 0.0;

    if(desc->delay_steps == 0 || global_step >= desc->delay_steps) {
        return 1.0;
    }

    ratio = (double)global_step / (double)desc->delay_steps;
    ratio = gsx_scheduler_clamp01(ratio);
    return (double)desc->delay_multiplier + (1.0 - (double)desc->delay_multiplier) * sin(0.5 * pi * ratio);
}

static gsx_error gsx_scheduler_evaluate_learning_rate(
    const gsx_scheduler_desc *desc,
    gsx_size_t global_step,
    gsx_float_t *out_learning_rate
)
{
    double learning_rate = 0.0;

    if(desc == NULL || out_learning_rate == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc and out_learning_rate must be non-null");
    }
    if(!gsx_scheduler_algorithm_is_valid(desc->algorithm)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scheduler algorithm is out of range");
    }

    switch(desc->algorithm) {
    case GSX_SCHEDULER_ALGORITHM_CONSTANT:
        learning_rate = (double)desc->initial_learning_rate;
        break;
    case GSX_SCHEDULER_ALGORITHM_DELAYED_EXPONENTIAL:
        learning_rate = gsx_scheduler_compute_base_exponential_lerp(desc, global_step)
            * gsx_scheduler_compute_delay_rate(desc, global_step);
        break;
    default:
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scheduler algorithm is out of range");
    }

    if(!isfinite(learning_rate) || learning_rate < 0.0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "scheduler evaluated an invalid learning_rate");
    }
    *out_learning_rate = (gsx_float_t)learning_rate;
    if(!gsx_scheduler_float_is_finite(*out_learning_rate) || *out_learning_rate < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "scheduler evaluated an invalid learning_rate");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_init(gsx_scheduler_t *out_scheduler, const gsx_scheduler_desc *desc)
{
    gsx_scheduler_t scheduler = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_scheduler == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_scheduler and desc must be non-null");
    }

    error = gsx_scheduler_validate_desc(desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    scheduler = (gsx_scheduler_t)calloc(1, sizeof(*scheduler));
    if(scheduler == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate scheduler");
    }

    memcpy(&scheduler->desc, desc, sizeof(*desc));
    scheduler->state.current_step = 0;
    scheduler->state.current_learning_rate = desc->initial_learning_rate;
    *out_scheduler = scheduler;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_free(gsx_scheduler_t scheduler)
{
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    free(scheduler);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_get_desc(gsx_scheduler_t scheduler, gsx_scheduler_desc *out_desc)
{
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }

    memcpy(out_desc, &scheduler->desc, sizeof(*out_desc));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_reset(gsx_scheduler_t scheduler)
{
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    scheduler->state.current_step = 0;
    scheduler->state.current_learning_rate = scheduler->desc.initial_learning_rate;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_get_state(gsx_scheduler_t scheduler, gsx_scheduler_state *out_state)
{
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_state == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_state must be non-null");
    }

    memcpy(out_state, &scheduler->state, sizeof(*out_state));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_set_state(gsx_scheduler_t scheduler, const gsx_scheduler_state *state)
{
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_scheduler_validate_state(state);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    memcpy(&scheduler->state, state, sizeof(*state));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_step(gsx_scheduler_t scheduler, gsx_size_t global_step, gsx_float_t *out_learning_rate)
{
    gsx_float_t learning_rate = 0.0f;
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_learning_rate == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_learning_rate must be non-null");
    }

    error = gsx_scheduler_evaluate_learning_rate(&scheduler->desc, global_step, &learning_rate);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    scheduler->state.current_step = global_step;
    scheduler->state.current_learning_rate = learning_rate;
    *out_learning_rate = learning_rate;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_scheduler_get_learning_rate(gsx_scheduler_t scheduler, gsx_float_t *out_learning_rate)
{
    gsx_error error = gsx_scheduler_require_handle(scheduler);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_learning_rate == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_learning_rate must be non-null");
    }

    *out_learning_rate = scheduler->state.current_learning_rate;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

enum {
    GSX_SESSION_CHECKPOINT_MAGIC = 0x314B5043u,
    GSX_SESSION_CHECKPOINT_FORMAT_VERSION = 1u
};

typedef struct gsx_session_checkpoint_payload_v1 {
    uint32_t magic;
    uint32_t format_version;
    gsx_size_t global_step;
    gsx_size_t epoch_index;
    gsx_size_t successful_step_count;
    gsx_size_t failed_step_count;
    gsx_backend_type backend_type;
    gsx_scheduler_algorithm scheduler_algorithm;
    gsx_size_t gaussian_count;
} gsx_session_checkpoint_payload_v1;

static gsx_error gsx_session_require_handle(gsx_session_t session)
{
    if(session == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session must be non-null");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_validate_desc(const gsx_session_desc *desc)
{
    gsx_size_t i = 0;
    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->backend == NULL || desc->gs == NULL || desc->optim == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, gs and optim must be non-null");
    }
    if(desc->dataloader == NULL || desc->renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dataloader and renderer must be non-null");
    }
    if(desc->loss_count == 0 || desc->loss_items == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "at least one loss item must be provided");
    }
    for(i = 0; i < desc->loss_count; ++i) {
        if(desc->loss_items[i].loss == NULL || desc->loss_items[i].context == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "each loss item must have non-null loss and context");
        }
        if(desc->loss_items[i].scale < 0.0f) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss scale must be non-negative");
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_free_step_tensors(gsx_session_t session)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(session->step_grad_prediction != NULL) {
        error = gsx_tensor_free(session->step_grad_prediction);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        session->step_grad_prediction = NULL;
    }
    if(session->step_loss_map != NULL) {
        error = gsx_tensor_free(session->step_loss_map);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        session->step_loss_map = NULL;
    }
    if(session->step_prediction != NULL) {
        error = gsx_tensor_free(session->step_prediction);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        session->step_prediction = NULL;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_session_tensor_desc_equal(const gsx_tensor_desc *a, const gsx_tensor_desc *b)
{
    gsx_index_t i = 0;

    if(a == NULL || b == NULL) {
        return false;
    }
    if(a->rank != b->rank || a->data_type != b->data_type || a->storage_format != b->storage_format) {
        return false;
    }
    for(i = 0; i < a->rank; ++i) {
        if(a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gsx_error gsx_session_prepare_step_tensors(gsx_session_t session, gsx_tensor_t target)
{
    gsx_tensor_desc target_desc = { 0 };
    gsx_tensor_desc current_desc = { 0 };
    gsx_tensor_desc alloc_desc = { 0 };
    bool has_compatible_tensors = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(target == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "target tensor must be non-null");
    }

    error = gsx_tensor_get_desc(target, &target_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(target_desc.storage_format != GSX_STORAGE_FORMAT_CHW) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session_step currently requires CHW dataloader output");
    }
    if(target_desc.data_type != GSX_DATA_TYPE_F32 && target_desc.data_type != GSX_DATA_TYPE_F16) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session_step currently supports F32/F16 image tensors only");
    }

    if(session->step_prediction != NULL && session->step_loss_map != NULL && session->step_grad_prediction != NULL) {
        error = gsx_tensor_get_desc(session->step_prediction, &current_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        has_compatible_tensors = gsx_session_tensor_desc_equal(&current_desc, &target_desc);
    }

    if(!has_compatible_tensors) {
        error = gsx_session_free_step_tensors(session);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(session->workspace_arena != NULL) {
            error = gsx_arena_reset(session->workspace_arena);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }

        memset(&alloc_desc, 0, sizeof(alloc_desc));
        alloc_desc.rank = target_desc.rank;
        memcpy(alloc_desc.shape, target_desc.shape, sizeof(alloc_desc.shape));
        alloc_desc.data_type = target_desc.data_type;
        alloc_desc.storage_format = target_desc.storage_format;
        alloc_desc.arena = session->workspace_arena;

        error = gsx_tensor_init(&session->step_prediction, &alloc_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_tensor_init(&session->step_loss_map, &alloc_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_tensor_init(&session->step_grad_prediction, &alloc_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    error = gsx_tensor_set_zero(session->step_loss_map);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(session->step_grad_prediction);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_session_is_optional_field_missing(gsx_error error)
{
    // GSX_ERROR_INVALID_STATE indicates that the gs tensor doesn't have the requested field
    // (returned by gsx_gsx_get_field when the field is not present in the gs tensor)
    return error.code == GSX_ERROR_INVALID_ARGUMENT || error.code == GSX_ERROR_OUT_OF_RANGE || error.code == GSX_ERROR_NOT_SUPPORTED
        || error.code == GSX_ERROR_INVALID_STATE;
}

static gsx_error gsx_session_try_get_optional_field(gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t *out_tensor)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor must be non-null");
    }
    *out_tensor = NULL;
    error = gsx_gs_get_field(gs, field, out_tensor);
    if(gsx_error_is_success(error)) {
        return error;
    }
    if(gsx_session_is_optional_field_missing(error)) {
        *out_tensor = NULL;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return error;
}

static gsx_error gsx_session_apply_scheduler_learning_rate(gsx_session_t session)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_float_t learning_rate = 0.0f;
    gsx_optim_param_role roles[] = {
        GSX_OPTIM_PARAM_ROLE_MEAN3D,
        GSX_OPTIM_PARAM_ROLE_LOGSCALE,
        GSX_OPTIM_PARAM_ROLE_ROTATION,
        GSX_OPTIM_PARAM_ROLE_OPACITY,
        GSX_OPTIM_PARAM_ROLE_SH0,
        GSX_OPTIM_PARAM_ROLE_SH1,
        GSX_OPTIM_PARAM_ROLE_SH2,
        GSX_OPTIM_PARAM_ROLE_SH3
    };
    gsx_index_t i = 0;

    if(session->scheduler == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_scheduler_step(session->scheduler, session->state.global_step, &learning_rate);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(i = 0; i < (gsx_index_t)(sizeof(roles) / sizeof(roles[0])); ++i) {
        error = gsx_optim_set_learning_rate_by_role(session->optim, roles[i], learning_rate);
        if(gsx_error_is_success(error)) {
            continue;
        }
        if(error.code == GSX_ERROR_OUT_OF_RANGE || error.code == GSX_ERROR_NOT_SUPPORTED) {
            continue;
        }
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_fill_checkpoint_info(gsx_session_t session, gsx_checkpoint_info *out_info)
{
    gsx_backend_info backend_info = { 0 };
    gsx_scheduler_desc scheduler_desc = { 0 };
    gsx_gs_info gs_info = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    memset(out_info, 0, sizeof(*out_info));
    out_info->format_version = GSX_SESSION_CHECKPOINT_FORMAT_VERSION;
    out_info->global_step = session->state.global_step;
    out_info->epoch_index = session->state.epoch_index;
    out_info->scheduler_algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;

    error = gsx_backend_get_info(session->backend, &backend_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    out_info->backend_type = backend_info.backend_type;

    if(session->scheduler != NULL) {
        error = gsx_scheduler_get_desc(session->scheduler, &scheduler_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        out_info->scheduler_algorithm = scheduler_desc.algorithm;
    }

    error = gsx_gs_get_info(session->gs, &gs_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    out_info->gaussian_count = gs_info.count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_step_fail(gsx_session_t session, gsx_error error)
{
    session->state.failed_step_count += 1;
    return error;
}

static gsx_error gsx_session_read_exact(const gsx_io_reader *reader, void *dst, gsx_size_t byte_count)
{
    gsx_size_t total_read = 0;

    while(total_read < byte_count) {
        gsx_size_t chunk_read = 0;
        gsx_error error = reader->read(reader->user_data, (uint8_t *)dst + total_read, byte_count - total_read, &chunk_read);

        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(chunk_read == 0) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "reader reached EOF before checkpoint payload completed");
        }
        total_read += chunk_read;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_init(gsx_session_t *out_session, const gsx_session_desc *desc)
{
    gsx_session_t session = NULL;
    gsx_backend_buffer_type_t device_buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t i = 0;

    if(out_session == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_session and desc must be non-null");
    }

    error = gsx_session_validate_desc(desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    session = (gsx_session_t)calloc(1, sizeof(*session));
    if(session == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate session");
    }

    session->backend = desc->backend;
    session->gs = desc->gs;
    session->optim = desc->optim;
    session->adc = desc->adc;
    session->dataloader = desc->dataloader;
    session->validation_dataloader = desc->validation_dataloader;
    session->scheduler = desc->scheduler;
    session->renderer = desc->renderer;
    session->state.global_step = desc->initial_global_step;
    session->state.epoch_index = desc->initial_epoch_index;
    session->state.successful_step_count = 0;
    session->state.failed_step_count = 0;

    session->loss_count = desc->loss_count;
    session->losses = (gsx_loss_t *)calloc(desc->loss_count, sizeof(gsx_loss_t));
    session->loss_contexts = (gsx_loss_context_t *)calloc(desc->loss_count, sizeof(gsx_loss_context_t));
    session->loss_scales = (gsx_float_t *)calloc(desc->loss_count, sizeof(gsx_float_t));
    if(session->losses == NULL || session->loss_contexts == NULL || session->loss_scales == NULL) {
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate loss arrays");
    }
    for(i = 0; i < desc->loss_count; ++i) {
        session->losses[i] = desc->loss_items[i].loss;
        session->loss_contexts[i] = desc->loss_items[i].context;
        session->loss_scales[i] = desc->loss_items[i].scale;
    }

    error = gsx_render_context_init(&session->render_context, session->renderer);
    if(!gsx_error_is_success(error)) {
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return error;
    }

    error = gsx_backend_find_buffer_type(session->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type);
    if(!gsx_error_is_success(error)) {
        (void)gsx_render_context_free(session->render_context);
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return error;
    }

    arena_desc.initial_capacity_bytes = 1u << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    error = gsx_arena_init(&session->workspace_arena, device_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        (void)gsx_render_context_free(session->render_context);
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return error;
    }

    *out_session = session;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_free(gsx_session_t session)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_session_free_step_tensors(session);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(session->workspace_arena != NULL) {
        error = gsx_arena_free(session->workspace_arena);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(session->render_context != NULL) {
        error = gsx_render_context_free(session->render_context);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    free(session->losses);
    free(session->loss_contexts);
    free(session->loss_scales);
    free(session);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_get_state(gsx_session_t session, gsx_session_state *out_state)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_state == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_state must be non-null");
    }

    memcpy(out_state, &session->state, sizeof(*out_state));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_set_state(gsx_session_t session, const gsx_session_state *state)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(state == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "state must be non-null");
    }

    memcpy(&session->state, state, sizeof(*state));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_get_checkpoint_info(gsx_session_t session, gsx_checkpoint_info *out_info)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_session_fill_checkpoint_info(session, out_info);
}

GSX_API gsx_error gsx_session_save_checkpoint(gsx_session_t session, const gsx_io_writer *writer, const gsx_checkpoint_info *info)
{
    gsx_checkpoint_info derived_info = { 0 };
    gsx_session_checkpoint_payload_v1 payload = { 0 };
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(writer == NULL || writer->write == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "writer and writer->write must be non-null");
    }

    if(info != NULL) {
        derived_info = *info;
    } else {
        error = gsx_session_fill_checkpoint_info(session, &derived_info);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    payload.magic = GSX_SESSION_CHECKPOINT_MAGIC;
    payload.format_version = GSX_SESSION_CHECKPOINT_FORMAT_VERSION;
    payload.global_step = session->state.global_step;
    payload.epoch_index = session->state.epoch_index;
    payload.successful_step_count = session->state.successful_step_count;
    payload.failed_step_count = session->state.failed_step_count;
    payload.backend_type = derived_info.backend_type;
    payload.scheduler_algorithm = derived_info.scheduler_algorithm;
    payload.gaussian_count = derived_info.gaussian_count;

    return writer->write(writer->user_data, &payload, (gsx_size_t)sizeof(payload));
}

GSX_API gsx_error gsx_session_load_checkpoint(gsx_session_t session, const gsx_io_reader *reader, gsx_checkpoint_info *out_info)
{
    gsx_session_checkpoint_payload_v1 payload = { 0 };
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(reader == NULL || reader->read == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reader and reader->read must be non-null");
    }

    error = gsx_session_read_exact(reader, &payload, (gsx_size_t)sizeof(payload));
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(payload.magic != GSX_SESSION_CHECKPOINT_MAGIC) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "checkpoint magic mismatch");
    }
    if(payload.format_version != GSX_SESSION_CHECKPOINT_FORMAT_VERSION) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "checkpoint format_version is not supported");
    }

    session->state.global_step = payload.global_step;
    session->state.epoch_index = payload.epoch_index;
    session->state.successful_step_count = payload.successful_step_count;
    session->state.failed_step_count = payload.failed_step_count;

    if(out_info != NULL) {
        memset(out_info, 0, sizeof(*out_info));
        out_info->format_version = payload.format_version;
        out_info->global_step = payload.global_step;
        out_info->epoch_index = payload.epoch_index;
        out_info->backend_type = payload.backend_type;
        out_info->scheduler_algorithm = payload.scheduler_algorithm;
        out_info->gaussian_count = payload.gaussian_count;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#define GSX_SESSION_STEP_TRY(call_expr) \
    do { \
        error = (call_expr); \
        if(!gsx_error_is_success(error)) { \
            return gsx_session_step_fail(session, error); \
        } \
    } while(0)

#define GSX_SESSION_STEP_GET_FIELD(field, out_tensor) \
    GSX_SESSION_STEP_TRY(gsx_gs_get_field(session->gs, (field), (out_tensor)))

#define GSX_SESSION_STEP_GET_OPTIONAL_FIELD(field, out_tensor) \
    GSX_SESSION_STEP_TRY(gsx_session_try_get_optional_field(session->gs, (field), (out_tensor)))

GSX_API gsx_error gsx_session_step(gsx_session_t session)
{
    gsx_dataloader_result batch = { 0 };
    gsx_tensor_t mean3d = NULL;
    gsx_tensor_t rotation = NULL;
    gsx_tensor_t logscale = NULL;
    gsx_tensor_t opacity = NULL;
    gsx_tensor_t sh0 = NULL;
    gsx_tensor_t sh1 = NULL;
    gsx_tensor_t sh2 = NULL;
    gsx_tensor_t sh3 = NULL;
    gsx_tensor_t visible_counter = NULL;
    gsx_tensor_t max_screen_radius = NULL;
    gsx_tensor_t grad_mean3d = NULL;
    gsx_tensor_t grad_rotation = NULL;
    gsx_tensor_t grad_logscale = NULL;
    gsx_tensor_t grad_opacity = NULL;
    gsx_tensor_t grad_sh0 = NULL;
    gsx_tensor_t grad_sh1 = NULL;
    gsx_tensor_t grad_sh2 = NULL;
    gsx_tensor_t grad_sh3 = NULL;
    gsx_tensor_t grad_acc = NULL;
    gsx_tensor_t absgrad_acc = NULL;
    gsx_tensor_desc target_desc = { 0 };
    gsx_render_forward_request forward_request = { 0 };
    gsx_render_backward_request backward_request = { 0 };
    gsx_loss_forward_request loss_forward_request = { 0 };
    gsx_loss_backward_request loss_backward_request = { 0 };
    gsx_optim_step_request optim_step_request = { 0 };
    gsx_error error = gsx_session_require_handle(session);
    gsx_index_t sh_degree = 0;
    gsx_index_t i = 0;

    if(!gsx_error_is_success(error)) {
        return error;
    }

    GSX_SESSION_STEP_TRY(gsx_dataloader_next_ex(session->dataloader, &batch));

    if((batch.boundary_flags & GSX_DATALOADER_BOUNDARY_NEW_EPOCH) != 0u) {
        session->state.epoch_index += 1;
    }

    GSX_SESSION_STEP_TRY(gsx_session_prepare_step_tensors(session, batch.rgb_image));

    GSX_SESSION_STEP_TRY(gsx_tensor_get_desc(batch.rgb_image, &target_desc));

    GSX_SESSION_STEP_TRY(gsx_gs_zero_gradients(session->gs));

    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_MEAN3D, &mean3d);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_ROTATION, &rotation);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_LOGSCALE, &logscale);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_OPACITY, &opacity);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_SH0, &sh0);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_SH1, &sh1);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_SH2, &sh2);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_SH3, &sh3);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_VISIBLE_COUNTER, &visible_counter);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_MAX_SCREEN_RADIUS, &max_screen_radius);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_GRAD_MEAN3D, &grad_mean3d);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_GRAD_ROTATION, &grad_rotation);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_GRAD_LOGSCALE, &grad_logscale);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_GRAD_OPACITY, &grad_opacity);
    GSX_SESSION_STEP_GET_FIELD(GSX_GS_FIELD_GRAD_SH0, &grad_sh0);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_GRAD_SH1, &grad_sh1);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_GRAD_SH2, &grad_sh2);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_GRAD_SH3, &grad_sh3);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_GRAD_ACC, &grad_acc);
    GSX_SESSION_STEP_GET_OPTIONAL_FIELD(GSX_GS_FIELD_ABSGRAD_ACC, &absgrad_acc);

    if(sh3 != NULL || grad_sh3 != NULL) {
        if(sh1 == NULL || sh2 == NULL || grad_sh1 == NULL || grad_sh2 == NULL || grad_sh3 == NULL) {
            return gsx_session_step_fail(session, gsx_make_error(GSX_ERROR_INVALID_STATE, "inconsistent SH3 field availability"));
        }
        sh_degree = 3;
    } else if(sh2 != NULL || grad_sh2 != NULL) {
        if(sh1 == NULL || grad_sh1 == NULL || grad_sh2 == NULL) {
            return gsx_session_step_fail(session, gsx_make_error(GSX_ERROR_INVALID_STATE, "inconsistent SH2 field availability"));
        }
        sh_degree = 2;
    } else if(sh1 != NULL || grad_sh1 != NULL) {
        if(grad_sh1 == NULL) {
            return gsx_session_step_fail(session, gsx_make_error(GSX_ERROR_INVALID_STATE, "inconsistent SH1 field availability"));
        }
        sh_degree = 1;
    }

    forward_request.intrinsics = &batch.intrinsics;
    forward_request.pose = &batch.pose;
    forward_request.near_plane = 0.01f;
    forward_request.far_plane = 1000.0f;
    forward_request.background_color = (gsx_vec3){ 0.0f, 0.0f, 0.0f };
    forward_request.precision = target_desc.data_type == GSX_DATA_TYPE_F16 ? GSX_RENDER_PRECISION_FLOAT16 : GSX_RENDER_PRECISION_FLOAT32;
    forward_request.sh_degree = sh_degree;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    forward_request.borrow_train_state = true;
    forward_request.gs_mean3d = mean3d;
    forward_request.gs_rotation = rotation;
    forward_request.gs_logscale = logscale;
    forward_request.gs_sh0 = sh0;
    forward_request.gs_sh1 = sh1;
    forward_request.gs_sh2 = sh2;
    forward_request.gs_sh3 = sh3;
    forward_request.gs_opacity = opacity;
    forward_request.out_rgb = session->step_prediction;
    forward_request.gs_visible_counter = visible_counter;
    forward_request.gs_max_screen_radius = max_screen_radius;

    GSX_SESSION_STEP_TRY(gsx_renderer_render(session->renderer, session->render_context, &forward_request));

    for(i = 0; i < (gsx_index_t)session->loss_count; ++i) {
        loss_forward_request.prediction = session->step_prediction;
        loss_forward_request.target = batch.rgb_image;
        loss_forward_request.loss_map_accumulator = session->step_loss_map;
        loss_forward_request.train = true;
        loss_forward_request.scale = session->loss_scales[i];
        GSX_SESSION_STEP_TRY(gsx_loss_forward(session->losses[i], session->loss_contexts[i], &loss_forward_request));
    }

    for(i = 0; i < (gsx_index_t)session->loss_count; ++i) {
        loss_backward_request.grad_prediction_accumulator = session->step_grad_prediction;
        loss_backward_request.scale = session->loss_scales[i];
        GSX_SESSION_STEP_TRY(gsx_loss_backward(session->losses[i], session->loss_contexts[i], &loss_backward_request));
    }

    backward_request.grad_rgb = session->step_grad_prediction;
    backward_request.grad_gs_mean3d = grad_mean3d;
    backward_request.grad_gs_rotation = grad_rotation;
    backward_request.grad_gs_logscale = grad_logscale;
    backward_request.grad_gs_sh0 = grad_sh0;
    backward_request.grad_gs_sh1 = grad_sh1;
    backward_request.grad_gs_sh2 = grad_sh2;
    backward_request.grad_gs_sh3 = grad_sh3;
    backward_request.grad_gs_opacity = grad_opacity;
    backward_request.gs_grad_acc = grad_acc;
    backward_request.gs_absgrad_acc = absgrad_acc;
    GSX_SESSION_STEP_TRY(gsx_renderer_backward(session->renderer, session->render_context, &backward_request));

    GSX_SESSION_STEP_TRY(gsx_session_apply_scheduler_learning_rate(session));

    optim_step_request.role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D
        | GSX_OPTIM_PARAM_ROLE_FLAG_LOGSCALE
        | GSX_OPTIM_PARAM_ROLE_FLAG_ROTATION
        | GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH0
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH1
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH2
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH3;
    optim_step_request.force_all = false;
    GSX_SESSION_STEP_TRY(gsx_optim_step(session->optim, &optim_step_request));

    if(session->adc != NULL) {
        gsx_adc_request adc_request = { 0 };
        gsx_adc_result adc_result = { 0 };

        adc_request.gs = session->gs;
        adc_request.optim = session->optim;
        adc_request.dataloader = session->dataloader;
        adc_request.renderer = session->renderer;
        adc_request.global_step = session->state.global_step;
        adc_request.scene_scale = 1.0f;
        GSX_SESSION_STEP_TRY(gsx_adc_step(session->adc, &adc_request, &adc_result));
    }

    session->state.global_step += 1;
    session->state.successful_step_count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#undef GSX_SESSION_STEP_GET_OPTIONAL_FIELD
#undef GSX_SESSION_STEP_GET_FIELD
#undef GSX_SESSION_STEP_TRY
