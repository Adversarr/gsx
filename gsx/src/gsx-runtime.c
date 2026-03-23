#include "gsx-impl.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    GSX_SESSION_CHECKPOINT_FORMAT_VERSION = 2u
};

typedef struct gsx_session_checkpoint_payload_v2 {
    uint32_t magic;
    uint32_t format_version;
    gsx_size_t global_step;
    gsx_size_t epoch_index;
    gsx_size_t successful_step_count;
    gsx_size_t failed_step_count;
    gsx_backend_type backend_type;
    gsx_scheduler_algorithm scheduler_algorithm;
    gsx_size_t gaussian_count;
    uint32_t has_scheduler_state;
    gsx_size_t scheduler_current_step;
    gsx_float_t scheduler_current_learning_rate;
} gsx_session_checkpoint_payload_v2;

static gsx_error gsx_session_require_handle(gsx_session_t session)
{
    if(session == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session must be non-null");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_validate_desc(const gsx_session_desc *desc)
{
    gsx_dataloader_info train_dataloader_info = { 0 };
    gsx_renderer_info renderer_info = { 0 };
    gsx_renderer_capabilities renderer_capabilities = { 0 };
    gsx_data_type renderer_output_data_type = GSX_DATA_TYPE_F32;
    gsx_size_t i = 0;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->backend == NULL || desc->gs == NULL || desc->optim == NULL || desc->renderer == NULL || desc->train_dataloader == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, gs, optim, renderer and train_dataloader must be non-null");
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

    if(desc->render.near_plane <= 0.0f || desc->render.far_plane <= desc->render.near_plane) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session render near/far planes are invalid");
    }
    if(desc->render.sh_degree_mode != GSX_SESSION_SH_DEGREE_MODE_AUTO_FROM_GS
        && desc->render.sh_degree_mode != GSX_SESSION_SH_DEGREE_MODE_EXPLICIT) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "session render sh_degree_mode is out of range");
    }
    if(desc->render.sh_degree_mode == GSX_SESSION_SH_DEGREE_MODE_EXPLICIT
        && (desc->render.sh_degree < 0 || desc->render.sh_degree > 3)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "session render sh_degree must be in [0, 3]");
    }
    if(desc->workspace.buffer_type_class < GSX_BACKEND_BUFFER_TYPE_HOST
        || desc->workspace.buffer_type_class > GSX_BACKEND_BUFFER_TYPE_UNIFIED) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "session workspace buffer_type_class is out of range");
    }
    if(desc->workspace.arena_desc.dry_run) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session workspace arena_desc.dry_run must be false");
    }
    if(!desc->optim_step.force_all && desc->optim_step.role_flags == 0u && desc->optim_step.param_group_index_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session optim_step must select at least one optimizer group");
    }
    if(desc->optim_step.param_group_index_count > 0 && desc->optim_step.param_group_indices == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session optim_step param_group_indices must be non-null when count is non-zero");
    }

    if(desc->adc_step.enabled) {
        if(desc->adc == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session adc must be non-null when adc_step is enabled");
        }
        if(!gsx_scheduler_float_is_finite(desc->adc_step.scene_scale) || desc->adc_step.scene_scale <= 0.0f) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session adc_step.scene_scale must be finite and positive");
        }
        if(desc->adc_step.dataloader != NULL) {
            gsx_error error = gsx_dataloader_get_info(desc->adc_step.dataloader, &train_dataloader_info);

            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
    } else if(desc->adc_step.dataloader != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session adc_step.dataloader requires adc_step.enabled");
    }

    {
        gsx_error error = gsx_dataloader_get_info(desc->train_dataloader, &train_dataloader_info);

        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(train_dataloader_info.storage_format != GSX_STORAGE_FORMAT_CHW) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session train_dataloader must produce CHW tensors");
    }
    if(train_dataloader_info.image_data_type != GSX_DATA_TYPE_F32 && train_dataloader_info.image_data_type != GSX_DATA_TYPE_F16) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session train_dataloader must produce F32 or F16 image tensors");
    }

    {
        gsx_error error = gsx_renderer_get_info(desc->renderer, &renderer_info);

        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_renderer_get_capabilities(desc->renderer, &renderer_capabilities);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_renderer_get_output_data_type(desc->renderer, desc->render.precision, &renderer_output_data_type);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    if(desc->render.precision == GSX_RENDER_PRECISION_FLOAT32
        && (renderer_capabilities.supported_precisions & GSX_RENDER_PRECISION_FLAG_FLOAT32) == 0u) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "session render precision FLOAT32 is not supported by the renderer");
    }
    if(desc->render.precision == GSX_RENDER_PRECISION_FLOAT16
        && (renderer_capabilities.supported_precisions & GSX_RENDER_PRECISION_FLAG_FLOAT16) == 0u) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "session render precision FLOAT16 is not supported by the renderer");
    }
    if(renderer_info.width != train_dataloader_info.output_width || renderer_info.height != train_dataloader_info.output_height) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session renderer geometry must match train_dataloader output geometry");
    }
    if(renderer_output_data_type != train_dataloader_info.image_data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "session renderer output type must match train_dataloader image_data_type");
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
    if(session->retained_target != NULL) {
        error = gsx_tensor_free(session->retained_target);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        session->retained_target = NULL;
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
    bool retain_target = false;
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

    retain_target = session->report_desc.retain_target;

    if(session->step_prediction != NULL && session->step_loss_map != NULL && session->step_grad_prediction != NULL
        && (!retain_target || session->retained_target != NULL)) {
        error = gsx_tensor_get_desc(session->step_prediction, &current_desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        has_compatible_tensors = gsx_session_tensor_desc_equal(&current_desc, &target_desc);
        if(has_compatible_tensors && retain_target) {
            error = gsx_tensor_get_desc(session->retained_target, &current_desc);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            has_compatible_tensors = gsx_session_tensor_desc_equal(&current_desc, &target_desc);
        }
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
        if(retain_target) {
            error = gsx_tensor_init(&session->retained_target, &alloc_desc);
            if(!gsx_error_is_success(error)) {
                return error;
            }
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

static bool gsx_session_collects_outputs(gsx_session_t session)
{
    return session->report_desc.retain_prediction || session->report_desc.retain_target || session->report_desc.retain_loss_map
        || session->report_desc.retain_grad_prediction;
}

static double gsx_session_get_time_us(void)
{
    struct timespec ts;

    if(timespec_get(&ts, TIME_UTC) != TIME_UTC) {
        return 0.0;
    }
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

static gsx_error gsx_session_sync_backend_if_needed(gsx_session_t session)
{
    gsx_backend_info info = { 0 };
    gsx_error error = gsx_backend_get_info(session->backend, &info);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(info.backend_type == GSX_BACKEND_TYPE_CPU) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_backend_major_stream_sync(session->backend);
}

static gsx_error gsx_session_begin_stage_timing(gsx_session_t session, bool enabled, double *out_start_us)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_start_us == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_start_us must be non-null");
    }
    *out_start_us = 0.0;
    if(!enabled) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_session_sync_backend_if_needed(session);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_start_us = gsx_session_get_time_us();
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_end_stage_timing(gsx_session_t session, bool enabled, double start_us, double *out_elapsed_us)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_elapsed_us == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_elapsed_us must be non-null");
    }
    *out_elapsed_us = 0.0;
    if(!enabled) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_session_sync_backend_if_needed(session);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    *out_elapsed_us = gsx_session_get_time_us() - start_us;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_session_optim_role_selected(gsx_session_t session, gsx_optim_param_role role)
{
    gsx_optim_param_role_flags role_flag = 0u;

    if(session->optim_step_desc.force_all) {
        return true;
    }

    switch(role) {
    case GSX_OPTIM_PARAM_ROLE_MEAN3D:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D;
        break;
    case GSX_OPTIM_PARAM_ROLE_LOGSCALE:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_LOGSCALE;
        break;
    case GSX_OPTIM_PARAM_ROLE_ROTATION:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_ROTATION;
        break;
    case GSX_OPTIM_PARAM_ROLE_OPACITY:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY;
        break;
    case GSX_OPTIM_PARAM_ROLE_SH0:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_SH0;
        break;
    case GSX_OPTIM_PARAM_ROLE_SH1:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_SH1;
        break;
    case GSX_OPTIM_PARAM_ROLE_SH2:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_SH2;
        break;
    case GSX_OPTIM_PARAM_ROLE_SH3:
        role_flag = GSX_OPTIM_PARAM_ROLE_FLAG_SH3;
        break;
    default:
        return false;
    }

    return (session->optim_step_desc.role_flags & role_flag) != 0u;
}

static gsx_error gsx_session_apply_scheduler_learning_rate(gsx_session_t session, bool *out_has_learning_rate, gsx_float_t *out_learning_rate)
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

    if(out_has_learning_rate == NULL || out_learning_rate == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_has_learning_rate and out_learning_rate must be non-null");
    }
    *out_has_learning_rate = false;
    *out_learning_rate = 0.0f;

    if(session->scheduler == NULL) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_scheduler_step(session->scheduler, session->state.global_step, &learning_rate);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(i = 0; i < (gsx_index_t)(sizeof(roles) / sizeof(roles[0])); ++i) {
        if(!gsx_session_optim_role_selected(session, roles[i])) {
            continue;
        }

        error = gsx_optim_set_learning_rate_by_role(session->optim, roles[i], learning_rate);
        if(gsx_error_is_success(error)) {
            continue;
        }
        if(error.code == GSX_ERROR_OUT_OF_RANGE || error.code == GSX_ERROR_NOT_SUPPORTED) {
            continue;
        }
        return error;
    }

    *out_has_learning_rate = true;
    *out_learning_rate = learning_rate;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_session_plan_workspace_capacity(gsx_session_t session, gsx_size_t *out_required_bytes)
{
    gsx_dataloader_info dataloader_info = { 0 };
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_tensor_desc descs[4] = { 0 };
    gsx_arena_desc arena_desc = { 0 };
    gsx_data_type output_data_type = GSX_DATA_TYPE_F32;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_index_t tensor_count = 0;

    if(out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_required_bytes must be non-null");
    }
    *out_required_bytes = 0;

    error = gsx_dataloader_get_info(session->train_dataloader, &dataloader_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_renderer_get_output_data_type(session->renderer, session->render_desc.precision, &output_data_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_backend_find_buffer_type(session->backend, session->workspace_desc.buffer_type_class, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    memset(descs, 0, sizeof(descs));
    descs[0].rank = 3;
    descs[0].shape[0] = 3;
    descs[0].shape[1] = dataloader_info.output_height;
    descs[0].shape[2] = dataloader_info.output_width;
    descs[0].data_type = output_data_type;
    descs[0].storage_format = GSX_STORAGE_FORMAT_CHW;
    descs[1] = descs[0];
    descs[2] = descs[0];
    tensor_count = 3;
    if(session->report_desc.retain_target) {
        descs[3] = descs[0];
        tensor_count = 4;
    }

    arena_desc = session->workspace_desc.arena_desc;
    return gsx_tensor_plan_required_bytes(buffer_type, &arena_desc, descs, tensor_count, out_required_bytes);
}

static gsx_error gsx_session_resolve_outputs(gsx_session_t session, gsx_session_outputs *out_outputs)
{
    if(out_outputs == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_outputs must be non-null");
    }

    memset(out_outputs, 0, sizeof(*out_outputs));
    if(!session->has_last_step_report || !gsx_session_collects_outputs(session)) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "session does not have retained outputs available yet");
    }

    if(session->report_desc.retain_prediction) {
        out_outputs->prediction = session->step_prediction;
    }
    if(session->report_desc.retain_target) {
        out_outputs->target = session->retained_target;
    }
    if(session->report_desc.retain_loss_map) {
        out_outputs->loss_map = session->step_loss_map;
    }
    if(session->report_desc.retain_grad_prediction) {
        out_outputs->grad_prediction = session->step_grad_prediction;
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
    gsx_backend_buffer_type_t workspace_buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t i = 0;
    gsx_size_t workspace_required_bytes = 0;

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
    session->renderer = desc->renderer;
    session->train_dataloader = desc->train_dataloader;
    session->adc = desc->adc;
    session->scheduler = desc->scheduler;
    session->adc_dataloader = desc->adc_step.dataloader != NULL ? desc->adc_step.dataloader : desc->train_dataloader;
    session->render_desc = desc->render;
    session->adc_step_desc = desc->adc_step;
    session->workspace_desc = desc->workspace;
    session->report_desc = desc->reporting;
    session->initial_state.global_step = desc->initial_global_step;
    session->initial_state.epoch_index = desc->initial_epoch_index;
    session->initial_state.successful_step_count = 0;
    session->initial_state.failed_step_count = 0;
    session->state = session->initial_state;
    session->has_last_step_report = false;

    session->loss_count = desc->loss_count;
    session->loss_items = (gsx_loss_item *)calloc(desc->loss_count, sizeof(gsx_loss_item));
    session->losses = (gsx_loss_t *)calloc(desc->loss_count, sizeof(gsx_loss_t));
    session->loss_contexts = (gsx_loss_context_t *)calloc(desc->loss_count, sizeof(gsx_loss_context_t));
    session->loss_scales = (gsx_float_t *)calloc(desc->loss_count, sizeof(gsx_float_t));
    if(session->loss_items == NULL || session->losses == NULL || session->loss_contexts == NULL || session->loss_scales == NULL) {
        free(session->loss_items);
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate loss arrays");
    }
    for(i = 0; i < desc->loss_count; ++i) {
        session->loss_items[i] = desc->loss_items[i];
        session->losses[i] = desc->loss_items[i].loss;
        session->loss_contexts[i] = desc->loss_items[i].context;
        session->loss_scales[i] = desc->loss_items[i].scale;
    }

    session->optim_step_desc.force_all = desc->optim_step.force_all;
    session->optim_step_desc.role_flags = desc->optim_step.role_flags;
    session->optim_step_desc.param_group_index_count = desc->optim_step.param_group_index_count;
    if(desc->optim_step.param_group_index_count > 0) {
        session->optim_param_group_indices = (gsx_index_t *)calloc(desc->optim_step.param_group_index_count, sizeof(gsx_index_t));
        if(session->optim_param_group_indices == NULL) {
            free(session->loss_items);
            free(session->losses);
            free(session->loss_contexts);
            free(session->loss_scales);
            free(session);
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate optimizer param-group indices");
        }
        memcpy(
            session->optim_param_group_indices,
            desc->optim_step.param_group_indices,
            (size_t)desc->optim_step.param_group_index_count * sizeof(gsx_index_t));
        session->optim_step_desc.param_group_indices = session->optim_param_group_indices;
    }

    error = gsx_render_context_init(&session->render_context, session->renderer);
    if(!gsx_error_is_success(error)) {
        free(session->optim_param_group_indices);
        free(session->loss_items);
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return error;
    }

    error = gsx_backend_find_buffer_type(session->backend, session->workspace_desc.buffer_type_class, &workspace_buffer_type);
    if(!gsx_error_is_success(error)) {
        (void)gsx_render_context_free(session->render_context);
        free(session->optim_param_group_indices);
        free(session->loss_items);
        free(session->losses);
        free(session->loss_contexts);
        free(session->loss_scales);
        free(session);
        return error;
    }

    arena_desc = session->workspace_desc.arena_desc;
    if(session->workspace_desc.auto_plan) {
        error = gsx_session_plan_workspace_capacity(session, &workspace_required_bytes);
        if(!gsx_error_is_success(error)) {
            (void)gsx_render_context_free(session->render_context);
            free(session->optim_param_group_indices);
            free(session->loss_items);
            free(session->losses);
            free(session->loss_contexts);
            free(session->loss_scales);
            free(session);
            return error;
        }
        arena_desc.initial_capacity_bytes = workspace_required_bytes;
        session->workspace_desc.arena_desc.initial_capacity_bytes = workspace_required_bytes;
    }

    error = gsx_arena_init(&session->workspace_arena, workspace_buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        (void)gsx_render_context_free(session->render_context);
        free(session->optim_param_group_indices);
        free(session->loss_items);
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

    free(session->optim_param_group_indices);
    free(session->loss_items);
    free(session->losses);
    free(session->loss_contexts);
    free(session->loss_scales);
    free(session);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_get_desc(gsx_session_t session, gsx_session_desc *out_desc)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_desc must be non-null");
    }

    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->backend = session->backend;
    out_desc->gs = session->gs;
    out_desc->optim = session->optim;
    out_desc->renderer = session->renderer;
    out_desc->train_dataloader = session->train_dataloader;
    out_desc->adc = session->adc;
    out_desc->scheduler = session->scheduler;
    out_desc->loss_count = session->loss_count;
    out_desc->loss_items = session->loss_items;
    out_desc->render = session->render_desc;
    out_desc->optim_step = session->optim_step_desc;
    out_desc->adc_step = session->adc_step_desc;
    out_desc->adc_step.dataloader = session->adc_dataloader;
    out_desc->workspace = session->workspace_desc;
    out_desc->reporting = session->report_desc;
    out_desc->initial_global_step = session->initial_state.global_step;
    out_desc->initial_epoch_index = session->initial_state.epoch_index;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_reset(gsx_session_t session)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_dataloader_reset(session->train_dataloader);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(session->adc_step_desc.enabled && session->adc_dataloader != NULL && session->adc_dataloader != session->train_dataloader) {
        error = gsx_dataloader_reset(session->adc_dataloader);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(session->scheduler != NULL) {
        error = gsx_scheduler_reset(session->scheduler);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    session->state = session->initial_state;
    session->has_last_step_report = false;
    memset(&session->last_step_report, 0, sizeof(session->last_step_report));
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

GSX_API gsx_error gsx_session_get_last_step_report(gsx_session_t session, gsx_session_step_report *out_report)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_report == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_report must be non-null");
    }
    if(!session->has_last_step_report) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "session does not have a successful step report yet");
    }

    memcpy(out_report, &session->last_step_report, sizeof(*out_report));
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_session_get_last_outputs(gsx_session_t session, gsx_session_outputs *out_outputs)
{
    gsx_error error = gsx_session_require_handle(session);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_session_resolve_outputs(session, out_outputs);
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
    gsx_session_checkpoint_payload_v2 payload = { 0 };
    gsx_scheduler_state scheduler_state = { 0 };
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
    payload.has_scheduler_state = session->scheduler != NULL ? 1u : 0u;
    if(session->scheduler != NULL) {
        error = gsx_scheduler_get_state(session->scheduler, &scheduler_state);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        payload.scheduler_current_step = scheduler_state.current_step;
        payload.scheduler_current_learning_rate = scheduler_state.current_learning_rate;
    }

    return writer->write(writer->user_data, &payload, (gsx_size_t)sizeof(payload));
}

GSX_API gsx_error gsx_session_load_checkpoint(gsx_session_t session, const gsx_io_reader *reader, gsx_checkpoint_info *out_info)
{
    gsx_session_checkpoint_payload_v2 payload = { 0 };
    gsx_checkpoint_info current_info = { 0 };
    gsx_scheduler_state scheduler_state = { 0 };
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

    error = gsx_session_fill_checkpoint_info(session, &current_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(current_info.backend_type != payload.backend_type) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "checkpoint backend_type is incompatible with the current session");
    }
    if(current_info.scheduler_algorithm != payload.scheduler_algorithm) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "checkpoint scheduler_algorithm is incompatible with the current session");
    }
    if(current_info.gaussian_count != payload.gaussian_count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "checkpoint gaussian_count is incompatible with the current session");
    }
    if(payload.has_scheduler_state != (session->scheduler != NULL ? 1u : 0u)) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "checkpoint scheduler presence is incompatible with the current session");
    }

    if(session->scheduler != NULL) {
        scheduler_state.current_step = payload.scheduler_current_step;
        scheduler_state.current_learning_rate = payload.scheduler_current_learning_rate;
        error = gsx_scheduler_set_state(session->scheduler, &scheduler_state);
        if(!gsx_error_is_success(error)) {
            return error;
        }
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
    gsx_session_step_report report = { 0 };
    gsx_session_step_timing timing = { 0 };
    gsx_session_state next_state = { 0 };
    gsx_adc_result adc_result = { 0 };
    gsx_error error = gsx_session_require_handle(session);
    gsx_index_t sh_degree = 0;
    gsx_index_t i = 0;
    bool collect_timings = false;
    bool has_learning_rate = false;
    gsx_float_t learning_rate = 0.0f;
    bool adc_result_available = false;
    double total_start_us = 0.0;
    double stage_start_us = 0.0;

    if(!gsx_error_is_success(error)) {
        return error;
    }

    collect_timings = session->report_desc.collect_timings;
    if(collect_timings) {
        total_start_us = gsx_session_get_time_us();
    }

    GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
    GSX_SESSION_STEP_TRY(gsx_dataloader_next_ex(session->train_dataloader, &batch));
    GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.dataloader_us));

    next_state = session->state;

    if((batch.boundary_flags & GSX_DATALOADER_BOUNDARY_NEW_EPOCH) != 0u) {
        next_state.epoch_index += 1;
    }

    GSX_SESSION_STEP_TRY(gsx_session_prepare_step_tensors(session, batch.rgb_image));

    GSX_SESSION_STEP_TRY(gsx_tensor_get_desc(batch.rgb_image, &target_desc));
    if(session->report_desc.retain_target) {
        GSX_SESSION_STEP_TRY(gsx_tensor_copy(batch.rgb_image, session->retained_target));
    }

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
    if(session->render_desc.sh_degree_mode == GSX_SESSION_SH_DEGREE_MODE_EXPLICIT) {
        sh_degree = session->render_desc.sh_degree;
    }
    if(sh_degree < 3) {
        sh3 = NULL;
        grad_sh3 = NULL;
    }
    if(sh_degree < 2) {
        sh2 = NULL;
        grad_sh2 = NULL;
    }
    if(sh_degree < 1) {
        sh1 = NULL;
        grad_sh1 = NULL;
    }

    forward_request.near_plane = session->render_desc.near_plane;
    forward_request.far_plane = session->render_desc.far_plane;
    forward_request.background_color = session->render_desc.background_color;
    forward_request.precision = session->render_desc.precision;
    forward_request.sh_degree = sh_degree;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    forward_request.borrow_train_state = session->render_desc.borrow_train_state;
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

    GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
    GSX_SESSION_STEP_TRY(gsx_renderer_render(session->renderer, session->render_context, &forward_request));
    GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.render_forward_us));

    GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
    for(i = 0; i < (gsx_index_t)session->loss_count; ++i) {
        loss_forward_request.prediction = session->step_prediction;
        loss_forward_request.target = batch.rgb_image;
        loss_forward_request.loss_map_accumulator = session->step_loss_map;
        loss_forward_request.train = true;
        loss_forward_request.scale = session->loss_scales[i];
        GSX_SESSION_STEP_TRY(gsx_loss_forward(session->losses[i], session->loss_contexts[i], &loss_forward_request));
    }
    GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.loss_forward_us));

    GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
    for(i = 0; i < (gsx_index_t)session->loss_count; ++i) {
        loss_backward_request.grad_prediction_accumulator = session->step_grad_prediction;
        loss_backward_request.scale = session->loss_scales[i];
        GSX_SESSION_STEP_TRY(gsx_loss_backward(session->losses[i], session->loss_contexts[i], &loss_backward_request));
    }
    GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.loss_backward_us));

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

    GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
    GSX_SESSION_STEP_TRY(gsx_renderer_backward(session->renderer, session->render_context, &backward_request));
    GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.render_backward_us));

    GSX_SESSION_STEP_TRY(gsx_session_apply_scheduler_learning_rate(session, &has_learning_rate, &learning_rate));

    optim_step_request.role_flags = session->optim_step_desc.role_flags;
    optim_step_request.param_group_indices = session->optim_step_desc.param_group_indices;
    optim_step_request.param_group_index_count = session->optim_step_desc.param_group_index_count;
    optim_step_request.force_all = session->optim_step_desc.force_all;
    GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
    GSX_SESSION_STEP_TRY(gsx_optim_step(session->optim, &optim_step_request));
    GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.optim_step_us));

    if(session->adc_step_desc.enabled) {
        gsx_adc_request adc_request = { 0 };

        adc_request.gs = session->gs;
        adc_request.optim = session->optim;
        adc_request.dataloader = session->adc_dataloader;
        adc_request.renderer = session->renderer;
        adc_request.global_step = session->state.global_step;
        adc_request.scene_scale = session->adc_step_desc.scene_scale;
        GSX_SESSION_STEP_TRY(gsx_session_begin_stage_timing(session, collect_timings, &stage_start_us));
        GSX_SESSION_STEP_TRY(gsx_adc_step(session->adc, &adc_request, &adc_result));
        GSX_SESSION_STEP_TRY(gsx_session_end_stage_timing(session, collect_timings, stage_start_us, &timing.adc_step_us));
        adc_result_available = true;
    }

    next_state.global_step += 1;
    next_state.successful_step_count += 1;
    report.global_step_before = session->state.global_step;
    report.global_step_after = next_state.global_step;
    report.epoch_index_before = session->state.epoch_index;
    report.epoch_index_after = next_state.epoch_index;
    report.batch_epoch_index = batch.epoch_index;
    report.boundary_flags = batch.boundary_flags;
    report.stable_sample_index = batch.stable_sample_index;
    report.stable_sample_id = batch.stable_sample_id;
    report.has_stable_sample_id = batch.has_stable_sample_id;
    report.has_applied_learning_rate = has_learning_rate;
    report.applied_learning_rate = learning_rate;
    report.outputs_available = gsx_session_collects_outputs(session);
    report.adc_result_available = adc_result_available;
    report.adc_result = adc_result;
    report.has_timings = collect_timings;
    if(collect_timings) {
        timing.total_step_us = gsx_session_get_time_us() - total_start_us;
        report.timings = timing;
    }

    session->state = next_state;
    session->last_step_report = report;
    session->has_last_step_report = true;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

#undef GSX_SESSION_STEP_GET_OPTIONAL_FIELD
#undef GSX_SESSION_STEP_GET_FIELD
#undef GSX_SESSION_STEP_TRY
