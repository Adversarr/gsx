#include "gsx-impl.h"

#include <math.h>
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
