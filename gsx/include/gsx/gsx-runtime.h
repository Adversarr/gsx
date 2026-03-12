#ifndef GSX_RUNTIME_H
#define GSX_RUNTIME_H

#include "gsx-adc.h"
#include "gsx-backend.h"

GSX_EXTERN_C_BEGIN

/** Reader callback used by checkpoint load paths. It must report bytes actually read. */
typedef gsx_error (*gsx_io_read_fn)(void *user_data, void *dst_bytes, gsx_size_t requested_bytes, gsx_size_t *out_bytes_read);
/** Writer callback used by checkpoint save paths. It must report transport errors via `gsx_error`. */
typedef gsx_error (*gsx_io_write_fn)(void *user_data, const void *src_bytes, gsx_size_t byte_count);

typedef struct gsx_io_reader {
    void *user_data;        /**< Opaque caller-owned context forwarded to `read`. */
    gsx_io_read_fn read;    /**< Required callback used for checkpoint deserialization. */
} gsx_io_reader;

typedef struct gsx_io_writer {
    void *user_data;        /**< Opaque caller-owned context forwarded to `write`. */
    gsx_io_write_fn write;  /**< Required callback used for checkpoint serialization. */
} gsx_io_writer;

typedef enum gsx_scheduler_algorithm {
    GSX_SCHEDULER_ALGORITHM_CONSTANT = 0,            /**< Fixed learning rate across all steps. */
    GSX_SCHEDULER_ALGORITHM_DELAYED_EXPONENTIAL = 1  /**< Delayed exponential schedule commonly used by 3DGS trainers. */
} gsx_scheduler_algorithm;

typedef struct gsx_scheduler_desc {
    gsx_scheduler_algorithm algorithm;    /**< Selected scheduler algorithm. */
    gsx_float_t initial_learning_rate;    /**< Learning rate at or before schedule start. */
    gsx_float_t final_learning_rate;      /**< Final asymptotic or terminal learning rate. */
    gsx_size_t delay_steps;               /**< Number of initial steps spent in the delay/warm-start region. */
    gsx_float_t delay_multiplier;         /**< Multiplier applied during the delay region when supported. */
    gsx_size_t decay_begin_step;          /**< Global step at which decay begins. */
    gsx_size_t decay_end_step;            /**< Global step at which decay reaches `final_learning_rate`. */
} gsx_scheduler_desc;

typedef struct gsx_scheduler_state {
    gsx_size_t current_step;         /**< Last step observed by the scheduler state. */
    gsx_float_t current_learning_rate; /**< Last evaluated learning rate. */
} gsx_scheduler_state;

typedef struct gsx_session_desc {
    gsx_backend_t backend;       /**< Borrowed backend used for compatibility metadata and runtime queries. */
    gsx_gs_t gs;                 /**< Borrowed Gaussian model bound into the session. */
    gsx_optim_t optim;           /**< Borrowed optimizer bound into the session. */
    gsx_adc_t adc;               /**< Optional borrowed ADC policy object. */
    gsx_dataloader_t dataloader; /**< Optional borrowed dataloader bound for lightweight iterator-order checkpointing. */
    gsx_scheduler_t scheduler;   /**< Optional borrowed scheduler bound for replay and checkpointing. */
    gsx_size_t initial_global_step; /**< Starting global step for a fresh session. */
    gsx_size_t initial_epoch_index; /**< Starting epoch index for a fresh session. */
} gsx_session_desc;

typedef struct gsx_session_state {
    gsx_size_t global_step;          /**< Current global training step. */
    gsx_size_t epoch_index;          /**< Current logical epoch index. */
    gsx_size_t successful_step_count; /**< Number of steps committed successfully. */
    gsx_size_t failed_step_count;    /**< Number of steps rejected by runtime policy. */
} gsx_session_state;

typedef struct gsx_checkpoint_info {
    gsx_size_t format_version;           /**< Serialized checkpoint format version. */
    gsx_size_t global_step;              /**< Global step stored in the checkpoint payload. */
    gsx_size_t epoch_index;              /**< Epoch index stored in the checkpoint payload. */
    gsx_backend_type backend_type;       /**< Backend family recorded for compatibility checks. */
    gsx_scheduler_algorithm scheduler_algorithm; /**< Scheduler algorithm recorded for compatibility checks. */
    gsx_size_t gaussian_count;           /**< Gaussian count recorded for quick compatibility validation. */
} gsx_checkpoint_info;

/** Create a scheduler object. `out_scheduler` owns the handle on success. */
GSX_API gsx_error gsx_scheduler_init(gsx_scheduler_t *out_scheduler, const gsx_scheduler_desc *desc);
/** Release a scheduler created by `gsx_scheduler_init`. */
GSX_API gsx_error gsx_scheduler_free(gsx_scheduler_t scheduler);
/** Query immutable scheduler configuration. */
GSX_API gsx_error gsx_scheduler_get_desc(gsx_scheduler_t scheduler, gsx_scheduler_desc *out_desc);
/** Reset a scheduler back to its initial state. */
GSX_API gsx_error gsx_scheduler_reset(gsx_scheduler_t scheduler);
/** Snapshot replay-critical scheduler state. */
GSX_API gsx_error gsx_scheduler_get_state(gsx_scheduler_t scheduler, gsx_scheduler_state *out_state);
/** Restore replay-critical scheduler state. */
GSX_API gsx_error gsx_scheduler_set_state(gsx_scheduler_t scheduler, const gsx_scheduler_state *state);
/** Advance scheduler state for `global_step` and return the current learning rate. */
GSX_API gsx_error gsx_scheduler_step(gsx_scheduler_t scheduler, gsx_size_t global_step, gsx_float_t *out_learning_rate);
/** Query the last evaluated learning rate without advancing scheduler state. */
GSX_API gsx_error gsx_scheduler_get_learning_rate(gsx_scheduler_t scheduler, gsx_float_t *out_learning_rate);

/** Create a thin runtime session that owns replay-critical orchestration state only. */
GSX_API gsx_error gsx_session_init(gsx_session_t *out_session, const gsx_session_desc *desc);
/** Release a session created by `gsx_session_init`. */
GSX_API gsx_error gsx_session_free(gsx_session_t session);
/** Query current replay-critical runtime counters. */
GSX_API gsx_error gsx_session_get_state(gsx_session_t session, gsx_session_state *out_state);
/** Restore runtime counters and replay-critical state. */
GSX_API gsx_error gsx_session_set_state(gsx_session_t session, const gsx_session_state *state);
/** Build lightweight checkpoint metadata for compatibility checks or logging. */
GSX_API gsx_error gsx_session_get_checkpoint_info(gsx_session_t session, gsx_checkpoint_info *out_info);
/** Serialize replay-critical session state through a caller-supplied writer callback. */
GSX_API gsx_error gsx_session_save_checkpoint(gsx_session_t session, const gsx_io_writer *writer, const gsx_checkpoint_info *info);
/** Deserialize replay-critical session state through a caller-supplied reader callback. */
GSX_API gsx_error gsx_session_load_checkpoint(gsx_session_t session, const gsx_io_reader *reader, gsx_checkpoint_info *out_info);

GSX_EXTERN_C_END

#endif /* GSX_RUNTIME_H */
