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

typedef struct gsx_loss_item {
    gsx_loss_t loss;                 /**< Borrowed loss object. */
    gsx_loss_context_t context;      /**< Borrowed loss context. */
    gsx_float_t scale;               /**< Scalar weight for forward and backward. */
} gsx_loss_item;

typedef enum gsx_session_sh_degree_mode {
    GSX_SESSION_SH_DEGREE_MODE_AUTO_FROM_GS = 0, /**< Derive the effective SH degree from the GS fields bound to the session. */
    GSX_SESSION_SH_DEGREE_MODE_EXPLICIT = 1      /**< Use `gsx_session_render_desc::sh_degree` exactly as provided. */
} gsx_session_sh_degree_mode;

typedef struct gsx_session_render_desc {
    gsx_float_t near_plane;                 /**< Camera-space near plane used for every training forward. Must be positive. */
    gsx_float_t far_plane;                  /**< Camera-space far plane used for every training forward. Must exceed `near_plane`. */
    gsx_vec3 background_color;              /**< RGB background color used for training forwards. */
    gsx_render_precision precision;         /**< Precision mode used for session-managed renders. */
    gsx_session_sh_degree_mode sh_degree_mode; /**< Whether SH degree is auto-derived from GS fields or taken from `sh_degree`. */
    gsx_index_t sh_degree;                  /**< Explicit SH degree used when `sh_degree_mode == GSX_SESSION_SH_DEGREE_MODE_EXPLICIT`. */
    bool borrow_train_state;                /**< Forward TRAIN retention mode forwarded into `gsx_render_forward_request`. */
} gsx_session_render_desc;

typedef struct gsx_session_optim_step_desc {
    gsx_optim_param_role_flags role_flags;      /**< Built-in optimizer roles selected for each step. Ignored when `force_all` is true. */
    const gsx_index_t *param_group_indices;     /**< Optional extra optimizer param-group indices selected for each step. */
    gsx_index_t param_group_index_count;        /**< Number of entries in `param_group_indices`. */
    bool force_all;                             /**< If true, step every optimizer param group regardless of selectors. */
} gsx_session_optim_step_desc;

typedef struct gsx_session_adc_step_desc {
    bool enabled;                           /**< If true, `gsx_session_step` invokes ADC after the optimizer step. */
    gsx_dataloader_t dataloader;            /**< Optional borrowed dataloader dedicated to ADC. NULL reuses `train_dataloader`. */
    gsx_float_t scene_scale;                /**< Scene normalization scale forwarded into `gsx_adc_request`. Must be positive when ADC is enabled. */
} gsx_session_adc_step_desc;

typedef struct gsx_session_workspace_desc {
    gsx_backend_buffer_type_class buffer_type_class; /**< Buffer type class used for session-owned scratch and retained tensors. */
    gsx_arena_desc arena_desc;                       /**< Arena policy used to create the session-owned workspace arena. */
    bool auto_plan;                                  /**< If true, size `arena_desc.initial_capacity_bytes` from the configured train geometry before arena creation. */
} gsx_session_workspace_desc;

typedef struct gsx_session_report_desc {
    bool retain_prediction;      /**< Retain the latest prediction tensor for `gsx_session_get_last_outputs`. */
    bool retain_target;          /**< Retain the latest target tensor for `gsx_session_get_last_outputs`. */
    bool retain_loss_map;        /**< Retain the latest accumulated loss-map tensor for `gsx_session_get_last_outputs`. */
    bool retain_grad_prediction; /**< Retain the latest dLoss/dPrediction tensor for `gsx_session_get_last_outputs`. */
    bool collect_timings;        /**< Collect coarse per-stage wall-clock timings for the latest successful step. */
} gsx_session_report_desc;

typedef struct gsx_session_desc {
    gsx_backend_t backend;                   /**< Borrowed backend used for compatibility metadata and runtime queries. */
    gsx_gs_t gs;                             /**< Borrowed Gaussian model bound into the session. */
    gsx_optim_t optim;                       /**< Borrowed optimizer bound into the session. */
    gsx_renderer_t renderer;                 /**< Borrowed renderer used by `gsx_session_step`. */
    gsx_dataloader_t train_dataloader;       /**< Borrowed training dataloader used by `gsx_session_step`. */
    gsx_adc_t adc;                           /**< Optional borrowed ADC policy object. */
    gsx_scheduler_t scheduler;               /**< Optional borrowed scheduler bound for replay and checkpointing. */
    gsx_size_t loss_count;                   /**< Number of loss items. Must be > 0. */
    const gsx_loss_item *loss_items;         /**< Array of loss items evaluated in order during `gsx_session_step`. */
    gsx_session_render_desc render;          /**< Render policy applied to every training step. */
    gsx_session_optim_step_desc optim_step;  /**< Optimizer-step selection policy applied to every training step. */
    gsx_session_adc_step_desc adc_step;      /**< ADC invocation policy applied to every training step. */
    gsx_session_workspace_desc workspace;    /**< Workspace and retained-output arena policy. */
    gsx_session_report_desc reporting;       /**< Retained-output and timing policy. */
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
    gsx_backend_type backend_type;       /**< Backend family recorded for compatibility checks. This is informational family-level compatibility metadata, not a backend-device pin. */
    gsx_scheduler_algorithm scheduler_algorithm; /**< Scheduler algorithm recorded for compatibility checks. */
    gsx_size_t gaussian_count;           /**< Gaussian count recorded for quick compatibility validation. */
} gsx_checkpoint_info;

typedef struct gsx_session_outputs {
    gsx_tensor_t prediction;      /**< Borrowed retained prediction tensor when enabled on the session; otherwise NULL. */
    gsx_tensor_t target;          /**< Borrowed retained target tensor when enabled on the session; otherwise NULL. */
    gsx_tensor_t loss_map;        /**< Borrowed retained loss-map tensor when enabled on the session; otherwise NULL. */
    gsx_tensor_t grad_prediction; /**< Borrowed retained dLoss/dPrediction tensor when enabled on the session; otherwise NULL. */
} gsx_session_outputs;

typedef struct gsx_session_step_timing {
    double render_forward_us; /**< Wall-clock duration of the render forward stage when timing collection is enabled. */
    double loss_forward_us;   /**< Wall-clock duration of the loss forward stage when timing collection is enabled. */
    double loss_backward_us;  /**< Wall-clock duration of the loss backward stage when timing collection is enabled. */
    double render_backward_us; /**< Wall-clock duration of the render backward stage when timing collection is enabled. */
    double optim_step_us;     /**< Wall-clock duration of the optimizer step when timing collection is enabled. */
    double adc_step_us;       /**< Wall-clock duration of the ADC stage when timing collection is enabled. */
    double total_step_us;     /**< Wall-clock duration of the full step when timing collection is enabled. */
} gsx_session_step_timing;

typedef struct gsx_session_step_report {
    gsx_size_t global_step_before;        /**< Global step value before the successful step. */
    gsx_size_t global_step_after;         /**< Global step value after the successful step. */
    gsx_size_t epoch_index_before;        /**< Session epoch index before the successful step. */
    gsx_size_t epoch_index_after;         /**< Session epoch index after the successful step. */
    gsx_size_t batch_epoch_index;         /**< Dataloader epoch index associated with the fetched batch. */
    gsx_dataloader_boundary_flags boundary_flags; /**< Boundary flags returned by the fetched batch. */
    gsx_size_t stable_sample_index;       /**< Stable sample index returned by the fetched batch. */
    gsx_id_t stable_sample_id;            /**< Optional stable sample identifier returned by the fetched batch. */
    bool has_stable_sample_id;            /**< True when `stable_sample_id` is meaningful. */
    gsx_float_t applied_learning_rate;    /**< Learning rate applied by the bound scheduler for this step. */
    bool has_applied_learning_rate;       /**< True when `applied_learning_rate` is meaningful. */
    bool outputs_available;               /**< True when retained outputs are available through `gsx_session_get_last_outputs`. */
    bool adc_result_available;            /**< True when `adc_result` was produced during the step. */
    gsx_adc_result adc_result;            /**< Result returned by ADC when `adc_result_available` is true. */
    bool has_timings;                     /**< True when `timings` was populated for the latest step. */
    gsx_session_step_timing timings;      /**< Coarse per-stage timing data. */
} gsx_session_step_report;

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

/** Create a trainer-oriented runtime session over borrowed backend, data, model, renderer, loss, optimizer, and optional ADC/scheduler handles. */
GSX_API gsx_error gsx_session_init(gsx_session_t *out_session, const gsx_session_desc *desc);
/** Release a session created by `gsx_session_init`. */
GSX_API gsx_error gsx_session_free(gsx_session_t session);
/** Query the effective session descriptor, including session-owned copies of loss items and optimizer index selectors. */
GSX_API gsx_error gsx_session_get_desc(gsx_session_t session, gsx_session_desc *out_desc);
/** Reset runtime counters, scheduler state, dataloader iteration state, and retained step reports/outputs back to the configured initial state. */
GSX_API gsx_error gsx_session_reset(gsx_session_t session);
/** Query current replay-critical runtime counters. */
GSX_API gsx_error gsx_session_get_state(gsx_session_t session, gsx_session_state *out_state);
/** Restore runtime counters and replay-critical state. */
GSX_API gsx_error gsx_session_set_state(gsx_session_t session, const gsx_session_state *state);
/** Query the latest successful step report. Returns `GSX_ERROR_INVALID_STATE` before the first successful step. */
GSX_API gsx_error gsx_session_get_last_step_report(gsx_session_t session, gsx_session_step_report *out_report);
/** Borrow the latest retained step outputs. Returns `GSX_ERROR_INVALID_STATE` when no retained outputs are available yet. */
GSX_API gsx_error gsx_session_get_last_outputs(gsx_session_t session, gsx_session_outputs *out_outputs);
/** Build lightweight checkpoint metadata for compatibility checks or logging. */
GSX_API gsx_error gsx_session_get_checkpoint_info(gsx_session_t session, gsx_checkpoint_info *out_info);
/** Serialize replay-critical session state through a caller-supplied writer callback. */
GSX_API gsx_error gsx_session_save_checkpoint(gsx_session_t session, const gsx_io_writer *writer, const gsx_checkpoint_info *info);
/** Deserialize replay-critical session state through a caller-supplied reader callback. */
GSX_API gsx_error gsx_session_load_checkpoint(gsx_session_t session, const gsx_io_reader *reader, gsx_checkpoint_info *out_info);
/** Advance session state for 1 step. */
GSX_API gsx_error gsx_session_step(gsx_session_t session);

GSX_EXTERN_C_END

#endif /* GSX_RUNTIME_H */
