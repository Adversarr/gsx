#ifndef GSX_IMPL_H
#define GSX_IMPL_H

#include "gsx/gsx.h"
#include "gsx/extra/gsx-image.h"

GSX_EXTERN_C_BEGIN
#include <stdbool.h>
#include <stdint.h>

#ifndef GSX_HAS_CUDA
#define GSX_HAS_CUDA 0
#endif

#ifndef GSX_HAS_METAL
#define GSX_HAS_METAL 0
#endif

/*
 * Internal implementation notes for the stable GSX public API surface:
 *
 * - Only headers under gsx/include/gsx are part of the stable ABI.
 * - Public descriptor/state/result structs are plain POD values in v0.
 * - Public backend-bound work is externally observed as one totally ordered
 *   major stream per backend.
 * - Private helper threads or streams are allowed for dataloader prefetch only.
 * - Builtin registry owns only pointer arrays for providers and devices.
 *   It never owns provider/device object storage and never destroys those
 *   objects during reset.
 * - Registered provider/device objects must remain valid for the full borrowed
 *   handle lifetime exposed by public APIs.
 * - Device identity fields must stay consistent with the owning provider:
 *   backend_type and backend_name must match provider metadata.
 * - Layering is one-way:
 *   core <- backend/render/data/loss/optim/adc <- runtime
 * - Runtime objects own replay-critical state only. They do not own the caller
 *   training loop.
 * - GS/optimizer structural mutation and ADC steps must remain transactional.
 */

typedef struct gsx_backend_provider gsx_backend_provider;
typedef gsx_backend_provider *gsx_backend_provider_t;
typedef struct gsx_backend_provider_i gsx_backend_provider_i;
typedef struct gsx_builtin_registry_state gsx_builtin_registry_state;
typedef struct gsx_backend_i gsx_backend_i;
typedef struct gsx_backend_buffer_type_i gsx_backend_buffer_type_i;
typedef struct gsx_backend_buffer_i gsx_backend_buffer_i;
typedef struct gsx_loss gsx_loss;
typedef struct gsx_loss_i gsx_loss_i;
typedef struct gsx_loss_context gsx_loss_context;
typedef struct gsx_loss_context_i gsx_loss_context_i;
typedef struct gsx_optim gsx_optim;
typedef struct gsx_optim_i gsx_optim_i;
typedef struct gsx_adc gsx_adc;
typedef struct gsx_adc_i gsx_adc_i;
typedef struct gsx_renderer gsx_renderer;
typedef struct gsx_render_context gsx_render_context;
typedef struct gsx_renderer_i gsx_renderer_i;
typedef struct gsx_render_context_i gsx_render_context_i;
typedef struct gsx_backend_tensor_view gsx_backend_tensor_view;
typedef struct gsx_async_dl* gsx_async_dl_t;
typedef struct gsx_async_dl_i gsx_async_dl_i;
typedef struct gsx_async_dl_ready_item gsx_async_dl_ready_item;
typedef struct gsx_async_dl_desc gsx_async_dl_desc;
typedef struct gsx_dataloader_slot gsx_dataloader_slot;

#define GSX_OPTIM_BUILTIN_ROLE_COUNT 8

struct gsx_backend_provider {
    const gsx_backend_provider_i *iface;
    gsx_backend_type backend_type;
    const char *backend_name;
};

struct gsx_backend_device {
    gsx_backend_provider_t provider;
    gsx_backend_device_info info;
};

struct gsx_backend {
    const gsx_backend_i *iface;
    gsx_backend_provider_t provider;
    gsx_backend_device_t device;
    gsx_size_t live_buffer_count;
    gsx_size_t live_arena_count;
    gsx_size_t live_renderer_count;
    gsx_size_t live_loss_count;
    gsx_size_t live_optim_count;
    gsx_size_t live_adc_count;
    gsx_size_t live_async_dl_count;
};

struct gsx_backend_buffer_type {
    const gsx_backend_buffer_type_i *iface;
    gsx_backend_t backend;
    gsx_size_t live_arena_count;
};

struct gsx_backend_buffer {
    const gsx_backend_buffer_i *iface;
    gsx_backend_buffer_type_t buffer_type;
    gsx_size_t size_bytes;
    gsx_size_t alignment_bytes;
};

struct gsx_arena {
    gsx_backend_buffer_type_t buffer_type;
    gsx_backend_buffer_t backing_buffer;
    gsx_size_t capacity_bytes;
    gsx_size_t cursor_bytes;
    gsx_size_t used_bytes;
    gsx_size_t peak_bytes;
    gsx_size_t required_bytes;
    gsx_size_t requested_alignment_bytes;
    gsx_size_t effective_alignment_bytes;
    bool dry_run;
    gsx_id_t reset_epoch;
    gsx_size_t active_tensor_count;
    gsx_size_t tensor_handle_count;
    struct gsx_tensor *active_head;
    struct gsx_tensor *active_tail;
};

struct gsx_tensor {
    gsx_arena_t arena;
    gsx_backend_buffer_t backing_buffer;
    gsx_size_t offset_bytes;
    gsx_size_t size_bytes;
    gsx_size_t alloc_span_bytes;
    gsx_size_t requested_alignment_bytes;
    gsx_size_t effective_alignment_bytes;
    gsx_size_t alloc_start_bytes;
    gsx_size_t alloc_end_bytes;
    gsx_index_t rank;
    gsx_index_t shape[GSX_TENSOR_MAX_DIM];
    gsx_data_type data_type;
    gsx_storage_format storage_format;
    struct gsx_tensor *prev_active;
    struct gsx_tensor *next_active;
};

struct gsx_optim {
    const gsx_optim_i *iface;
    gsx_backend_t backend;
    gsx_backend_buffer_type_t state_buffer_type;
    gsx_optim_algorithm algorithm;
    gsx_index_t param_group_count;
    gsx_optim_param_group_desc *param_groups;
    gsx_float_t *learning_rates;
    gsx_index_t role_to_index[GSX_OPTIM_BUILTIN_ROLE_COUNT];
};

struct gsx_adc {
    const gsx_adc_i *iface;
    gsx_backend_t backend;
    gsx_adc_desc desc;
};

struct gsx_renderer {
    const gsx_renderer_i *iface;
    gsx_backend_t backend;
    gsx_renderer_info info;
    gsx_renderer_capabilities capabilities;
    gsx_size_t live_context_count;
};

struct gsx_render_context {
    const gsx_render_context_i *iface;
    gsx_renderer_t renderer;
};

struct gsx_loss {
    const gsx_loss_i *iface;
    gsx_backend_t backend;
    gsx_loss_algorithm algorithm;
    gsx_loss_grad_normalization_type grad_normalization;
    gsx_size_t live_context_count;
};

struct gsx_loss_context {
    const gsx_loss_context_i *iface;
    gsx_loss_t loss;
    gsx_tensor_t retained_prediction;
    gsx_tensor_t retained_target;
    bool has_forward_state;
    bool forward_is_training;
};

struct gsx_backend_tensor_view {
    gsx_backend_buffer_t buffer;
    gsx_size_t offset_bytes;
    gsx_size_t size_bytes;
    gsx_size_t effective_alignment_bytes;
    gsx_data_type data_type;
};

struct gsx_session {
    gsx_backend_t backend;
    gsx_gs_t gs;
    gsx_optim_t optim;
    gsx_renderer_t renderer;
    gsx_dataloader_t train_dataloader;
    gsx_adc_t adc;
    gsx_scheduler_t scheduler;
    gsx_dataloader_t adc_dataloader;
    gsx_session_render_desc render_desc;
    gsx_session_optim_step_desc optim_step_desc;
    gsx_session_adc_step_desc adc_step_desc;
    gsx_session_workspace_desc workspace_desc;
    gsx_session_report_desc report_desc;
    gsx_render_context_t render_context;
    gsx_arena_t workspace_arena;
    gsx_tensor_t step_prediction;
    gsx_tensor_t step_loss_map;
    gsx_tensor_t step_grad_prediction;
    gsx_tensor_t retained_target;
    gsx_session_step_report last_step_report;
    bool has_last_step_report;
    gsx_session_state state;
    gsx_session_state initial_state;
    gsx_size_t loss_count;
    gsx_loss_t *losses;
    gsx_loss_context_t *loss_contexts;
    gsx_float_t *loss_scales;
    gsx_loss_item *loss_items;
    gsx_index_t *optim_param_group_indices;
};

struct gsx_backend_provider_i {
    gsx_error (*discover_devices)(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
    gsx_error (*create_backend)(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);
};

typedef enum gsx_impl_unary_reduce_op {
    GSX_IMPL_UNARY_REDUCE_OP_SUM = 0,
    GSX_IMPL_UNARY_REDUCE_OP_MEAN = 1,
    GSX_IMPL_UNARY_REDUCE_OP_MAX = 2
} gsx_impl_unary_reduce_op;

typedef enum gsx_impl_binary_reduce_op {
    GSX_IMPL_BINARY_REDUCE_OP_MSE = 0,
    GSX_IMPL_BINARY_REDUCE_OP_MAE = 1
} gsx_impl_binary_reduce_op;

struct gsx_backend_i {
    gsx_error (*free)(gsx_backend_t backend);
    gsx_error (*get_info)(gsx_backend_t backend, gsx_backend_info *out_info);
    gsx_error (*get_capabilities)(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
    gsx_error (*get_major_stream)(gsx_backend_t backend, void **out_stream);
    gsx_error (*major_stream_sync)(gsx_backend_t backend);
    gsx_error (*count_buffer_types)(gsx_backend_t backend, gsx_index_t *out_count);
    gsx_error (*get_buffer_type)(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
    gsx_error (*find_buffer_type)(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);
    gsx_error (*query_unary_reduce_workspace_size)(
        gsx_backend_t backend,
        gsx_backend_buffer_type_class workspace_buffer_type,
        gsx_data_type data_type,
        gsx_index_t x_rank,
        const gsx_index_t *x_shape,
        gsx_index_t out_rank,
        const gsx_index_t *out_shape,
        gsx_index_t start_axis,
        gsx_impl_unary_reduce_op op,
        gsx_size_t *out_workspace_size_bytes,
        gsx_size_t *out_workspace_alignment_bytes
    );
    gsx_error (*query_binary_reduce_workspace_size)(
        gsx_backend_t backend,
        gsx_backend_buffer_type_class workspace_buffer_type,
        gsx_data_type data_type,
        gsx_index_t lhs_rank,
        const gsx_index_t *lhs_shape,
        gsx_index_t rhs_rank,
        const gsx_index_t *rhs_shape,
        gsx_index_t out_rank,
        const gsx_index_t *out_shape,
        gsx_index_t start_axis,
        gsx_impl_binary_reduce_op op,
        gsx_size_t *out_workspace_size_bytes,
        gsx_size_t *out_workspace_alignment_bytes
    );
    gsx_error (*create_renderer)(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer);
    gsx_error (*create_loss)(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss);
    gsx_error (*create_optim)(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim);
    gsx_error (*create_adc)(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc);
    gsx_error (*create_async_dl)(gsx_backend_t backend, const gsx_async_dl_desc *desc, gsx_async_dl_t *out_async_dl);
};

struct gsx_backend_buffer_type_i {
    gsx_error (*get_info)(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
    gsx_error (*get_alloc_size)(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
    gsx_error (*init_buffer)(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);
};

typedef enum gsx_impl_unary_op {
    GSX_IMPL_UNARY_OP_EXP = 0,
    GSX_IMPL_UNARY_OP_SIGMOID = 1,
    GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE = 2,
    GSX_IMPL_UNARY_OP_ABS = 3
} gsx_impl_unary_op;

struct gsx_backend_buffer_i {
    gsx_error (*free)(gsx_backend_buffer_t buffer);
    gsx_error (*get_info)(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
    gsx_error (*get_native_handle)(gsx_backend_buffer_t buffer, void **out_handle);
    gsx_error (*upload)(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
    gsx_error (*download)(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
    gsx_error (*set_zero)(gsx_backend_buffer_t buffer);
    gsx_error (*memset_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        uint8_t value,
        gsx_size_t offset_bytes,
        gsx_size_t size_bytes
    );
    gsx_error (*set_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        const void *src_bytes,
        gsx_size_t offset_bytes,
        gsx_size_t size_bytes
    );
    gsx_error (*get_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        void *dst_bytes,
        gsx_size_t offset_bytes,
        gsx_size_t size_bytes
    );
    gsx_error (*copy_tensor)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *src_view,
        const gsx_backend_tensor_view *dst_view
    );
    gsx_error (*fill_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        const void *value_bytes,
        gsx_size_t value_size_bytes
    );
    gsx_error (*fill_rand_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        uint64_t rng_state,
        uint64_t rng_inc
    );
    gsx_error (*fill_randn_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        uint64_t rng_state,
        uint64_t rng_inc,
        gsx_float_t sigma
    );
    gsx_error (*fill_randint_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        uint64_t rng_state,
        uint64_t rng_inc,
        uint32_t bound
    );
    gsx_error (*multinomial_tensor)(
        gsx_backend_buffer_t out_buffer,
        const gsx_backend_tensor_view *out_view,
        const gsx_backend_tensor_view *cdf_view,
        uint64_t rng_state,
        uint64_t rng_inc
    );
    gsx_error (*check_finite_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        bool *out_is_finite
    );
    gsx_error (*gather_tensor)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *x_view,
        const gsx_backend_tensor_view *index_view,
        const gsx_backend_tensor_view *out_view,
        gsx_index_t x_rank,
        const gsx_index_t *x_shape,
        gsx_index_t out_rank,
        const gsx_index_t *out_shape
    );
    gsx_error (*unary_tensor)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *x_view,
        const gsx_backend_tensor_view *out_view,
        gsx_index_t rank,
        const gsx_index_t *shape,
        gsx_impl_unary_op op
    );
    gsx_error (*unary_tensor_inplace)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        gsx_impl_unary_op op
    );
    gsx_error (*unary_reduce_tensor)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *x_view,
        const gsx_backend_tensor_view *out_view,
        const gsx_backend_tensor_view *workspace_view,
        gsx_index_t x_rank,
        const gsx_index_t *x_shape,
        gsx_index_t out_rank,
        const gsx_index_t *out_shape,
        gsx_index_t start_axis,
        gsx_impl_unary_reduce_op op
    );
    gsx_error (*binary_reduce_tensor)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *lhs_view,
        const gsx_backend_tensor_view *rhs_view,
        const gsx_backend_tensor_view *out_view,
        const gsx_backend_tensor_view *workspace_view,
        gsx_index_t lhs_rank,
        const gsx_index_t *lhs_shape,
        gsx_index_t rhs_rank,
        const gsx_index_t *rhs_shape,
        gsx_index_t out_rank,
        const gsx_index_t *out_shape,
        gsx_index_t start_axis,
        gsx_impl_binary_reduce_op op
    );
    gsx_error (*clamp_inplace_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        const void *min_value,
        const void *max_value
    );
    gsx_error (*image_convert_colorspace)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *src_view,
        gsx_storage_format storage_format,
        gsx_index_t rank,
        const gsx_index_t *shape,
        gsx_image_colorspace src_colorspace,
        const gsx_backend_tensor_view *dst_view,
        gsx_image_colorspace dst_colorspace
    );
    gsx_error (*image_convert_storage_format)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *src_view,
        gsx_index_t src_rank,
        const gsx_index_t *src_shape,
        gsx_storage_format src_storage_format,
        const gsx_backend_tensor_view *dst_view,
        gsx_index_t dst_rank,
        const gsx_index_t *dst_shape,
        gsx_storage_format dst_storage_format
    );
    gsx_error (*image_convert_data_type)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *src_view,
        gsx_storage_format storage_format,
        gsx_index_t rank,
        const gsx_index_t *shape,
        const gsx_backend_tensor_view *dst_view
    );
};

struct gsx_optim_i {
    gsx_error (*destroy)(gsx_optim_t optim);
    gsx_error (*step_selected)(gsx_optim_t optim, const bool *selected);
    gsx_error (*permute)(gsx_optim_t optim, gsx_tensor_t permutation);
    gsx_error (*gather)(gsx_optim_t optim, gsx_tensor_t indices);
    gsx_error (*resize)(gsx_optim_t optim, gsx_size_t new_count);
    gsx_error (*reset_all)(gsx_optim_t optim);
    gsx_error (*reset_by_index)(gsx_optim_t optim, gsx_index_t index);
};

struct gsx_adc_i {
    gsx_error (*destroy)(gsx_adc_t adc);
    gsx_error (*step)(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);
};

struct gsx_renderer_i {
    gsx_error (*destroy)(gsx_renderer_t renderer);
    gsx_error (*create_context)(gsx_renderer_t renderer, gsx_render_context_t *out_context);
    gsx_error (*render)(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request);
    gsx_error (*backward)(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request);
};

struct gsx_render_context_i {
    gsx_error (*destroy)(gsx_render_context_t context);
};

struct gsx_loss_i {
    gsx_error (*destroy)(gsx_loss_t loss);
    gsx_error (*create_context)(gsx_loss_t loss, gsx_loss_context_t *out_context);
    gsx_error (*forward)(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request);
    gsx_error (*backward)(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request);
};

struct gsx_loss_context_i {
    gsx_error (*destroy)(gsx_loss_context_t context);
};

struct gsx_builtin_registry_state {
    bool is_initialized;
    gsx_index_t backend_provider_count;
    gsx_index_t backend_provider_capacity;
    gsx_backend_provider_t *backend_providers;
    gsx_index_t backend_device_count;
    gsx_index_t backend_device_capacity;
    gsx_backend_device_t *backend_devices;
};

struct gsx_async_dl_desc {
    gsx_backend_t backend;
    gsx_dataset_desc dataset_desc;
    gsx_data_type output_data_type;
    gsx_size_t slot_count;
    gsx_dataloader_slot *slots;
};

struct gsx_async_dl_ready_item {
    gsx_size_t stable_sample_index;
    gsx_size_t slot_index;
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    gsx_id_t stable_sample_id;
    bool has_stable_sample_id;
};

struct gsx_async_dl_i {
    gsx_error (*destroy)(gsx_async_dl_t async_dl);
    gsx_error (*submit)(gsx_async_dl_t async_dl, gsx_size_t stable_sample_index, gsx_size_t slot_index);
    gsx_error (*wait)(gsx_async_dl_t async_dl, gsx_async_dl_ready_item *out_item);
    gsx_error (*inflight_count)(gsx_async_dl_t async_dl, gsx_size_t *out_count);
};

struct gsx_async_dl {
    const gsx_async_dl_i *iface;
};

static inline gsx_error gsx_make_error(gsx_error_code code, const char *message)
{
    gsx_error error = { code, message };
    return error;
}

static inline gsx_error gsx_data_type_get_size_bytes(gsx_data_type data_type, gsx_size_t *out_size_bytes)
{
    if(out_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_size_bytes must be non-null");
    }

    switch(data_type) {
    case GSX_DATA_TYPE_F32:
    case GSX_DATA_TYPE_I32:
    case GSX_DATA_TYPE_U32:
        *out_size_bytes = 4;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_F16:
        *out_size_bytes = 2;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    case GSX_DATA_TYPE_U8:
        *out_size_bytes = 1;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "data type is unsupported");
    }
}

static inline bool gsx_is_power_of_two(gsx_size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static inline bool gsx_size_add_overflows(gsx_size_t lhs, gsx_size_t rhs, gsx_size_t *out_sum)
{
    if(lhs > UINT64_MAX - rhs) {
        return true;
    }
    if(out_sum != NULL) {
        *out_sum = lhs + rhs;
    }
    return false;
}

static inline bool gsx_size_mul_overflows(gsx_size_t lhs, gsx_size_t rhs, gsx_size_t *out_product)
{
    if(lhs != 0 && rhs > UINT64_MAX / lhs) {
        return true;
    }
    if(out_product != NULL) {
        *out_product = lhs * rhs;
    }
    return false;
}

static inline bool gsx_round_up_overflows(gsx_size_t value, gsx_size_t alignment_bytes, gsx_size_t *out_rounded_value)
{
    gsx_size_t adjustment = 0;

    if(alignment_bytes == 0) {
        if(out_rounded_value != NULL) {
            *out_rounded_value = value;
        }
        return false;
    }

    adjustment = alignment_bytes - 1;
    if(gsx_size_add_overflows(value, adjustment, out_rounded_value)) {
        return true;
    }
    if(out_rounded_value != NULL) {
        *out_rounded_value &= ~(alignment_bytes - 1);
    }
    return false;
}

// Loggers
void gsx_log_callback_default(enum gsx_log_level level, const char * text, void * user_data);
void gsx_log_internal(enum gsx_log_level level, const char * format, ...);

#ifdef NDEBUG
#define GSX_LOG_DEBUG(...) ((void)0)
#else
#define GSX_LOG_DEBUG(...) gsx_log_internal(GSX_LOG_LEVEL_DEBUG, __VA_ARGS__)
#endif
// The logger will not automatically append a newline, please manually include `\n` in the format string as needed.
#define GSX_LOG_INFO(...)  gsx_log_internal(GSX_LOG_LEVEL_INFO, __VA_ARGS__)
#define GSX_LOG_WARN(...)  gsx_log_internal(GSX_LOG_LEVEL_WARNING, __VA_ARGS__)
#define GSX_LOG_ERROR(...) gsx_log_internal(GSX_LOG_LEVEL_ERROR, __VA_ARGS__)

// Registry and Backend
gsx_builtin_registry_state *gsx_builtin_registry_get(void);
void gsx_builtin_registry_reset(gsx_builtin_registry_state *registry);
gsx_error gsx_builtin_registry_append_provider(gsx_builtin_registry_state *registry, gsx_backend_provider_t backend_provider);
gsx_error gsx_builtin_registry_append_device(gsx_builtin_registry_state *registry, gsx_backend_device_t backend_device);
// Optimizer
bool gsx_optim_algorithm_is_valid(gsx_optim_algorithm algorithm);
bool gsx_optim_param_role_is_valid(gsx_optim_param_role role);
bool gsx_optim_param_role_is_builtin(gsx_optim_param_role role);
bool gsx_optim_float_is_finite(gsx_float_t value);
gsx_error gsx_optim_validate_desc(gsx_backend_t backend, const gsx_optim_desc *desc);
gsx_error gsx_optim_base_init(
    gsx_optim *optim,
    const gsx_optim_i *iface,
    gsx_backend_t backend,
    const gsx_optim_desc *desc
);
void gsx_optim_base_deinit(gsx_optim *optim);
gsx_error gsx_optim_lookup_role_index(const gsx_optim *optim, gsx_optim_param_role role, gsx_index_t *out_index);
gsx_error gsx_optim_copy_param_group_desc(const gsx_optim *optim, gsx_index_t index, gsx_optim_param_group_desc *out_desc);
gsx_error gsx_optim_select_param_groups(const gsx_optim *optim, const gsx_optim_step_request *request, bool *selected);
// Loss
bool gsx_loss_algorithm_is_valid(gsx_loss_algorithm algorithm);
bool gsx_loss_grad_normalization_type_is_valid(gsx_loss_grad_normalization_type normalization_type);
gsx_error gsx_loss_validate_desc(gsx_backend_t backend, const gsx_loss_desc *desc);
gsx_error gsx_loss_base_init(gsx_loss *loss, const gsx_loss_i *iface, gsx_backend_t backend, const gsx_loss_desc *desc);
void gsx_loss_base_deinit(gsx_loss *loss);
gsx_error gsx_loss_context_base_init(gsx_loss_context *context, const gsx_loss_context_i *iface, gsx_loss_t loss);
void gsx_loss_context_base_deinit(gsx_loss_context *context);
// Renderer
gsx_error gsx_renderer_validate_desc(gsx_backend_t backend, const gsx_renderer_desc *desc);
gsx_error gsx_renderer_base_init(
    gsx_renderer *renderer,
    const gsx_renderer_i *iface,
    gsx_backend_t backend,
    const gsx_renderer_desc *desc,
    const gsx_renderer_capabilities *capabilities
);
void gsx_renderer_base_deinit(gsx_renderer *renderer);
gsx_error gsx_render_context_base_init(gsx_render_context *context, const gsx_render_context_i *iface, gsx_renderer_t renderer);
void gsx_render_context_base_deinit(gsx_render_context *context);
// ADC
bool gsx_adc_algorithm_is_valid(gsx_adc_algorithm algorithm);
gsx_error gsx_adc_validate_desc(gsx_backend_t backend, const gsx_adc_desc *desc);
gsx_error gsx_adc_base_init(gsx_adc *adc, const gsx_adc_i *iface, gsx_backend_t backend, const gsx_adc_desc *desc);
void gsx_adc_base_deinit(gsx_adc *adc);

// Backend bootstrappers
gsx_error gsx_cpu_backend_provider_bootstrap(gsx_builtin_registry_state *registry);
#if GSX_HAS_CUDA
gsx_error gsx_cuda_backend_provider_bootstrap(gsx_builtin_registry_state *registry);
#endif
#if GSX_HAS_METAL
gsx_error gsx_metal_backend_provider_bootstrap(gsx_builtin_registry_state *registry);
#endif
GSX_EXTERN_C_END

#endif /* GSX_IMPL_H */
