#ifndef GSX_RENDER_H
#define GSX_RENDER_H

#include "gsx-core.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_render_precision {
    GSX_RENDER_PRECISION_FLOAT32 = 0, /**< Float32 accumulation/output path. */
    GSX_RENDER_PRECISION_FLOAT16 = 1  /**< Float16 accumulation/output path when supported. */
} gsx_render_precision;

typedef gsx_flags32_t gsx_render_precision_flags;
enum {
    GSX_RENDER_PRECISION_FLAG_FLOAT32 = 1u << 0, /**< Renderer supports float32 precision mode. */
    GSX_RENDER_PRECISION_FLAG_FLOAT16 = 1u << 1  /**< Renderer supports float16 precision mode. */
};

typedef gsx_flags32_t gsx_renderer_feature_flags;
enum {
    GSX_RENDERER_FEATURE_ANTIALIASING = 1u << 0, /**< Enable backend-specific anti-aliasing behavior when available. */
    GSX_RENDERER_FEATURE_DEBUG = 1u << 1         /**< Enable debug-oriented validation or instrumentation paths. */
};

typedef struct gsx_renderer_desc {
    gsx_index_t width;                   /**< Target render width in pixels. */
    gsx_index_t height;                  /**< Target render height in pixels. */
    gsx_data_type output_data_type;      /**< Default output tensor element type for renderer-managed outputs. */
    gsx_renderer_feature_flags feature_flags; /**< Optional renderer feature toggles. */
    bool enable_invdepth_output;         /**< Allow forward requests to bind inverse-depth outputs. */
    bool enable_alpha_output;            /**< Allow forward requests to bind alpha outputs. */
} gsx_renderer_desc;

typedef struct gsx_renderer_info {
    gsx_index_t width;                   /**< Configured render width in pixels. */
    gsx_index_t height;                  /**< Configured render height in pixels. */
    gsx_data_type output_data_type;      /**< Default output tensor element type. */
    gsx_renderer_feature_flags feature_flags; /**< Enabled feature flags on this renderer instance. */
    bool enable_invdepth_output;         /**< True if inverse-depth outputs are enabled on this instance. */
    bool enable_alpha_output;            /**< True if alpha outputs are enabled on this instance. */
} gsx_renderer_info;

typedef struct gsx_renderer_capabilities {
    gsx_render_precision_flags supported_precisions; /**< Supported precision modes for render requests. */
    bool supports_invdepth_output;          /**< True if inverse-depth outputs are implemented. */
    bool supports_alpha_output;             /**< True if alpha outputs are implemented. */
    bool supports_cov3d_input;              /**< True if callers may supply precomputed covariance tensors. */
} gsx_renderer_capabilities;

typedef enum gsx_render_forward_type {
    GSX_RENDER_FORWARD_TYPE_INFERENCE = 0, /**< Forward render for inference outputs only. */
    GSX_RENDER_FORWARD_TYPE_TRAIN     = 1, /**< Forward render that retains backward-required context state. */
    GSX_RENDER_FORWARD_TYPE_METRIC    = 2, /**< Forward render that accumulates per-Gaussian metric attribution. */
} gsx_render_forward_type;

/*
 * Forward render requests are explicit and self-contained.
 * - INFERENCE renders outputs only and does not retain backward state.
 * - TRAIN renders outputs and retains the intermediates required by a later
 *   backward call on the same context.
 * - METRIC consumes `metric_map` and writes `gs_metric_accumulator` without
 *   retaining backward state.
 * NOTE:
 * - All render related tensors must have GSX_BACKEND_BUFFER_TYPE_DEVICE backing buffer!
 */
typedef struct gsx_render_forward_request {
    const gsx_camera_intrinsics *intrinsics;  /**< Borrowed camera intrinsics for this render. */
    const gsx_camera_pose *pose;              /**< Borrowed world-to-camera pose for this render. */
    gsx_float_t near_plane;                   /**< Camera-space near plane; must be positive. */
    gsx_float_t far_plane;                    /**< Camera-space far plane; must exceed `near_plane`. */
    gsx_vec3 background_color;                /**< RGB background used where no Gaussian contributes. */
    gsx_render_precision precision;           /**< Precision mode for this render call. CUDA currently supports float32 only. */
    gsx_index_t sh_degree;                    /**< Effective SH degree; must be in `[0, 3]`. */
    gsx_render_forward_type forward_type;     /**< Type of this forward request. */
    bool borrow_train_state;                  /**< TRAIN mode retention policy: false clones retained tensors, true borrows caller tensors for backward. Ignored for non-TRAIN forwards. */

    // render in
    gsx_tensor_t gs_mean3d;                   /**< Input means with GS-compatible shape and dtype. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_rotation;                 /**< Input rotations in xyzw quaternion order. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_logscale;                 /**< Input log-scale parameters. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_cov3d;                    /**< TODO: covariance input is reserved for a future iteration and is not implemented by CUDA yet. */
    gsx_tensor_t gs_sh0;                      /**< Input SH degree-0 coefficients. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_sh1;                      /**< Optional, input SH degree-1 coefficients, must match sh_degree. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_sh2;                      /**< Optional, input SH degree-2 coefficients, must match sh_degree. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_sh3;                      /**< Optional, input SH degree-3 coefficients, must match sh_degree. CUDA currently requires float32 CHW. */
    gsx_tensor_t gs_opacity;                  /**< Input pre-sigmoid opacities. CUDA currently requires float32 CHW. */

    // render out
    gsx_tensor_t out_rgb;                     /**< RGB output tensor for inference and train forwards in public CHW layout. */
    gsx_tensor_t out_invdepth;                /**< TODO: inverse-depth output is reserved for a future iteration and is not implemented by CUDA yet. */
    gsx_tensor_t out_alpha;                   /**< TODO: alpha output is reserved for a future iteration and is not implemented by CUDA yet. */

    // metric -> gs accumulator
    gsx_tensor_t metric_map;                  /**< TODO: metric mode is reserved for a future iteration and is not implemented by CUDA yet. */
    gsx_tensor_t gs_metric_accumulator;       /**< TODO: metric mode is reserved for a future iteration and is not implemented by CUDA yet. */
} gsx_render_forward_request;

typedef struct gsx_render_backward_request {
    gsx_tensor_t grad_rgb;         /**< Upstream RGB gradient; required for RGB loss backpropagation. */
    gsx_tensor_t grad_invdepth;    /**< TODO: inverse-depth backward is reserved for a future iteration and is not implemented by CUDA yet. */
    gsx_tensor_t grad_alpha;       /**< TODO: alpha backward is reserved for a future iteration and is not implemented by CUDA yet. */

    gsx_tensor_t grad_gs_mean3d;   /**< Output gradient sink for Gaussian means. */
    gsx_tensor_t grad_gs_rotation; /**< Output gradient sink for Gaussian rotations. */
    gsx_tensor_t grad_gs_logscale; /**< Output gradient sink for Gaussian log-scales. */
    gsx_tensor_t grad_gs_cov3d;    /**< TODO: covariance backward is reserved for a future iteration and is not implemented by CUDA yet. */
    gsx_tensor_t grad_gs_sh0;      /**< Output gradient sink for SH degree-0 coefficients. */
    gsx_tensor_t grad_gs_sh1;      /**< Output gradient sink for SH degree-1 coefficients. */
    gsx_tensor_t grad_gs_sh2;      /**< Output gradient sink for SH degree-2 coefficients. */
    gsx_tensor_t grad_gs_sh3;      /**< Output gradient sink for SH degree-3 coefficients. */
    gsx_tensor_t grad_gs_opacity;  /**< Output gradient sink for opacities. */
} gsx_render_backward_request;

/** Create a renderer bound to a backend and fixed output geometry. */
GSX_API gsx_error gsx_renderer_init(gsx_renderer_t *out_renderer, gsx_backend_t backend, const gsx_renderer_desc *desc);
/** Release a renderer created by `gsx_renderer_init`. */
GSX_API gsx_error gsx_renderer_free(gsx_renderer_t renderer);
/** Query immutable renderer configuration. */
GSX_API gsx_error gsx_renderer_get_info(gsx_renderer_t renderer, gsx_renderer_info *out_info);
/** Query supported precision modes and optional output features. */
GSX_API gsx_error gsx_renderer_get_capabilities(gsx_renderer_t renderer, gsx_renderer_capabilities *out_capabilities);
/** Query the effective output dtype for a given precision mode. */
GSX_API gsx_error gsx_renderer_get_output_data_type(gsx_renderer_t renderer, gsx_render_precision precision, gsx_data_type *out_data_type);

/** Create a reusable render context that owns its internal retained render state. */
GSX_API gsx_error gsx_render_context_init(gsx_render_context_t *out_context, gsx_renderer_t renderer);
/** Release a render context created by `gsx_render_context_init`. */
GSX_API gsx_error gsx_render_context_free(gsx_render_context_t context);
/** Execute a forward render pass using explicit camera, scene, and output bindings. */
GSX_API gsx_error gsx_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request);
/** Execute a backward render pass against the most recent `GSX_RENDER_FORWARD_TYPE_TRAIN` forward pass on the same context. */
GSX_API gsx_error gsx_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request);

GSX_EXTERN_C_END

#endif /* GSX_RENDER_H */
