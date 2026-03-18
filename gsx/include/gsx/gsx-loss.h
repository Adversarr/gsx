#ifndef GSX_LOSS_H
#define GSX_LOSS_H

#include "gsx-core.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_loss_algorithm {
    GSX_LOSS_ALGORITHM_L1 = 0,   /**< Differentiable L1 image loss. */
    GSX_LOSS_ALGORITHM_MSE = 1,  /**< Differentiable mean squared error loss. */
    GSX_LOSS_ALGORITHM_SSIM = 2  /**< Differentiable structural similarity loss. */
} gsx_loss_algorithm;

typedef enum gsx_loss_grad_normalization_type {
    GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN = 0, /**< Normalize gradients by the total tensor element count. */
    GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM = 1   /**< Do not normalize gradients by the tensor element count. */
} gsx_loss_grad_normalization_type;

typedef struct gsx_loss_desc {
    gsx_loss_algorithm algorithm;                              /**< Selected differentiable loss algorithm. */
    gsx_loss_grad_normalization_type grad_normalization;       /**< Gradient normalization mode applied by `gsx_loss_backward`. */
} gsx_loss_desc;

typedef struct gsx_loss_request {
    gsx_tensor_t prediction;            /**< Read-only predicted tensor. */
    gsx_tensor_t target;                /**< Read-only target tensor. */
    gsx_tensor_t loss_map_accumulator;  /**< Required accumulator for scaled per-element loss before any normalization. */
    gsx_tensor_t grad_prediction_accumulator; /**< Optional accumulator for scaled dLoss/dPrediction after `grad_normalization`. */
    gsx_float_t scale;                  /**< Scalar weight applied to both loss-map and gradient accumulation. */
} gsx_loss_request;

typedef struct gsx_loss_forward_request {
    gsx_tensor_t prediction;            /**< Read-only predicted tensor. */
    gsx_tensor_t target;                /**< Read-only target tensor. */
    gsx_tensor_t loss_map_accumulator;  /**< Required accumulator for scaled per-element loss before any normalization. */
    bool train;                         /**< Enables retaining backend-specific backward buffers for a subsequent backward call. */
    gsx_float_t scale;                  /**< Scalar weight applied to loss-map accumulation. */
} gsx_loss_forward_request;

typedef struct gsx_loss_backward_request {
    gsx_tensor_t grad_prediction_accumulator; /**< Required accumulator for scaled dLoss/dPrediction after `grad_normalization`. */
    gsx_float_t scale;                        /**< Scalar weight applied to gradient accumulation. */
} gsx_loss_backward_request;

typedef enum gsx_metric_algorithm {
    GSX_METRIC_ALGORITHM_PSNR = 0, /**< Scalar peak signal-to-noise ratio metric. */
    GSX_METRIC_ALGORITHM_SSIM = 1  /**< Scalar structural similarity metric. */
} gsx_metric_algorithm;

typedef struct gsx_metric_desc {
    gsx_metric_algorithm algorithm;   /**< Selected scalar metric algorithm. */
} gsx_metric_desc;

typedef struct gsx_metric_request {
    gsx_tensor_t prediction; /**< Predicted image tensor. */
    gsx_tensor_t target;     /**< Target image tensor. */
} gsx_metric_request;

/*
 * Loss and metric APIs are split by intent:
 * - loss forward accumulates a map and may retain state for one backward pass;
 * - loss backward consumes that retained state exactly once;
 * - metrics do not retain gradients and are only for evaluation/logging.
 *
 * Best practice:
 * - keep one loss context per training stream;
 * - keep prediction, target, and accumulator tensors on the same backend;
 * - let the backend size any internal scratch through its own planning path
 *   instead of guessing bytes manually.
 *
 * Example:
 *   gsx_loss_forward_request fwd = { 0 };
 *   fwd.prediction = prediction;
 *   fwd.target = target;
 *   fwd.loss_map_accumulator = loss_map;
 *   fwd.train = true;
 *   gsx_loss_forward(loss, context, &fwd);
 *   gsx_loss_backward_request bwd = { .grad_prediction_accumulator = grad_pred, .scale = 1.0f };
 *   gsx_loss_backward(loss, context, &bwd);
 *   gsx_metric_request metric_req = { .prediction = prediction, .target = target };
 *   gsx_metric_evaluate(metric, &metric_req, &metric_value);
 */

/** Create a differentiable loss object. `out_loss` owns the handle on success. */
GSX_API gsx_error gsx_loss_init(gsx_loss_t *out_loss, gsx_backend_t backend, const gsx_loss_desc *desc);
/** Release a loss object created by `gsx_loss_init`. */
GSX_API gsx_error gsx_loss_free(gsx_loss_t loss);
/** Query immutable loss configuration. */
GSX_API gsx_error gsx_loss_get_desc(gsx_loss_t loss, gsx_loss_desc *out_desc);
/** Create a reusable loss context that owns retained algorithm-specific state. */
GSX_API gsx_error gsx_loss_context_init(gsx_loss_context_t *out_context, gsx_loss_t loss);
/** Release a loss context created by `gsx_loss_context_init`. */
GSX_API gsx_error gsx_loss_context_free(gsx_loss_context_t context);
/** Evaluate forward loss-map accumulation and retain prediction/target state on `context` for one backward call. */
GSX_API gsx_error gsx_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request);
/** Evaluate gradient accumulation and consume the retained forward state on `context`. */
GSX_API gsx_error gsx_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request);
/** Map a stable loss algorithm enum to a static diagnostic name string. */
GSX_API gsx_error gsx_loss_get_algorithm_name(gsx_loss_algorithm algorithm, const char **out_name);

/** Create a scalar metric object. `out_metric` owns the handle on success. */
GSX_API gsx_error gsx_metric_init(gsx_metric_t *out_metric, gsx_backend_t backend, const gsx_metric_desc *desc);
/** Release a metric object created by `gsx_metric_init`. */
GSX_API gsx_error gsx_metric_free(gsx_metric_t metric);
/** Query immutable metric configuration. */
GSX_API gsx_error gsx_metric_get_desc(gsx_metric_t metric, gsx_metric_desc *out_desc);
/** Evaluate a scalar metric. No gradients are written by this API. */
GSX_API gsx_error gsx_metric_evaluate(gsx_metric_t metric, const gsx_metric_request *request, gsx_float_t *out_value);
/** Map a stable metric algorithm enum to a static diagnostic name string. */
GSX_API gsx_error gsx_metric_get_algorithm_name(gsx_metric_algorithm algorithm, const char **out_name);

GSX_EXTERN_C_END

#endif /* GSX_LOSS_H */
