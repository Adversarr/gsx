#ifndef GSX_LOSS_H
#define GSX_LOSS_H

#include "gsx-core.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_loss_algorithm {
    GSX_LOSS_ALGORITHM_L1 = 0,   /**< Differentiable L1 image loss. */
    GSX_LOSS_ALGORITHM_MSE = 1,  /**< Differentiable mean squared error loss. */
    GSX_LOSS_ALGORITHM_SSIM = 2  /**< Differentiable structural similarity loss. */
} gsx_loss_algorithm;

typedef struct gsx_loss_desc {
    gsx_loss_algorithm algorithm;       /**< Selected differentiable loss algorithm. */
    bool requires_individual_loss_map;  /**< Request a reusable per-pixel loss map when supported. */
} gsx_loss_desc;

typedef struct gsx_loss_request {
    gsx_tensor_t prediction;            /**< Predicted image tensor. */
    gsx_tensor_t target;                /**< Target image tensor. */
    gsx_tensor_t loss_map_accumulator;  /**< Optional accumulator for per-pixel loss values. */
    gsx_tensor_t grad_prediction_accumulator; /**< Optional accumulator for dLoss/dPrediction. */
    gsx_float_t scale;                  /**< Scalar weight applied before accumulation. */
} gsx_loss_request;

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

/** Create a differentiable loss object. `out_loss` owns the handle on success. */
GSX_API gsx_error gsx_loss_init(gsx_loss_t *out_loss, gsx_backend_t backend, const gsx_loss_desc *desc);
/** Release a loss object created by `gsx_loss_init`. */
GSX_API gsx_error gsx_loss_free(gsx_loss_t loss);
/** Query immutable loss configuration. */
GSX_API gsx_error gsx_loss_get_desc(gsx_loss_t loss, gsx_loss_desc *out_desc);
/** Evaluate a differentiable loss and accumulate into the provided tensors. */
GSX_API gsx_error gsx_loss_evaluate(gsx_loss_t loss, const gsx_loss_request *request);
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
