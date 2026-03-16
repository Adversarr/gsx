#ifndef GSX_ADC_H
#define GSX_ADC_H

#include "gsx-data.h"
#include "gsx-optim.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_adc_algorithm {
    GSX_ADC_ALGORITHM_DEFAULT = 0, /**< Default density-control policy chosen by the implementation. */
    GSX_ADC_ALGORITHM_ABSGS = 1,   /**< Absolute-gradient-driven densification policy. */
    GSX_ADC_ALGORITHM_MCMC = 2,    /**< MCMC-style density-control policy. */
    GSX_ADC_ALGORITHM_FASTGS = 3   /**< FastGS-inspired density-control policy. */
} gsx_adc_algorithm;

typedef struct gsx_adc_desc {
    gsx_adc_algorithm algorithm;         /**< Selected ADC policy. */

    gsx_float_t pruning_opacity_threshold; /**< Prune entries below this opacity criterion. */
    gsx_float_t opacity_clamp_value;       /**< Clamp opacity to this ceiling after reset/prune steps when applicable. */
    gsx_float_t max_world_scale;           /**< Prune or reject Gaussians larger than this world-space scale. */
    gsx_float_t max_screen_scale;          /**< Prune or reject Gaussians larger than this screen-space radius/scale. */
    gsx_float_t duplicate_grad_threshold;  /**< Duplicate candidates above this accumulated gradient threshold. */
    gsx_float_t duplicate_scale_threshold; /**< Duplicate candidates below this scale threshold. */
    gsx_index_t refine_every;              /**< Run structural refinement every N global steps. */
    gsx_index_t start_refine;              /**< Do not refine before this global step. */
    gsx_index_t end_refine;                /**< Stop refinement after this global step. */
    gsx_index_t max_num_gaussians;         /**< Hard ceiling for structural growth. */
    gsx_index_t reset_every;               /**< Reset ADC statistics every N global steps when non-zero. */
    gsx_size_t seed;                       /**< Deterministic seed for stochastic ADC policies. */
    bool prune_degenerate_rotation;        /**< Remove Gaussians with invalid or degenerate rotation state when supported. */

    gsx_float_t duplicate_absgrad_threshold; /**< Extra absolute-gradient threshold used by ABSGS-like policies. */
    gsx_float_t noise_strength;              /**< Stochastic perturbation scale for MCMC-like policies. */
    gsx_float_t grow_ratio;                  /**< Target growth ratio for MCMC-like policies. */
    gsx_float_t loss_threshold;              /**< Loss gate for FastGS-like selective updates. */
    gsx_index_t max_sampled_cameras;         /**< Maximum sampled cameras for policies that score visibility by sampling. */
    gsx_float_t importance_threshold;        /**< Importance threshold used by FastGS-like pruning/duplication. */
    gsx_float_t prune_budget_ratio;          /**< Budget ratio for bounded prune passes. */
} gsx_adc_desc;

typedef struct gsx_adc_request {
    gsx_gs_t gs;                  /**< Gaussian set to mutate on success. */
    gsx_optim_t optim;            /**< Optimizer whose state must stay count-aligned with `gs`. */
    gsx_dataloader_t dataloader;  /**< Dataloader consulted for replayable visibility/statistics queries. */
    gsx_renderer_t renderer;      /**< Renderer used for policy-specific sampling or validation. */
    gsx_size_t global_step;       /**< Global training step associated with this ADC invocation. */
    gsx_float_t scene_scale;      /**< Scene normalization scale used for grow/prune world-scale thresholds. */
} gsx_adc_request;

typedef struct gsx_adc_result {
    gsx_size_t gaussians_before; /**< Number of Gaussians visible to ADC before mutation. */
    gsx_size_t gaussians_after;  /**< Number of Gaussians visible to ADC after mutation. */
    gsx_size_t pruned_count;     /**< Number of entries removed by the step. */
    gsx_size_t duplicated_count; /**< Number of entries duplicated by the step. */
    gsx_size_t grown_count;      /**< Number of entirely new entries grown by the step. */
    gsx_size_t reset_count;      /**< Number of statistic reset events performed by the step. */
    bool mutated;                /**< True if any structural or stateful change was committed. */
} gsx_adc_result;

/** Create an ADC policy object. `out_adc` owns the handle on success. */
GSX_API gsx_error gsx_adc_init(gsx_adc_t *out_adc, gsx_backend_t backend, const gsx_adc_desc *desc);
/** Release an ADC policy created by `gsx_adc_init`. */
GSX_API gsx_error gsx_adc_free(gsx_adc_t adc);
/** Query the current ADC configuration. */
GSX_API gsx_error gsx_adc_get_desc(gsx_adc_t adc, gsx_adc_desc *out_desc);
/** Replace the current ADC configuration. */
GSX_API gsx_error gsx_adc_set_desc(gsx_adc_t adc, const gsx_adc_desc *desc);

/*
 * ADC steps are transactional across GS and optimizer mutation.
 * A failing step must leave both objects unchanged from the caller's point of
 * view.
 * NOTE:
 * - All related tensors must have GSX_BACKEND_BUFFER_TYPE_DEVICE backing buffer, except CPU!
 */
/** Execute one transactional ADC step. On error, GS and optimizer state must remain unchanged. */
GSX_API gsx_error gsx_adc_step(gsx_adc_t adc, const gsx_adc_request *request, gsx_adc_result *out_result);

GSX_EXTERN_C_END

#endif /* GSX_ADC_H */
