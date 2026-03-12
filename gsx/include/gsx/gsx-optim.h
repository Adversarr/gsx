#ifndef GSX_OPTIM_H
#define GSX_OPTIM_H

#include "gsx-core.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_optim_algorithm {
    GSX_OPTIM_ALGORITHM_ADAM = 0 /**< Adam optimizer with implementation-owned moment state. */
} gsx_optim_algorithm;

typedef enum gsx_optim_param_role {
    GSX_OPTIM_PARAM_ROLE_MEAN3D = 0,   /**< Built-in optimizer role for Gaussian means. */
    GSX_OPTIM_PARAM_ROLE_LOGSCALE = 1, /**< Built-in optimizer role for Gaussian log-scales. */
    GSX_OPTIM_PARAM_ROLE_ROTATION = 2, /**< Built-in optimizer role for Gaussian rotations. */
    GSX_OPTIM_PARAM_ROLE_OPACITY = 3,  /**< Built-in optimizer role for Gaussian opacities. */
    GSX_OPTIM_PARAM_ROLE_SH0 = 4,      /**< Built-in optimizer role for degree-0 SH coefficients. */
    GSX_OPTIM_PARAM_ROLE_SH1 = 5,      /**< Built-in optimizer role for degree-1 SH coefficients. */
    GSX_OPTIM_PARAM_ROLE_SH2 = 6,      /**< Built-in optimizer role for degree-2 SH coefficients. */
    GSX_OPTIM_PARAM_ROLE_SH3 = 7,      /**< Built-in optimizer role for degree-3 SH coefficients. */
    GSX_OPTIM_PARAM_ROLE_CUSTOM = 255  /**< Custom role with index-only addressing. Role-based APIs reject it. */
} gsx_optim_param_role;

typedef gsx_flags32_t gsx_optim_param_role_flags;
enum {
    GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D = 1u << 0,   /**< Select the built-in mean3d role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_LOGSCALE = 1u << 1, /**< Select the built-in logscale role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_ROTATION = 1u << 2, /**< Select the built-in rotation role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY = 1u << 3,  /**< Select the built-in opacity role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_SH0 = 1u << 4,      /**< Select the built-in SH0 role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_SH1 = 1u << 5,      /**< Select the built-in SH1 role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_SH2 = 1u << 6,      /**< Select the built-in SH2 role. */
    GSX_OPTIM_PARAM_ROLE_FLAG_SH3 = 1u << 7       /**< Select the built-in SH3 role. */
};

/*
 * Parameter group descriptors are immutable after gsx_optim_init succeeds.
 * Each built-in role may appear at most once. `GSX_OPTIM_PARAM_ROLE_CUSTOM`
 * may appear multiple times and is addressed through index-based APIs only.
 * Learning rate updates happen through dedicated setter APIs.
 */
typedef struct gsx_optim_param_group_desc {
    gsx_optim_param_role role; /**< Built-in GS role or `GSX_OPTIM_PARAM_ROLE_CUSTOM`. */
    const char *label;         /**< Optional diagnostics-only label. It is never used for lookup, targeting, or scheduler logic. */
    gsx_tensor_t parameter;    /**< Borrowed tensor updated by the optimizer. */
    gsx_tensor_t gradient;     /**< Borrowed gradient tensor consumed by optimization steps. */
    gsx_float_t learning_rate; /**< Initial learning rate for the group. */
    gsx_float_t beta1;         /**< First-moment decay coefficient for Adam. */
    gsx_float_t beta2;         /**< Second-moment decay coefficient for Adam. */
    gsx_float_t weight_decay;  /**< Weight-decay coefficient applied during step when supported. */
    gsx_float_t epsilon;       /**< Numerical stability term for denominator regularization. */
    gsx_float_t max_grad_norm; /**< Optional per-group clipping threshold; non-positive disables per-group clipping. */
} gsx_optim_param_group_desc;

typedef struct gsx_optim_desc {
    gsx_optim_algorithm algorithm;             /**< Selected optimizer algorithm. */
    const gsx_optim_param_group_desc *param_groups; /**< Immutable array of parameter-group descriptors. */
    gsx_index_t param_group_count;             /**< Number of descriptors in `param_groups`. */
} gsx_optim_desc;

typedef struct gsx_optim_info {
    gsx_optim_algorithm algorithm;  /**< Optimizer algorithm used by this instance. */
    gsx_index_t param_group_count;  /**< Number of parameter groups bound to this instance. */
} gsx_optim_info;

typedef struct gsx_optim_step_request {
    gsx_optim_param_role_flags role_flags; /**< Built-in GS roles to step. `GSX_OPTIM_PARAM_ROLE_CUSTOM` is not representable here. */
    const gsx_index_t *param_group_indices; /**< Optional extra param-group indices to step, typically for custom groups. */
    gsx_index_t param_group_index_count;    /**< Number of entries in `param_group_indices`. */
    bool force_all;                         /**< If true, ignore the selectors and step every group. */
} gsx_optim_step_request;

typedef struct gsx_grad_norm_result {
    gsx_float_t global_norm;              /**< Measured norm across all contributing gradients. */
    bool was_clipped;                     /**< True if the immediately preceding clipping operation modified gradients. */
    gsx_index_t contributing_param_group_count; /**< Number of groups included in the norm computation. */
} gsx_grad_norm_result;

/** Create an optimizer instance. `out_optim` owns the handle on success. */
GSX_API gsx_error gsx_optim_init(gsx_optim_t *out_optim, gsx_backend_t backend, const gsx_optim_desc *desc);
/** Release an optimizer created by `gsx_optim_init`. */
GSX_API gsx_error gsx_optim_free(gsx_optim_t optim);
/** Query basic optimizer metadata. */
GSX_API gsx_error gsx_optim_get_info(gsx_optim_t optim, gsx_optim_info *out_info);
/** Query a copied view of a parameter-group descriptor by index. */
GSX_API gsx_error gsx_optim_get_param_group_desc_by_index(gsx_optim_t optim, gsx_index_t index, gsx_optim_param_group_desc *out_desc);
/** Query a copied view of a built-in parameter-group descriptor by role. `GSX_OPTIM_PARAM_ROLE_CUSTOM` is rejected with `GSX_ERROR_INVALID_ARGUMENT`. */
GSX_API gsx_error gsx_optim_get_param_group_desc_by_role(gsx_optim_t optim, gsx_optim_param_role role, gsx_optim_param_group_desc *out_desc);

/** Apply one optimizer step to the selected parameter groups. */
GSX_API gsx_error gsx_optim_step(gsx_optim_t optim, const gsx_optim_step_request *request);
/** Query the current learning rate for a parameter group by index. */
GSX_API gsx_error gsx_optim_get_learning_rate_by_index(gsx_optim_t optim, gsx_index_t index, gsx_float_t *out_learning_rate);
/** Query the current learning rate for a built-in parameter group by role. `GSX_OPTIM_PARAM_ROLE_CUSTOM` is rejected with `GSX_ERROR_INVALID_ARGUMENT`. */
GSX_API gsx_error gsx_optim_get_learning_rate_by_role(gsx_optim_t optim, gsx_optim_param_role role, gsx_float_t *out_learning_rate);
/** Override the current learning rate for a parameter group by index. */
GSX_API gsx_error gsx_optim_set_learning_rate_by_index(gsx_optim_t optim, gsx_index_t index, gsx_float_t learning_rate);
/** Override the current learning rate for a built-in parameter group by role. `GSX_OPTIM_PARAM_ROLE_CUSTOM` is rejected with `GSX_ERROR_INVALID_ARGUMENT`. */
GSX_API gsx_error gsx_optim_set_learning_rate_by_role(gsx_optim_t optim, gsx_optim_param_role role, gsx_float_t learning_rate);
/** Measure the aggregate gradient norm across optimizer-owned parameter groups. */
GSX_API gsx_error gsx_optim_get_grad_norm(gsx_optim_t optim, gsx_grad_norm_result *out_result);
/** Clip gradients in-place to `max_norm` and report the measured norm. */
GSX_API gsx_error gsx_optim_clip_grad_norm(gsx_optim_t optim, gsx_float_t max_norm, gsx_grad_norm_result *out_result);

/** Apply a permutation tensor to all optimizer-owned parameter and state tensors transactionally. */
GSX_API gsx_error gsx_optim_permute(gsx_optim_t optim, gsx_tensor_t permutation);
/** Remove optimizer-owned entries rejected by `keep_mask` transactionally. */
GSX_API gsx_error gsx_optim_prune(gsx_optim_t optim, gsx_tensor_t keep_mask);
/** Grow optimizer-owned state tensors by `growth_count` entries, matching GS growth. */
GSX_API gsx_error gsx_optim_grow(gsx_optim_t optim, gsx_size_t growth_count);

/** Reset all optimizer state such as moments and accumulators. */
GSX_API gsx_error gsx_optim_reset(gsx_optim_t optim);
/** Reset one parameter-group state by index. */
GSX_API gsx_error gsx_optim_reset_param_group_by_index(gsx_optim_t optim, gsx_index_t index);
/** Reset one built-in parameter-group state by role. `GSX_OPTIM_PARAM_ROLE_CUSTOM` is rejected with `GSX_ERROR_INVALID_ARGUMENT`. */
GSX_API gsx_error gsx_optim_reset_param_group_by_role(gsx_optim_t optim, gsx_optim_param_role role);

GSX_EXTERN_C_END

#endif /* GSX_OPTIM_H */
