#ifndef GSX_FLANN_H
#define GSX_FLANN_H

#include "gsx/gsx-core.h"

GSX_EXTERN_C_BEGIN

/**
 * Recompute gaussian logscale and rotation from current mean3d positions with FLANN KNN.
 *
 * This helper updates only `GSX_GS_FIELD_LOGSCALE` and `GSX_GS_FIELD_ROTATION`.
 * It does not modify opacity or any SH fields.
 *
 * @param in_out_gs         Gaussian set to update. Must not be NULL.
 * @param num_neighbors     Number of nearest neighbors used per gaussian. Must be > 0.
 * @param init_scaling      Multiplicative factor applied to the neighbor distance estimate.
 * @param min_distance      Lower clamp bound for linear scale before log transform.
 * @param max_distance      Upper clamp bound for linear scale before log transform.
 * @param default_distance  Fallback distance used when no valid neighbors are found.
 * @param radius            Optional radius filter applied after KNN lookup. Set <= 0 to disable.
 * @param use_anisotropic   When true, estimate anisotropic covariance and rotation from local neighbors.
 * @return                  GSX_ERROR_SUCCESS on success, or an appropriate error code.
 */
GSX_API gsx_error gsx_gs_recompute_scale_rotation_flann(
    gsx_gs_t in_out_gs,
    gsx_index_t num_neighbors,
    gsx_float_t init_scaling,
    gsx_float_t min_distance,
    gsx_float_t max_distance,
    gsx_float_t default_distance,
    gsx_float_t radius,
    bool use_anisotropic);

GSX_EXTERN_C_END

#endif // GSX_FLANN_H
