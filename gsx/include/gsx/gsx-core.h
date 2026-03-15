#ifndef GSX_CORE_H
#define GSX_CORE_H

#include "gsx-base.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_data_type {
    GSX_DATA_TYPE_F32   = 0,  /**< IEEE float32 element type. */
    GSX_DATA_TYPE_F16   = 1,  /**< IEEE float16 element type when supported by the backend. */
    GSX_DATA_TYPE_BF16  = 2,  /**< Brain floating point 16-bit type when supported by the backend. */
    GSX_DATA_TYPE_U8    = 8,  /**< Unsigned 8-bit integer element type. */
    GSX_DATA_TYPE_I8    = 9,  /**< Signed 8-bit integer element type. */
    GSX_DATA_TYPE_U16   = 10, /**< Unsigned 16-bit integer element type. */
    GSX_DATA_TYPE_I16   = 11, /**< Signed 16-bit integer element type. */
    GSX_DATA_TYPE_I32   = 12, /**< Signed 32-bit integer element type. */
    GSX_DATA_TYPE_U32   = 13, /**< Unsigned 32-bit integer element type. */
    GSX_DATA_TYPE_U64   = 14, /**< Unsigned 64-bit integer element type. */
    GSX_DATA_TYPE_I64   = 15  /**< Signed 64-bit integer element type. */
} gsx_data_type;

typedef gsx_flags64_t gsx_data_type_flags;
enum {
    GSX_DATA_TYPE_FLAG_F32  = 1ull << 0,
    GSX_DATA_TYPE_FLAG_F16  = 1ull << 1,
    GSX_DATA_TYPE_FLAG_BF16 = 1ull << 2,
    GSX_DATA_TYPE_FLAG_U8   = 1ull << 8,
    GSX_DATA_TYPE_FLAG_I8   = 1ull << 9,
    GSX_DATA_TYPE_FLAG_U16  = 1ull << 10,
    GSX_DATA_TYPE_FLAG_I16  = 1ull << 11,
    GSX_DATA_TYPE_FLAG_I32  = 1ull << 12,
    GSX_DATA_TYPE_FLAG_U32  = 1ull << 13,
    GSX_DATA_TYPE_FLAG_U64  = 1ull << 14,
    GSX_DATA_TYPE_FLAG_I64  = 1ull << 15
};

#ifdef __cplusplus
#define GSX_STATIC_ASSERT_EXPR(condition, message) static_assert(condition, message)
#else
#define GSX_STATIC_ASSERT_EXPR(condition, message) _Static_assert(condition, message)
#endif

GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_F32 == (1ull << GSX_DATA_TYPE_F32), "gsx_data_type F32 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_F16 == (1ull << GSX_DATA_TYPE_F16), "gsx_data_type F16 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_BF16 == (1ull << GSX_DATA_TYPE_BF16), "gsx_data_type BF16 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_U8 == (1ull << GSX_DATA_TYPE_U8), "gsx_data_type U8 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_I8 == (1ull << GSX_DATA_TYPE_I8), "gsx_data_type I8 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_U16 == (1ull << GSX_DATA_TYPE_U16), "gsx_data_type U16 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_I16 == (1ull << GSX_DATA_TYPE_I16), "gsx_data_type I16 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_I32 == (1ull << GSX_DATA_TYPE_I32), "gsx_data_type I32 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_U32 == (1ull << GSX_DATA_TYPE_U32), "gsx_data_type U32 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_U64 == (1ull << GSX_DATA_TYPE_U64), "gsx_data_type U64 flag mapping must stay in sync");
GSX_STATIC_ASSERT_EXPR(GSX_DATA_TYPE_FLAG_I64 == (1ull << GSX_DATA_TYPE_I64), "gsx_data_type I64 flag mapping must stay in sync");

#undef GSX_STATIC_ASSERT_EXPR

typedef enum gsx_storage_format {
    GSX_STORAGE_FORMAT_CHW = 0,       /**< Channel-major contiguous layout. */
    GSX_STORAGE_FORMAT_HWC = 1,       /**< Pixel-major contiguous layout. */
    GSX_STORAGE_FORMAT_TILED_CHW = 2  /**< Tiled CHW layout with backend-defined tile interpretation. */
} gsx_storage_format;

typedef enum gsx_arena_growth_mode {
    GSX_ARENA_GROWTH_MODE_FIXED = 0,          /**< Allocation fails once the current arena capacity is insufficient. */
    GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND = 1  /**< Allocation may reserve more capacity; non-dry-run requires no live tensors, while dry-run may grow for sizing-only planning. */
} gsx_arena_growth_mode;

typedef struct gsx_arena_mark {
    gsx_size_t offset_bytes;  /**< Arena cursor position captured by `gsx_arena_get_mark`. */
    gsx_id_t reset_epoch;     /**< Reset epoch captured with the mark so stale marks can be rejected. */
} gsx_arena_mark;

/*
 * Arena objects provide suballocated storage for tensors as views over one
 * arena-managed backing buffer.
 */
typedef struct gsx_arena_desc {
    gsx_size_t initial_capacity_bytes;      /**< Initial arena capacity request in bytes. Implementations may round it with `gsx_backend_buffer_type_get_alloc_size`. */
    gsx_size_t requested_alignment_bytes;   /**< Minimum alignment requested for arena-backed allocations. Zero means use the buffer-type default. */
    gsx_arena_growth_mode growth_mode;      /**< Whether the arena is fixed-capacity or may reserve more storage on demand. */
    bool dry_run;                           /**< If true, the arena mirrors allocation layout and accounting without reserving backing memory. */
} gsx_arena_desc;

typedef struct gsx_arena_info {
    gsx_size_t capacity_bytes;               /**< Currently allocated scratch capacity in bytes after backend rounding. */
    gsx_size_t used_bytes;                   /**< Sum of live allocation spans, including per-allocation alignment padding. */
    gsx_size_t peak_bytes;                   /**< Maximum observed `used_bytes` since the most recent full reset. */
    gsx_size_t effective_alignment_bytes;    /**< Effective allocation alignment in bytes. It is never lower than the owning buffer-type alignment. */
    gsx_size_t active_tensor_count;          /**< Number of live tensors that still reference arena storage. */
    gsx_arena_growth_mode growth_mode;       /**< Effective growth policy for this arena. */
    bool dry_run;                            /**< True if the arena only tracks size requirements. */
    gsx_backend_buffer_type_t buffer_type;   /**< Buffer type that owns arena allocations; borrowed handle valid while the arena lives. */
} gsx_arena_info;

/** Create an arena bound to a backend buffer type. `out_arena` owns the handle on success. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs or NULL buffer types; `GSX_ERROR_OUT_OF_RANGE` if the requested capacity overflows or exceeds the buffer-type limit. */
GSX_API gsx_error gsx_arena_init(gsx_arena_t *out_arena, gsx_backend_buffer_type_t buffer_type, const gsx_arena_desc *desc);
/** Release an arena previously created by `gsx_arena_init`. Returns `GSX_ERROR_INVALID_ARGUMENT` if `arena` is NULL and `GSX_ERROR_INVALID_STATE` while any tensor handles created from the arena still exist. */
GSX_API gsx_error gsx_arena_free(gsx_arena_t arena);
/** Query the backend associated with an arena. The returned handle is borrowed and must not be freed. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_backend(gsx_arena_t arena, gsx_backend_t *out_backend);
/** Query the buffer type associated with an arena. The returned handle is borrowed and remains valid while the arena lives. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_buffer_type(gsx_arena_t arena, gsx_backend_buffer_type_t *out_buffer_type);
/** Query stable capacity, alignment, and ownership information for an arena. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_info(gsx_arena_t arena, gsx_arena_info *out_info);
/** Reserve arena backing storage for at least `capacity_bytes`. This call returns `GSX_ERROR_INVALID_STATE` when live tensors still exist and returns `GSX_ERROR_OUT_OF_RANGE` if `capacity_bytes` exceeds the buffer-type limit after rounding. */
GSX_API gsx_error gsx_arena_reserve(gsx_arena_t arena, gsx_size_t capacity_bytes);
/** Reset the arena cursor to zero and clear episode statistics. Returns `GSX_ERROR_INVALID_STATE` if live tensors still exist. */
GSX_API gsx_error gsx_arena_reset(gsx_arena_t arena);
/** Capture the current arena cursor as a rewind target. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_mark(gsx_arena_t arena, gsx_arena_mark *out_mark);
/** Rewind the arena cursor to a previously captured mark. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL handles or stale marks and `GSX_ERROR_INVALID_STATE` if the rewind would invalidate any live tensor allocation. */
GSX_API gsx_error gsx_arena_rewind(gsx_arena_t arena, gsx_arena_mark mark);
/** Report the high-water required bytes since the most recent full reset. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_required_bytes(gsx_arena_t arena, gsx_size_t *out_required_bytes);

/*
 * Tensor descriptors describe contiguous tensors backed by byte ranges inside
 * an arena-managed buffer.
 */
typedef struct gsx_tensor_desc {
    gsx_index_t rank;                        /**< Tensor rank; must be in `[1, GSX_TENSOR_MAX_DIM]`. Scalar values use rank `1` with shape `(1)`. */
    gsx_index_t shape[GSX_TENSOR_MAX_DIM];   /**< Positive extents for the first `rank` dimensions. */
    gsx_size_t requested_alignment_bytes;    /**< Minimum required alignment for the tensor storage. Zero means use the arena default. */
    gsx_data_type data_type;                 /**< Element type for storage and arithmetic validation. */
    gsx_storage_format storage_format;       /**< Logical contiguous layout contract for the tensor. */
    gsx_arena_t arena;                       /**< Owning arena used for storage allocation; borrowed by the descriptor. */
} gsx_tensor_desc;

typedef struct gsx_tensor_info {
    gsx_index_t rank;                        /**< Effective tensor rank. */
    gsx_index_t shape[GSX_TENSOR_MAX_DIM];   /**< Effective extents for the first `rank` dimensions. */
    gsx_size_t size_bytes;                   /**< Total accessible storage in bytes. */
    gsx_size_t effective_alignment_bytes;    /**< Effective storage alignment in bytes. */
    gsx_data_type data_type;                 /**< Effective tensor element type. */
    gsx_storage_format storage_format;       /**< Effective logical storage format. */
    gsx_arena_t arena;                       /**< Arena that owns the storage; borrowed handle. */
    gsx_backend_buffer_type_t buffer_type;   /**< Buffer type that backs the owning arena; borrowed handle valid while the tensor lives. */
} gsx_tensor_info;

/** Allocate and initialize a tensor according to `desc`. `out_tensor` owns the handle on success. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs, NULL arenas, unsupported dtypes, or malformed shapes; returns `GSX_ERROR_OUT_OF_RANGE` when the computed storage size overflows or the required allocation exceeds current arena limits. Zero-extent tensors are not supported; use shape `(1)` for scalar-like values. */
GSX_API gsx_error gsx_tensor_init(gsx_tensor_t *out_tensor, const gsx_tensor_desc *desc);
/** Release a tensor created by `gsx_tensor_init`. Releasing the handle never rewinds or compacts the arena cursor. */
GSX_API gsx_error gsx_tensor_free(gsx_tensor_t tensor);
/** Return the descriptor view for an existing tensor. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_get_desc(gsx_tensor_t tensor, gsx_tensor_desc *out_desc);
/** Return derived information such as effective size, alignment, and buffer-type ownership. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_get_info(gsx_tensor_t tensor, gsx_tensor_info *out_info);
/** Return the total byte capacity addressable through the tensor handle. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_get_size_bytes(gsx_tensor_t tensor, gsx_size_t *out_size_bytes);
/** Upload raw bytes from host memory. `byte_count` must not exceed the tensor capacity. The transfer is ordered on the owning backend major stream; backend-specific transfer mechanics remain internal. Dry-run tensors return `GSX_ERROR_INVALID_STATE`. */
GSX_API gsx_error gsx_tensor_upload(gsx_tensor_t tensor, const void *src_bytes, gsx_size_t byte_count);
/** Download raw bytes to host memory. `byte_count` must not exceed the tensor capacity. The transfer is ordered on the owning backend major stream; backend-specific transfer mechanics remain internal. Dry-run tensors return `GSX_ERROR_INVALID_STATE`. */
GSX_API gsx_error gsx_tensor_download(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count);
/** Fill the tensor storage with zeros using backend-native semantics. Returns `GSX_ERROR_INVALID_ARGUMENT` if `tensor` is NULL and `GSX_ERROR_INVALID_STATE` for dry-run tensors. */
GSX_API gsx_error gsx_tensor_set_zero(gsx_tensor_t tensor);
/** Copy one tensor into another compatible tensor. Source and destination shapes, dtypes, and storage formats must match and both tensors must belong to the same backend. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL handles or incompatible tensors. */
GSX_API gsx_error gsx_tensor_copy(gsx_tensor_t src, gsx_tensor_t dst);
/** Broadcast a scalar byte-pattern into the full tensor. `value_size_bytes` must match the element size or the call returns `GSX_ERROR_INVALID_ARGUMENT`. */
GSX_API gsx_error gsx_tensor_fill(gsx_tensor_t tensor, const void *value_bytes, gsx_size_t value_size_bytes);

/** Gather elements from `x` using `index` (1d, int32 tensor) as the indices. The leading dimension of `index` must match the leading dimension of `out`. Repeated indices are undefined behavior. */
GSX_API gsx_error gsx_tensor_gather(gsx_tensor_t x, gsx_tensor_t index, gsx_tensor_t out);
/** Resize `x` into `out`. The leading dimension of x vs. out could be different, but the rest of the shape must match. (zero init remainings) */
GSX_API gsx_error gsx_tensor_resize(gsx_tensor_t x, gsx_tensor_t out);
/** Check tensor values for NaN or infinity and return whether all values are finite. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output and `GSX_ERROR_NOT_SUPPORTED` for unsupported floating-point dtypes. */
GSX_API gsx_error gsx_tensor_check_finite(gsx_tensor_t tensor, bool *out_is_finite);
/** Elementwise exponential. Input and output tensors must be shape-compatible. */
GSX_API gsx_error gsx_tensor_exp(gsx_tensor_t x, gsx_tensor_t out);
/** Elementwise sigmoid. Input and output tensors must be shape-compatible. */
GSX_API gsx_error gsx_tensor_sigmoid(gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out);
/** Elementwise sigmoid derivative helper. */
GSX_API gsx_error gsx_tensor_sigmoid_grad(gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out);
/** Elementwise absolute value. */
GSX_API gsx_error gsx_tensor_abs(gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out);

/** In-place exponential. Tensor must be writable and use a supported data type. */
GSX_API gsx_error gsx_tensor_exp_inplace(gsx_arena_t arena, gsx_tensor_t x);
/** In-place sigmoid. Tensor must be writable and use a supported data type. */
GSX_API gsx_error gsx_tensor_sigmoid_inplace(gsx_arena_t arena, gsx_tensor_t x);
/** In-place sigmoid derivative helper. */
GSX_API gsx_error gsx_tensor_sigmoid_grad_inplace(gsx_arena_t arena, gsx_tensor_t x);
/** In-place absolute value. */
GSX_API gsx_error gsx_tensor_abs_inplace(gsx_arena_t arena, gsx_tensor_t x);
/** In-place clamp in the range [`min`, `max`]. */
GSX_API gsx_error gsx_tensor_clamp_inplace(gsx_arena_t arena, gsx_tensor_t x, void* min_value, void* max_value);

/** Reduce by summation from `start_axis` to the final axis into `tensor_out`. */
GSX_API gsx_error gsx_tensor_sum(gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis);
/** Reduce by mean from `start_axis` to the final axis into `tensor_out`. */
GSX_API gsx_error gsx_tensor_mean(gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis);
/** Reduce by maximum from `start_axis` to the final axis into `tensor_out`. */
GSX_API gsx_error gsx_tensor_max(gsx_arena_t arena, gsx_tensor_t tensor_in, gsx_tensor_t tensor_out, gsx_index_t start_axis);

/** Fused mean squared error helper. Inputs must be shape-compatible. */
GSX_API gsx_error gsx_tensor_mse(gsx_arena_t arena, gsx_tensor_t pred, gsx_tensor_t target, gsx_tensor_t out);
/** Fused mean absolute error helper. Inputs must be shape-compatible. */
GSX_API gsx_error gsx_tensor_mae(gsx_arena_t arena, gsx_tensor_t pred, gsx_tensor_t target, gsx_tensor_t out);

typedef gsx_flags32_t gsx_gs_aux_flags;
enum {
    GSX_GS_AUX_NONE              = 0u,      /**< No auxiliary storage. */
    GSX_GS_AUX_VISIBLE_COUNTER   = 1u << 0, /**< Per-Gaussian visible-count statistics for replayable training heuristics. */
    GSX_GS_AUX_MAX_SCREEN_RADIUS = 1u << 1, /**< Per-Gaussian maximum observed screen radius. */
    GSX_GS_AUX_GRAD_ACC          = 1u << 2, /**< Per-Gaussian accumulated gradient statistics. */
    GSX_GS_AUX_ABSGRAD_ACC       = 1u << 3, /**< Per-Gaussian accumulated absolute image-gradient statistics. */
    GSX_GS_AUX_METRICS_ACC       = 1u << 4, /**< Per-Gaussian customizable metric accumulation storage. */
    GSX_GS_AUX_SH1               = 1u << 5, /**< Per-Gaussian SH1 coefficients. */
    GSX_GS_AUX_SH2               = 1u << 6, /**< Per-Gaussian SH2 coefficients. */
    GSX_GS_AUX_SH3               = 1u << 7, /**< Per-Gaussian SH3 coefficients. */
    GSX_GS_AUX_DEFAULT           = GSX_GS_AUX_SH1 | GSX_GS_AUX_SH2 | GSX_GS_AUX_SH3, /**< Default set of auxiliary storage. */
};

typedef enum gsx_gs_field {
    GSX_GS_FIELD_MEAN3D = 0,
    GSX_GS_FIELD_LOGSCALE = 1,
    GSX_GS_FIELD_ROTATION = 2,
    GSX_GS_FIELD_OPACITY = 3,
    GSX_GS_FIELD_SH0 = 4,
    GSX_GS_FIELD_SH1 = 5,
    GSX_GS_FIELD_SH2 = 6,
    GSX_GS_FIELD_SH3 = 7,
    GSX_GS_FIELD_GRAD_MEAN3D = 8,
    GSX_GS_FIELD_GRAD_LOGSCALE = 9,
    GSX_GS_FIELD_GRAD_ROTATION = 10,
    GSX_GS_FIELD_GRAD_OPACITY = 11,
    GSX_GS_FIELD_GRAD_SH0 = 12,
    GSX_GS_FIELD_GRAD_SH1 = 13,
    GSX_GS_FIELD_GRAD_SH2 = 14,
    GSX_GS_FIELD_GRAD_SH3 = 15,
    GSX_GS_FIELD_VISIBLE_COUNTER = 16,
    GSX_GS_FIELD_MAX_SCREEN_RADIUS = 17,
    GSX_GS_FIELD_GRAD_ACC = 18,
    GSX_GS_FIELD_ABSGRAD_ACC = 19,
    GSX_GS_FIELD_METRICS_ACC = 20
} gsx_gs_field;

typedef struct gsx_gs_desc {
    gsx_arena_t arena;          /**< Arena that owns all Gaussian parameter storage; borrowed by the descriptor. */
    gsx_size_t count;           /**< Initial number of Gaussians to allocate. */
    gsx_gs_aux_flags aux_flags; /**< Auxiliary statistic tensors to allocate eagerly. */
} gsx_gs_desc;

typedef struct gsx_gs_info {
    gsx_arena_t arena;          /**< Arena that currently owns the Gaussian storage; borrowed handle. */
    gsx_size_t count;           /**< Current number of active Gaussians. */
    gsx_gs_aux_flags aux_flags; /**< Auxiliary statistic tensors currently allocated. */
} gsx_gs_info;

typedef struct gsx_gs_finite_check_result {
    bool is_finite;                        /**< True if all checked parameter tensors contain only finite values. */
    gsx_gs_field first_non_finite_field;   /**< First field that failed the finite check when available. */
    gsx_size_t first_non_finite_flat_index; /**< First flattened element index in that field when available. */
    gsx_size_t non_finite_count;           /**< Number of detected invalid values when available. */
} gsx_gs_finite_check_result;

/*
 * GS and optimizer mutation contracts are transactional.
 * A successful mutation applies the full count change; an error applies none.
 */
GSX_API gsx_error gsx_gs_init(gsx_gs_t *out_gs, const gsx_gs_desc *desc);
/** Release a Gaussian set created by `gsx_gs_init`. */
GSX_API gsx_error gsx_gs_free(gsx_gs_t gs);
/** Query current allocation and auxiliary-state information for a Gaussian set. */
GSX_API gsx_error gsx_gs_get_info(gsx_gs_t gs, gsx_gs_info *out_info);

/** Borrow one tensor field owned by the Gaussian set. `field` may select parameter, gradient, or enabled auxiliary storage. The returned tensor must not be freed by the caller. */
GSX_API gsx_error gsx_gs_get_field(gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t *out_tensor);
/** Zero all optimizer-facing gradient tensors owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_zero_gradients(gsx_gs_t gs);

/** Replace one Gaussian-set tensor view selected by `field`. Tensor shape and data type must match the GS contract; implementations may reject non-replaceable fields. */
GSX_API gsx_error gsx_gs_set_field(gsx_gs_t gs, gsx_gs_field field, gsx_tensor_t tensor);

/** Enable or disable auxiliary statistic tensors. Disabling may release backing storage. */
GSX_API gsx_error gsx_gs_set_aux_enabled(gsx_gs_t gs, gsx_gs_aux_flags aux_flags, bool enabled);
/** Zero selected auxiliary tensors owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_zero_aux_tensors(gsx_gs_t gs, gsx_gs_aux_flags aux_flags);

/** Apply a permutation tensor to all Gaussian-owned fields transactionally. It is user's duty to ensure permutation is unique. Otherwise, the result is undefined. */
GSX_API gsx_error gsx_gs_permute(gsx_gs_t gs, gsx_tensor_t permutation);
/** Gather Gaussian rows by `index` (1d, int32 tensor), transactionally replacing all GS-owned fields. */
GSX_API gsx_error gsx_gs_gather(gsx_gs_t gs, gsx_tensor_t index);
/** Resize the Gaussian set to `new_count`, preserving existing prefix rows and zero-initializing grown rows. */
GSX_API gsx_error gsx_gs_resize(gsx_gs_t gs, gsx_size_t new_count);
/** Check all Gaussian-owned parameter tensors for NaN or infinity. */
GSX_API gsx_error gsx_gs_check_finite(gsx_gs_t gs, gsx_gs_finite_check_result *out_result);

/** Helper math for quick computation host side */
GSX_API gsx_float_t gsx_expf(gsx_float_t x);
GSX_API gsx_float_t gsx_sigmoid(gsx_float_t x);
GSX_API gsx_float_t gsx_sigmoid_derivative(gsx_float_t x);


GSX_EXTERN_C_END

#endif /* GSX_CORE_H */
