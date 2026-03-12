#ifndef GSX_CORE_H
#define GSX_CORE_H

#include "gsx-base.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_data_type {
    GSX_DATA_TYPE_F32 = 0,  /**< IEEE float32 element type. */
    GSX_DATA_TYPE_F16 = 1,  /**< IEEE float16 element type when supported by the backend. */
    GSX_DATA_TYPE_BF16 = 2, /**< Brain floating point 16-bit type when supported by the backend. */
    GSX_DATA_TYPE_U8 = 8,   /**< Unsigned 8-bit integer element type. */
    GSX_DATA_TYPE_I8 = 9,   /**< Signed 8-bit integer element type. */
    GSX_DATA_TYPE_U16 = 10, /**< Unsigned 16-bit integer element type. */
    GSX_DATA_TYPE_I16 = 11, /**< Signed 16-bit integer element type. */
    GSX_DATA_TYPE_I32 = 12, /**< Signed 32-bit integer element type. */
    GSX_DATA_TYPE_U32 = 13, /**< Unsigned 32-bit integer element type. */
    GSX_DATA_TYPE_U64 = 14, /**< Unsigned 64-bit integer element type. */
    GSX_DATA_TYPE_I64 = 15  /**< Signed 64-bit integer element type. */
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

typedef enum gsx_storage_format {
    GSX_STORAGE_FORMAT_CHW = 0,       /**< Channel-major contiguous layout. */
    GSX_STORAGE_FORMAT_HWC = 1,       /**< Pixel-major contiguous layout. */
    GSX_STORAGE_FORMAT_TILED_CHW = 2  /**< Tiled CHW layout with backend-defined tile interpretation. */
} gsx_storage_format;

/*
 * Arena objects provide temporary allocation storage for backend operations.
 * The stable contract treats them as backend-owned scratch allocators.
 */
typedef struct gsx_arena_desc {
    gsx_size_t initial_capacity_bytes;  /**< Initial scratch allocation request in bytes. Implementations may round it with `gsx_backend_buffer_type_get_alloc_size`. */
    gsx_size_t alignment_bytes;         /**< Minimum alignment requested for arena-backed allocations. Zero means use the buffer-type default. */
    bool dry_run;                       /**< If true, the arena reports sizing and alignment decisions without reserving backing memory. */
} gsx_arena_desc;

typedef struct gsx_arena_info {
    gsx_size_t capacity_bytes;               /**< Currently allocated scratch capacity in bytes after backend rounding. */
    gsx_size_t used_bytes;                   /**< Bytes currently considered live by the implementation. */
    gsx_size_t alignment_bytes;              /**< Effective allocation alignment in bytes. It is never lower than the owning buffer-type alignment. */
    bool dry_run;                            /**< True if the arena only tracks size requirements. */
    gsx_backend_buffer_type_t buffer_type;   /**< Buffer type that owns arena allocations; borrowed handle valid while the arena lives. */
} gsx_arena_info;

/** Create a scratch arena bound to a backend buffer type. `out_arena` owns the handle on success. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs or NULL buffer types; `GSX_ERROR_OUT_OF_RANGE` if the requested capacity overflows or exceeds the buffer-type limit. */
GSX_API gsx_error gsx_arena_init(gsx_arena_t *out_arena, gsx_backend_buffer_type_t buffer_type, const gsx_arena_desc *desc);
/** Release an arena previously created by `gsx_arena_init`. Returns `GSX_ERROR_INVALID_ARGUMENT` if `arena` is NULL. */
GSX_API gsx_error gsx_arena_free(gsx_arena_t arena);
/** Query the backend associated with an arena. The returned handle is borrowed and must not be freed. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_backend(gsx_arena_t arena, gsx_backend_t *out_backend);
/** Query the buffer type associated with an arena. The returned handle is borrowed and remains valid while the arena lives. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_buffer_type(gsx_arena_t arena, gsx_backend_buffer_type_t *out_buffer_type);
/** Query stable capacity, alignment, and ownership information for an arena. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_info(gsx_arena_t arena, gsx_arena_info *out_info);
/** Resize arena backing storage. Implementations may reject resize while the arena is in active use. Returns `GSX_ERROR_OUT_OF_RANGE` if `capacity_bytes` exceeds the buffer-type limit after rounding. */
GSX_API gsx_error gsx_arena_resize(gsx_arena_t arena, gsx_size_t capacity_bytes);
/** Report required bytes for dry-run sizing or implementation-defined planning. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_arena_get_required_bytes(gsx_arena_t arena, gsx_size_t *out_required_bytes);

/*
 * Tensor descriptors describe contiguous tensors owned by a backend arena.
 */
typedef struct gsx_tensor_desc {
    gsx_index_t rank;                        /**< Tensor rank; must be in `[0, GSX_TENSOR_MAX_DIM]`. */
    gsx_index_t shape[GSX_TENSOR_MAX_DIM];   /**< Extents for the first `rank` dimensions. */
    gsx_size_t alignment_bytes;              /**< Minimum required alignment for the tensor storage. Zero means use the arena default. */
    gsx_data_type data_type;                 /**< Element type for storage and arithmetic validation. */
    gsx_storage_format storage_format;       /**< Logical contiguous layout contract for the tensor. */
    gsx_arena_t arena;                       /**< Owning arena used for storage allocation; borrowed by the descriptor. */
} gsx_tensor_desc;

typedef struct gsx_tensor_info {
    gsx_index_t rank;                        /**< Effective tensor rank. */
    gsx_index_t shape[GSX_TENSOR_MAX_DIM];   /**< Effective extents for the first `rank` dimensions. */
    gsx_size_t size_bytes;                   /**< Total accessible storage in bytes. */
    gsx_size_t alignment_bytes;              /**< Effective storage alignment in bytes. */
    gsx_data_type data_type;                 /**< Effective tensor element type. */
    gsx_storage_format storage_format;       /**< Effective logical storage format. */
    gsx_arena_t arena;                       /**< Arena that owns the storage; borrowed handle. */
    gsx_backend_buffer_type_t buffer_type;   /**< Buffer type that backs the owning arena; borrowed handle valid while the tensor lives. */
} gsx_tensor_info;

typedef struct gsx_finite_check_result {
    bool is_finite;                       /**< True if all inspected values are finite. */
    gsx_size_t first_non_finite_flat_index; /**< First flattened element index that failed the check. */
    gsx_size_t non_finite_count;          /**< Number of non-finite values detected when available. */
} gsx_finite_check_result;

/** Allocate and initialize a tensor according to `desc`. `out_tensor` owns the handle on success. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs, NULL arenas, unsupported dtypes, or malformed shapes; returns `GSX_ERROR_OUT_OF_RANGE` when the computed storage size overflows or exceeds arena limits. */
GSX_API gsx_error gsx_tensor_init(gsx_tensor_t *out_tensor, const gsx_tensor_desc *desc);
/** Release a tensor created by `gsx_tensor_init`. Returns `GSX_ERROR_INVALID_ARGUMENT` if `tensor` is NULL. */
GSX_API gsx_error gsx_tensor_free(gsx_tensor_t tensor);
/** Return the descriptor view for an existing tensor. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_get_desc(gsx_tensor_t tensor, gsx_tensor_desc *out_desc);
/** Return derived information such as effective size, alignment, and buffer-type ownership. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_get_info(gsx_tensor_t tensor, gsx_tensor_info *out_info);
/** Return the total byte capacity addressable through the tensor handle. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_get_size_bytes(gsx_tensor_t tensor, gsx_size_t *out_size_bytes);
/** Upload raw bytes from host memory. `byte_count` must not exceed the tensor capacity. The transfer is ordered on the owning backend major stream; backend-specific transfer mechanics remain internal. */
GSX_API gsx_error gsx_tensor_upload(gsx_tensor_t tensor, const void *src_bytes, gsx_size_t byte_count);
/** Download raw bytes to host memory. `byte_count` must not exceed the tensor capacity. The transfer is ordered on the owning backend major stream; backend-specific transfer mechanics remain internal. */
GSX_API gsx_error gsx_tensor_download(gsx_tensor_t tensor, void *dst_bytes, gsx_size_t byte_count);
/** Fill the tensor storage with zeros using backend-native semantics. Returns `GSX_ERROR_INVALID_ARGUMENT` if `tensor` is NULL. */
GSX_API gsx_error gsx_tensor_set_zero(gsx_tensor_t tensor);
/** Copy one tensor into another compatible tensor. Source and destination shapes, dtypes, and storage formats must match. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL handles. */
GSX_API gsx_error gsx_tensor_copy(gsx_arena_t arena, gsx_tensor_t src, gsx_tensor_t dst);
/** Broadcast a scalar byte-pattern into the full tensor. `value_size_bytes` must match the element size or the call returns `GSX_ERROR_INVALID_ARGUMENT`. */
GSX_API gsx_error gsx_tensor_fill(gsx_arena_t arena, gsx_tensor_t tensor, const void *value_bytes, gsx_size_t value_size_bytes);
/** Check tensor values for NaN or infinity and report the first offending location when available. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_tensor_check_finite(gsx_tensor_t tensor, gsx_finite_check_result *out_result);

/** Elementwise exponential. Input and output tensors must be shape-compatible. */
GSX_API gsx_error gsx_tensor_exp(gsx_arena_t arena, gsx_tensor_t x, gsx_tensor_t out);
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
    GSX_GS_AUX_VISIBLE_COUNTER   = 1u << 0, /**< Per-Gaussian visible-count statistics for replayable training heuristics. */
    GSX_GS_AUX_MAX_SCREEN_RADIUS = 1u << 1, /**< Per-Gaussian maximum observed screen radius. */
    GSX_GS_AUX_GRAD_ACC          = 1u << 2, /**< Per-Gaussian accumulated gradient statistics. */
    GSX_GS_AUX_ABSGRAD_ACC       = 1u << 3, /**< Per-Gaussian accumulated absolute image-gradient statistics. */
    GSX_GS_AUX_METRICS_ACC       = 1u << 4  /**< Per-Gaussian customizable metric accumulation storage. */
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

/** Borrow the mean3d tensor owned by the Gaussian set. The returned tensor must not be freed by the caller. */
GSX_API gsx_error gsx_gs_get_mean3d(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the logscale tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_logscale(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the rotation tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_rotation(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the opacity tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_opacity(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the SH degree-0 tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_sh0(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the SH degree-1 tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_sh1(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the SH degree-2 tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_sh2(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the SH degree-3 tensor owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_get_sh3(gsx_gs_t gs, gsx_tensor_t *out_tensor);

/** Borrow the accumulated gradient tensor for mean3d. */
GSX_API gsx_error gsx_gs_get_grad_mean3d(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for logscale. */
GSX_API gsx_error gsx_gs_get_grad_logscale(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for rotation. */
GSX_API gsx_error gsx_gs_get_grad_rotation(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for opacity. */
GSX_API gsx_error gsx_gs_get_grad_opacity(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for SH degree 0. */
GSX_API gsx_error gsx_gs_get_grad_sh0(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for SH degree 1. */
GSX_API gsx_error gsx_gs_get_grad_sh1(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for SH degree 2. */
GSX_API gsx_error gsx_gs_get_grad_sh2(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient tensor for SH degree 3. */
GSX_API gsx_error gsx_gs_get_grad_sh3(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Zero all optimizer-facing gradient tensors owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_zero_gradients(gsx_gs_t gs);

/** Replace the mean3d tensor view. Tensor shape and data type must match the GS contract. */
GSX_API gsx_error gsx_gs_set_mean3d(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the logscale tensor view. Tensor shape and data type must match the GS contract. */
GSX_API gsx_error gsx_gs_set_logscale(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the rotation tensor view. Tensor shape and data type must match the GS contract. */
GSX_API gsx_error gsx_gs_set_rotation(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the opacity tensor view. Tensor shape and data type must match the GS contract. */
GSX_API gsx_error gsx_gs_set_opacity(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the SH degree-0 tensor view. */
GSX_API gsx_error gsx_gs_set_sh0(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the SH degree-1 tensor view. */
GSX_API gsx_error gsx_gs_set_sh1(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the SH degree-2 tensor view. */
GSX_API gsx_error gsx_gs_set_sh2(gsx_gs_t gs, gsx_tensor_t tensor);
/** Replace the SH degree-3 tensor view. */
GSX_API gsx_error gsx_gs_set_sh3(gsx_gs_t gs, gsx_tensor_t tensor);
/** Clamp opacity parameter values in-place into the closed interval `[min_value, max_value]`. */
GSX_API gsx_error gsx_gs_clamp_opacity(gsx_gs_t gs, gsx_float_t min_value, gsx_float_t max_value);

/** Enable or disable auxiliary statistic tensors. Disabling may release backing storage. */
GSX_API gsx_error gsx_gs_set_aux_enabled(gsx_gs_t gs, gsx_gs_aux_flags aux_flags, bool enabled);
/** Borrow the visible-counter tensor when enabled. */
GSX_API gsx_error gsx_gs_get_visible_counter(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the max-screen-radius tensor when enabled. */
GSX_API gsx_error gsx_gs_get_max_screen_radius(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated gradient-statistics tensor when enabled. */
GSX_API gsx_error gsx_gs_get_grad_acc(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the accumulated absolute-gradient tensor when enabled. */
GSX_API gsx_error gsx_gs_get_absgrad_acc(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Borrow the custom metric-accumulator tensor when enabled. */
GSX_API gsx_error gsx_gs_get_metrics_acc(gsx_gs_t gs, gsx_tensor_t *out_tensor);
/** Zero selected auxiliary tensors owned by the Gaussian set. */
GSX_API gsx_error gsx_gs_zero_aux_tensors(gsx_gs_t gs, gsx_gs_aux_flags aux_flags);

/** Apply a permutation tensor to all Gaussian-owned fields transactionally. */
GSX_API gsx_error gsx_gs_permute(gsx_gs_t gs, gsx_tensor_t permutation);
/** Remove Gaussians where `keep_mask` indicates rejection, transactionally. */
GSX_API gsx_error gsx_gs_prune(gsx_gs_t gs, gsx_tensor_t keep_mask);
/** Grow the Gaussian set by `growth_count` entries, preserving existing data. */
GSX_API gsx_error gsx_gs_grow(gsx_gs_t gs, gsx_size_t growth_count);
/** Check all Gaussian-owned parameter tensors for NaN or infinity. */
GSX_API gsx_error gsx_gs_check_finite(gsx_gs_t gs, gsx_gs_finite_check_result *out_result);

GSX_EXTERN_C_END

#endif /* GSX_CORE_H */
