#ifndef GSX_DATA_H
#define GSX_DATA_H

#include "gsx-render.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_image_resize_policy {
    GSX_IMAGE_RESIZE_PIXEL_CENTER = 0 /**< Rescale intrinsics by image-size ratio using pixel-center semantics. */
} gsx_image_resize_policy;

/*
 * Temporary host image contract for dataset callbacks.
 * Storage is host-readable, row-major, interleaved, and tightly packed within
 * each row. `row_stride_bytes` may add padding between rows only.
 */
typedef struct gsx_cpu_image_view {
    const void *data;            /**< First byte of the image payload. `NULL` marks an absent optional view. */
    gsx_data_type data_type;     /**< Scalar element type stored in the image payload. */
    gsx_index_t width;           /**< Image width in pixels. */
    gsx_index_t height;          /**< Image height in pixels. */
    gsx_index_t channel_count;   /**< Channel count per pixel. RGB uses 3, alpha and inverse-depth use 1 when present. */
    gsx_size_t row_stride_bytes; /**< Byte stride between adjacent rows. */
} gsx_cpu_image_view;

typedef struct gsx_dataset_cpu_sample gsx_dataset_cpu_sample;

/*
 * Dataset callbacks define a stable indexed CPU sample source.
 * For one dataloader:
 * - callbacks never overlap;
 * - if async prefetch is disabled, callbacks run on the caller thread;
 * - if async prefetch is enabled, callbacks run on one GSX-owned worker
 *   thread, and `get_sample` and `release_sample` stay on that same thread;
 * - each successful `get_sample` is followed by exactly one later
 *   `release_sample`;
 * - up to `prefetch_count + 1` samples may be outstanding at once.
 *
 * The same dataset must not be attached to multiple live dataloaders
 * concurrently in this version.
 */
typedef gsx_error (*gsx_dataset_get_length_fn)(void *object, gsx_size_t *out_length);
typedef gsx_error (*gsx_dataset_get_sample_fn)(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample);
typedef void (*gsx_dataset_release_sample_fn)(void *object, gsx_dataset_cpu_sample *sample);

typedef struct gsx_dataset_desc {
    const char *name;                         /**< Optional dataset label for logging and diagnostics. */
    void *object;                            /**< Caller-owned dataset object forwarded to all callbacks. */
    gsx_dataset_get_length_fn get_length;    /**< Required callback used once at init time to cache the stable dataset length. */
    gsx_dataset_get_sample_fn get_sample;    /**< Required callback that borrows one CPU sample by stable index. */
    gsx_dataset_release_sample_fn release_sample; /**< Required callback that releases one previously borrowed CPU sample. */
} gsx_dataset_desc;

typedef struct gsx_dataset_cpu_sample {
    gsx_camera_intrinsics intrinsics; /**< Source-geometry intrinsics for this sample before dataloader resize. */
    gsx_camera_pose pose;             /**< Pose associated with the sample. `pose.camera_id` must match `intrinsics.camera_id`. */
    gsx_cpu_image_view rgb;           /**< Required RGB image view. `data` must not be `NULL` and `channel_count` must be 3. */
    gsx_cpu_image_view alpha;         /**< Optional alpha view. When present, it must match the RGB extent and use 1 channel. */
    gsx_cpu_image_view invdepth;      /**< Optional inverse-depth view. When present, it must match the RGB extent and use 1 channel. */
    gsx_id_t stable_sample_id;        /**< Optional sample identifier distinct from camera and frame identifiers. */
    bool has_stable_sample_id;        /**< True when `stable_sample_id` is meaningful. */
    void *release_token;              /**< Opaque dataset-owned token forwarded back through `release_sample` unchanged. */
} gsx_dataset_cpu_sample;

typedef struct gsx_dataset_info {
    gsx_size_t length; /**< Cached number of samples exposed by the dataset. Empty datasets are rejected at init time. */
} gsx_dataset_info;

typedef struct gsx_dataloader_desc {
    bool shuffle_each_epoch;      /**< Shuffle sample order at epoch boundaries while preserving deterministic replay with the same seed. */
    bool enable_async_prefetch;   /**< Run dataset callbacks on one GSX-owned worker thread when supported and when `prefetch_count` is non-zero. */
    gsx_size_t prefetch_count;    /**< Maximum number of prefetched samples to stage ahead. Zero disables staged lookahead. */
    gsx_size_t seed;              /**< Deterministic seed for permutation and prefetch order. */
    gsx_data_type image_data_type; /**< Desired tensor element type for returned RGB/alpha/inverse-depth tensors. */
    gsx_index_t output_width;     /**< Requested output image width in pixels. */
    gsx_index_t output_height;    /**< Requested output image height in pixels. */
    gsx_image_resize_policy resize_policy; /**< Intrinsics update policy applied with output resize. */
} gsx_dataloader_desc;

typedef struct gsx_dataloader_info {
    gsx_size_t length;           /**< Number of samples in one logical epoch. */
    bool shuffle_each_epoch;     /**< Effective epoch-shuffle setting. */
    bool enable_async_prefetch;  /**< Effective internal background-prefetch setting. Public backend-bound work still executes on the backend major stream. */
    gsx_size_t prefetch_count;   /**< Effective prefetch depth. */
    gsx_data_type image_data_type; /**< Effective tensor element type for returned RGB/alpha/inverse-depth tensors. */
    gsx_index_t output_width;    /**< Effective output width in pixels. */
    gsx_index_t output_height;   /**< Effective output height in pixels. */
} gsx_dataloader_info;

/*
 * Iterator-order state is intentionally lightweight.
 * Implementations may internally use different mechanisms, but restoring a
 * state snapshot must resume logical sample ordering for the same dataset and
 * loader configuration assumptions.
 */
typedef struct gsx_dataloader_state {
    gsx_size_t epoch_index;       /**< Current epoch index at the time of snapshot. */
    gsx_size_t next_sample_ordinal; /**< Zero-based sample ordinal that will be produced next. */
    gsx_size_t permutation_index; /**< Current permutation generation index. */
    gsx_size_t rng_state[4];      /**< RNG payload used to resume iterator order for the same process/configuration assumptions. */
} gsx_dataloader_state;

typedef gsx_flags32_t gsx_dataloader_boundary_flags;
enum {
    GSX_DATALOADER_BOUNDARY_NEW_EPOCH = 1u << 0,       /**< This result begins a new logical epoch. */
    GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION = 1u << 1  /**< This result begins a new sample permutation/order. */
};

typedef struct gsx_dataloader_result {
    gsx_camera_intrinsics intrinsics;      /**< Effective intrinsics matching the returned tensor geometry after resize. */
    gsx_camera_pose pose;                  /**< Pose associated with the returned sample. */
    gsx_tensor_t rgb_image;                /**< Borrowed RGB tensor owned by the dataloader. */
    gsx_tensor_t alpha_image;              /**< Borrowed optional alpha tensor. It is a null handle when absent. */
    gsx_tensor_t invdepth_image;           /**< Borrowed optional inverse-depth tensor. It is a null handle when absent. */
    gsx_size_t stable_sample_index;        /**< Deterministic sample ordinal within the dataset. */
    gsx_id_t stable_sample_id;             /**< Optional dataset-defined stable sample identifier. */
    bool has_stable_sample_id;             /**< True when `stable_sample_id` is meaningful. */
    gsx_size_t epoch_index;                /**< Epoch index associated with this sample. */
    gsx_dataloader_boundary_flags boundary_flags; /**< Epoch/permutation boundary markers for replay logic. */
} gsx_dataloader_result;

/** Validate callback presence, cache the stable dataset length, and create a dataset handle. Empty datasets are rejected. */
GSX_API gsx_error gsx_dataset_init(gsx_dataset_t *out_dataset, const gsx_dataset_desc *desc);
/** Release a dataset created by `gsx_dataset_init`. The dataset must outlive any dataloader that still borrows it. */
GSX_API gsx_error gsx_dataset_free(gsx_dataset_t dataset);
/** Query the cached dataset length. */
GSX_API gsx_error gsx_dataset_get_info(gsx_dataset_t dataset, gsx_dataset_info *out_info);

/** Create a dataloader over a borrowed dataset. The dataset must remain alive until `gsx_dataloader_free` returns. */
GSX_API gsx_error gsx_dataloader_init(gsx_dataloader_t *out_dataloader, gsx_backend_t backend, gsx_dataset_t dataset, const gsx_dataloader_desc *desc);
/** Release a dataloader created by `gsx_dataloader_init`. */
GSX_API gsx_error gsx_dataloader_free(gsx_dataloader_t dataloader);
/** Reset the iterator to its initial state using the configured seed and ordering rules. */
GSX_API gsx_error gsx_dataloader_reset(gsx_dataloader_t dataloader);
/** Query effective output geometry and ordering settings. */
GSX_API gsx_error gsx_dataloader_get_info(gsx_dataloader_t dataloader, gsx_dataloader_info *out_info);
/** Snapshot lightweight iterator-order state for the same dataset and loader configuration assumptions. */
GSX_API gsx_error gsx_dataloader_get_state(gsx_dataloader_t dataloader, gsx_dataloader_state *out_state);
/** Restore iterator-order state previously produced by `gsx_dataloader_get_state`. Outstanding prefetched items are flushed first. */
GSX_API gsx_error gsx_dataloader_set_state(gsx_dataloader_t dataloader, const gsx_dataloader_state *state);
/** Update the positive output resize target. Outstanding prefetched items are flushed before the new shape takes effect. */
GSX_API gsx_error gsx_dataloader_set_output_shape(gsx_dataloader_t dataloader, gsx_index_t width, gsx_index_t height);
/** Produce the next sample plus epoch/permutation boundary information. Returned tensor handles remain valid until the next mutating dataloader call. */
GSX_API gsx_error gsx_dataloader_next_ex(gsx_dataloader_t dataloader, gsx_dataloader_result *out_result);

GSX_EXTERN_C_END

#endif /* GSX_DATA_H */
