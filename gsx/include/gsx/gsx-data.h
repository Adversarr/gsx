#ifndef GSX_DATA_H
#define GSX_DATA_H

#include "gsx-render.h"

GSX_EXTERN_C_BEGIN

typedef struct gsx_dataset_cpu_sample gsx_dataset_cpu_sample;

/*
 * Dataset callbacks define a stable indexed CPU sample source.
 * For one dataloader:
 * - callbacks never overlap;
 * - synchronous iteration runs callbacks on the caller thread;
 * - async prefetch runs callbacks on one internal helper thread;
 * - each successful `get_sample` is followed by exactly one later
 *   `release_sample`;
 * - at most one sample is outstanding at once.
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
    gsx_data_type image_data_type;           /**< Fixed host image element type used by every sample payload. v1 supports U8 and F32 only. */
    gsx_index_t width;                       /**< Fixed sample width in pixels for every modality exposed by the dataset. */
    gsx_index_t height;                      /**< Fixed sample height in pixels for every modality exposed by the dataset. */
    bool has_rgb;                            /**< True when every sample provides a tightly packed HWC RGB payload with 3 channels. */
    bool has_alpha;                          /**< True when every sample provides a tightly packed HW alpha payload with 1 channel. */
    bool has_invdepth;                       /**< True when every sample provides a tightly packed HW inverse-depth payload with 1 channel. */
    gsx_dataset_get_length_fn get_length;    /**< Required callback used once at init time to cache the stable dataset length. */
    gsx_dataset_get_sample_fn get_sample;    /**< Required callback that borrows one CPU sample by stable index. */
    gsx_dataset_release_sample_fn release_sample; /**< Required callback that releases one previously borrowed CPU sample. */
} gsx_dataset_desc;

typedef struct gsx_dataset_cpu_sample {
    gsx_camera_intrinsics intrinsics; /**< Source-geometry intrinsics for this sample. `intrinsics.width/height` must match the dataset descriptor. */
    gsx_camera_pose pose;             /**< Pose associated with the sample. `pose.camera_id` must match `intrinsics.camera_id`. */
    const void *rgb_data;             /**< Fixed tightly packed HWC RGB payload with 3 channels when `gsx_dataset_desc.has_rgb` is true. */
    const void *alpha_data;           /**< Fixed tightly packed HW alpha payload when `gsx_dataset_desc.has_alpha` is true. */
    const void *invdepth_data;        /**< Fixed tightly packed HW inverse-depth payload when `gsx_dataset_desc.has_invdepth` is true. */
    gsx_id_t stable_sample_id;        /**< Optional sample identifier distinct from camera and frame identifiers. */
    bool has_stable_sample_id;        /**< True when `stable_sample_id` is meaningful. */
    void *release_token;              /**< Opaque dataset-owned token forwarded back through `release_sample` unchanged. */
} gsx_dataset_cpu_sample;

typedef struct gsx_dataset_info {
    gsx_size_t length;          /**< Cached number of samples exposed by the dataset. Empty datasets are rejected at init time. */
    gsx_data_type image_data_type; /**< Fixed host payload element type. */
    gsx_index_t width;          /**< Fixed dataset width in pixels. */
    gsx_index_t height;         /**< Fixed dataset height in pixels. */
    bool has_rgb;               /**< True when RGB payloads are present. */
    bool has_alpha;             /**< True when alpha payloads are present. */
    bool has_invdepth;          /**< True when inverse-depth payloads are present. */
} gsx_dataset_info;

typedef struct gsx_dataloader_desc {
    bool shuffle_each_epoch;      /**< Shuffle sample order at epoch boundaries while preserving deterministic ordering with the same seed. */
    bool enable_async_prefetch;   /**< Enable background dataset fetch plus backend transfer on an internal helper thread and helper stream/queue. */
    gsx_size_t prefetch_count;    /**< Async prefetch depth. Must be positive when `enable_async_prefetch` is true and zero otherwise. */
    gsx_size_t seed;              /**< Deterministic seed for permutation and prefetch order. */
    gsx_data_type image_data_type; /**< Desired tensor element type for returned RGB/alpha/inverse-depth tensors. Currently only F32 is supported. */
} gsx_dataloader_desc;

typedef struct gsx_dataloader_info {
    gsx_size_t length;           /**< Number of samples in one logical epoch. */
    bool shuffle_each_epoch;     /**< Effective epoch-shuffle setting. */
    bool enable_async_prefetch;  /**< Effective internal background-prefetch setting. Public backend-bound work still executes on the backend major stream. */
    gsx_size_t prefetch_count;   /**< Effective prefetch depth. */
    gsx_data_type image_data_type; /**< Effective tensor element type for returned RGB/alpha/inverse-depth tensors. */
    gsx_storage_format storage_format; /**< Effective returned image tensor layout. The dataloader always returns CHW. */
    gsx_index_t output_width;    /**< Effective output width in pixels copied from the dataset descriptor. */
    gsx_index_t output_height;   /**< Effective output height in pixels copied from the dataset descriptor. */
} gsx_dataloader_info;

typedef gsx_flags32_t gsx_dataloader_boundary_flags;
enum {
    GSX_DATALOADER_BOUNDARY_NEW_EPOCH = 1u << 0,       /**< This result begins a new logical epoch. */
    GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION = 1u << 1  /**< This result begins a new sample permutation/order. */
};

typedef struct gsx_dataloader_result {
    gsx_camera_intrinsics intrinsics;      /**< Effective intrinsics matching the returned tensor geometry. */
    gsx_camera_pose pose;                  /**< Pose associated with the returned sample. */
    gsx_tensor_t rgb_image;                /**< Borrowed RGB tensor owned by the dataloader with CHW shape `[3,H,W]`. */
    gsx_tensor_t alpha_image;              /**< Borrowed optional alpha tensor. It is a null handle when absent and otherwise has CHW shape `[1,H,W]`. */
    gsx_tensor_t invdepth_image;           /**< Borrowed optional inverse-depth tensor. It is a null handle when absent and otherwise has CHW shape `[1,H,W]`. */
    gsx_size_t stable_sample_index;        /**< Deterministic sample ordinal within the dataset. */
    gsx_id_t stable_sample_id;             /**< Optional dataset-defined stable sample identifier. */
    bool has_stable_sample_id;             /**< True when `stable_sample_id` is meaningful. */
    gsx_size_t epoch_index;                /**< Epoch index associated with this sample. */
    gsx_dataloader_boundary_flags boundary_flags; /**< Epoch/permutation boundary markers for data-loop control. */
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
/** Produce the next sample plus epoch/permutation boundary information. Returned tensor handles remain valid until the next mutating dataloader call. */
GSX_API gsx_error gsx_dataloader_next_ex(gsx_dataloader_t dataloader, gsx_dataloader_result *out_result);

GSX_EXTERN_C_END

#endif /* GSX_DATA_H */
