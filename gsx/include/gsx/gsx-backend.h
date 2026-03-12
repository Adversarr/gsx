#ifndef GSX_BACKEND_H
#define GSX_BACKEND_H

#include "gsx-core.h"

GSX_EXTERN_C_BEGIN

typedef enum gsx_backend_type {
    GSX_BACKEND_TYPE_CPU = 0,   /**< CPU execution backend. */
    GSX_BACKEND_TYPE_CUDA = 1,  /**< CUDA execution backend. */
    GSX_BACKEND_TYPE_METAL = 2  /**< Metal execution backend. */
} gsx_backend_type;

typedef enum gsx_backend_buffer_type_class {
    GSX_BACKEND_BUFFER_TYPE_HOST = 0,        /**< Plain host allocation. Upload and download access are direct. */
    GSX_BACKEND_BUFFER_TYPE_HOST_PINNED = 1, /**< Page-locked host allocation optimized for transfer. Upload and download access are direct. */
    GSX_BACKEND_BUFFER_TYPE_DEVICE = 2,      /**< Device-local allocation scheduled on the backend major stream. */
    GSX_BACKEND_BUFFER_TYPE_UNIFIED = 3      /**< A single allocation visible to host and device. Visibility does not guarantee optimal performance. */
} gsx_backend_buffer_type_class;

typedef struct gsx_device_info {
    gsx_backend_type backend_type; /**< Backend family that owns this device. */
    gsx_index_t device_index;    /**< Zero-based device ordinal for enumeration. */
    gsx_size_t total_memory_bytes; /**< Best-effort total addressable memory for planning. */
} gsx_device_info;

typedef struct gsx_backend_desc {
    gsx_device_t device;           /**< Target device; borrowed by the descriptor and copied at init time. */
    const void *options;           /**< Optional backend-specific initialization blob; may be NULL. */
    gsx_size_t options_size_bytes; /**< Byte size of the backend-specific options blob. */
} gsx_backend_desc;

typedef struct gsx_backend_info {
    gsx_backend_type backend_type; /**< Backend family for this backend instance. */
    gsx_device_t device;         /**< Borrowed device handle associated with this backend. */
} gsx_backend_info;

typedef struct gsx_backend_capabilities {
    gsx_data_type_flags supported_data_types; /**< Bitmask of tensor element types accepted by backend compute kernels. */
    bool supports_async_prefetch;        /**< True if the implementation may use private helper threads or streams for dataloader prefetch. */
} gsx_backend_capabilities;

typedef struct gsx_backend_buffer_type_info {
    gsx_backend_t backend;                     /**< Borrowed backend that owns this immutable buffer-type handle. */
    gsx_backend_buffer_type_class type;        /**< Portable class used for cross-backend selection. */
    const char *name;                          /**< Borrowed backend-defined name valid until the owning backend is freed. Do not free it. */
    gsx_size_t alignment_bytes;                /**< Minimum allocation alignment guaranteed for buffers of this type. */
    gsx_size_t max_allocation_size_bytes;      /**< Maximum single allocation size accepted by this type when known; zero means unknown. */
} gsx_backend_buffer_type_info;

/** Count visible devices for all compiled backends. Returns `GSX_ERROR_INVALID_ARGUMENT` if `out_count` is NULL. */
GSX_API gsx_error gsx_count_devices(gsx_index_t *out_count);
/** Query a device handle by zero-based index. The returned handle is borrowed and must not be freed. Returns `GSX_ERROR_OUT_OF_RANGE` for an invalid index. */
GSX_API gsx_error gsx_get_device(gsx_index_t index, gsx_device_t *out_device);
/** Query stable device information for logging and memory planning. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_device_get_info(gsx_device_t device, gsx_device_info *out_info);

/** Construct a backend instance bound to a specific device. `out_backend` owns the handle on success. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs or NULL devices. */
GSX_API gsx_error gsx_backend_init(gsx_backend_t *out_backend, const gsx_backend_desc *desc);
/** Release a backend instance created by `gsx_backend_init`. Returns `GSX_ERROR_INVALID_ARGUMENT` if `backend` is NULL. */
GSX_API gsx_error gsx_backend_free(gsx_backend_t backend);
/** Query backend identity and bound device information. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
/** Query backend-wide compute and runtime capabilities. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
/** Query the backend-owned major stream or command-queue handle. The returned pointer is borrowed, backend-specific, valid while the backend lives, and must not be freed or replaced by the caller. CPU backends return success with `*out_stream = NULL`. */
GSX_API gsx_error gsx_backend_get_major_stream(gsx_backend_t backend, void **out_stream);

/** Count public buffer types exported by a backend. Returns `GSX_ERROR_INVALID_ARGUMENT` if `backend` or `out_count` is NULL. */
GSX_API gsx_error gsx_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
/** Query a backend-owned immutable buffer-type handle by zero-based index. The returned handle is borrowed and remains valid until `backend` is freed. Returns `GSX_ERROR_OUT_OF_RANGE` for an invalid index. */
GSX_API gsx_error gsx_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
/** Resolve the backend-preferred public buffer type for a portable class. Returns `GSX_ERROR_NOT_SUPPORTED` if the backend does not expose that class. */
GSX_API gsx_error gsx_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);
/** Query stable metadata for a buffer type. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
/** Query the portable class for a buffer type. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_type(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_class *out_type);
/** Query the backend-owned stable name for a buffer type. The returned string is borrowed and valid until the owning backend is freed. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_name(gsx_backend_buffer_type_t buffer_type, const char **out_name);
/** Query the backend that owns a buffer type. The returned backend handle is borrowed and must not be freed. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_backend(gsx_backend_buffer_type_t buffer_type, gsx_backend_t *out_backend);
/** Query the minimum allocation alignment for a buffer type. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_alignment(gsx_backend_buffer_type_t buffer_type, gsx_size_t *out_alignment_bytes);
/** Query the maximum single allocation size for a buffer type when known. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_max_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t *out_max_size_bytes);
/** Round `requested_size_bytes` to the backend allocation size for this type. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs, `GSX_ERROR_OUT_OF_RANGE` on overflow, and `GSX_ERROR_NOT_SUPPORTED` if the type cannot serve the request. */
GSX_API gsx_error gsx_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);

GSX_EXTERN_C_END

#endif /* GSX_BACKEND_H */
