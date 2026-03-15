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

typedef struct gsx_backend_device_info {
    gsx_backend_type backend_type; /**< Backend family that exposes this device. */
    const char *backend_name;      /**< Stable short backend-family name such as `"cpu"` or `"cuda"`. */
    gsx_index_t device_index;      /**< Zero-based device ordinal within the visible devices of `backend_type`. */
    const char *name;              /**< Stable backend-defined device name such as `"cpu0"` or a GPU model name. */
    gsx_size_t total_memory_bytes; /**< Best-effort total addressable memory for planning. Comparisons across backend families are heuristic only. */
} gsx_backend_device_info;

typedef struct gsx_backend_desc {
    gsx_backend_device_t device;   /**< Target backend device. This is a backend-specific execution target, not necessarily a physical device. */
    const void *options;           /**< Optional backend-specific initialization blob; may be NULL. */
    gsx_size_t options_size_bytes; /**< Byte size of the backend-specific options blob. */
} gsx_backend_desc;

typedef struct gsx_backend_info {
    gsx_backend_type backend_type; /**< Backend family for this backend instance. */
    gsx_backend_device_t device;   /**< Borrowed backend device handle associated with this backend. */
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

typedef struct gsx_backend_buffer_desc {
    gsx_backend_buffer_type_t buffer_type; /**< Buffer type that owns placement rules for the allocation. */
    gsx_size_t size_bytes;                 /**< Logical accessible size in bytes. Must be non-zero. */
    gsx_size_t alignment_bytes;            /**< Requested minimum allocation alignment. Zero means use the buffer-type default. Non-zero values must be powers of two. */
} gsx_backend_buffer_desc;

typedef struct gsx_backend_buffer_info {
    gsx_backend_t backend;                 /**< Borrowed backend that owns this buffer. */
    gsx_backend_buffer_type_t buffer_type; /**< Borrowed buffer type used to allocate this buffer. */
    gsx_size_t size_bytes;                 /**< Logical accessible size in bytes. */
    gsx_size_t alignment_bytes;            /**< Effective allocation alignment in bytes. */
} gsx_backend_buffer_info;

/** Initialize the builtin backend registry singleton. This function is explicit and idempotent. */
GSX_API gsx_error gsx_backend_registry_init(void);
/** Count backend devices visible in the builtin registry. The registry presents one flat device inventory across all backend families. */
GSX_API gsx_error gsx_count_backend_devices(gsx_index_t *out_count);
/** Query a backend-device handle by zero-based index within the global backend-device inventory. The returned handle is borrowed and remains valid until process exit. */
GSX_API gsx_error gsx_get_backend_device(gsx_index_t index, gsx_backend_device_t *out_backend_device);
/** Count backend devices exposed by a specific backend family. Returns zero when no devices of that family are currently visible. */
GSX_API gsx_error gsx_count_backend_devices_by_type(gsx_backend_type backend_type, gsx_index_t *out_count);
/** Query a backend-device handle by zero-based index within the filtered device inventory for `backend_type`. The returned handle is borrowed and remains valid until process exit. */
GSX_API gsx_error gsx_get_backend_device_by_type(gsx_backend_type backend_type, gsx_index_t index, gsx_backend_device_t *out_backend_device);
/** Query stable backend-device information for logging and memory planning. The same physical accelerator may appear multiple times when exposed by multiple backend families. */
GSX_API gsx_error gsx_backend_device_get_info(gsx_backend_device_t backend_device, gsx_backend_device_info *out_info);

/** Construct a backend instance bound to a specific backend device. `out_backend` owns the handle on success. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs or NULL devices and `GSX_ERROR_INVALID_STATE` before `gsx_backend_registry_init`. */
GSX_API gsx_error gsx_backend_init(gsx_backend_t *out_backend, const gsx_backend_desc *desc);
/** Release a backend instance created by `gsx_backend_init`. Returns `GSX_ERROR_INVALID_ARGUMENT` if `backend` is NULL and `GSX_ERROR_INVALID_STATE` while backend-owned buffers are still live. */
GSX_API gsx_error gsx_backend_free(gsx_backend_t backend);
/** Query backend identity and bound device information. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
/** Query backend-wide compute and runtime capabilities. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
/** Query the backend-owned major stream or command-queue handle. The returned pointer is borrowed, backend-specific, valid while the backend lives, and must not be freed or replaced by the caller. CPU backends return success with `*out_stream = NULL`. */
GSX_API gsx_error gsx_backend_get_major_stream(gsx_backend_t backend, void **out_stream);
/** Synchronize work submitted to the backend-owned major stream or command queue. CPU backends return success immediately because there is no asynchronous device queue. */
GSX_API gsx_error gsx_backend_major_stream_sync(gsx_backend_t backend);
/** Count public buffer types exported by a backend. Returns `GSX_ERROR_INVALID_ARGUMENT` if `backend` or `out_count` is NULL. */
GSX_API gsx_error gsx_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
/** Query a backend-owned immutable buffer-type handle by zero-based index. The returned handle is borrowed and remains valid until `backend` is freed. Returns `GSX_ERROR_OUT_OF_RANGE` for an invalid index. */
GSX_API gsx_error gsx_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
/** Resolve the backend-preferred public buffer type for a portable class. Returns `GSX_ERROR_NOT_SUPPORTED` if the backend does not expose that class. */
GSX_API gsx_error gsx_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);

/** Query stable metadata for a buffer type, including class, backend-defined name, alignment, and maximum allocation size. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
/** Query the backend that owns a buffer type. The returned backend handle is borrowed and must not be freed. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_type_get_backend(gsx_backend_buffer_type_t buffer_type, gsx_backend_t *out_backend);
/** Round `requested_size_bytes` to the backend allocation size for this type. Returns `GSX_ERROR_INVALID_ARGUMENT` for NULL outputs, `GSX_ERROR_OUT_OF_RANGE` on overflow, and `GSX_ERROR_NOT_SUPPORTED` if the type cannot serve the request. */
GSX_API gsx_error gsx_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);

/** Create a raw backend-owned buffer according to `desc`. `out_buffer` owns the handle on success. `desc->size_bytes` must be non-zero, and non-zero explicit alignments must be powers of two. */
GSX_API gsx_error gsx_backend_buffer_init(gsx_backend_buffer_t *out_buffer, const gsx_backend_buffer_desc *desc);
/** Release a backend buffer created by `gsx_backend_buffer_init`. Returns `GSX_ERROR_INVALID_ARGUMENT` if `buffer` is NULL. */
GSX_API gsx_error gsx_backend_buffer_free(gsx_backend_buffer_t buffer);
/** Query stable buffer placement and size metadata. Returns `GSX_ERROR_INVALID_ARGUMENT` for a NULL handle or NULL output. */
GSX_API gsx_error gsx_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
/** Query the backend-native handle for a buffer. The returned pointer is borrowed, backend-specific, valid while `buffer` remains alive, and must not be freed or replaced by the caller. Backends may return success with `*out_handle = NULL` when no native handle is available. */
GSX_API gsx_error gsx_backend_buffer_get_native_handle(gsx_backend_buffer_t buffer, void **out_handle);

//! upload/download/set_zero operations are async to CPU, but running in the major stream (command-queue) for GPU backends. Synchronization is the responsibility of the caller.
/** Upload bytes into a backend buffer at `dst_offset_bytes`. Returns `GSX_ERROR_OUT_OF_RANGE` if the write would exceed `size_bytes`. */
GSX_API gsx_error gsx_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
/** Download bytes from a backend buffer at `src_offset_bytes`. Returns `GSX_ERROR_OUT_OF_RANGE` if the read would exceed `size_bytes`. */
GSX_API gsx_error gsx_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
/** Fill the full logical buffer range with zero bytes. Returns `GSX_ERROR_INVALID_ARGUMENT` if `buffer` is NULL. */
GSX_API gsx_error gsx_backend_buffer_set_zero(gsx_backend_buffer_t buffer);


GSX_EXTERN_C_END

#endif /* GSX_BACKEND_H */
