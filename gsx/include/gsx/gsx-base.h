#ifndef GSX_BASE_H
#define GSX_BASE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(GSX_BUILD_SHARED)
#if defined(GSX_BUILDING_LIBRARY)
#define GSX_API __declspec(dllexport)
#else
#define GSX_API __declspec(dllimport)
#endif
#else
#define GSX_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define GSX_API __attribute__((visibility("default")))
#else
#define GSX_API
#endif

#ifdef __cplusplus
#define GSX_DECL_ALIGNAS(bytes) alignas(bytes)
#define GSX_TYPEDEF_ALIGNAS(bytes)
#elif defined(_MSC_VER)
#define GSX_DECL_ALIGNAS(bytes) __declspec(align(bytes))
#define GSX_TYPEDEF_ALIGNAS(bytes)
#elif defined(__GNUC__) || defined(__clang__)
#define GSX_DECL_ALIGNAS(bytes) _Alignas(bytes)
#define GSX_TYPEDEF_ALIGNAS(bytes) __attribute__((aligned(bytes)))
#else
#define GSX_DECL_ALIGNAS(bytes)
#define GSX_TYPEDEF_ALIGNAS(bytes) __attribute__((aligned(bytes)))
#endif

#ifdef __cplusplus
#define GSX_EXTERN_C_BEGIN extern "C" {
#define GSX_EXTERN_C_END }
#else
#define GSX_EXTERN_C_BEGIN
#define GSX_EXTERN_C_END
#endif

#define GSX_VERSION_MAJOR 0
#define GSX_VERSION_MINOR 0
#define GSX_VERSION_PATCH 1

#define GSX_MAKE_VERSION(major, minor, patch) \
    ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))

#define GSX_VERSION GSX_MAKE_VERSION(GSX_VERSION_MAJOR, GSX_VERSION_MINOR, GSX_VERSION_PATCH)

#ifndef GSX_TENSOR_MAX_DIM
#define GSX_TENSOR_MAX_DIM ((gsx_index_t)4)
#endif

GSX_EXTERN_C_BEGIN

typedef int32_t  gsx_index_t;   /**< Common signed index type used by tensor and image dimensions. */
typedef uint64_t gsx_size_t;    /**< Unsigned size and byte-count type for stable ABI fields. */
typedef uint64_t gsx_id_t;      /**< Stable identifier type for samples and replay metadata. */
typedef float    gsx_float_t;   /**< Public floating-point scalar used by metrics and schedules. */
typedef uint32_t gsx_flags32_t; /**< Packed 32-bit bitfield for feature and option flags. */
typedef uint64_t gsx_flags64_t; /**< Packed 64-bit bitfield for capability and dtype masks. */

typedef enum gsx_error_code {
    GSX_ERROR_SUCCESS = 0,              /**< Operation completed successfully. */
    GSX_ERROR_OUT_OF_MEMORY = 1,        /**< Allocation failed or the backend cannot reserve required memory. */
    GSX_ERROR_INVALID_ARGUMENT = 2,     /**< Caller supplied a null, malformed, or incompatible argument. */
    GSX_ERROR_OUT_OF_RANGE = 3,         /**< Caller supplied an index, size, or enum value outside the supported range. */
    GSX_ERROR_INVALID_STATE = 4,        /**< Object state does not permit the requested operation at this time. */
    GSX_ERROR_NOT_SUPPORTED = 5,        /**< Backend or object does not implement the requested feature. */
    GSX_ERROR_INCOMPATIBLE_VERSION = 6, /**< Checkpoint format or ABI version is incompatible. */
    GSX_ERROR_CHECKPOINT_CORRUPT = 7,   /**< Serialized runtime state is malformed or incomplete. */
    GSX_ERROR_IO = 8,                   /**< Reader or writer callback reported a transport failure. */
    GSX_ERROR_UNKNOWN = 255             /**< Implementation hit an unspecified internal failure. */
} gsx_error_code;

typedef struct gsx_error {
    gsx_error_code code;    /**< Stable machine-readable status code. */
    const char *message;    /**< Optional diagnostic string owned by the implementation; do not free it. */
} gsx_error;

static inline bool gsx_error_is_success(gsx_error error)
{
    return error.code == GSX_ERROR_SUCCESS;
}

/*
 * Opaque operational handles.
 *
 * Execution model for the stable ABI:
 * - public backend-bound operations execute in-order on one backend-owned major
 *   stream or command queue;
 * - unless a type is documented as an immutable value, public GSX handles are
 *   not safe for concurrent calls from multiple threads;
 * - callers must dispatch backend-bound public calls from one main thread, or
 *   externally serialize them so they observe the same total order;
 * - GSX does not support public overlap of render, optimizer, ADC, or tensor
 *   transfer operations on the same backend;
 * - implementations may use private helper threads or streams internally for
 *   dataloader prefetch only;
 * - callers must not race object destruction with any use of borrowed handles
 *   or output pointers derived from that object;
 * - tensor handles returned through the public API are ready for use on the
 *   backend major stream when the call returns;
 * - immutable value structs declared below are safe to copy by value.
 *
 * Best practice:
 * - pick one backend and one caller-visible thread;
 * - size storage with a dry-run arena instead of guessing alignment-padded
 *   byte counts by hand;
 * - create tensors from the arena, then build higher-level modules on top;
 * - free objects in reverse order after the backend queue is idle.
 *
 * Example:
 *   gsx_backend_init(&backend, &backend_desc);
 *   gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_type);
 *   gsx_arena_desc arena_desc = { 0 };
 *   arena_desc.dry_run = true;
 *   gsx_arena_init(&dry_arena, device_type, &arena_desc);
 *   gsx_tensor_init(&probe, &tensor_desc);
 *   gsx_arena_get_required_bytes(dry_arena, &required_bytes);
 *   gsx_arena_free(dry_arena);
 *   arena_desc.dry_run = false;
 *   arena_desc.initial_capacity_bytes = required_bytes;
 *   gsx_arena_init(&arena, device_type, &arena_desc);
 *   gsx_tensor_init(&tensor, &tensor_desc);
 */
typedef struct gsx_backend_device    *gsx_backend_device_t;
typedef struct gsx_backend           *gsx_backend_t;
typedef struct gsx_backend_buffer_type *gsx_backend_buffer_type_t;
typedef struct gsx_backend_buffer    *gsx_backend_buffer_t;
typedef struct gsx_arena             *gsx_arena_t;
typedef struct gsx_tensor            *gsx_tensor_t;
typedef struct gsx_gs                *gsx_gs_t;
typedef struct gsx_renderer          *gsx_renderer_t;
typedef struct gsx_render_context    *gsx_render_context_t;
typedef struct gsx_dataset           *gsx_dataset_t;
typedef struct gsx_dataloader        *gsx_dataloader_t;
typedef struct gsx_loss              *gsx_loss_t;
typedef struct gsx_loss_context      *gsx_loss_context_t;
typedef struct gsx_metric            *gsx_metric_t;
typedef struct gsx_optim             *gsx_optim_t;
typedef struct gsx_adc               *gsx_adc_t;
typedef struct gsx_session           *gsx_session_t;
typedef struct gsx_scheduler         *gsx_scheduler_t;

typedef struct gsx_vec3 {
    gsx_float_t x;
    gsx_float_t y;
    gsx_float_t z;
} gsx_vec3;

#ifdef __cplusplus
typedef struct GSX_DECL_ALIGNAS(16) gsx_vec4 {
    gsx_float_t x;
    gsx_float_t y;
    gsx_float_t z;
    gsx_float_t w;
} gsx_vec4 GSX_TYPEDEF_ALIGNAS(16);
#else
typedef struct gsx_vec4 {
    gsx_float_t x;
    gsx_float_t y;
    gsx_float_t z;
    gsx_float_t w;
} gsx_vec4 GSX_TYPEDEF_ALIGNAS(16);
#endif

typedef gsx_vec4 gsx_quat;

typedef struct gsx_mat3 {
    gsx_float_t m[9];
} gsx_mat3;

typedef struct gsx_mat4 {
    gsx_float_t m[16];
} gsx_mat4;

typedef enum gsx_camera_model {
    GSX_CAMERA_MODEL_PINHOLE = 0 /**< Basic pinhole projection using fx, fy, cx, cy. */
} gsx_camera_model;

/*
 * Camera intrinsics are plain immutable-by-convention value types.
 * The associated image geometry is defined by the API that produces them.
 */
typedef struct gsx_camera_intrinsics {
    gsx_camera_model model; /**< Projection model; callers must set a supported enum value. */
    gsx_float_t fx;         /**< Horizontal focal length in pixels for the associated image geometry. */
    gsx_float_t fy;         /**< Vertical focal length in pixels for the associated image geometry. */
    gsx_float_t cx;         /**< Principal point x-coordinate in the associated image pixel space. */
    gsx_float_t cy;         /**< Principal point y-coordinate in the associated image pixel space. */
    gsx_float_t k1;         /**< Radial distortion coefficient preserved across dataloader resize. */
    gsx_float_t k2;         /**< Radial distortion coefficient preserved across dataloader resize. */
    gsx_float_t k3;         /**< Radial distortion coefficient preserved across dataloader resize. */
    gsx_float_t p1;         /**< Tangential distortion coefficient preserved across dataloader resize. */
    gsx_float_t p2;         /**< Tangential distortion coefficient preserved across dataloader resize. */
    gsx_index_t camera_id;  /**< Dataset-defined camera identifier for stable joins and logging. */
    gsx_index_t width;      /**< Image width associated with these intrinsics. */
    gsx_index_t height;     /**< Image height associated with these intrinsics. */
} gsx_camera_intrinsics;

/*
 * Camera pose uses world-to-camera extrinsics.
 * Quaternion ordering is fixed as xyzw.
 */
typedef struct gsx_camera_pose {
    gsx_quat rot;           /**< World-to-camera rotation quaternion in xyzw order. Callers should provide a normalized value. */
    gsx_vec3 transl;        /**< World-to-camera translation in scene units. It uses the same convention as Gaussian means and renderer camera space. */
    gsx_index_t camera_id;  /**< Camera identifier associated with these extrinsics. */
    gsx_index_t frame_id;   /**< Dataset-defined frame identifier for deterministic replay. */
} gsx_camera_pose;

GSX_EXTERN_C_END

#endif /* GSX_BASE_H */
