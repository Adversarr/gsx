#ifndef GSX_IMPL_H
#define GSX_IMPL_H

#include "gsx/gsx.h"

GSX_EXTERN_C_BEGIN
#include <stdbool.h>
#include <stdint.h>

/*
 * Internal implementation notes for the stable GSX public API surface:
 *
 * - Only headers under gsx/include/gsx are part of the stable ABI.
 * - Public descriptor/state/result structs are plain POD values in v0.
 * - Public backend-bound work is externally observed as one totally ordered
 *   major stream per backend.
 * - Private helper threads or streams are allowed for dataloader prefetch only.
 * - Builtin registry owns only pointer arrays for providers and devices.
 *   It never owns provider/device object storage and never destroys those
 *   objects during reset.
 * - Registered provider/device objects must remain valid for the full borrowed
 *   handle lifetime exposed by public APIs.
 * - Device identity fields must stay consistent with the owning provider:
 *   backend_type and backend_name must match provider metadata.
 * - Layering is one-way:
 *   core <- backend/render/data/loss/optim/adc <- runtime
 * - Runtime objects own replay-critical state only. They do not own the caller
 *   training loop.
 * - GS/optimizer structural mutation and ADC steps must remain transactional.
 */

typedef struct gsx_backend_provider gsx_backend_provider;
typedef gsx_backend_provider *gsx_backend_provider_t;
typedef struct gsx_backend_provider_i gsx_backend_provider_i;
typedef struct gsx_builtin_registry_state gsx_builtin_registry_state;
typedef struct gsx_backend_i gsx_backend_i;
typedef struct gsx_backend_buffer_type_i gsx_backend_buffer_type_i;
typedef struct gsx_backend_buffer_i gsx_backend_buffer_i;
typedef struct gsx_backend_tensor_view gsx_backend_tensor_view;

struct gsx_backend_provider {
    const gsx_backend_provider_i *iface;
    gsx_backend_type backend_type;
    const char *backend_name;
};

struct gsx_backend_device {
    gsx_backend_provider_t provider;
    gsx_backend_device_info info;
};

struct gsx_backend {
    const gsx_backend_i *iface;
    gsx_backend_provider_t provider;
    gsx_backend_device_t device;
    gsx_size_t live_buffer_count;
    gsx_size_t live_arena_count;
};

struct gsx_backend_buffer_type {
    const gsx_backend_buffer_type_i *iface;
    gsx_backend_t backend;
    gsx_size_t live_arena_count;
};

struct gsx_backend_buffer {
    const gsx_backend_buffer_i *iface;
    gsx_backend_buffer_type_t buffer_type;
    gsx_size_t size_bytes;
    gsx_size_t alignment_bytes;
};

struct gsx_backend_tensor_view {
    gsx_backend_buffer_t buffer;
    gsx_size_t offset_bytes;
    gsx_size_t size_bytes;
    gsx_data_type data_type;
};

struct gsx_backend_provider_i {
    gsx_error (*discover_devices)(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
    gsx_error (*create_backend)(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);
};

struct gsx_backend_i {
    gsx_error (*free)(gsx_backend_t backend);
    gsx_error (*get_info)(gsx_backend_t backend, gsx_backend_info *out_info);
    gsx_error (*get_capabilities)(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
    gsx_error (*get_major_stream)(gsx_backend_t backend, void **out_stream);
    gsx_error (*count_buffer_types)(gsx_backend_t backend, gsx_index_t *out_count);
    gsx_error (*get_buffer_type)(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
    gsx_error (*find_buffer_type)(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);
};

struct gsx_backend_buffer_type_i {
    gsx_error (*get_info)(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
    gsx_error (*get_alloc_size)(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
    gsx_error (*init_buffer)(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);
};

struct gsx_backend_buffer_i {
    gsx_error (*free)(gsx_backend_buffer_t buffer);
    gsx_error (*get_info)(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
    gsx_error (*upload)(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
    gsx_error (*download)(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
    gsx_error (*set_zero)(gsx_backend_buffer_t buffer);
    gsx_error (*memset_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        uint8_t value,
        gsx_size_t offset_bytes,
        gsx_size_t size_bytes
    );
    gsx_error (*set_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        const void *src_bytes,
        gsx_size_t offset_bytes,
        gsx_size_t size_bytes
    );
    gsx_error (*get_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        void *dst_bytes,
        gsx_size_t offset_bytes,
        gsx_size_t size_bytes
    );
    gsx_error (*copy_tensor)(
        gsx_backend_buffer_t dst_buffer,
        const gsx_backend_tensor_view *src_view,
        const gsx_backend_tensor_view *dst_view
    );
    gsx_error (*fill_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        const void *value_bytes,
        gsx_size_t value_size_bytes
    );
    gsx_error (*check_finite_tensor)(
        gsx_backend_buffer_t buffer,
        const gsx_backend_tensor_view *tensor_view,
        gsx_finite_check_result *out_result
    );
};

struct gsx_builtin_registry_state {
    bool is_initialized;
    gsx_index_t backend_provider_count;
    gsx_index_t backend_provider_capacity;
    gsx_backend_provider_t *backend_providers;
    gsx_index_t backend_device_count;
    gsx_index_t backend_device_capacity;
    gsx_backend_device_t *backend_devices;
};

static inline gsx_error gsx_make_error(gsx_error_code code, const char *message)
{
    gsx_error error = { code, message };
    return error;
}

static inline bool gsx_is_power_of_two(gsx_size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static inline bool gsx_size_add_overflows(gsx_size_t lhs, gsx_size_t rhs, gsx_size_t *out_sum)
{
    if(lhs > UINT64_MAX - rhs) {
        return true;
    }
    if(out_sum != NULL) {
        *out_sum = lhs + rhs;
    }
    return false;
}

static inline bool gsx_size_mul_overflows(gsx_size_t lhs, gsx_size_t rhs, gsx_size_t *out_product)
{
    if(lhs != 0 && rhs > UINT64_MAX / lhs) {
        return true;
    }
    if(out_product != NULL) {
        *out_product = lhs * rhs;
    }
    return false;
}

static inline bool gsx_round_up_overflows(gsx_size_t value, gsx_size_t alignment_bytes, gsx_size_t *out_rounded_value)
{
    gsx_size_t adjustment = 0;

    if(alignment_bytes == 0) {
        if(out_rounded_value != NULL) {
            *out_rounded_value = value;
        }
        return false;
    }

    adjustment = alignment_bytes - 1;
    if(gsx_size_add_overflows(value, adjustment, out_rounded_value)) {
        return true;
    }
    if(out_rounded_value != NULL) {
        *out_rounded_value &= ~(alignment_bytes - 1);
    }
    return false;
}

gsx_builtin_registry_state *gsx_builtin_registry_get(void);
void gsx_builtin_registry_reset(gsx_builtin_registry_state *registry);
gsx_error gsx_builtin_registry_append_provider(gsx_builtin_registry_state *registry, gsx_backend_provider_t backend_provider);
gsx_error gsx_builtin_registry_append_device(gsx_builtin_registry_state *registry, gsx_backend_device_t backend_device);
gsx_error gsx_cpu_backend_provider_bootstrap(gsx_builtin_registry_state *registry);
GSX_EXTERN_C_END

#endif /* GSX_IMPL_H */
