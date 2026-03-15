#ifndef GSX_METAL_INTERNAL_H
#define GSX_METAL_INTERNAL_H

#include "../gsx-impl.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct gsx_metal_backend_provider {
    struct gsx_backend_provider base;
} gsx_metal_backend_provider;

typedef struct gsx_metal_backend_device {
    struct gsx_backend_device base;
    void *mtl_device;
    char device_name[256];
} gsx_metal_backend_device;

typedef struct gsx_metal_backend_buffer_type {
    struct gsx_backend_buffer_type base;
    gsx_backend_buffer_type_info info;
} gsx_metal_backend_buffer_type;

typedef struct gsx_metal_backend {
    struct gsx_backend base;
    gsx_backend_capabilities capabilities;
    void *mtl_device;
    void *major_command_queue;
    void *optim_adam_pipeline;    /* cached MTLComputePipelineState, NULL until first use */
    void *optim_row_gather_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    gsx_metal_backend_buffer_type device_buffer_type;
    gsx_metal_backend_buffer_type host_pinned_buffer_type;
    gsx_metal_backend_buffer_type unified_buffer_type;
} gsx_metal_backend;

typedef struct gsx_metal_backend_buffer {
    struct gsx_backend_buffer base;
    void *mtl_buffer;
    gsx_size_t alloc_size_bytes;
    gsx_backend_buffer_type_class type_class;
    uint32_t resource_options;
} gsx_metal_backend_buffer;

typedef struct gsx_metal_adam_step_params {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_grad;
    float inv_beta1_correction;
    float inv_beta2_correction;
    uint32_t element_count;
} gsx_metal_adam_step_params;

typedef struct gsx_metal_row_gather_params {
    uint32_t row_floats; /* number of float32 elements per row */
    uint32_t row_count;  /* number of destination rows */
} gsx_metal_row_gather_params;

extern gsx_metal_backend_provider gsx_metal_backend_provider_singleton;
extern gsx_metal_backend_device *gsx_metal_backend_devices;
extern int gsx_metal_device_count;
extern int gsx_metal_device_capacity;

extern const gsx_backend_provider_i gsx_metal_backend_provider_iface;
extern const gsx_backend_i gsx_metal_backend_iface;
extern const gsx_backend_buffer_type_i gsx_metal_backend_buffer_type_iface;
extern const gsx_backend_buffer_i gsx_metal_backend_buffer_iface;

gsx_error gsx_metal_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
gsx_error gsx_metal_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);
gsx_error gsx_metal_backend_free(gsx_backend_t backend);
gsx_error gsx_metal_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
gsx_error gsx_metal_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
gsx_error gsx_metal_backend_get_major_stream(gsx_backend_t backend, void **out_stream);
gsx_error gsx_metal_backend_major_stream_sync(gsx_backend_t backend);
gsx_error gsx_metal_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
gsx_error gsx_metal_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
gsx_error gsx_metal_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);
gsx_error gsx_metal_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer);
gsx_error gsx_metal_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss);
gsx_error gsx_metal_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim);
gsx_error gsx_metal_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc);

gsx_error gsx_metal_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
gsx_error gsx_metal_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
gsx_error gsx_metal_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);

gsx_error gsx_metal_backend_buffer_free(gsx_backend_buffer_t buffer);
gsx_error gsx_metal_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
gsx_error gsx_metal_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
gsx_error gsx_metal_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
gsx_error gsx_metal_backend_buffer_set_zero(gsx_backend_buffer_t buffer);
gsx_error gsx_metal_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
gsx_error gsx_metal_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
gsx_error gsx_metal_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
gsx_error gsx_metal_backend_buffer_copy_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view
);
gsx_error gsx_metal_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
);
gsx_error gsx_metal_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
);

gsx_metal_backend *gsx_metal_backend_from_base(gsx_backend_t backend);
gsx_metal_backend_buffer_type *gsx_metal_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type);
gsx_metal_backend_buffer *gsx_metal_backend_buffer_from_base(gsx_backend_buffer_t buffer);
gsx_backend_buffer_type_class gsx_metal_backend_buffer_get_type_class(gsx_backend_buffer_t buffer);
void gsx_metal_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes);
bool gsx_metal_backend_f16_is_finite(uint16_t value);
bool gsx_metal_backend_bf16_is_finite(uint16_t value);
gsx_error gsx_metal_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count);
gsx_error gsx_metal_backend_dispatch_adam_step(
    gsx_backend_t backend,
    gsx_tensor_t parameter,
    gsx_tensor_t gradient,
    gsx_tensor_t first_moment,
    gsx_tensor_t second_moment,
    const gsx_metal_adam_step_params *params
);
/* Gather row_count rows from src into dst using indices[dst_row] as source row per entry.
 * Dispatched on the major command queue; safe to free indices_buffer immediately after return. */
gsx_error gsx_metal_backend_dispatch_row_gather(
    gsx_backend_t backend,
    gsx_tensor_t dst,
    gsx_tensor_t src,
    gsx_backend_buffer_t indices_buffer,
    gsx_size_t indices_offset_bytes,
    uint32_t row_floats,
    uint32_t row_count
);
/* Copy copy_bytes from src to dst then zero-fill the remaining suffix via blit encoder. */
gsx_error gsx_metal_backend_dispatch_grow_blit(
    gsx_backend_t backend,
    gsx_tensor_t dst,
    gsx_tensor_t src,
    gsx_size_t copy_bytes,
    gsx_size_t total_dst_bytes
);
void gsx_metal_backend_init_buffer_type(
    gsx_metal_backend *metal_backend,
    gsx_metal_backend_buffer_type *buffer_type,
    gsx_backend_buffer_type_class type,
    const char *name,
    gsx_size_t alignment_bytes
);
gsx_error gsx_metal_backend_tensor_view_validate(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view);
gsx_error gsx_metal_backend_tensor_view_check_range(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_size_t offset_bytes,
    gsx_size_t byte_count
);

#endif /* GSX_METAL_INTERNAL_H */
