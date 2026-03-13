#ifndef GSX_CUDA_INTERNAL_H
#define GSX_CUDA_INTERNAL_H

#include "../gsx-impl.h"

#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct gsx_cuda_backend_provider {
    struct gsx_backend_provider base;
} gsx_cuda_backend_provider;

typedef struct gsx_cuda_backend_device {
    struct gsx_backend_device base;
    int cuda_device_ordinal;
    int compute_capability_major;
    int compute_capability_minor;
    char device_name[256];
} gsx_cuda_backend_device;

typedef struct gsx_cuda_backend_buffer_type {
    struct gsx_backend_buffer_type base;
    gsx_backend_buffer_type_info info;
} gsx_cuda_backend_buffer_type;

typedef struct gsx_cuda_backend {
    struct gsx_backend base;
    gsx_backend_capabilities capabilities;
    cudaStream_t major_stream;
    gsx_cuda_backend_buffer_type device_buffer_type;
    gsx_cuda_backend_buffer_type host_pinned_buffer_type;
} gsx_cuda_backend;

typedef struct gsx_cuda_backend_buffer {
    struct gsx_backend_buffer base;
    void *ptr;
    gsx_size_t alloc_size_bytes;
} gsx_cuda_backend_buffer;

extern gsx_cuda_backend_provider gsx_cuda_backend_provider_singleton;
extern gsx_cuda_backend_device *gsx_cuda_backend_devices;
extern int gsx_cuda_device_count;
extern int gsx_cuda_device_capacity;

extern const gsx_backend_provider_i gsx_cuda_backend_provider_iface;
extern const gsx_backend_i gsx_cuda_backend_iface;
extern const gsx_backend_buffer_type_i gsx_cuda_backend_buffer_type_iface;
extern const gsx_backend_buffer_i gsx_cuda_backend_buffer_iface;

gsx_error gsx_cuda_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry);
gsx_error gsx_cuda_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend);
gsx_error gsx_cuda_backend_free(gsx_backend_t backend);
gsx_error gsx_cuda_backend_get_info(gsx_backend_t backend, gsx_backend_info *out_info);
gsx_error gsx_cuda_backend_get_capabilities(gsx_backend_t backend, gsx_backend_capabilities *out_capabilities);
gsx_error gsx_cuda_backend_get_major_stream(gsx_backend_t backend, void **out_stream);
gsx_error gsx_cuda_backend_count_buffer_types(gsx_backend_t backend, gsx_index_t *out_count);
gsx_error gsx_cuda_backend_get_buffer_type(gsx_backend_t backend, gsx_index_t index, gsx_backend_buffer_type_t *out_buffer_type);
gsx_error gsx_cuda_backend_find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type, gsx_backend_buffer_type_t *out_buffer_type);

gsx_error gsx_cuda_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
gsx_error gsx_cuda_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
gsx_error gsx_cuda_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);

gsx_error gsx_cuda_backend_buffer_free(gsx_backend_buffer_t buffer);
gsx_error gsx_cuda_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
gsx_error gsx_cuda_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count);
gsx_error gsx_cuda_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count);
gsx_error gsx_cuda_backend_buffer_set_zero(gsx_backend_buffer_t buffer);
gsx_error gsx_cuda_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
gsx_error gsx_cuda_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
gsx_error gsx_cuda_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
);
gsx_error gsx_cuda_backend_buffer_copy_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view
);
gsx_error gsx_cuda_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
);
gsx_error gsx_cuda_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
);

gsx_error gsx_cuda_make_error(cudaError_t cuda_err, const char *context);
gsx_cuda_backend *gsx_cuda_backend_from_base(gsx_backend_t backend);
gsx_cuda_backend_buffer_type *gsx_cuda_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type);
gsx_cuda_backend_buffer *gsx_cuda_backend_buffer_from_base(gsx_backend_buffer_t buffer);
gsx_backend_buffer_type_class gsx_cuda_backend_buffer_get_type_class(gsx_backend_buffer_t buffer);
void gsx_cuda_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes);
bool gsx_cuda_backend_f16_is_finite(uint16_t value);
bool gsx_cuda_backend_bf16_is_finite(uint16_t value);
gsx_error gsx_cuda_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count);
void gsx_cuda_backend_init_buffer_type(
    gsx_cuda_backend *cuda_backend,
    gsx_cuda_backend_buffer_type *buffer_type,
    gsx_backend_buffer_type_class type,
    const char *name,
    gsx_size_t alignment_bytes
);

void gsx_cuda_fill_tensor_kernel_launch(
    void *dst,
    const void *value,
    gsx_size_t value_size,
    gsx_size_t total_bytes,
    gsx_size_t alignment_bytes,
    cudaStream_t stream
);
void gsx_cuda_check_finite_tensor_f32_kernel_launch(
    const void *src,
    gsx_size_t total_elements,
    gsx_size_t alignment_bytes,
    int *out_has_non_finite,
    cudaStream_t stream
);
void gsx_cuda_check_finite_tensor_f16_kernel_launch(
    const void *src,
    gsx_size_t total_elements,
    gsx_size_t alignment_bytes,
    int *out_has_non_finite,
    cudaStream_t stream
);
void gsx_cuda_check_finite_tensor_bf16_kernel_launch(
    const void *src,
    gsx_size_t total_elements,
    gsx_size_t alignment_bytes,
    int *out_has_non_finite,
    cudaStream_t stream
);

gsx_error gsx_cuda_backend_provider_bootstrap(gsx_builtin_registry_state *registry);

#endif /* GSX_CUDA_INTERNAL_H */
