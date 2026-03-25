#ifndef GSX_CPU_INTERNAL_H
#define GSX_CPU_INTERNAL_H

#include "../gsx-impl.h"

GSX_EXTERN_C_BEGIN

typedef struct gsx_cpu_backend_provider {
    struct gsx_backend_provider base;
} gsx_cpu_backend_provider;

typedef struct gsx_cpu_backend_device {
    struct gsx_backend_device base;
} gsx_cpu_backend_device;

typedef struct gsx_cpu_backend_buffer_type {
    struct gsx_backend_buffer_type base;
    gsx_backend_buffer_type_info info;
} gsx_cpu_backend_buffer_type;

typedef struct gsx_cpu_backend {
    struct gsx_backend base;
    gsx_backend_capabilities capabilities;
    gsx_cpu_backend_buffer_type host_buffer_type;
    gsx_cpu_backend_buffer_type device_buffer_type;
} gsx_cpu_backend;

typedef struct gsx_cpu_backend_buffer {
    struct gsx_backend_buffer base;
    void *data;
    gsx_size_t alloc_size_bytes;
} gsx_cpu_backend_buffer;

gsx_error gsx_cpu_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss);
gsx_error gsx_cpu_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim);
gsx_error gsx_cpu_optim_zero_appended_rows(gsx_optim_t optim, gsx_size_t old_count, gsx_size_t new_count);
gsx_error gsx_cpu_optim_zero_rows(gsx_optim_t optim, const gsx_size_t *indices, gsx_size_t index_count, gsx_size_t row_count);
gsx_error gsx_cpu_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc);
gsx_error gsx_cpu_backend_create_async_dl(gsx_backend_t backend, const gsx_async_dl_desc *desc, gsx_async_dl_t *out_async_dl);
gsx_error gsx_cpu_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer);
gsx_error gsx_cpu_backend_provider_bootstrap(gsx_builtin_registry_state *registry);

#include "tensor-helpers.h"

GSX_EXTERN_C_END

#endif /* GSX_CPU_INTERNAL_H */
