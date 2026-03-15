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

typedef char *(*gsx_cuda_resize_buffer_fn)(void *user_data, gsx_size_t size_bytes);

GSX_EXTERN_C_BEGIN

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
gsx_error gsx_cuda_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer);
gsx_error gsx_cuda_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss);
gsx_error gsx_cuda_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim);
gsx_error gsx_cuda_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc);

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
gsx_error gsx_cuda_backend_buffer_gather_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape
);
gsx_error gsx_cuda_backend_buffer_exp_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape
);
gsx_error gsx_cuda_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
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
void gsx_cuda_adam_step_f32_kernel_launch(
    float *parameter,
    const float *gradient,
    float *first_moment,
    float *second_moment,
    gsx_size_t total_elements,
    float beta1,
    float beta2,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float max_grad,
    double inv_beta1_correction,
    double inv_beta2_correction,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_mse_f32_forward_kernel_launch(
    float *loss_map,
    const float *prediction,
    const float *target,
    gsx_size_t total_elements,
    float scale,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_mse_f32_backward_kernel_launch(
    float *grad_prediction,
    const float *prediction,
    const float *target,
    gsx_size_t total_elements,
    float grad_scale,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_l1_f32_forward_kernel_launch(
    float *loss_map,
    const float *prediction,
    const float *target,
    gsx_size_t total_elements,
    float scale,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_l1_f32_backward_kernel_launch(
    float *grad_prediction,
    const float *prediction,
    const float *target,
    gsx_size_t total_elements,
    float grad_scale,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_ssim_chw_f32_forward_kernel_launch(
    float *loss_map,
    const float *prediction,
    const float *target,
    gsx_size_t outer_count,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    float scale,
    float *ssim_buffer_a,
    float *ssim_buffer_b,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_ssim_chw_f32_backward_kernel_launch(
    float *grad_prediction,
    const float *prediction,
    const float *target,
    gsx_size_t outer_count,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    float grad_scale,
    float *ssim_buffer_a,
    float *ssim_buffer_b,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_ssim_hwc_f32_forward_kernel_launch(
    float *loss_map,
    const float *prediction,
    const float *target,
    gsx_size_t outer_count,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    float scale,
    float *ssim_buffer_a,
    float *ssim_buffer_b,
    cudaStream_t stream
);
cudaError_t gsx_cuda_loss_ssim_hwc_f32_backward_kernel_launch(
    float *grad_prediction,
    const float *prediction,
    const float *target,
    gsx_size_t outer_count,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    float grad_scale,
    float *ssim_buffer_a,
    float *ssim_buffer_b,
    cudaStream_t stream
);
cudaError_t gsx_cuda_gather_rows_kernel_launch(
    const void *src,
    void *dst,
    gsx_size_t row_bytes,
    gsx_size_t row_count,
    const int32_t *src_indices,
    gsx_size_t src_row_count,
    int *out_has_out_of_range,
    cudaStream_t stream
);
cudaError_t gsx_cuda_render_tiled_to_chw_f32_kernel_launch(
    const float *src_tiled,
    const float *alpha_tiled,
    float *dst_chw,
    gsx_index_t width,
    gsx_index_t height,
    gsx_vec3 background_color,
    cudaStream_t stream
);
cudaError_t gsx_cuda_render_chw_to_tiled_f32_kernel_launch(
    const float *src_chw,
    float *dst_tiled,
    gsx_index_t width,
    gsx_index_t height,
    cudaStream_t stream
);
cudaError_t gsx_cuda_render_compose_background_tiled_f32_kernel_launch(
    float *image_tiled,
    const float *alpha_tiled,
    gsx_index_t width,
    gsx_index_t height,
    gsx_vec3 background_color,
    cudaStream_t stream
);
cudaError_t gsx_cuda_render_clear_tiled_f32_kernel_launch(
    float *dst_tiled,
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channels,
    cudaStream_t stream
);
cudaError_t gsx_cuda_fastgs_forward_launch(
    gsx_cuda_resize_buffer_fn per_primitive_buffers_func,
    void *per_primitive_user_data,
    gsx_cuda_resize_buffer_fn per_tile_buffers_func,
    void *per_tile_user_data,
    gsx_cuda_resize_buffer_fn per_instance_buffers_func,
    void *per_instance_user_data,
    gsx_cuda_resize_buffer_fn per_bucket_buffers_func,
    void *per_bucket_user_data,
    const float3 *means,
    const float3 *scales_raw,
    const float4 *rotations_raw,
    const float *opacities_raw,
    const float *sh0,
    const float *sh1,
    const float *sh2,
    const float *sh3,
    const float4 *w2c,
    const float3 *cam_position,
    float *image,
    float *alpha,
    int n_primitives,
    int active_sh_bases,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    float near_plane,
    float far_plane,
    cudaStream_t major_stream,
    cudaStream_t helper_stream,
    char *zero_copy,
    cudaEvent_t memset_per_tile_done,
    cudaEvent_t copy_n_instances_done,
    cudaEvent_t preprocess_done,
    int *out_n_visible_primitives,
    int *out_n_instances,
    int *out_n_buckets,
    int *out_primitive_selector,
    int *out_instance_selector
);
cudaError_t gsx_cuda_fastgs_backward_launch(
    const float *grad_image,
    const float *image,
    const float3 *means,
    const float3 *scales_raw,
    const float4 *rotations_raw,
    const float *sh1,
    const float *sh2,
    const float *sh3,
    const float4 *w2c,
    const float3 *cam_position,
    char *per_primitive_buffers_blob,
    char *per_tile_buffers_blob,
    char *per_instance_buffers_blob,
    char *per_bucket_buffers_blob,
    float3 *grad_means,
    float3 *grad_scales_raw,
    float4 *grad_rotations_raw,
    float *grad_opacities_raw,
    float *grad_sh0,
    float *grad_sh1,
    float *grad_sh2,
    float *grad_sh3,
    float2 *grad_mean2d_helper,
    float *grad_conic_helper,
    float3 *grad_color,
    float4 *grad_w2c,
    float2 *absgrad_mean2d_helper,
    int n_primitives,
    int n_visible_primitives,
    int n_instances,
    int n_buckets,
    int primitive_selector,
    int instance_selector,
    int active_sh_bases,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    cudaStream_t stream
);

gsx_error gsx_cuda_backend_provider_bootstrap(gsx_builtin_registry_state *registry);

GSX_EXTERN_C_END

#endif /* GSX_CUDA_INTERNAL_H */
