#ifndef GSX_METAL_INTERNAL_H
#define GSX_METAL_INTERNAL_H

#include "../gsx-impl.h"

#include <stdbool.h>
#include <stdint.h>

GSX_EXTERN_C_BEGIN

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

typedef struct gsx_metal_sort_profile {
    double histogram_ns;
    double prefix_offsets_ns;
    double scatter_ns;
    double total_ns;
    uint32_t count;
    uint32_t significant_bits;
    uint32_t pass_count;
    uint32_t num_threadgroups;
    bool valid;
} gsx_metal_sort_profile;

typedef struct gsx_metal_backend {
    struct gsx_backend base;
    gsx_backend_capabilities capabilities;
    void *mtl_device;
    void *major_command_queue;
    void *tensor_library;            /* cached MTLLibrary loaded from embedded metallib bytes */
    void *tensor_gather_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_exp_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_sigmoid_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_sigmoid_derivative_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_abs_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_rand_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_randn_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_randint_i32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_sum_reduce_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_mean_reduce_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_max_reduce_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_mse_reduce_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_mae_reduce_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_clamp_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_clamp_i32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_check_finite_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *tensor_check_finite_f16_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *image_linear_to_srgb_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *image_srgb_to_linear_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *image_chw_to_hwc_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *image_hwc_to_chw_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *image_f32_to_u8_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *image_u8_to_f32_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *adc_library;               /* cached MTLLibrary loaded from embedded metallib bytes */
    void *adc_classify_growth_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *adc_apply_split_pipeline;  /* cached MTLComputePipelineState, NULL until first use */
    void *adc_keep_mask_pipeline;    /* cached MTLComputePipelineState, NULL until first use */
    void *optim_library;             /* cached MTLLibrary loaded from embedded metallib bytes */
    void *optim_adam_pipeline;       /* cached MTLComputePipelineState, NULL until first use */
    void *loss_library;              /* cached MTLLibrary loaded from embedded metallib bytes */
    void *loss_mse_pipeline;         /* cached MTLComputePipelineState, NULL until first use */
    void *loss_l1_pipeline;          /* cached MTLComputePipelineState, NULL until first use */
    void *loss_mse_backward_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *loss_l1_backward_pipeline; /* cached MTLComputePipelineState, NULL until first use */
    void *loss_ssim_chw_pipeline;    /* cached MTLComputePipelineState, NULL until first use */
    void *loss_ssim_hwc_pipeline;    /* cached MTLComputePipelineState, NULL until first use */
    void *loss_ssim_backward_chw_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *loss_ssim_backward_hwc_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_library;            /* cached MTLLibrary loaded from embedded metallib bytes */
    void *render_preprocess_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_apply_depth_ordering_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_create_instances_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_extract_instance_ranges_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_extract_bucket_counts_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_finalize_bucket_offsets_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_blend_pipeline;     /* cached MTLComputePipelineState, NULL until first use */
    void *render_preprocess_backward_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_blend_backward_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *render_compose_pipeline;   /* cached MTLComputePipelineState, NULL until first use */
    void *sort_library;              /* cached MTLLibrary loaded from embedded metallib bytes */
    void *sort_histogram_pipeline;   /* cached MTLComputePipelineState, NULL until first use */
    void *sort_reduce_pipeline;      /* cached MTLComputePipelineState, NULL until first use */
    void *sort_scan_pipeline;        /* cached MTLComputePipelineState, NULL until first use */
    void *sort_scatter_offsets_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *sort_scatter_pipeline;     /* cached MTLComputePipelineState, NULL until first use */
    void *sort_scatter_tail_pipeline;/* cached MTLComputePipelineState, NULL until first use */
    void *scan_library;              /* cached MTLLibrary loaded from embedded metallib bytes */
    void *scan_blocks_pipeline;      /* cached MTLComputePipelineState, NULL until first use */
    void *scan_block_sums_pipeline;  /* cached MTLComputePipelineState, NULL until first use */
    void *scan_add_offsets_pipeline; /* cached MTLComputePipelineState, NULL until first use */
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

typedef struct gsx_metal_tensor_gather_params {
    uint32_t x_row_count;
    uint32_t out_row_count;
    uint32_t row_bytes;
} gsx_metal_tensor_gather_params;

typedef struct gsx_metal_tensor_unary_f32_params {
    uint32_t element_count;
} gsx_metal_tensor_unary_f32_params;

typedef struct gsx_metal_tensor_rand_f32_params {
    uint64_t rng_state;
    uint64_t rng_inc;
    uint32_t element_count;
} gsx_metal_tensor_rand_f32_params;

typedef struct gsx_metal_tensor_randn_f32_params {
    uint64_t rng_state;
    uint64_t rng_inc;
    float sigma;
    uint32_t element_count;
} gsx_metal_tensor_randn_f32_params;

typedef struct gsx_metal_tensor_randint_i32_params {
    uint64_t rng_state;
    uint64_t rng_inc;
    uint32_t bound;
    uint32_t element_count;
} gsx_metal_tensor_randint_i32_params;

typedef struct gsx_metal_tensor_unary_reduce_f32_params {
    uint32_t outer_count;
    uint32_t reduce_count;
} gsx_metal_tensor_unary_reduce_f32_params;

typedef struct gsx_metal_tensor_binary_reduce_f32_params {
    uint32_t outer_count;
    uint32_t reduce_count;
} gsx_metal_tensor_binary_reduce_f32_params;

typedef struct gsx_metal_tensor_clamp_f32_params {
    float min_value;
    float max_value;
    uint32_t element_count;
} gsx_metal_tensor_clamp_f32_params;

typedef struct gsx_metal_tensor_clamp_i32_params {
    int32_t min_value;
    int32_t max_value;
    uint32_t element_count;
} gsx_metal_tensor_clamp_i32_params;

typedef struct gsx_metal_adc_classify_growth_params {
    uint32_t gaussian_count;
    uint32_t has_visible_counter;
    float duplicate_grad_threshold;
    float duplicate_scale_threshold;
    float scene_scale;
} gsx_metal_adc_classify_growth_params;

typedef struct gsx_metal_adc_apply_split_params {
    uint32_t split_count;
    uint64_t rng_state;
    uint64_t rng_inc;
} gsx_metal_adc_apply_split_params;

typedef struct gsx_metal_adc_keep_mask_params {
    uint32_t gaussian_count;
    uint32_t has_max_screen_radius;
    uint32_t count_before_growth;
    uint32_t prune_large;
    float scene_scale;
    float pruning_opacity_threshold;
    float max_world_scale;
    float max_screen_scale;
} gsx_metal_adc_keep_mask_params;

typedef struct gsx_metal_tensor_check_finite_params {
    uint32_t element_count;
} gsx_metal_tensor_check_finite_params;

typedef struct gsx_metal_image_tensor_params {
    uint32_t element_count;
} gsx_metal_image_tensor_params;

typedef struct gsx_metal_image_layout_params {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t element_size_bytes;
} gsx_metal_image_layout_params;

typedef struct gsx_metal_loss_pointwise_params {
    uint32_t element_count;
    float scale;
} gsx_metal_loss_pointwise_params;

typedef struct gsx_metal_loss_ssim_params {
    uint32_t outer_count;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t element_count;
    uint32_t has_scratch;
    float scale;
} gsx_metal_loss_ssim_params;

typedef struct gsx_metal_render_compose_params {
    uint32_t width;
    uint32_t height;
    uint32_t channel_stride;
    float background_r;
    float background_g;
    float background_b;
} gsx_metal_render_compose_params;

typedef struct gsx_metal_render_preprocess_params {
    uint32_t gaussian_count;
    uint32_t width;
    uint32_t height;
    uint32_t sh_degree;
    uint32_t has_visible_counter;
    uint32_t has_max_screen_radius;
    uint32_t grid_width;
    uint32_t grid_height;
    float fx;
    float fy;
    float cx;
    float cy;
    float near_plane;
    float far_plane;
    float pose_qx;
    float pose_qy;
    float pose_qz;
    float pose_qw;
    float pose_tx;
    float pose_ty;
    float pose_tz;
} gsx_metal_render_preprocess_params;

typedef struct gsx_metal_render_create_instances_params {
    uint32_t visible_count;
    uint32_t grid_width;
    uint32_t grid_height;
} gsx_metal_render_create_instances_params;

typedef struct gsx_metal_render_blend_params {
    uint32_t width;
    uint32_t height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t tile_count;
    uint32_t channel_stride;
} gsx_metal_render_blend_params;

typedef struct gsx_metal_render_preprocess_backward_params {
    uint32_t gaussian_count;
    uint32_t width;
    uint32_t height;
    uint32_t sh_degree;
    uint32_t has_grad_acc;
    uint32_t has_absgrad_acc;
    float fx;
    float fy;
    float cx;
    float cy;
    float near_plane;
    float far_plane;
    float pose_qx;
    float pose_qy;
    float pose_qz;
    float pose_qw;
    float pose_tx;
    float pose_ty;
    float pose_tz;
} gsx_metal_render_preprocess_backward_params;

typedef struct gsx_metal_render_blend_backward_params {
    uint32_t gaussian_count;
    uint32_t width;
    uint32_t height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t tile_count;
    uint32_t total_bucket_count;
    uint32_t channel_stride;
    float background_r;
    float background_g;
    float background_b;
} gsx_metal_render_blend_backward_params;

typedef struct gsx_metal_sort_histogram_params {
    uint32_t count;
    uint32_t shift;
    uint32_t num_threadgroups;
} gsx_metal_sort_histogram_params;

typedef struct gsx_metal_scan_blocks_params {
    uint32_t count;
} gsx_metal_scan_blocks_params;

typedef struct gsx_metal_sort_pair_u32 {
    uint32_t key;
    uint32_t value;
    uint32_t stable_index;
} gsx_metal_sort_pair_u32;

typedef gsx_error (*gsx_metal_render_dry_run_plan_fn)(gsx_arena_t dry_run_arena, void *user_data);

typedef struct gsx_metal_renderer {
    struct gsx_renderer base;
    gsx_backend_buffer_type_t device_buffer_type;
} gsx_metal_renderer;

typedef struct gsx_metal_render_context {
    struct gsx_render_context base;
    gsx_arena_t helper_arena;
    gsx_arena_t scratch_arena;
    gsx_arena_t forward_per_primitive_arena;
    gsx_arena_t forward_per_tile_arena;
    gsx_arena_t forward_per_instance_arena;
    gsx_arena_t forward_per_bucket_arena;
    gsx_tensor_t helper_image_chw;
    gsx_tensor_t helper_alpha_hw;
    gsx_tensor_t optional_dummy_f32;
    gsx_arena_t retain_arena;
    gsx_tensor_t saved_mean3d;
    gsx_tensor_t saved_rotation;
    gsx_tensor_t saved_logscale;
    gsx_tensor_t saved_sh0;
    gsx_tensor_t saved_sh1;
    gsx_tensor_t saved_sh2;
    gsx_tensor_t saved_sh3;
    gsx_tensor_t saved_opacity;
    gsx_tensor_t saved_mean2d;
    gsx_tensor_t saved_conic_opacity;
    gsx_tensor_t saved_color;
    gsx_tensor_t saved_instance_primitive_ids;
    gsx_tensor_t saved_tile_ranges;
    gsx_tensor_t saved_tile_bucket_offsets;
    gsx_tensor_t saved_bucket_tile_index;
    gsx_tensor_t saved_bucket_color_transmittance;
    gsx_tensor_t saved_tile_max_n_contributions;
    gsx_tensor_t saved_tile_n_contributions;
    uint32_t saved_bucket_count;
    gsx_camera_intrinsics saved_intrinsics;
    gsx_camera_pose saved_pose;
    gsx_vec3 saved_background_color;
    gsx_float_t saved_near_plane;
    gsx_float_t saved_far_plane;
    gsx_index_t saved_sh_degree;
    bool has_train_state;
    bool train_state_borrowed;
    /* Persistent sort-scratch buffers: owned for context lifetime, grown lazily, freed in dispose. */
    gsx_metal_sort_pair_u32 *host_visible_pairs;  /* capacity: host_gaussian_capacity entries */
    gsx_metal_sort_pair_u32 *host_instance_pairs; /* capacity: host_instance_capacity entries */
    gsx_size_t host_gaussian_capacity;
    gsx_size_t host_instance_capacity;
} gsx_metal_render_context;

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
gsx_error gsx_metal_backend_query_unary_reduce_workspace_size(
    gsx_backend_t backend,
    gsx_backend_buffer_type_class workspace_buffer_type,
    gsx_data_type data_type,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
);
gsx_error gsx_metal_backend_query_binary_reduce_workspace_size(
    gsx_backend_t backend,
    gsx_backend_buffer_type_class workspace_buffer_type,
    gsx_data_type data_type,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
);
gsx_error gsx_metal_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer);
gsx_error gsx_metal_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss);
gsx_error gsx_metal_backend_create_optim(gsx_backend_t backend, const gsx_optim_desc *desc, gsx_optim_t *out_optim);
gsx_error gsx_metal_optim_zero_appended_rows(gsx_optim_t optim, gsx_size_t old_count, gsx_size_t new_count);
gsx_error gsx_metal_backend_create_adc(gsx_backend_t backend, const gsx_adc_desc *desc, gsx_adc_t *out_adc);
gsx_error gsx_metal_backend_create_async_dl(gsx_backend_t backend, const gsx_async_dl_desc *desc, gsx_async_dl_t *out_async_dl);

gsx_error gsx_metal_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info);
gsx_error gsx_metal_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes);
gsx_error gsx_metal_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer);

gsx_error gsx_metal_backend_buffer_free(gsx_backend_buffer_t buffer);
gsx_error gsx_metal_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info);
gsx_error gsx_metal_backend_buffer_get_native_handle(gsx_backend_buffer_t buffer, void **out_handle);
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
gsx_error gsx_metal_backend_buffer_fill_rand_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc
);
gsx_error gsx_metal_backend_buffer_fill_randn_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc,
    gsx_float_t sigma
);
gsx_error gsx_metal_backend_buffer_fill_randint_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc,
    uint32_t bound
);
gsx_error gsx_metal_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
);
gsx_error gsx_metal_backend_buffer_gather_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape
);
gsx_error gsx_metal_backend_buffer_unary_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
);
gsx_error gsx_metal_backend_buffer_unary_tensor_inplace(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
);
gsx_error gsx_metal_backend_buffer_unary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
);
gsx_error gsx_metal_backend_buffer_binary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
);
gsx_error gsx_metal_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
);
gsx_error gsx_metal_backend_buffer_image_convert_colorspace(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    gsx_storage_format storage_format,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_image_colorspace src_colorspace,
    const gsx_backend_tensor_view *dst_view,
    gsx_image_colorspace dst_colorspace
);
gsx_error gsx_metal_backend_buffer_image_convert_storage_format(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    gsx_index_t src_rank,
    const gsx_index_t *src_shape,
    gsx_storage_format src_storage_format,
    const gsx_backend_tensor_view *dst_view,
    gsx_index_t dst_rank,
    const gsx_index_t *dst_shape,
    gsx_storage_format dst_storage_format
);
gsx_error gsx_metal_backend_buffer_image_convert_data_type(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    gsx_storage_format storage_format,
    gsx_index_t rank,
    const gsx_index_t *shape,
    const gsx_backend_tensor_view *dst_view
);

gsx_metal_backend *gsx_metal_backend_from_base(gsx_backend_t backend);
gsx_metal_backend_buffer_type *gsx_metal_backend_buffer_type_from_base(gsx_backend_buffer_type_t buffer_type);
gsx_metal_backend_buffer *gsx_metal_backend_buffer_from_base(gsx_backend_buffer_t buffer);
gsx_backend_buffer_type_class gsx_metal_backend_buffer_get_type_class(gsx_backend_buffer_t buffer);
void gsx_metal_backend_fill_host_bytes(void *dst_bytes, gsx_size_t total_bytes, const void *value_bytes, gsx_size_t value_size_bytes);
bool gsx_metal_backend_f16_is_finite(uint16_t value);
gsx_error gsx_metal_backend_buffer_check_range(gsx_backend_buffer_t buffer, gsx_size_t offset_bytes, gsx_size_t byte_count);
gsx_error gsx_metal_backend_reduce_validate_shape_contract(
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_reduce_count
);
gsx_error gsx_metal_backend_dispatch_adam_step(
    gsx_backend_t backend,
    gsx_tensor_t parameter,
    gsx_tensor_t gradient,
    gsx_tensor_t first_moment,
    gsx_tensor_t second_moment,
    const gsx_metal_adam_step_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_gather(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_gather_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_exp(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_sigmoid(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_sigmoid_derivative(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_abs(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_rand_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_rand_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_randn_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_randn_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_randint_i32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_randint_i32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_unary_reduce_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_unary_reduce_f32_params *params,
    gsx_impl_unary_reduce_op op
);
gsx_error gsx_metal_backend_dispatch_tensor_binary_reduce_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_binary_reduce_f32_params *params,
    gsx_impl_binary_reduce_op op
);
gsx_error gsx_metal_backend_dispatch_tensor_clamp_f32_inplace(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_clamp_f32_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_clamp_i32_inplace(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_clamp_i32_params *params
);
gsx_error gsx_metal_backend_dispatch_adc_classify_growth(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *grad_acc_view,
    const gsx_backend_tensor_view *visible_counter_view,
    const gsx_backend_tensor_view *logscale_view,
    gsx_backend_buffer_t out_mode_buffer,
    const gsx_metal_adc_classify_growth_params *params
);
gsx_error gsx_metal_backend_dispatch_adc_apply_split(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *mean3d_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *rotation_view,
    gsx_backend_buffer_t split_source_buffer,
    gsx_backend_buffer_t split_target_buffer,
    const gsx_metal_adc_apply_split_params *params
);
gsx_error gsx_metal_backend_dispatch_adc_keep_mask(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *rotation_view,
    const gsx_backend_tensor_view *max_screen_radius_view,
    gsx_backend_buffer_t out_keep_mask_buffer,
    const gsx_metal_adc_keep_mask_params *params
);
gsx_error gsx_metal_backend_dispatch_tensor_check_finite_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_check_finite_params *params,
    uint32_t *out_has_non_finite
);
gsx_error gsx_metal_backend_dispatch_tensor_check_finite_f16(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tensor_view,
    const gsx_metal_tensor_check_finite_params *params,
    uint32_t *out_has_non_finite
);
gsx_error gsx_metal_backend_dispatch_image_linear_to_srgb_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
);
gsx_error gsx_metal_backend_dispatch_image_srgb_to_linear_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
);
gsx_error gsx_metal_backend_dispatch_image_chw_to_hwc(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_layout_params *params
);
gsx_error gsx_metal_backend_dispatch_image_hwc_to_chw(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_layout_params *params
);
gsx_error gsx_metal_backend_dispatch_image_f32_to_u8(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
);
gsx_error gsx_metal_backend_dispatch_image_u8_to_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *src_view,
    const gsx_backend_tensor_view *dst_view,
    const gsx_metal_image_tensor_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_mse_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_l1_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_mse_backward_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_metal_loss_pointwise_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_l1_backward_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_metal_loss_pointwise_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_ssim_chw_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *loss_map_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_ssim_hwc_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *loss_map_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_ssim_backward_chw_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params
);
gsx_error gsx_metal_backend_dispatch_loss_ssim_backward_hwc_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_backend_tensor_view *scratch_a_view,
    const gsx_backend_tensor_view *scratch_b_view,
    const gsx_metal_loss_ssim_params *params
);
gsx_error gsx_metal_backend_dispatch_render_compose_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *image_view,
    const gsx_backend_tensor_view *alpha_view,
    const gsx_backend_tensor_view *out_rgb_view,
    const gsx_metal_render_compose_params *params
);
gsx_error gsx_metal_backend_dispatch_render_preprocess(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *mean3d_view,
    const gsx_backend_tensor_view *rotation_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *sh0_view,
    const gsx_backend_tensor_view *sh1_view,
    const gsx_backend_tensor_view *sh2_view,
    const gsx_backend_tensor_view *sh3_view,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *depth_keys_view,
    const gsx_backend_tensor_view *visible_primitive_ids_view,
    const gsx_backend_tensor_view *touched_tiles_view,
    const gsx_backend_tensor_view *bounds_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_backend_tensor_view *visible_count_view,
    const gsx_backend_tensor_view *instance_count_view,
    const gsx_backend_tensor_view *visible_counter_view,
    const gsx_backend_tensor_view *max_screen_radius_view,
    const gsx_metal_render_preprocess_params *params
);
gsx_error gsx_metal_backend_dispatch_render_apply_depth_ordering(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *sorted_primitive_ids_view,
    const gsx_backend_tensor_view *touched_tiles_view,
    const gsx_backend_tensor_view *primitive_offsets_view,
    uint32_t visible_count
);
gsx_error gsx_metal_backend_dispatch_render_create_instances(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *sorted_primitive_ids_view,
    const gsx_backend_tensor_view *primitive_offsets_view,
    const gsx_backend_tensor_view *bounds_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *instance_keys_view,
    const gsx_backend_tensor_view *instance_primitive_ids_view,
    const gsx_metal_render_create_instances_params *params
);
gsx_error gsx_metal_backend_dispatch_render_extract_instance_ranges(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *instance_keys_view,
    const gsx_backend_tensor_view *tile_ranges_view,
    uint32_t instance_count,
    uint32_t tile_count
);
gsx_error gsx_metal_backend_dispatch_render_extract_bucket_counts(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_ranges_view,
    const gsx_backend_tensor_view *tile_bucket_counts_view,
    uint32_t tile_count
);
gsx_error gsx_metal_backend_dispatch_render_finalize_bucket_offsets(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_bucket_counts_view,
    const gsx_backend_tensor_view *tile_bucket_offsets_view,
    uint32_t tile_count
);
gsx_error gsx_metal_backend_dispatch_render_blend(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_ranges_view,
    const gsx_backend_tensor_view *tile_bucket_offsets_view,
    const gsx_backend_tensor_view *instance_primitive_ids_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_backend_tensor_view *image_view,
    const gsx_backend_tensor_view *alpha_view,
    const gsx_backend_tensor_view *tile_max_n_contributions_view,
    const gsx_backend_tensor_view *tile_n_contributions_view,
    const gsx_backend_tensor_view *bucket_tile_index_view,
    const gsx_backend_tensor_view *bucket_color_transmittance_view,
    const gsx_metal_render_blend_params *params
);
gsx_error gsx_metal_backend_dispatch_render_blend_backward(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *tile_ranges_view,
    const gsx_backend_tensor_view *tile_bucket_offsets_view,
    const gsx_backend_tensor_view *bucket_tile_index_view,
    const gsx_backend_tensor_view *instance_primitive_ids_view,
    const gsx_backend_tensor_view *mean2d_view,
    const gsx_backend_tensor_view *conic_opacity_view,
    const gsx_backend_tensor_view *color_view,
    const gsx_backend_tensor_view *image_view,
    const gsx_backend_tensor_view *alpha_view,
    const gsx_backend_tensor_view *tile_max_n_contributions_view,
    const gsx_backend_tensor_view *tile_n_contributions_view,
    const gsx_backend_tensor_view *bucket_color_transmittance_view,
    const gsx_backend_tensor_view *grad_rgb_view,
    const gsx_backend_tensor_view *grad_mean2d_view,
    const gsx_backend_tensor_view *absgrad_mean2d_view,
    const gsx_backend_tensor_view *grad_conic_view,
    const gsx_backend_tensor_view *grad_raw_opacity_view,
    const gsx_backend_tensor_view *grad_color_view,
    const gsx_metal_render_blend_backward_params *params
);
gsx_error gsx_metal_backend_dispatch_render_preprocess_backward(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *mean3d_view,
    const gsx_backend_tensor_view *rotation_view,
    const gsx_backend_tensor_view *logscale_view,
    const gsx_backend_tensor_view *sh1_view,
    const gsx_backend_tensor_view *sh2_view,
    const gsx_backend_tensor_view *sh3_view,
    const gsx_backend_tensor_view *opacity_view,
    const gsx_backend_tensor_view *grad_mean2d_view,
    const gsx_backend_tensor_view *absgrad_mean2d_view,
    const gsx_backend_tensor_view *grad_conic_view,
    const gsx_backend_tensor_view *grad_raw_opacity_partial_view,
    const gsx_backend_tensor_view *grad_color_view,
    const gsx_backend_tensor_view *grad_mean3d_view,
    const gsx_backend_tensor_view *grad_rotation_view,
    const gsx_backend_tensor_view *grad_logscale_view,
    const gsx_backend_tensor_view *grad_sh0_view,
    const gsx_backend_tensor_view *grad_sh1_view,
    const gsx_backend_tensor_view *grad_sh2_view,
    const gsx_backend_tensor_view *grad_sh3_view,
    const gsx_backend_tensor_view *grad_opacity_view,
    const gsx_backend_tensor_view *grad_acc_view,
    const gsx_backend_tensor_view *absgrad_acc_view,
    const gsx_metal_render_preprocess_backward_params *params
);
gsx_error gsx_metal_backend_dispatch_sort_pairs_u32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *keys_in_view,
    const gsx_backend_tensor_view *values_in_view,
    const gsx_backend_tensor_view *keys_out_view,
    const gsx_backend_tensor_view *values_out_view,
    const gsx_backend_tensor_view *histogram_view,
    const gsx_backend_tensor_view *global_histogram_view,
    const gsx_backend_tensor_view *scatter_offsets_view,
    uint32_t count,
    uint32_t significant_bits,
    gsx_metal_sort_profile *out_profile
);
gsx_error gsx_metal_backend_dispatch_scan_exclusive_u32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *data_view,
    const gsx_backend_tensor_view *block_sums_view,
    const gsx_backend_tensor_view *scanned_block_sums_view,
    uint32_t count
);

bool gsx_metal_render_tensor_is_device_f32(gsx_tensor_t tensor);
bool gsx_metal_render_tensor_is_optional_device_f32(gsx_tensor_t tensor);
bool gsx_metal_render_tensor_is_backed_f32(gsx_tensor_t tensor);
bool gsx_metal_render_tensor_is_backed_i32(gsx_tensor_t tensor);
gsx_error gsx_metal_render_make_tensor(
    gsx_arena_t arena,
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_tensor_t *out_tensor
);
gsx_error gsx_metal_render_make_tensor_aligned(
    gsx_arena_t arena,
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_size_t requested_alignment_bytes,
    gsx_tensor_t *out_tensor
);
gsx_error gsx_metal_render_validate_tensor_alignment(gsx_tensor_t tensor, gsx_size_t required_alignment_bytes, const char *tensor_name);
void gsx_metal_render_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view);
void gsx_metal_render_release_tensor(gsx_tensor_t *tensor);
gsx_error gsx_metal_render_reserve_arena_with_dry_run(
    gsx_arena_t target_arena,
    gsx_metal_render_dry_run_plan_fn plan_fn,
    void *plan_user_data
);
gsx_error gsx_metal_render_validate_train_state_for_backward(const gsx_metal_render_context *metal_context);
gsx_error gsx_metal_render_tensor_map_host_bytes(gsx_tensor_t tensor, void **out_bytes, gsx_size_t *out_size_bytes);
gsx_index_t gsx_metal_render_get_grid_width(gsx_index_t width);
gsx_index_t gsx_metal_render_get_grid_height(gsx_index_t height);
gsx_size_t gsx_metal_render_get_tile_count(gsx_index_t width, gsx_index_t height);
gsx_size_t gsx_metal_render_get_channel_stride(gsx_index_t width, gsx_index_t height);

gsx_error gsx_metal_render_context_init(gsx_metal_render_context *metal_context, gsx_backend_buffer_type_t buffer_type, gsx_index_t width, gsx_index_t height);
gsx_error gsx_metal_render_context_dispose(gsx_metal_render_context *metal_context);
gsx_error gsx_metal_render_context_clear_train_state(gsx_metal_render_context *metal_context);
gsx_error gsx_metal_render_context_snapshot_train_state(
    gsx_metal_render_context *metal_context,
    const gsx_render_forward_request *request,
    gsx_tensor_t mean2d,
    gsx_tensor_t conic_opacity,
    gsx_tensor_t color,
    gsx_tensor_t instance_primitive_ids,
    gsx_tensor_t tile_ranges,
    gsx_tensor_t tile_bucket_offsets,
    gsx_tensor_t bucket_tile_index,
    gsx_tensor_t bucket_color_transmittance,
    gsx_tensor_t tile_max_n_contributions,
    gsx_tensor_t tile_n_contributions,
    uint32_t bucket_count
);
gsx_error gsx_metal_renderer_forward_impl(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request);
gsx_error gsx_metal_renderer_backward_impl(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request);
void gsx_metal_render_sort_pairs_u32(gsx_metal_sort_pair_u32 *pairs, uint32_t count);
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

GSX_EXTERN_C_END

#endif /* GSX_METAL_INTERNAL_H */
