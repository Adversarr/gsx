#ifndef GSX_METAL_ADC_INTERNAL_H
#define GSX_METAL_ADC_INTERNAL_H

#include "../internal.h"
#include "../../pcg32.h"

#include "gsx/gsx-random.h"

typedef struct gsx_metal_adc {
    struct gsx_adc base;
    gsx_pcg32_t rng;
} gsx_metal_adc;

typedef struct gsx_metal_adc_refine_data {
    gsx_size_t count;
    gsx_tensor_t mean3d_tensor;
    gsx_backend_tensor_view mean3d_view;
    gsx_tensor_t grad_acc_tensor;
    gsx_backend_tensor_view grad_acc_view;
    gsx_tensor_t visible_counter_tensor;
    gsx_backend_tensor_view visible_counter_view;
    gsx_tensor_t logscale_tensor;
    gsx_backend_tensor_view logscale_view;
    gsx_tensor_t opacity_tensor;
    gsx_backend_tensor_view opacity_view;
    gsx_tensor_t rotation_tensor;
    gsx_backend_tensor_view rotation_view;
    gsx_tensor_t sh0_tensor;
    gsx_backend_tensor_view sh0_view;
    gsx_tensor_t sh1_tensor;
    gsx_backend_tensor_view sh1_view;
    gsx_tensor_t sh2_tensor;
    gsx_backend_tensor_view sh2_view;
    gsx_tensor_t sh3_tensor;
    gsx_backend_tensor_view sh3_view;
    gsx_tensor_t max_screen_radius_tensor;
    gsx_backend_tensor_view max_screen_radius_view;
    bool has_visible_counter;
    bool has_sh1;
    bool has_sh2;
    bool has_sh3;
    bool has_max_screen_radius;
} gsx_metal_adc_refine_data;

typedef enum gsx_metal_adc_grow_mode {
    GSX_METAL_ADC_GROW_NONE = 0,
    GSX_METAL_ADC_GROW_DUPLICATE = 1,
    GSX_METAL_ADC_GROW_SPLIT = 2
} gsx_metal_adc_grow_mode;

gsx_size_t gsx_metal_adc_non_negative_index(gsx_index_t value);
void gsx_metal_adc_zero_result(gsx_adc_result *out_result);
bool gsx_metal_adc_optim_enabled(const gsx_adc_request *request);
bool gsx_metal_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step);
bool gsx_metal_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step);
gsx_error gsx_metal_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count);
void gsx_metal_adc_make_tensor_view(gsx_tensor_t tensor, gsx_backend_tensor_view *out_view);
gsx_error gsx_metal_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    bool optional,
    gsx_tensor_t *out_tensor,
    gsx_backend_tensor_view *out_view
);
void gsx_metal_adc_free_refine_data(gsx_metal_adc_refine_data *data);
gsx_error gsx_metal_adc_load_refine_data(gsx_gs_t gs, gsx_size_t count, bool require_grad_acc, gsx_metal_adc_refine_data *out_data);
gsx_error gsx_metal_adc_init_temp_buffer_for_tensor(
    gsx_tensor_t reference_tensor,
    gsx_size_t byte_count,
    gsx_backend_buffer_t *out_buffer
);
gsx_error gsx_metal_adc_build_index_tensor(
    gsx_tensor_t reference_tensor,
    const int32_t *indices,
    gsx_size_t index_count,
    gsx_backend_buffer_t *out_buffer,
    struct gsx_tensor *out_index_tensor
);
gsx_error gsx_metal_adc_apply_gs_and_optim_gather(const gsx_adc_request *request, const int32_t *indices, gsx_size_t index_count);
gsx_error gsx_metal_adc_apply_gs_gather_and_rebind_optim(const gsx_adc_request *request, const int32_t *indices, gsx_size_t index_count);
gsx_error gsx_metal_adc_zero_growth_optim_state(const gsx_adc_request *request, gsx_size_t old_count, gsx_size_t new_count);
gsx_error gsx_metal_adc_reset_post_refine_aux(gsx_gs_t gs);
gsx_error gsx_metal_adc_download_temp_buffer(gsx_backend_buffer_t buffer, void *dst_bytes, gsx_size_t byte_count);
gsx_error gsx_metal_adc_upload_temp_buffer(gsx_backend_buffer_t buffer, const void *src_bytes, gsx_size_t byte_count);
gsx_error gsx_metal_adc_advance_rng(gsx_metal_adc *metal_adc, gsx_size_t draw_count, const char *context);
gsx_error gsx_metal_adc_apply_default_reset(const gsx_adc_desc *desc, const gsx_adc_request *request);
gsx_error gsx_metal_adc_apply_default_refine(
    gsx_metal_adc *metal_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
);
gsx_error gsx_metal_adc_apply_mcmc_noise(
    gsx_metal_adc *metal_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request
);
gsx_error gsx_metal_adc_apply_mcmc_refine(
    gsx_metal_adc *metal_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
);

#endif /* GSX_METAL_ADC_INTERNAL_H */
