#ifndef GSX_CPU_ADC_INTERNAL_H
#define GSX_CPU_ADC_INTERNAL_H

#include "../internal.h"

#include "gsx/gsx-random.h"

typedef struct gsx_cpu_adc {
    struct gsx_adc base;
    gsx_pcg32_t rng;
} gsx_cpu_adc;

typedef struct gsx_cpu_adc_refine_data {
    gsx_size_t count;
    float *mean3d;
    float *grad_acc;
    float *absgrad_acc;
    float *visible_counter;
    float *logscale;
    float *opacity;
    float *rotation;
    float *sh0;
    float *sh1;
    float *sh2;
    float *sh3;
    float *max_screen_radius;
    bool has_visible_counter;
    bool has_max_screen_radius;
} gsx_cpu_adc_refine_data;

gsx_size_t gsx_cpu_adc_non_negative_index(gsx_index_t value);
void gsx_cpu_adc_zero_result(gsx_adc_result *out_result);
bool gsx_cpu_adc_optim_enabled(const gsx_adc_request *request);
bool gsx_cpu_adc_in_refine_window(const gsx_adc_desc *desc, gsx_size_t global_step);
bool gsx_cpu_adc_in_reset_window(const gsx_adc_desc *desc, gsx_size_t global_step);
gsx_error gsx_cpu_adc_load_count(gsx_gs_t gs, gsx_size_t *out_count);
gsx_error gsx_cpu_adc_load_refine_field(
    gsx_gs_t gs,
    gsx_gs_field field,
    gsx_size_t count,
    gsx_size_t expected_dim1,
    bool optional,
    float **out_values
);
void gsx_cpu_adc_free_refine_data(gsx_cpu_adc_refine_data *data);
gsx_error gsx_cpu_adc_load_refine_data(gsx_gs_t gs, gsx_size_t count, bool require_grad_acc, gsx_cpu_adc_refine_data *out_data);
float gsx_cpu_adc_probability_to_logit(float probability);
gsx_error gsx_cpu_adc_sample_normal(gsx_pcg32_t rng, float *out_value);
gsx_error gsx_cpu_adc_copy_slice(float *dst, gsx_size_t dst_index, const float *src, gsx_size_t src_index, gsx_size_t width);
gsx_error gsx_cpu_adc_copy_optional_slice(float *dst, gsx_size_t dst_index, const float *src, gsx_size_t src_index, gsx_size_t width);
float gsx_cpu_adc_clamp_probability(float value);
void gsx_cpu_adc_normalize_quaternion(float *qx, float *qy, float *qz, float *qw);
void gsx_cpu_adc_build_rotation_matrix(
    float qx,
    float qy,
    float qz,
    float qw,
    float *m00,
    float *m01,
    float *m02,
    float *m10,
    float *m11,
    float *m12,
    float *m20,
    float *m21,
    float *m22
);
gsx_error gsx_cpu_adc_apply_gs_and_optim_gather(const gsx_adc_request *request, const int32_t *indices, gsx_size_t index_count);
gsx_error gsx_cpu_adc_zero_growth_optim_state(const gsx_adc_request *request, gsx_size_t old_count, gsx_size_t new_count);
gsx_error gsx_cpu_adc_reset_post_refine_aux(gsx_gs_t gs);

gsx_error gsx_cpu_adc_apply_default_refine(
    gsx_cpu_adc *cpu_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
);
gsx_error gsx_cpu_adc_apply_default_reset(const gsx_adc_desc *desc, const gsx_adc_request *request);
gsx_error gsx_cpu_adc_apply_mcmc_noise(
    gsx_cpu_adc *cpu_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request
);
gsx_error gsx_cpu_adc_apply_mcmc_refine(
    gsx_cpu_adc *cpu_adc,
    const gsx_adc_desc *desc,
    const gsx_adc_request *request,
    gsx_adc_result *out_result
);

#endif /* GSX_CPU_ADC_INTERNAL_H */
