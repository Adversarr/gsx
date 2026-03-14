#include "internal.h"

#include <stdint.h>
#include <math.h>
#include <stdlib.h>

typedef struct gsx_cpu_loss {
    struct gsx_loss base;
} gsx_cpu_loss;

typedef struct gsx_cpu_loss_context {
    struct gsx_loss_context base;
} gsx_cpu_loss_context;

typedef struct gsx_cpu_loss_ssim_point_terms {
    double mu1;
    double mu2;
    double sigma1_sq;
    double sigma2_sq;
    double sigma12;
    double ssim;
} gsx_cpu_loss_ssim_point_terms;

static gsx_error gsx_cpu_loss_destroy(gsx_loss_t loss);
static gsx_error gsx_cpu_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context);
static gsx_error gsx_cpu_loss_context_destroy(gsx_loss_context_t context);
static gsx_error gsx_cpu_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request);
static gsx_error gsx_cpu_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request);
static gsx_error gsx_cpu_loss_forward_mse(
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    gsx_size_t element_count,
    gsx_float_t loss_scale
);
static gsx_error gsx_cpu_loss_backward_mse(
    const float *prediction_values,
    const float *target_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t grad_scale
);
static gsx_error gsx_cpu_loss_forward_l1(
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    gsx_size_t element_count,
    gsx_float_t loss_scale
);
static gsx_error gsx_cpu_loss_backward_l1(
    const float *prediction_values,
    const float *target_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t grad_scale
);
static gsx_error gsx_cpu_loss_evaluate_ssim(
    const gsx_cpu_loss *cpu_loss,
    gsx_tensor_t prediction,
    gsx_tensor_t target,
    gsx_tensor_t loss_map_accumulator,
    gsx_tensor_t grad_prediction_accumulator,
    gsx_float_t loss_scale,
    gsx_float_t grad_scale
);
static bool gsx_cpu_loss_ssim_extract_layout(
    gsx_tensor_t prediction,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_channels,
    gsx_size_t *out_height,
    gsx_size_t *out_width
);
static gsx_size_t gsx_cpu_loss_ssim_linear_index_chw(
    gsx_size_t outer, gsx_size_t channel, gsx_size_t y, gsx_size_t x, gsx_size_t channels, gsx_size_t height, gsx_size_t width);
static gsx_size_t gsx_cpu_loss_ssim_linear_index_hwc(
    gsx_size_t outer, gsx_size_t channel, gsx_size_t y, gsx_size_t x, gsx_size_t channels, gsx_size_t height, gsx_size_t width);
static gsx_size_t gsx_cpu_loss_ssim_linear_index(
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    gsx_size_t y,
    gsx_size_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
);
static double gsx_cpu_loss_ssim_sample_or_zero_f32(
    const float *values,
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    int64_t y,
    int64_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
);
static double gsx_cpu_loss_ssim_sample_or_zero_f64(
    const double *values,
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    int64_t y,
    int64_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
);
static gsx_cpu_loss_ssim_point_terms gsx_cpu_loss_ssim_point_terms_eval(
    const float *prediction_values,
    const float *target_values,
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    gsx_size_t y,
    gsx_size_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
);

enum {
    GSX_CPU_LOSS_SSIM_KERNEL_RADIUS = 5,
    GSX_CPU_LOSS_SSIM_KERNEL_SIZE = 2 * GSX_CPU_LOSS_SSIM_KERNEL_RADIUS + 1
};

static const double gsx_cpu_loss_ssim_gauss[GSX_CPU_LOSS_SSIM_KERNEL_SIZE] = {
    0.001028380123898387,
    0.0075987582094967365,
    0.036000773310661316,
    0.10936068743467331,
    0.21300552785396576,
    0.26601171493530273,
    0.21300552785396576,
    0.10936068743467331,
    0.036000773310661316,
    0.0075987582094967365,
    0.001028380123898387
};

static const gsx_loss_i gsx_cpu_loss_iface = {
    gsx_cpu_loss_destroy,
    gsx_cpu_loss_create_context,
    gsx_cpu_loss_forward,
    gsx_cpu_loss_backward
};

static const gsx_loss_context_i gsx_cpu_loss_context_iface = {
    gsx_cpu_loss_context_destroy
};

/* CPU-private access boundary: only these helpers may dereference cpu_buffer->data. */
static unsigned char *gsx_cpu_loss_tensor_data_bytes(gsx_tensor_t tensor)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)tensor->backing_buffer;

    return (unsigned char *)cpu_buffer->data + (size_t)tensor->offset_bytes;
}

static float *gsx_cpu_loss_tensor_data_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cpu_loss_tensor_data_bytes(tensor);
}

static gsx_error gsx_cpu_loss_validate_tensor_f32(gsx_tensor_t tensor)
{
    if(tensor->data_type != GSX_DATA_TYPE_F32 || tensor->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu loss tensors must use float32 storage");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_cpu_loss_grad_scale(const gsx_cpu_loss *cpu_loss, gsx_size_t element_count, gsx_float_t scale)
{
    if(cpu_loss->base.grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN) {
        return scale / (float)element_count;
    }

    return scale;
}

gsx_error gsx_cpu_backend_create_loss(gsx_backend_t backend, const gsx_loss_desc *desc, gsx_loss_t *out_loss)
{
    gsx_cpu_loss *cpu_loss = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_loss == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_loss and desc must be non-null");
    }

    *out_loss = NULL;
    cpu_loss = (gsx_cpu_loss *)calloc(1, sizeof(*cpu_loss));
    if(cpu_loss == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu loss");
    }

    error = gsx_loss_base_init(&cpu_loss->base, &gsx_cpu_loss_iface, backend, desc);
    if(!gsx_error_is_success(error)) {
        free(cpu_loss);
        return error;
    }

    *out_loss = &cpu_loss->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_destroy(gsx_loss_t loss)
{
    gsx_cpu_loss *cpu_loss = (gsx_cpu_loss *)loss;

    if(loss == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss must be non-null");
    }

    gsx_loss_base_deinit(&cpu_loss->base);
    free(cpu_loss);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_create_context(gsx_loss_t loss, gsx_loss_context_t *out_context)
{
    gsx_cpu_loss_context *cpu_context = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(loss == NULL || out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "loss and out_context must be non-null");
    }

    *out_context = NULL;
    cpu_context = (gsx_cpu_loss_context *)calloc(1, sizeof(*cpu_context));
    if(cpu_context == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu loss context");
    }
    error = gsx_loss_context_base_init(&cpu_context->base, &gsx_cpu_loss_context_iface, loss);
    if(!gsx_error_is_success(error)) {
        free(cpu_context);
        return error;
    }

    *out_context = &cpu_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_context_destroy(gsx_loss_context_t context)
{
    gsx_cpu_loss_context *cpu_context = (gsx_cpu_loss_context *)context;

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    gsx_loss_context_base_deinit(&cpu_context->base);
    free(cpu_context);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_forward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_forward_request *request)
{
    gsx_cpu_loss *cpu_loss = (gsx_cpu_loss *)loss;
    const float *prediction_values = NULL;
    const float *target_values = NULL;
    float *loss_map_values = NULL;
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    (void)context;
    error = gsx_cpu_loss_validate_tensor_f32(request->prediction);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_loss_validate_tensor_f32(request->target);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_loss_validate_tensor_f32(request->loss_map_accumulator);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = request->prediction->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu loss tensors must contain at least one element");
    }

    prediction_values = gsx_cpu_loss_tensor_data_f32(request->prediction);
    target_values = gsx_cpu_loss_tensor_data_f32(request->target);
    loss_map_values = gsx_cpu_loss_tensor_data_f32(request->loss_map_accumulator);

    switch(cpu_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        return gsx_cpu_loss_forward_mse(prediction_values, target_values, loss_map_values, element_count, request->scale);
    case GSX_LOSS_ALGORITHM_L1:
        return gsx_cpu_loss_forward_l1(prediction_values, target_values, loss_map_values, element_count, request->scale);
    case GSX_LOSS_ALGORITHM_SSIM:
        return gsx_cpu_loss_evaluate_ssim(
            cpu_loss, request->prediction, request->target, request->loss_map_accumulator, NULL, request->scale, 0.0f);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

static gsx_error gsx_cpu_loss_backward(gsx_loss_t loss, gsx_loss_context_t context, const gsx_loss_backward_request *request)
{
    gsx_cpu_loss *cpu_loss = (gsx_cpu_loss *)loss;
    const float *prediction_values = NULL;
    const float *target_values = NULL;
    float *grad_values = NULL;
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    float grad_scale = 0.0f;

    error = gsx_cpu_loss_validate_tensor_f32(context->retained_prediction);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_loss_validate_tensor_f32(context->retained_target);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_loss_validate_tensor_f32(request->grad_prediction_accumulator);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    element_count = context->retained_prediction->size_bytes / sizeof(float);
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu loss tensors must contain at least one element");
    }
    prediction_values = gsx_cpu_loss_tensor_data_f32(context->retained_prediction);
    target_values = gsx_cpu_loss_tensor_data_f32(context->retained_target);
    grad_values = gsx_cpu_loss_tensor_data_f32(request->grad_prediction_accumulator);
    grad_scale = gsx_cpu_loss_grad_scale(cpu_loss, element_count, request->scale);

    switch(cpu_loss->base.algorithm) {
    case GSX_LOSS_ALGORITHM_MSE:
        return gsx_cpu_loss_backward_mse(prediction_values, target_values, grad_values, element_count, grad_scale);
    case GSX_LOSS_ALGORITHM_L1:
        return gsx_cpu_loss_backward_l1(prediction_values, target_values, grad_values, element_count, grad_scale);
    case GSX_LOSS_ALGORITHM_SSIM:
        return gsx_cpu_loss_evaluate_ssim(
            cpu_loss,
            context->retained_prediction,
            context->retained_target,
            NULL,
            request->grad_prediction_accumulator,
            0.0f,
            grad_scale);
    }

    return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "loss algorithm is out of range");
}

static gsx_error gsx_cpu_loss_forward_mse(
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    gsx_size_t element_count,
    gsx_float_t loss_scale
)
{
    gsx_size_t element_index = 0;

    for(element_index = 0; element_index < element_count; ++element_index) {
        float diff = prediction_values[element_index] - target_values[element_index];

        loss_map_values[element_index] += loss_scale * (diff * diff);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_backward_mse(
    const float *prediction_values,
    const float *target_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t grad_scale
)
{
    gsx_size_t element_index = 0;

    for(element_index = 0; element_index < element_count; ++element_index) {
        float diff = prediction_values[element_index] - target_values[element_index];

        grad_values[element_index] += 2.0f * diff * grad_scale;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_forward_l1(
    const float *prediction_values,
    const float *target_values,
    float *loss_map_values,
    gsx_size_t element_count,
    gsx_float_t loss_scale
)
{
    gsx_size_t element_index = 0;

    for(element_index = 0; element_index < element_count; ++element_index) {
        float diff = prediction_values[element_index] - target_values[element_index];
        loss_map_values[element_index] += loss_scale * fabsf(diff);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_backward_l1(
    const float *prediction_values,
    const float *target_values,
    float *grad_values,
    gsx_size_t element_count,
    gsx_float_t grad_scale
)
{
    gsx_size_t element_index = 0;

    for(element_index = 0; element_index < element_count; ++element_index) {
        float diff = prediction_values[element_index] - target_values[element_index];
        float sign = 0.0f;

        if(diff > 0.0f) {
            sign = 1.0f;
        } else if(diff < 0.0f) {
            sign = -1.0f;
        }
        grad_values[element_index] += sign * grad_scale;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_loss_evaluate_ssim(
    const gsx_cpu_loss *cpu_loss,
    gsx_tensor_t prediction,
    gsx_tensor_t target,
    gsx_tensor_t loss_map_accumulator,
    gsx_tensor_t grad_prediction_accumulator,
    gsx_float_t loss_scale_value,
    gsx_float_t grad_scale_value)
{
    (void)cpu_loss;
    const gsx_storage_format storage_format = prediction->storage_format;
    const float *prediction_values = gsx_cpu_loss_tensor_data_f32(prediction);
    const float *target_values = gsx_cpu_loss_tensor_data_f32(target);
    float *loss_map_values = NULL;
    float *grad_values = NULL;
    double *dm_dmu1 = NULL;
    double *dm_dsigma1_sq = NULL;
    double *dm_dsigma12 = NULL;
    gsx_size_t outer_count = 0;
    gsx_size_t channels = 0;
    gsx_size_t height = 0;
    gsx_size_t width = 0;
    gsx_size_t outer = 0;
    gsx_size_t channel = 0;
    gsx_size_t y = 0;
    gsx_size_t x = 0;
    const gsx_size_t element_count = prediction->size_bytes / sizeof(float);
    gsx_size_t temp_bytes = 0;
    const double loss_scale = (double)loss_scale_value;
    const double grad_scale = (double)grad_scale_value;

    if(storage_format == GSX_STORAGE_FORMAT_TILED_CHW) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu backend does not define tiled CHW SSIM neighborhood indexing");
    }
    if(storage_format != GSX_STORAGE_FORMAT_CHW && storage_format != GSX_STORAGE_FORMAT_HWC) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "ssim storage_format is out of range");
    }
    if(!gsx_cpu_loss_ssim_extract_layout(prediction, &outer_count, &channels, &height, &width)) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "ssim loss expects rank>=3 with finite contiguous shape for image dimensions");
    }
    if(grad_prediction_accumulator != NULL) {
        grad_values = gsx_cpu_loss_tensor_data_f32(grad_prediction_accumulator);
        if(gsx_size_mul_overflows(element_count, sizeof(double), &temp_bytes)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "ssim temporary workspace size overflow");
        }
        dm_dmu1 = (double *)calloc(1, temp_bytes);
        dm_dsigma1_sq = (double *)calloc(1, temp_bytes);
        dm_dsigma12 = (double *)calloc(1, temp_bytes);
        if(dm_dmu1 == NULL || dm_dsigma1_sq == NULL || dm_dsigma12 == NULL) {
            free(dm_dsigma12);
            free(dm_dsigma1_sq);
            free(dm_dmu1);
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate ssim backward workspace");
        }
    }
    if(loss_map_accumulator != NULL) {
        loss_map_values = gsx_cpu_loss_tensor_data_f32(loss_map_accumulator);
    }
    for(outer = 0; outer < outer_count; ++outer) {
        for(channel = 0; channel < channels; ++channel) {
            for(y = 0; y < height; ++y) {
                for(x = 0; x < width; ++x) {
                    const gsx_size_t element_index =
                        gsx_cpu_loss_ssim_linear_index(storage_format, outer, channel, y, x, channels, height, width);
                    const gsx_cpu_loss_ssim_point_terms point_terms = gsx_cpu_loss_ssim_point_terms_eval(
                        prediction_values, target_values, storage_format, outer, channel, y, x, channels, height, width);
                    const double mu1_sq = point_terms.mu1 * point_terms.mu1;
                    const double mu2_sq = point_terms.mu2 * point_terms.mu2;
                    const double a = mu1_sq + mu2_sq + 0.01 * 0.01;
                    const double b = point_terms.sigma1_sq + point_terms.sigma2_sq + 0.03 * 0.03;
                    const double c_term = 2.0 * point_terms.mu1 * point_terms.mu2 + 0.01 * 0.01;
                    const double d_term = 2.0 * point_terms.sigma12 + 0.03 * 0.03;

                    if(loss_map_values != NULL) {
                        loss_map_values[element_index] += (float)((1.0 - point_terms.ssim) * loss_scale);
                    }
                    if(grad_values != NULL) {
                        if(a == 0.0 || b == 0.0) {
                            dm_dmu1[element_index] = 0.0;
                            dm_dsigma1_sq[element_index] = 0.0;
                            dm_dsigma12[element_index] = 0.0;
                        } else {
                            const double ab = a * b;
                            const double aab = a * ab;
                            const double abb = ab * b;

                            dm_dmu1[element_index] = (2.0 * point_terms.mu2 * d_term) / ab
                                - (2.0 * point_terms.mu2 * c_term) / ab
                                - (2.0 * point_terms.mu1 * c_term * d_term) / aab
                                + (2.0 * point_terms.mu1 * c_term * d_term) / abb;
                            dm_dsigma1_sq[element_index] = -(c_term * d_term) / abb;
                            dm_dsigma12[element_index] = (2.0 * c_term) / ab;
                        }
                    }
                }
            }
        }
    }
    if(grad_values != NULL) {
        for(outer = 0; outer < outer_count; ++outer) {
            for(channel = 0; channel < channels; ++channel) {
                for(y = 0; y < height; ++y) {
                    for(x = 0; x < width; ++x) {
                        const gsx_size_t element_index =
                            gsx_cpu_loss_ssim_linear_index(storage_format, outer, channel, y, x, channels, height, width);
                        const int64_t y_center = (int64_t)y;
                        const int64_t x_center = (int64_t)x;
                        double conv_mu = 0.0;
                        double conv_sigma1 = 0.0;
                        double conv_sigma12 = 0.0;
                        int64_t ky = 0;
                        int64_t kx = 0;

                        for(ky = -GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; ky <= GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; ++ky) {
                            const double wy = gsx_cpu_loss_ssim_gauss[ky + GSX_CPU_LOSS_SSIM_KERNEL_RADIUS];

                            for(kx = -GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; kx <= GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; ++kx) {
                                const double wx = gsx_cpu_loss_ssim_gauss[kx + GSX_CPU_LOSS_SSIM_KERNEL_RADIUS];
                                const double w = wy * wx;

                                conv_mu += gsx_cpu_loss_ssim_sample_or_zero_f64(
                                    dm_dmu1, storage_format, outer, channel, y_center + ky, x_center + kx, channels, height, width) * w;
                                conv_sigma1 += gsx_cpu_loss_ssim_sample_or_zero_f64(
                                    dm_dsigma1_sq, storage_format, outer, channel, y_center + ky, x_center + kx, channels, height, width) * w;
                                conv_sigma12 += gsx_cpu_loss_ssim_sample_or_zero_f64(
                                    dm_dsigma12, storage_format, outer, channel, y_center + ky, x_center + kx, channels, height, width) * w;
                            }
                        }
                        {
                            const double prediction_sample = (double)prediction_values[element_index];
                            const double target_sample = (double)target_values[element_index];
                            const double grad = -(conv_mu + 2.0 * prediction_sample * conv_sigma1 + target_sample * conv_sigma12)
                                * grad_scale;

                            grad_values[element_index] += (float)grad;
                        }
                    }
                }
            }
        }
    }
    free(dm_dsigma12);
    free(dm_dsigma1_sq);
    free(dm_dmu1);

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_cpu_loss_ssim_extract_layout(
    gsx_tensor_t prediction,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_channels,
    gsx_size_t *out_height,
    gsx_size_t *out_width
)
{
    gsx_size_t outer_count = 1;
    gsx_index_t axis = 0;
    gsx_size_t channels = 0;
    gsx_size_t height = 0;
    gsx_size_t width = 0;

    if(prediction == NULL || out_outer_count == NULL || out_channels == NULL || out_height == NULL || out_width == NULL) {
        return false;
    }
    if(prediction->rank < 3) {
        return false;
    }
    for(axis = 0; axis < prediction->rank - 3; ++axis) {
        gsx_size_t dim_extent = (gsx_size_t)prediction->shape[axis];
        gsx_size_t next_outer_count = 0;

        if(dim_extent == 0 || gsx_size_mul_overflows(outer_count, dim_extent, &next_outer_count)) {
            return false;
        }
        outer_count = next_outer_count;
    }
    if(prediction->storage_format == GSX_STORAGE_FORMAT_HWC) {
        height = (gsx_size_t)prediction->shape[prediction->rank - 3];
        width = (gsx_size_t)prediction->shape[prediction->rank - 2];
        channels = (gsx_size_t)prediction->shape[prediction->rank - 1];
    } else {
        channels = (gsx_size_t)prediction->shape[prediction->rank - 3];
        height = (gsx_size_t)prediction->shape[prediction->rank - 2];
        width = (gsx_size_t)prediction->shape[prediction->rank - 1];
    }
    if(channels == 0 || height == 0 || width == 0) {
        return false;
    }

    *out_outer_count = outer_count;
    *out_channels = channels;
    *out_height = height;
    *out_width = width;
    return true;
}

static gsx_size_t gsx_cpu_loss_ssim_linear_index_chw(
    gsx_size_t outer, gsx_size_t channel, gsx_size_t y, gsx_size_t x, gsx_size_t channels, gsx_size_t height, gsx_size_t width)
{
    return (((outer * channels + channel) * height + y) * width + x);
}

static gsx_size_t gsx_cpu_loss_ssim_linear_index_hwc(
    gsx_size_t outer, gsx_size_t channel, gsx_size_t y, gsx_size_t x, gsx_size_t channels, gsx_size_t height, gsx_size_t width)
{
    return (((outer * height + y) * width + x) * channels + channel);
}

static gsx_size_t gsx_cpu_loss_ssim_linear_index(
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    gsx_size_t y,
    gsx_size_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
)
{
    if(storage_format == GSX_STORAGE_FORMAT_HWC) {
        return gsx_cpu_loss_ssim_linear_index_hwc(outer, channel, y, x, channels, height, width);
    }

    return gsx_cpu_loss_ssim_linear_index_chw(outer, channel, y, x, channels, height, width);
}

static double gsx_cpu_loss_ssim_sample_or_zero_f32(
    const float *values,
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    int64_t y,
    int64_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
)
{
    gsx_size_t element_index = 0;

    if(y < 0 || x < 0 || y >= (int64_t)height || x >= (int64_t)width) {
        return 0.0;
    }
    element_index = gsx_cpu_loss_ssim_linear_index(
        storage_format, outer, channel, (gsx_size_t)y, (gsx_size_t)x, channels, height, width);
    return (double)values[element_index];
}

static double gsx_cpu_loss_ssim_sample_or_zero_f64(
    const double *values,
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    int64_t y,
    int64_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
)
{
    gsx_size_t element_index = 0;

    if(y < 0 || x < 0 || y >= (int64_t)height || x >= (int64_t)width) {
        return 0.0;
    }
    element_index = gsx_cpu_loss_ssim_linear_index(
        storage_format, outer, channel, (gsx_size_t)y, (gsx_size_t)x, channels, height, width);
    return values[element_index];
}

static gsx_cpu_loss_ssim_point_terms gsx_cpu_loss_ssim_point_terms_eval(
    const float *prediction_values,
    const float *target_values,
    gsx_storage_format storage_format,
    gsx_size_t outer,
    gsx_size_t channel,
    gsx_size_t y,
    gsx_size_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
)
{
    gsx_cpu_loss_ssim_point_terms out = { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
    double mu1 = 0.0;
    double mu2 = 0.0;
    double ex2 = 0.0;
    double ey2 = 0.0;
    double exy = 0.0;
    int64_t ky = 0;
    int64_t kx = 0;
    const double c1 = 0.01 * 0.01;
    const double c2 = 0.03 * 0.03;
    const int64_t y_center = (int64_t)y;
    const int64_t x_center = (int64_t)x;

    for(ky = -GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; ky <= GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; ++ky) {
        const double wy = gsx_cpu_loss_ssim_gauss[ky + GSX_CPU_LOSS_SSIM_KERNEL_RADIUS];

        for(kx = -GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; kx <= GSX_CPU_LOSS_SSIM_KERNEL_RADIUS; ++kx) {
            const double wx = gsx_cpu_loss_ssim_gauss[kx + GSX_CPU_LOSS_SSIM_KERNEL_RADIUS];
            const double w = wy * wx;
            const double prediction_sample = gsx_cpu_loss_ssim_sample_or_zero_f32(
                prediction_values,
                storage_format,
                outer,
                channel,
                y_center + ky,
                x_center + kx,
                channels,
                height,
                width);
            const double target_sample = gsx_cpu_loss_ssim_sample_or_zero_f32(
                target_values, storage_format, outer, channel, y_center + ky, x_center + kx, channels, height, width);

            mu1 += prediction_sample * w;
            mu2 += target_sample * w;
            ex2 += prediction_sample * prediction_sample * w;
            ey2 += target_sample * target_sample * w;
            exy += prediction_sample * target_sample * w;
        }
    }

    {
        const double mu1_sq = mu1 * mu1;
        const double mu2_sq = mu2 * mu2;
        const double sigma1_sq = ex2 - mu1_sq;
        const double sigma2_sq = ey2 - mu2_sq;
        const double sigma12 = exy - mu1 * mu2;
        const double a = mu1_sq + mu2_sq + c1;
        const double b = sigma1_sq + sigma2_sq + c2;
        const double c_term = 2.0 * mu1 * mu2 + c1;
        const double d_term = 2.0 * sigma12 + c2;
        const double denominator = a * b;

        out.mu1 = mu1;
        out.mu2 = mu2;
        out.sigma1_sq = sigma1_sq;
        out.sigma2_sq = sigma2_sq;
        out.sigma12 = sigma12;
        if(denominator == 0.0) {
            out.ssim = 1.0;
            return out;
        }
        out.ssim = (c_term * d_term) / denominator;
        return out;
    }
}
