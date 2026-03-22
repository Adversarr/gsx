#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {

#define ASSERT_GSX_SUCCESS(expr)                                                                                     \
    do {                                                                                                             \
        const gsx_error gsx_assert_success_error__ = (expr);                                                         \
        ASSERT_EQ(gsx_assert_success_error__.code, GSX_ERROR_SUCCESS)                                                \
            << (gsx_assert_success_error__.message != nullptr ? gsx_assert_success_error__.message : "");           \
    } while(false)

#define EXPECT_GSX_CODE(expr, expected_code)                                                                         \
    do {                                                                                                             \
        const gsx_error gsx_expect_code_error__ = (expr);                                                            \
        EXPECT_EQ(gsx_expect_code_error__.code, (expected_code))                                                     \
            << (gsx_expect_code_error__.message != nullptr ? gsx_expect_code_error__.message : "");                \
    } while(false)

static gsx_backend_device_t get_cpu_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device), GSX_ERROR_SUCCESS);
    return backend_device;
}

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};

    backend_desc.device = get_cpu_backend_device();
    EXPECT_NE(backend_desc.device, nullptr);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, type, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static gsx_arena_t create_arena(gsx_backend_buffer_type_t buffer_type)
{
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};

    arena_desc.initial_capacity_bytes = 4096;
    EXPECT_GSX_CODE(gsx_arena_init(&arena, buffer_type, &arena_desc), GSX_ERROR_SUCCESS);
    return arena;
}

static gsx_size_t product_of_shape(std::initializer_list<gsx_index_t> shape)
{
    gsx_size_t product = 1;

    for(gsx_index_t dim : shape) {
        product *= (gsx_size_t)dim;
    }
    return product;
}

static gsx_tensor_t make_f32_tensor(
    gsx_arena_t arena,
    std::initializer_list<gsx_index_t> shape,
    const std::vector<float> &values,
    gsx_storage_format storage_format = GSX_STORAGE_FORMAT_CHW)
{
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc desc{};
    std::size_t dim_index = 0;

    EXPECT_EQ(values.size(), (std::size_t)product_of_shape(shape));
    desc.rank = (gsx_index_t)shape.size();
    for(gsx_index_t dim : shape) {
        desc.shape[dim_index++] = dim;
    }
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = storage_format;
    desc.arena = arena;
    EXPECT_GSX_CODE(gsx_tensor_init(&tensor, &desc), GSX_ERROR_SUCCESS);
    if(tensor != nullptr && !values.empty()) {
        EXPECT_GSX_CODE(gsx_tensor_upload(tensor, values.data(), (gsx_size_t)values.size() * sizeof(float)), GSX_ERROR_SUCCESS);
    }
    return tensor;
}

static std::vector<float> download_f32_tensor(gsx_tensor_t tensor, std::size_t element_count)
{
    std::vector<float> values(element_count);

    if(!values.empty()) {
        EXPECT_GSX_CODE(gsx_tensor_download(tensor, values.data(), (gsx_size_t)values.size() * sizeof(float)), GSX_ERROR_SUCCESS);
    }
    return values;
}

static gsx_error evaluate_loss_once(gsx_loss_t loss, const gsx_loss_request *request)
{
    gsx_loss_context_t context = nullptr;
    gsx_loss_forward_request forward_request = {};
    gsx_loss_backward_request backward_request = {};
    gsx_error error = gsx_loss_context_init(&context, loss);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    forward_request.prediction = request->prediction;
    forward_request.target = request->target;
    forward_request.loss_map_accumulator = request->loss_map_accumulator;
    forward_request.train = request->grad_prediction_accumulator != nullptr;
    forward_request.scale = request->scale;
    error = gsx_loss_forward(loss, context, &forward_request);
    if(!gsx_error_is_success(error)) {
        (void)gsx_loss_context_free(context);
        return error;
    }
    if(request->grad_prediction_accumulator != nullptr) {
        backward_request.grad_prediction_accumulator = request->grad_prediction_accumulator;
        backward_request.scale = request->scale;
        error = gsx_loss_backward(loss, context, &backward_request);
        if(!gsx_error_is_success(error)) {
            (void)gsx_loss_context_free(context);
            return error;
        }
    }
    return gsx_loss_context_free(context);
}

static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1e-6f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index=" << i;
    }
}

enum {
    k_ssim_kernel_radius = 5,
    k_ssim_kernel_size = 2 * k_ssim_kernel_radius + 1
};

static const double k_ssim_gauss[k_ssim_kernel_size] = {
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

static std::size_t image_index(
    gsx_storage_format storage_format, std::size_t c, std::size_t y, std::size_t x, std::size_t channels, std::size_t height, std::size_t width)
{
    if(storage_format == GSX_STORAGE_FORMAT_HWC) {
        return (y * width + x) * channels + c;
    }

    return (c * height + y) * width + x;
}

static double image_sample_or_zero(
    const std::vector<float> &values,
    gsx_storage_format storage_format,
    std::size_t c,
    int64_t y,
    int64_t x,
    std::size_t channels,
    std::size_t height,
    std::size_t width)
{
    if(y < 0 || x < 0 || y >= (int64_t)height || x >= (int64_t)width) {
        return 0.0;
    }

    return values[image_index(storage_format, c, (std::size_t)y, (std::size_t)x, channels, height, width)];
}

typedef struct SsimPointTerms {
    double mu1;
    double mu2;
    double sigma1_sq;
    double sigma2_sq;
    double sigma12;
    double ssim;
} SsimPointTerms;

static SsimPointTerms compute_ssim_point_terms_reference(
    const std::vector<float> &prediction,
    const std::vector<float> &target,
    gsx_storage_format storage_format,
    std::size_t c,
    std::size_t y,
    std::size_t x,
    std::size_t channels,
    std::size_t height,
    std::size_t width)
{
    const double c1 = 0.01 * 0.01;
    const double c2 = 0.03 * 0.03;
    SsimPointTerms out{};
    double ex2 = 0.0;
    double ey2 = 0.0;
    double exy = 0.0;
    int64_t ky = 0;
    int64_t kx = 0;

    for(ky = -k_ssim_kernel_radius; ky <= k_ssim_kernel_radius; ++ky) {
        const double wy = k_ssim_gauss[ky + k_ssim_kernel_radius];
        for(kx = -k_ssim_kernel_radius; kx <= k_ssim_kernel_radius; ++kx) {
            const double wx = k_ssim_gauss[kx + k_ssim_kernel_radius];
            const double w = wy * wx;
            const double p = image_sample_or_zero(
                prediction, storage_format, c, (int64_t)y + ky, (int64_t)x + kx, channels, height, width);
            const double t = image_sample_or_zero(
                target, storage_format, c, (int64_t)y + ky, (int64_t)x + kx, channels, height, width);

            out.mu1 += p * w;
            out.mu2 += t * w;
            ex2 += p * p * w;
            ey2 += t * t * w;
            exy += p * t * w;
        }
    }

    {
        const double mu1_sq = out.mu1 * out.mu1;
        const double mu2_sq = out.mu2 * out.mu2;
        const double a = mu1_sq + mu2_sq + c1;
        const double b = (ex2 - mu1_sq) + (ey2 - mu2_sq) + c2;
        const double c_term = 2.0 * out.mu1 * out.mu2 + c1;
        const double d_term = 2.0 * (exy - out.mu1 * out.mu2) + c2;
        const double denominator = a * b;

        out.sigma1_sq = ex2 - mu1_sq;
        out.sigma2_sq = ey2 - mu2_sq;
        out.sigma12 = exy - out.mu1 * out.mu2;
        if(denominator == 0.0) {
            out.ssim = 1.0;
        } else {
            out.ssim = (c_term * d_term) / denominator;
        }
    }

    return out;
}

static double image_sample_or_zero_f64(
    const std::vector<double> &values,
    gsx_storage_format storage_format,
    std::size_t c,
    int64_t y,
    int64_t x,
    std::size_t channels,
    std::size_t height,
    std::size_t width)
{
    if(y < 0 || x < 0 || y >= (int64_t)height || x >= (int64_t)width) {
        return 0.0;
    }

    return values[image_index(storage_format, c, (std::size_t)y, (std::size_t)x, channels, height, width)];
}

static std::vector<float> compute_ssim_loss_map_reference(
    const std::vector<float> &prediction,
    const std::vector<float> &target,
    const std::vector<float> &initial_loss_map,
    gsx_storage_format storage_format,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    float scale)
{
    const double actual_scale = (double)scale;
    std::vector<float> output = initial_loss_map;
    std::size_t c = 0;
    std::size_t y = 0;
    std::size_t x = 0;

    for(c = 0; c < channels; ++c) {
        for(y = 0; y < height; ++y) {
            for(x = 0; x < width; ++x) {
                const SsimPointTerms terms =
                    compute_ssim_point_terms_reference(prediction, target, storage_format, c, y, x, channels, height, width);
                const std::size_t idx = image_index(storage_format, c, y, x, channels, height, width);

                output[idx] += (float)((1.0 - terms.ssim) * actual_scale);
            }
        }
    }

    return output;
}

static std::vector<float> compute_ssim_grad_map_reference(
    const std::vector<float> &prediction,
    const std::vector<float> &target,
    const std::vector<float> &initial_grad,
    gsx_storage_format storage_format,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    float scale,
    gsx_loss_grad_normalization_type grad_normalization)
{
    const std::size_t element_count = channels * height * width;
    const double c1 = 0.01 * 0.01;
    const double c2 = 0.03 * 0.03;
    const double grad_scale = grad_normalization == GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN
        ? (double)scale / (double)element_count
        : (double)scale;
    std::vector<float> output = initial_grad;
    std::vector<double> dm_dmu1(element_count, 0.0);
    std::vector<double> dm_dsigma1_sq(element_count, 0.0);
    std::vector<double> dm_dsigma12(element_count, 0.0);
    std::size_t c = 0;
    std::size_t y = 0;
    std::size_t x = 0;

    for(c = 0; c < channels; ++c) {
        for(y = 0; y < height; ++y) {
            for(x = 0; x < width; ++x) {
                const SsimPointTerms terms =
                    compute_ssim_point_terms_reference(prediction, target, storage_format, c, y, x, channels, height, width);
                const double mu1_sq = terms.mu1 * terms.mu1;
                const double mu2_sq = terms.mu2 * terms.mu2;
                const double a = mu1_sq + mu2_sq + c1;
                const double b = terms.sigma1_sq + terms.sigma2_sq + c2;
                const double c_term = 2.0 * terms.mu1 * terms.mu2 + c1;
                const double d_term = 2.0 * terms.sigma12 + c2;
                const std::size_t idx = image_index(storage_format, c, y, x, channels, height, width);

                if(a == 0.0 || b == 0.0) {
                    dm_dmu1[idx] = 0.0;
                    dm_dsigma1_sq[idx] = 0.0;
                    dm_dsigma12[idx] = 0.0;
                } else {
                    const double ab = a * b;
                    const double aab = a * ab;
                    const double abb = ab * b;

                    dm_dmu1[idx] = (2.0 * terms.mu2 * d_term) / ab
                        - (2.0 * terms.mu2 * c_term) / ab
                        - (2.0 * terms.mu1 * c_term * d_term) / aab
                        + (2.0 * terms.mu1 * c_term * d_term) / abb;
                    dm_dsigma1_sq[idx] = -(c_term * d_term) / abb;
                    dm_dsigma12[idx] = (2.0 * c_term) / ab;
                }
            }
        }
    }
    for(c = 0; c < channels; ++c) {
        for(y = 0; y < height; ++y) {
            for(x = 0; x < width; ++x) {
                double conv_mu = 0.0;
                double conv_sigma1 = 0.0;
                double conv_sigma12 = 0.0;
                int64_t ky = 0;
                int64_t kx = 0;
                const std::size_t idx = image_index(storage_format, c, y, x, channels, height, width);

                for(ky = -k_ssim_kernel_radius; ky <= k_ssim_kernel_radius; ++ky) {
                    const double wy = k_ssim_gauss[ky + k_ssim_kernel_radius];

                    for(kx = -k_ssim_kernel_radius; kx <= k_ssim_kernel_radius; ++kx) {
                        const double wx = k_ssim_gauss[kx + k_ssim_kernel_radius];
                        const double w = wy * wx;

                        conv_mu += image_sample_or_zero_f64(
                            dm_dmu1, storage_format, c, (int64_t)y + ky, (int64_t)x + kx, channels, height, width) * w;
                        conv_sigma1 += image_sample_or_zero_f64(
                            dm_dsigma1_sq, storage_format, c, (int64_t)y + ky, (int64_t)x + kx, channels, height, width) * w;
                        conv_sigma12 += image_sample_or_zero_f64(
                            dm_dsigma12, storage_format, c, (int64_t)y + ky, (int64_t)x + kx, channels, height, width) * w;
                    }
                }
                output[idx] += (float)(
                    -(conv_mu + 2.0 * (double)prediction[idx] * conv_sigma1 + (double)target[idx] * conv_sigma12) * grad_scale);
            }
        }
    }

    return output;
}

static void destroy_tensor(gsx_tensor_t tensor)
{
    if(tensor != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    }
}

static void destroy_arena(gsx_arena_t arena)
{
    if(arena != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    }
}

static void destroy_loss(gsx_loss_t loss)
{
    if(loss != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    }
}

static void destroy_backend(gsx_backend_t backend)
{
    if(backend != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    }
}

TEST(LossRuntime, InitMetadataAndAlgorithmNamesMatchContract)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_loss_desc desc{};
    gsx_loss_desc out_desc{};
    gsx_loss_t loss = nullptr;
    const char *name = nullptr;

    ASSERT_NE(backend, nullptr);

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_get_desc(loss, &out_desc));
    EXPECT_EQ(out_desc.algorithm, GSX_LOSS_ALGORITHM_MSE);
    EXPECT_EQ(out_desc.grad_normalization, GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN);
    ASSERT_GSX_SUCCESS(gsx_loss_get_algorithm_name(GSX_LOSS_ALGORITHM_MSE, &name));
    EXPECT_STREQ(name, "mse");
    destroy_loss(loss);
    loss = nullptr;

    desc.algorithm = GSX_LOSS_ALGORITHM_L1;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_get_desc(loss, &out_desc));
    EXPECT_EQ(out_desc.grad_normalization, GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM);
    ASSERT_GSX_SUCCESS(gsx_loss_get_algorithm_name(GSX_LOSS_ALGORITHM_L1, &name));
    EXPECT_STREQ(name, "l1");
    destroy_loss(loss);
    loss = nullptr;

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));
    destroy_loss(loss);
    loss = nullptr;
    ASSERT_GSX_SUCCESS(gsx_loss_get_algorithm_name(GSX_LOSS_ALGORITHM_SSIM, &name));
    EXPECT_STREQ(name, "ssim");
    EXPECT_GSX_CODE(gsx_loss_get_algorithm_name((gsx_loss_algorithm)99, &name), GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(gsx_loss_init(&loss, backend, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    desc.algorithm = (gsx_loss_algorithm)99;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    EXPECT_GSX_CODE(gsx_loss_init(&loss, backend, &desc), GSX_ERROR_OUT_OF_RANGE);

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = (gsx_loss_grad_normalization_type)99;
    EXPECT_GSX_CODE(gsx_loss_init(&loss, backend, &desc), GSX_ERROR_OUT_OF_RANGE);

    destroy_backend(backend);
}

TEST(LossRuntime, SsimForwardBackwardAccumulatesAndRejectsUnsupportedTiled)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_tensor_t grad = nullptr;
    gsx_tensor_t tiled_prediction = nullptr;
    gsx_tensor_t tiled_target = nullptr;
    gsx_tensor_t tiled_loss_map = nullptr;
    gsx_loss_request request{};
    const std::vector<float> prediction_values = {
        0.05f, 0.25f, 0.45f,
        0.15f, 0.35f, 0.55f,
        0.20f, 0.40f, 0.60f
    };
    const std::vector<float> target_values = {
        0.08f, 0.30f, 0.40f,
        0.18f, 0.28f, 0.50f,
        0.25f, 0.38f, 0.62f
    };
    const std::vector<float> initial_loss_map = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };
    const std::vector<float> expected_loss_map = compute_ssim_loss_map_reference(
        prediction_values, target_values, initial_loss_map, GSX_STORAGE_FORMAT_CHW, 1, 3, 3, 0.75f);
    const std::vector<float> initial_grad = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
    const std::vector<float> expected_grad = compute_ssim_grad_map_reference(
        prediction_values,
        target_values,
        initial_grad,
        GSX_STORAGE_FORMAT_CHW,
        1,
        3,
        3,
        0.75f,
        GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN);
    std::vector<float> actual_loss_map;

    prediction = make_f32_tensor(arena, { 1, 3, 3 }, prediction_values, GSX_STORAGE_FORMAT_CHW);
    target = make_f32_tensor(arena, { 1, 3, 3 }, target_values, GSX_STORAGE_FORMAT_CHW);
    loss_map = make_f32_tensor(arena, { 1, 3, 3 }, initial_loss_map, GSX_STORAGE_FORMAT_CHW);
    grad = make_f32_tensor(arena, { 1, 3, 3 }, initial_grad, GSX_STORAGE_FORMAT_CHW);
    tiled_prediction = make_f32_tensor(arena, { 1, 3, 3 }, prediction_values, GSX_STORAGE_FORMAT_TILED_CHW);
    tiled_target = make_f32_tensor(arena, { 1, 3, 3 }, target_values, GSX_STORAGE_FORMAT_TILED_CHW);
    tiled_loss_map = make_f32_tensor(arena, { 1, 3, 3 }, std::vector<float>(9, 1.0f), GSX_STORAGE_FORMAT_TILED_CHW);

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 0.75f;
    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    actual_loss_map = download_f32_tensor(loss_map, 9);
    expect_near_vectors(actual_loss_map, expected_loss_map, 1e-5f);
    expect_near_vectors(download_f32_tensor(grad, 9), expected_grad, 5e-5f);

    request.prediction = tiled_prediction;
    request.target = tiled_target;
    request.loss_map_accumulator = tiled_loss_map;
    request.grad_prediction_accumulator = nullptr;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_NOT_SUPPORTED);
    expect_near_vectors(download_f32_tensor(tiled_loss_map, 9), std::vector<float>(9, 1.0f));
    expect_near_vectors(download_f32_tensor(grad, 9), expected_grad, 5e-5f);

    destroy_loss(loss);
    destroy_tensor(tiled_loss_map);
    destroy_tensor(tiled_target);
    destroy_tensor(tiled_prediction);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, SsimGradientNormalizationChangesOnlyGradient)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc mean_desc{};
    gsx_loss_desc sum_desc{};
    gsx_loss_t mean_loss = nullptr;
    gsx_loss_t sum_loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t mean_loss_map = nullptr;
    gsx_tensor_t sum_loss_map = nullptr;
    gsx_tensor_t mean_grad = nullptr;
    gsx_tensor_t sum_grad = nullptr;
    gsx_loss_request mean_request{};
    gsx_loss_request sum_request{};

    prediction = make_f32_tensor(
        arena,
        { 3, 2, 2 },
        {
            0.05f, 0.25f, 0.45f, 0.15f,
            0.35f, 0.55f, 0.20f, 0.40f,
            0.60f, 0.11f, 0.33f, 0.77f
        },
        GSX_STORAGE_FORMAT_HWC);
    target = make_f32_tensor(
        arena,
        { 3, 2, 2 },
        {
            0.08f, 0.30f, 0.40f, 0.18f,
            0.28f, 0.50f, 0.25f, 0.38f,
            0.62f, 0.13f, 0.29f, 0.70f
        },
        GSX_STORAGE_FORMAT_HWC);
    mean_loss_map = make_f32_tensor(arena, { 3, 2, 2 }, std::vector<float>(12, 0.25f), GSX_STORAGE_FORMAT_HWC);
    sum_loss_map = make_f32_tensor(arena, { 3, 2, 2 }, std::vector<float>(12, 0.25f), GSX_STORAGE_FORMAT_HWC);
    mean_grad = make_f32_tensor(arena, { 3, 2, 2 }, std::vector<float>(12, 0.0f), GSX_STORAGE_FORMAT_HWC);
    sum_grad = make_f32_tensor(arena, { 3, 2, 2 }, std::vector<float>(12, 0.0f), GSX_STORAGE_FORMAT_HWC);

    mean_desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    mean_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    sum_desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    sum_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&mean_loss, backend, &mean_desc));
    ASSERT_GSX_SUCCESS(gsx_loss_init(&sum_loss, backend, &sum_desc));

    mean_request.prediction = prediction;
    mean_request.target = target;
    mean_request.loss_map_accumulator = mean_loss_map;
    mean_request.grad_prediction_accumulator = mean_grad;
    mean_request.scale = 1.2f;

    sum_request = mean_request;
    sum_request.loss_map_accumulator = sum_loss_map;
    sum_request.grad_prediction_accumulator = sum_grad;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(mean_loss, &mean_request));
    ASSERT_GSX_SUCCESS(evaluate_loss_once(sum_loss, &sum_request));
    expect_near_vectors(download_f32_tensor(mean_loss_map, 12), download_f32_tensor(sum_loss_map, 12), 1e-5f);

    {
        const std::vector<float> mean_grad_values = download_f32_tensor(mean_grad, 12);
        const std::vector<float> sum_grad_values = download_f32_tensor(sum_grad, 12);
        std::size_t i = 0;

        ASSERT_EQ(mean_grad_values.size(), sum_grad_values.size());
        for(i = 0; i < mean_grad_values.size(); ++i) {
            EXPECT_NEAR(sum_grad_values[i], mean_grad_values[i] * 12.0f, 2e-4f) << "index=" << i;
        }
    }

    destroy_loss(sum_loss);
    destroy_loss(mean_loss);
    destroy_tensor(sum_grad);
    destroy_tensor(mean_grad);
    destroy_tensor(sum_loss_map);
    destroy_tensor(mean_loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, SsimRejectsRankLessThanThree)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_loss_request request{};

    prediction = make_f32_tensor(arena, { 2, 2 }, { 0.1f, 0.2f, 0.3f, 0.4f });
    target = make_f32_tensor(arena, { 2, 2 }, { 0.1f, 0.1f, 0.1f, 0.1f });
    loss_map = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 1.0f, 1.0f, 1.0f });

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = nullptr;
    request.scale = 1.0f;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 1.0f, 1.0f, 1.0f, 1.0f });

    destroy_loss(loss);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, SsimIdenticalImagesProduceZeroLoss)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_loss_request request{};
    const std::vector<float> image = {
        0.05f, 0.25f, 0.45f,
        0.15f, 0.35f, 0.55f,
        0.20f, 0.40f, 0.60f
    };
    const std::vector<float> initial_loss_map = {
        0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f
    };

    prediction = make_f32_tensor(arena, { 1, 3, 3 }, image, GSX_STORAGE_FORMAT_CHW);
    target = make_f32_tensor(arena, { 1, 3, 3 }, image, GSX_STORAGE_FORMAT_CHW);
    loss_map = make_f32_tensor(arena, { 1, 3, 3 }, initial_loss_map, GSX_STORAGE_FORMAT_CHW);

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = nullptr;
    request.scale = 1.0f;
    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 9), initial_loss_map, 1e-5f);

    destroy_loss(loss);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, EmptyTensorRejection)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_tensor_t empty_tensor = nullptr;
    gsx_loss_request request{};
    gsx_tensor_desc empty_desc{};

    prediction = make_f32_tensor(arena, { 1, 1 }, { 0.0f });
    target = make_f32_tensor(arena, { 1, 1 }, { 0.0f });
    loss_map = make_f32_tensor(arena, { 1, 1 }, { 0.0f });

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = nullptr;
    request.scale = 1.0f;
    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 1), { 0.0f });

    empty_desc.rank = 2;
    empty_desc.shape[0] = 0;
    empty_desc.shape[1] = 2;
    empty_desc.data_type = GSX_DATA_TYPE_F32;
    empty_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    empty_desc.arena = arena;
    EXPECT_GSX_CODE(gsx_tensor_init(&empty_tensor, &empty_desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(empty_tensor, nullptr);

    destroy_loss(loss);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, MseMeanAccumulatesRawLossMapAndNormalizedGradient)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_tensor_t grad = nullptr;
    gsx_loss_request request{};

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);
    ASSERT_NE(arena, nullptr);

    prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, -2.0f, 3.0f, 4.0f });
    target = make_f32_tensor(arena, { 2, 2 }, { 0.0f, -1.0f, 1.0f, 5.0f });
    loss_map = make_f32_tensor(arena, { 2, 2 }, { 10.0f, 20.0f, 30.0f, 40.0f });
    grad = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f });

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 0.5f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 10.5f, 20.5f, 32.0f, 40.5f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 1.25f, 1.75f, 3.5f, 3.75f });

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 11.0f, 21.0f, 34.0f, 41.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 1.5f, 1.5f, 4.0f, 3.5f });

    destroy_loss(loss);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, L1SumAccumulatesRawLossMapAndOptionalGradient)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_tensor_t grad = nullptr;
    gsx_loss_request request{};

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);
    ASSERT_NE(arena, nullptr);

    prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 2.0f, -1.0f });
    target = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 0.0f, 4.0f, -2.0f });
    loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });

    desc.algorithm = GSX_LOSS_ALGORITHM_L1;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 2.0f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 0.0f, 4.0f, 4.0f, 2.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 0.0f, 2.0f, -2.0f, 2.0f });

    request.grad_prediction_accumulator = nullptr;
    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 0.0f, 8.0f, 8.0f, 4.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 0.0f, 2.0f, -2.0f, 2.0f });

    destroy_loss(loss);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, GradientNormalizationChangesOnlyGradient)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc mean_desc{};
    gsx_loss_desc sum_desc{};
    gsx_loss_t mean_loss = nullptr;
    gsx_loss_t sum_loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t mean_loss_map = nullptr;
    gsx_tensor_t sum_loss_map = nullptr;
    gsx_tensor_t mean_grad = nullptr;
    gsx_tensor_t sum_grad = nullptr;
    gsx_loss_request mean_request{};
    gsx_loss_request sum_request{};

    prediction = make_f32_tensor(arena, { 2, 2 }, { 3.0f, 0.0f, -1.0f, 4.0f });
    target = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 1.0f, -2.0f, 0.0f });
    mean_loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    sum_loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    mean_grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    sum_grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });

    mean_desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    mean_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    sum_desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    sum_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&mean_loss, backend, &mean_desc));
    ASSERT_GSX_SUCCESS(gsx_loss_init(&sum_loss, backend, &sum_desc));

    mean_request.prediction = prediction;
    mean_request.target = target;
    mean_request.loss_map_accumulator = mean_loss_map;
    mean_request.grad_prediction_accumulator = mean_grad;
    mean_request.scale = 1.5f;

    sum_request = mean_request;
    sum_request.loss_map_accumulator = sum_loss_map;
    sum_request.grad_prediction_accumulator = sum_grad;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(mean_loss, &mean_request));
    ASSERT_GSX_SUCCESS(evaluate_loss_once(sum_loss, &sum_request));

    expect_near_vectors(download_f32_tensor(mean_loss_map, 4), download_f32_tensor(sum_loss_map, 4));
    expect_near_vectors(download_f32_tensor(mean_loss_map, 4), { 6.0f, 1.5f, 1.5f, 24.0f });
    expect_near_vectors(download_f32_tensor(mean_grad, 4), { 1.5f, -0.75f, 0.75f, 3.0f });
    expect_near_vectors(download_f32_tensor(sum_grad, 4), { 6.0f, -3.0f, 3.0f, 12.0f });

    destroy_loss(sum_loss);
    destroy_loss(mean_loss);
    destroy_tensor(sum_grad);
    destroy_tensor(mean_grad);
    destroy_tensor(sum_loss_map);
    destroy_tensor(mean_loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, ForwardBackwardRejectsInvalidRequestsWithExpectedAccumulatorEffects)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_t backend2 = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t buffer_type2 = find_buffer_type(backend2, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_arena_t other_arena = create_arena(buffer_type2);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_tensor_t grad = nullptr;
    gsx_tensor_t bad_loss_map = nullptr;
    gsx_tensor_t bad_grad = nullptr;
    gsx_tensor_t bad_target = nullptr;
    gsx_loss_request request{};

    prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f });
    target = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 1.0f, 2.0f, 3.0f });
    loss_map = make_f32_tensor(arena, { 2, 2 }, { 9.0f, 8.0f, 7.0f, 6.0f });
    grad = make_f32_tensor(arena, { 2, 2 }, { 5.0f, 4.0f, 3.0f, 2.0f });
    bad_loss_map = make_f32_tensor(arena, { 4, 1 }, { 0.0f, 0.0f, 0.0f, 0.0f }, GSX_STORAGE_FORMAT_HWC);
    bad_grad = make_f32_tensor(arena, { 2, 1 }, { 0.0f, 0.0f });
    bad_target = make_f32_tensor(other_arena, { 2, 2 }, { 0.0f, 1.0f, 2.0f, 3.0f });

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = nullptr;
    request.grad_prediction_accumulator = grad;
    request.scale = 1.0f;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.loss_map_accumulator = bad_loss_map;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = bad_grad;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 10.0f, 9.0f, 8.0f, 7.0f });

    request.grad_prediction_accumulator = grad;
    request.target = bad_target;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 10.0f, 9.0f, 8.0f, 7.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.target = target;
    request.loss_map_accumulator = prediction;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(prediction, 4), { 1.0f, 2.0f, 3.0f, 4.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = prediction;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 11.0f, 10.0f, 9.0f, 8.0f });
    expect_near_vectors(download_f32_tensor(prediction, 4), { 1.0f, 2.0f, 3.0f, 4.0f });

    request.grad_prediction_accumulator = grad;
    request.scale = std::numeric_limits<float>::infinity();
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 11.0f, 10.0f, 9.0f, 8.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    destroy_loss(loss);
    destroy_tensor(bad_target);
    destroy_tensor(bad_grad);
    destroy_tensor(bad_loss_map);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(other_arena);
    destroy_arena(arena);
    destroy_backend(backend2);
    destroy_backend(backend);
}

TEST(LossRuntime, LossContextOwnershipAndFreeOrderContracts)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss_a = nullptr;
    gsx_loss_t loss_b = nullptr;
    gsx_loss_context_t context_a = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, -2.0f, 3.0f, 4.0f });
    gsx_tensor_t target = make_f32_tensor(arena, { 2, 2 }, { 0.0f, -1.0f, 1.0f, 5.0f });
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_tensor_t grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_loss_forward_request forward_request{};
    gsx_loss_backward_request backward_request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss_a, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss_b, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_context_init(&context_a, loss_a));

    EXPECT_GSX_CODE(gsx_loss_free(loss_a), GSX_ERROR_INVALID_STATE);

    forward_request.prediction = prediction;
    forward_request.target = target;
    forward_request.loss_map_accumulator = loss_map;
    forward_request.train = true;
    forward_request.scale = 0.5f;
    ASSERT_GSX_SUCCESS(gsx_loss_forward(loss_a, context_a, &forward_request));

    backward_request.grad_prediction_accumulator = grad;
    backward_request.scale = 0.5f;
    ASSERT_GSX_SUCCESS(gsx_loss_backward(loss_a, context_a, &backward_request));
    EXPECT_GSX_CODE(gsx_loss_backward(loss_b, context_a, &backward_request), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_loss_context_free(context_a));
    context_a = nullptr;
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss_b));
    loss_b = nullptr;
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss_a));
    loss_a = nullptr;

    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, LossBackwardRequiresForwardAndContextCanBeReused)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t context = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 2.0f, -1.0f });
    gsx_tensor_t target = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 0.0f, 4.0f, -2.0f });
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_tensor_t grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_loss_forward_request forward_request{};
    gsx_loss_backward_request backward_request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_L1;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_context_init(&context, loss));

    backward_request.grad_prediction_accumulator = grad;
    backward_request.scale = 2.0f;
    EXPECT_GSX_CODE(gsx_loss_backward(loss, context, &backward_request), GSX_ERROR_INVALID_STATE);

    forward_request.prediction = prediction;
    forward_request.target = target;
    forward_request.loss_map_accumulator = loss_map;
    forward_request.train = true;
    forward_request.scale = 2.0f;
    ASSERT_GSX_SUCCESS(gsx_loss_forward(loss, context, &forward_request));
    destroy_tensor(loss_map);
    loss_map = nullptr;
    loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    ASSERT_GSX_SUCCESS(gsx_loss_backward(loss, context, &backward_request));
    EXPECT_GSX_CODE(gsx_loss_backward(loss, context, &backward_request), GSX_ERROR_INVALID_STATE);
    forward_request.loss_map_accumulator = loss_map;
    ASSERT_GSX_SUCCESS(gsx_loss_forward(loss, context, &forward_request));
    ASSERT_GSX_SUCCESS(gsx_loss_backward(loss, context, &backward_request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 0.0f, 4.0f, 4.0f, 2.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 0.0f, 4.0f, -4.0f, 4.0f });

    ASSERT_GSX_SUCCESS(gsx_loss_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST(LossRuntime, LossBackwardRejectsNonTrainingForward)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t context = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 2.0f, -1.0f });
    gsx_tensor_t target = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 0.0f, 4.0f, -2.0f });
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_tensor_t grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_loss_forward_request forward_request{};
    gsx_loss_backward_request backward_request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_context_init(&context, loss));

    forward_request.prediction = prediction;
    forward_request.target = target;
    forward_request.loss_map_accumulator = loss_map;
    forward_request.train = false;
    forward_request.scale = 1.0f;
    ASSERT_GSX_SUCCESS(gsx_loss_forward(loss, context, &forward_request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 0.0f, 4.0f, 4.0f, 1.0f });

    backward_request.grad_prediction_accumulator = grad;
    backward_request.scale = 1.0f;
    EXPECT_GSX_CODE(gsx_loss_backward(loss, context, &backward_request), GSX_ERROR_INVALID_STATE);
    expect_near_vectors(download_f32_tensor(grad, 4), { 0.0f, 0.0f, 0.0f, 0.0f });

    forward_request.train = true;
    ASSERT_GSX_SUCCESS(gsx_loss_forward(loss, context, &forward_request));
    ASSERT_GSX_SUCCESS(gsx_loss_backward(loss, context, &backward_request));
    expect_near_vectors(download_f32_tensor(grad, 4), { 0.0f, 4.0f, -4.0f, 2.0f });

    ASSERT_GSX_SUCCESS(gsx_loss_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

}  // namespace
