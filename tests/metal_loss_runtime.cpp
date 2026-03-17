#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
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

static bool has_metal_device()
{
    gsx_index_t metal_device_count = 0;
    gsx_error error = gsx_backend_registry_init();

    if(error.code != GSX_ERROR_SUCCESS) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_METAL, &metal_device_count);
    return error.code == GSX_ERROR_SUCCESS && metal_device_count > 0;
}

static gsx_backend_device_t get_metal_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device), GSX_ERROR_SUCCESS);
    return backend_device;
}

static gsx_backend_t create_metal_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};

    backend_desc.device = get_metal_backend_device();
    EXPECT_NE(backend_desc.device, nullptr);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_desc.device), GSX_ERROR_SUCCESS);
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

    arena_desc.initial_capacity_bytes = 32768;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
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

static std::vector<float> chw_to_hwc(const std::vector<float> &chw_values, gsx_index_t channels, gsx_index_t height, gsx_index_t width)
{
    std::vector<float> hwc_values((std::size_t)channels * (std::size_t)height * (std::size_t)width, 0.0f);

    for(gsx_index_t channel = 0; channel < channels; ++channel) {
        for(gsx_index_t y = 0; y < height; ++y) {
            for(gsx_index_t x = 0; x < width; ++x) {
                std::size_t chw_index =
                    (std::size_t)channel * (std::size_t)height * (std::size_t)width + (std::size_t)y * (std::size_t)width + (std::size_t)x;
                std::size_t hwc_index =
                    ((std::size_t)y * (std::size_t)width + (std::size_t)x) * (std::size_t)channels + (std::size_t)channel;

                hwc_values[hwc_index] = chw_values[chw_index];
            }
        }
    }

    return hwc_values;
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

static std::vector<float> download_f32_tensor(gsx_backend_t backend, gsx_tensor_t tensor, std::size_t element_count)
{
    std::vector<float> values(element_count);

    if(!values.empty()) {
        EXPECT_GSX_CODE(gsx_tensor_download(tensor, values.data(), (gsx_size_t)values.size() * sizeof(float)), GSX_ERROR_SUCCESS);
        EXPECT_GSX_CODE(gsx_backend_major_stream_sync(backend), GSX_ERROR_SUCCESS);
    }
    return values;
}

static std::vector<float> download_f32_tensor_no_sync(gsx_tensor_t tensor, std::size_t element_count)
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

static void sync_backend(gsx_backend_t backend)
{
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
}

static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1e-5f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index=" << i;
    }
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

class MetalLossTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if(!has_metal_device()) {
            GTEST_SKIP() << "No Metal devices available";
        }
    }
};

TEST_F(MetalLossTest, MseMatchesCpuSemantics)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, -2.0f, 3.0f, 4.0f });
    gsx_tensor_t target = make_f32_tensor(arena, { 2, 2 }, { 0.0f, -1.0f, 1.0f, 5.0f });
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 2, 2 }, { 10.0f, 20.0f, 30.0f, 40.0f });
    gsx_tensor_t grad = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f });
    gsx_loss_request request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 0.5f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    sync_backend(backend);
    expect_near_vectors(download_f32_tensor(backend, loss_map, 4), { 10.5f, 20.5f, 32.0f, 40.5f });
    expect_near_vectors(download_f32_tensor(backend, grad, 4), { 1.25f, 1.75f, 3.5f, 3.75f });

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    sync_backend(backend);
    expect_near_vectors(download_f32_tensor(backend, loss_map, 4), { 11.0f, 21.0f, 34.0f, 41.0f });
    expect_near_vectors(download_f32_tensor(backend, grad, 4), { 1.5f, 1.5f, 4.0f, 3.5f });

    destroy_loss(loss);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST_F(MetalLossTest, L1MatchesCpuSemantics)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 2.0f, -1.0f });
    gsx_tensor_t target = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 0.0f, 4.0f, -2.0f });
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_tensor_t grad = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_loss_request request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_L1;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 2.0f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    sync_backend(backend);
    expect_near_vectors(download_f32_tensor(backend, loss_map, 4), { 0.0f, 4.0f, 4.0f, 2.0f });
    expect_near_vectors(download_f32_tensor(backend, grad, 4), { 0.0f, 2.0f, -2.0f, 2.0f });

    request.grad_prediction_accumulator = nullptr;
    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    sync_backend(backend);
    expect_near_vectors(download_f32_tensor(backend, loss_map, 4), { 0.0f, 8.0f, 8.0f, 4.0f });
    expect_near_vectors(download_f32_tensor(backend, grad, 4), { 0.0f, 2.0f, -2.0f, 2.0f });

    destroy_loss(loss);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST_F(MetalLossTest, SsimSupportsMetalAndCpu)
{
    gsx_backend_t metal_backend = create_metal_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, metal_backend, &desc));
    destroy_loss(loss);
    loss = nullptr;

    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, cpu_backend, &desc));
    destroy_loss(loss);

    destroy_backend(cpu_backend);
    destroy_backend(metal_backend);
}

TEST_F(MetalLossTest, SsimRank4ChwMatchesCpuForwardAndGradient)
{
    gsx_backend_t metal_backend = create_metal_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
    gsx_backend_buffer_type_t metal_buffer_type = find_buffer_type(metal_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t cpu_buffer_type = find_buffer_type(cpu_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t metal_arena = create_arena(metal_buffer_type);
    gsx_arena_t cpu_arena = create_arena(cpu_buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t metal_loss = nullptr;
    gsx_loss_t cpu_loss = nullptr;
    std::vector<float> prediction_values((std::size_t)2 * 3 * 6 * 5, 0.0f);
    std::vector<float> target_values((std::size_t)2 * 3 * 6 * 5, 0.0f);
    gsx_tensor_t metal_prediction = nullptr;
    gsx_tensor_t metal_target = nullptr;
    gsx_tensor_t metal_loss_map = nullptr;
    gsx_tensor_t metal_grad = nullptr;
    gsx_tensor_t cpu_prediction = nullptr;
    gsx_tensor_t cpu_target = nullptr;
    gsx_tensor_t cpu_loss_map = nullptr;
    gsx_tensor_t cpu_grad = nullptr;
    gsx_loss_request metal_request{};
    gsx_loss_request cpu_request{};

    for(std::size_t i = 0; i < prediction_values.size(); ++i) {
        prediction_values[i] = 0.6f * std::sin((float)i * 0.19f) + 0.3f * std::cos((float)i * 0.07f) + 0.1f;
        target_values[i] = 0.55f * std::sin((float)i * 0.23f + 0.4f) - 0.25f * std::cos((float)i * 0.11f) + 0.05f;
    }

    metal_prediction = make_f32_tensor(metal_arena, { 2, 3, 6, 5 }, prediction_values, GSX_STORAGE_FORMAT_CHW);
    metal_target = make_f32_tensor(metal_arena, { 2, 3, 6, 5 }, target_values, GSX_STORAGE_FORMAT_CHW);
    metal_loss_map = make_f32_tensor(
        metal_arena, { 2, 3, 6, 5 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_CHW);
    metal_grad = make_f32_tensor(
        metal_arena, { 2, 3, 6, 5 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_CHW);
    cpu_prediction = make_f32_tensor(cpu_arena, { 2, 3, 6, 5 }, prediction_values, GSX_STORAGE_FORMAT_CHW);
    cpu_target = make_f32_tensor(cpu_arena, { 2, 3, 6, 5 }, target_values, GSX_STORAGE_FORMAT_CHW);
    cpu_loss_map = make_f32_tensor(
        cpu_arena, { 2, 3, 6, 5 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_CHW);
    cpu_grad = make_f32_tensor(
        cpu_arena, { 2, 3, 6, 5 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_CHW);

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&metal_loss, metal_backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_init(&cpu_loss, cpu_backend, &desc));

    metal_request.prediction = metal_prediction;
    metal_request.target = metal_target;
    metal_request.loss_map_accumulator = metal_loss_map;
    metal_request.grad_prediction_accumulator = metal_grad;
    metal_request.scale = 0.8f;
    cpu_request.prediction = cpu_prediction;
    cpu_request.target = cpu_target;
    cpu_request.loss_map_accumulator = cpu_loss_map;
    cpu_request.grad_prediction_accumulator = cpu_grad;
    cpu_request.scale = 0.8f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(metal_loss, &metal_request));
    ASSERT_GSX_SUCCESS(evaluate_loss_once(cpu_loss, &cpu_request));
    sync_backend(metal_backend);
    expect_near_vectors(
        download_f32_tensor(metal_backend, metal_loss_map, prediction_values.size()),
        download_f32_tensor_no_sync(cpu_loss_map, prediction_values.size()),
        2e-3f);
    expect_near_vectors(
        download_f32_tensor(metal_backend, metal_grad, prediction_values.size()),
        download_f32_tensor_no_sync(cpu_grad, prediction_values.size()),
        3e-3f);

    destroy_loss(cpu_loss);
    destroy_loss(metal_loss);
    destroy_tensor(cpu_grad);
    destroy_tensor(cpu_loss_map);
    destroy_tensor(cpu_target);
    destroy_tensor(cpu_prediction);
    destroy_tensor(metal_grad);
    destroy_tensor(metal_loss_map);
    destroy_tensor(metal_target);
    destroy_tensor(metal_prediction);
    destroy_arena(cpu_arena);
    destroy_arena(metal_arena);
    destroy_backend(cpu_backend);
    destroy_backend(metal_backend);
}

TEST_F(MetalLossTest, SsimRank4HwcMatchesCpuForwardAndGradient)
{
    gsx_backend_t metal_backend = create_metal_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
    gsx_backend_buffer_type_t metal_buffer_type = find_buffer_type(metal_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t cpu_buffer_type = find_buffer_type(cpu_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t metal_arena = create_arena(metal_buffer_type);
    gsx_arena_t cpu_arena = create_arena(cpu_buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t metal_loss = nullptr;
    gsx_loss_t cpu_loss = nullptr;
    std::vector<float> prediction_values((std::size_t)2 * 6 * 5 * 3, 0.0f);
    std::vector<float> target_values((std::size_t)2 * 6 * 5 * 3, 0.0f);
    gsx_tensor_t metal_prediction = nullptr;
    gsx_tensor_t metal_target = nullptr;
    gsx_tensor_t metal_loss_map = nullptr;
    gsx_tensor_t metal_grad = nullptr;
    gsx_tensor_t cpu_prediction = nullptr;
    gsx_tensor_t cpu_target = nullptr;
    gsx_tensor_t cpu_loss_map = nullptr;
    gsx_tensor_t cpu_grad = nullptr;
    gsx_loss_request metal_request{};
    gsx_loss_request cpu_request{};

    for(std::size_t i = 0; i < prediction_values.size(); ++i) {
        prediction_values[i] = 0.45f * std::sin((float)i * 0.13f) + 0.35f * std::cos((float)i * 0.05f) + 0.2f;
        target_values[i] = 0.5f * std::sin((float)i * 0.17f + 0.2f) - 0.3f * std::cos((float)i * 0.09f) - 0.05f;
    }

    metal_prediction = make_f32_tensor(metal_arena, { 2, 6, 5, 3 }, prediction_values, GSX_STORAGE_FORMAT_HWC);
    metal_target = make_f32_tensor(metal_arena, { 2, 6, 5, 3 }, target_values, GSX_STORAGE_FORMAT_HWC);
    metal_loss_map = make_f32_tensor(
        metal_arena, { 2, 6, 5, 3 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_HWC);
    metal_grad = make_f32_tensor(
        metal_arena, { 2, 6, 5, 3 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_HWC);
    cpu_prediction = make_f32_tensor(cpu_arena, { 2, 6, 5, 3 }, prediction_values, GSX_STORAGE_FORMAT_HWC);
    cpu_target = make_f32_tensor(cpu_arena, { 2, 6, 5, 3 }, target_values, GSX_STORAGE_FORMAT_HWC);
    cpu_loss_map = make_f32_tensor(
        cpu_arena, { 2, 6, 5, 3 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_HWC);
    cpu_grad = make_f32_tensor(
        cpu_arena, { 2, 6, 5, 3 }, std::vector<float>(prediction_values.size(), 0.0f), GSX_STORAGE_FORMAT_HWC);

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&metal_loss, metal_backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_init(&cpu_loss, cpu_backend, &desc));

    metal_request.prediction = metal_prediction;
    metal_request.target = metal_target;
    metal_request.loss_map_accumulator = metal_loss_map;
    metal_request.grad_prediction_accumulator = metal_grad;
    metal_request.scale = 0.6f;
    cpu_request.prediction = cpu_prediction;
    cpu_request.target = cpu_target;
    cpu_request.loss_map_accumulator = cpu_loss_map;
    cpu_request.grad_prediction_accumulator = cpu_grad;
    cpu_request.scale = 0.6f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(metal_loss, &metal_request));
    ASSERT_GSX_SUCCESS(evaluate_loss_once(cpu_loss, &cpu_request));
    sync_backend(metal_backend);
    expect_near_vectors(
        download_f32_tensor(metal_backend, metal_loss_map, prediction_values.size()),
        download_f32_tensor_no_sync(cpu_loss_map, prediction_values.size()),
        2e-3f);
    expect_near_vectors(
        download_f32_tensor(metal_backend, metal_grad, prediction_values.size()),
        download_f32_tensor_no_sync(cpu_grad, prediction_values.size()),
        3e-3f);

    destroy_loss(cpu_loss);
    destroy_loss(metal_loss);
    destroy_tensor(cpu_grad);
    destroy_tensor(cpu_loss_map);
    destroy_tensor(cpu_target);
    destroy_tensor(cpu_prediction);
    destroy_tensor(metal_grad);
    destroy_tensor(metal_loss_map);
    destroy_tensor(metal_target);
    destroy_tensor(metal_prediction);
    destroy_arena(cpu_arena);
    destroy_arena(metal_arena);
    destroy_backend(cpu_backend);
    destroy_backend(metal_backend);
}

TEST_F(MetalLossTest, SsimRejectsUnsupportedLayoutsAndBuffers)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t host_pinned_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED);
    gsx_arena_t arena = create_arena(device_buffer_type);
    gsx_arena_t host_arena = create_arena(host_pinned_buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.5f));
    gsx_tensor_t target = make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.5f));
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.0f));
    gsx_tensor_t grad = make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.0f));
    gsx_tensor_t tiled_prediction =
        make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.5f), GSX_STORAGE_FORMAT_TILED_CHW);
    gsx_tensor_t tiled_target =
        make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.5f), GSX_STORAGE_FORMAT_TILED_CHW);
    gsx_tensor_t tiled_loss_map =
        make_f32_tensor(arena, { 3, 4, 4 }, std::vector<float>(48, 0.0f), GSX_STORAGE_FORMAT_TILED_CHW);
    gsx_tensor_t host_loss_map = make_f32_tensor(host_arena, { 3, 4, 4 }, std::vector<float>(48, 0.0f));
    gsx_loss_request request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = tiled_prediction;
    request.target = tiled_target;
    request.loss_map_accumulator = tiled_loss_map;
    request.grad_prediction_accumulator = nullptr;
    request.scale = 1.0f;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_NOT_SUPPORTED);

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = host_loss_map;
    request.grad_prediction_accumulator = grad;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_NOT_SUPPORTED);

    request.loss_map_accumulator = loss_map;
    destroy_tensor(grad);
    grad = make_f32_tensor(arena, { 48 }, std::vector<float>(48, 0.0f));
    request.grad_prediction_accumulator = grad;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_INVALID_ARGUMENT);

    destroy_loss(loss);
    destroy_tensor(host_loss_map);
    destroy_tensor(tiled_loss_map);
    destroy_tensor(tiled_target);
    destroy_tensor(tiled_prediction);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(host_arena);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST_F(MetalLossTest, SsimProducesNearZeroLossForIdenticalImages)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    std::vector<float> image(3 * 8 * 8, 0.5f);
    gsx_tensor_t prediction = make_f32_tensor(arena, { 3, 8, 8 }, image);
    gsx_tensor_t target = make_f32_tensor(arena, { 3, 8, 8 }, image);
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 3, 8, 8 }, std::vector<float>(image.size(), 0.0f));
    gsx_tensor_t grad = make_f32_tensor(arena, { 3, 8, 8 }, std::vector<float>(image.size(), 0.0f));
    gsx_loss_request request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 1.0f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(loss, &request));
    sync_backend(backend);
    for(float value : download_f32_tensor(backend, loss_map, image.size())) {
        EXPECT_NEAR(value, 0.0f, 2e-5f);
    }
    for(float value : download_f32_tensor(backend, grad, image.size())) {
        EXPECT_NEAR(value, 0.0f, 2e-5f);
    }

    destroy_loss(loss);
    destroy_tensor(grad);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(arena);
    destroy_backend(backend);
}

TEST_F(MetalLossTest, SsimHwcMatchesChwForSameLogicalImage)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t chw_loss = nullptr;
    gsx_loss_t hwc_loss = nullptr;
    std::vector<float> prediction_chw(3 * 8 * 8, 0.4f);
    std::vector<float> target_chw(3 * 8 * 8, 0.35f);
    std::vector<float> prediction_hwc;
    std::vector<float> target_hwc;
    gsx_tensor_t prediction_chw_tensor = nullptr;
    gsx_tensor_t target_chw_tensor = nullptr;
    gsx_tensor_t loss_map_chw_tensor = nullptr;
    gsx_tensor_t grad_chw_tensor = nullptr;
    gsx_tensor_t prediction_hwc_tensor = nullptr;
    gsx_tensor_t target_hwc_tensor = nullptr;
    gsx_tensor_t loss_map_hwc_tensor = nullptr;
    gsx_tensor_t grad_hwc_tensor = nullptr;
    gsx_loss_request chw_request{};
    gsx_loss_request hwc_request{};

    prediction_chw[0] = 0.2f;
    prediction_chw[17] = 0.7f;
    prediction_chw[63] = 0.9f;
    prediction_chw[91] = 0.1f;
    target_chw[0] = 0.4f;
    target_chw[17] = 0.1f;
    target_chw[63] = 0.5f;
    target_chw[91] = 0.8f;
    prediction_hwc = chw_to_hwc(prediction_chw, 3, 8, 8);
    target_hwc = chw_to_hwc(target_chw, 3, 8, 8);

    prediction_chw_tensor = make_f32_tensor(arena, { 3, 8, 8 }, prediction_chw, GSX_STORAGE_FORMAT_CHW);
    target_chw_tensor = make_f32_tensor(arena, { 3, 8, 8 }, target_chw, GSX_STORAGE_FORMAT_CHW);
    loss_map_chw_tensor = make_f32_tensor(
        arena, { 3, 8, 8 }, std::vector<float>(prediction_chw.size(), 0.0f), GSX_STORAGE_FORMAT_CHW);
    grad_chw_tensor = make_f32_tensor(
        arena, { 3, 8, 8 }, std::vector<float>(prediction_chw.size(), 0.0f), GSX_STORAGE_FORMAT_CHW);
    prediction_hwc_tensor = make_f32_tensor(arena, { 8, 8, 3 }, prediction_hwc, GSX_STORAGE_FORMAT_HWC);
    target_hwc_tensor = make_f32_tensor(arena, { 8, 8, 3 }, target_hwc, GSX_STORAGE_FORMAT_HWC);
    loss_map_hwc_tensor = make_f32_tensor(
        arena, { 8, 8, 3 }, std::vector<float>(prediction_hwc.size(), 0.0f), GSX_STORAGE_FORMAT_HWC);
    grad_hwc_tensor = make_f32_tensor(
        arena, { 8, 8, 3 }, std::vector<float>(prediction_hwc.size(), 0.0f), GSX_STORAGE_FORMAT_HWC);

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&chw_loss, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_loss_init(&hwc_loss, backend, &desc));

    chw_request.prediction = prediction_chw_tensor;
    chw_request.target = target_chw_tensor;
    chw_request.loss_map_accumulator = loss_map_chw_tensor;
    chw_request.grad_prediction_accumulator = grad_chw_tensor;
    chw_request.scale = 0.75f;
    hwc_request.prediction = prediction_hwc_tensor;
    hwc_request.target = target_hwc_tensor;
    hwc_request.loss_map_accumulator = loss_map_hwc_tensor;
    hwc_request.grad_prediction_accumulator = grad_hwc_tensor;
    hwc_request.scale = 0.75f;

    ASSERT_GSX_SUCCESS(evaluate_loss_once(chw_loss, &chw_request));
    ASSERT_GSX_SUCCESS(evaluate_loss_once(hwc_loss, &hwc_request));
    sync_backend(backend);
    expect_near_vectors(
        download_f32_tensor(backend, loss_map_hwc_tensor, prediction_hwc.size()),
        chw_to_hwc(download_f32_tensor(backend, loss_map_chw_tensor, prediction_chw.size()), 3, 8, 8),
        5e-4f);
    expect_near_vectors(
        download_f32_tensor(backend, grad_hwc_tensor, prediction_hwc.size()),
        chw_to_hwc(download_f32_tensor(backend, grad_chw_tensor, prediction_chw.size()), 3, 8, 8),
        5e-4f);

    destroy_loss(hwc_loss);
    destroy_loss(chw_loss);
    destroy_tensor(grad_hwc_tensor);
    destroy_tensor(loss_map_hwc_tensor);
    destroy_tensor(target_hwc_tensor);
    destroy_tensor(prediction_hwc_tensor);
    destroy_tensor(grad_chw_tensor);
    destroy_tensor(loss_map_chw_tensor);
    destroy_tensor(target_chw_tensor);
    destroy_tensor(prediction_chw_tensor);
    destroy_arena(arena);
    destroy_backend(backend);
}

} // namespace
