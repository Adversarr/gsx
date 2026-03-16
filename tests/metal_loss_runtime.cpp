#include "gsx/gsx.h"

#include <gtest/gtest.h>

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

static gsx_tensor_t make_f32_tensor(gsx_arena_t arena, std::initializer_list<gsx_index_t> shape, const std::vector<float> &values)
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
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
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

TEST_F(MetalLossTest, RejectsSsimAndNonDeviceLossMap)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t host_pinned_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED);
    gsx_arena_t arena = create_arena(device_buffer_type);
    gsx_arena_t host_arena = create_arena(host_pinned_buffer_type);
    gsx_loss_desc desc{};
    gsx_loss_t loss = nullptr;
    gsx_tensor_t prediction = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f });
    gsx_tensor_t target = make_f32_tensor(arena, { 2, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f });
    gsx_tensor_t loss_map = make_f32_tensor(arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_tensor_t host_loss_map = make_f32_tensor(host_arena, { 2, 2 }, { 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_loss_request request{};

    desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = nullptr;
    request.scale = 1.0f;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_NOT_SUPPORTED);

    destroy_loss(loss);
    loss = nullptr;

    desc.algorithm = GSX_LOSS_ALGORITHM_L1;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_SUM;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &desc));

    request.loss_map_accumulator = host_loss_map;
    EXPECT_GSX_CODE(evaluate_loss_once(loss, &request), GSX_ERROR_NOT_SUPPORTED);

    destroy_loss(loss);
    destroy_tensor(host_loss_map);
    destroy_tensor(loss_map);
    destroy_tensor(target);
    destroy_tensor(prediction);
    destroy_arena(host_arena);
    destroy_arena(arena);
    destroy_backend(backend);
}

} // namespace
