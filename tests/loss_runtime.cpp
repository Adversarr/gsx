#include "gsx/gsx.h"

#include <gtest/gtest.h>

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

static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1e-6f)
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
    EXPECT_GSX_CODE(gsx_loss_init(&loss, backend, &desc), GSX_ERROR_NOT_SUPPORTED);
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

    ASSERT_GSX_SUCCESS(gsx_loss_evaluate(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 10.5f, 20.5f, 32.0f, 40.5f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 1.25f, 1.75f, 3.5f, 3.75f });

    ASSERT_GSX_SUCCESS(gsx_loss_evaluate(loss, &request));
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

    ASSERT_GSX_SUCCESS(gsx_loss_evaluate(loss, &request));
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 0.0f, 4.0f, 4.0f, 2.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 0.0f, 2.0f, -2.0f, 2.0f });

    request.grad_prediction_accumulator = nullptr;
    ASSERT_GSX_SUCCESS(gsx_loss_evaluate(loss, &request));
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

    ASSERT_GSX_SUCCESS(gsx_loss_evaluate(mean_loss, &mean_request));
    ASSERT_GSX_SUCCESS(gsx_loss_evaluate(sum_loss, &sum_request));

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

TEST(LossRuntime, EvaluateRejectsInvalidRequestsWithoutWritingAccumulators)
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
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.loss_map_accumulator = bad_loss_map;
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = bad_grad;
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });

    request.grad_prediction_accumulator = grad;
    request.target = bad_target;
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.target = target;
    request.loss_map_accumulator = prediction;
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(prediction, 4), { 1.0f, 2.0f, 3.0f, 4.0f });
    expect_near_vectors(download_f32_tensor(grad, 4), { 5.0f, 4.0f, 3.0f, 2.0f });

    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = prediction;
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
    expect_near_vectors(download_f32_tensor(prediction, 4), { 1.0f, 2.0f, 3.0f, 4.0f });

    request.grad_prediction_accumulator = grad;
    request.scale = std::numeric_limits<float>::infinity();
    EXPECT_GSX_CODE(gsx_loss_evaluate(loss, &request), GSX_ERROR_INVALID_ARGUMENT);
    expect_near_vectors(download_f32_tensor(loss_map, 4), { 9.0f, 8.0f, 7.0f, 6.0f });
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

}  // namespace
