#include "gsx/gsx.h"
#include "gsx/gsx-random.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
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

static gsx_backend_t create_backend_by_type(gsx_backend_type backend_type)
{
    gsx_backend_t backend = nullptr;
    gsx_backend_device_t device = nullptr;
    gsx_backend_desc desc{};
    gsx_error error{};

    error = gsx_backend_registry_init();
    if(!gsx_error_is_success(error)) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    error = gsx_get_backend_device_by_type(backend_type, 0, &device);
    if(!gsx_error_is_success(error)) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    desc.device = device;
    error = gsx_backend_init(&backend, &desc);
    if(!gsx_error_is_success(error)) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return backend;
}

static bool has_backend_device(gsx_backend_type backend_type)
{
    gsx_index_t device_count = 0;
    gsx_error error = gsx_backend_registry_init();

    if(!gsx_error_is_success(error)) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(backend_type, &device_count);
    return gsx_error_is_success(error) && device_count > 0;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_error error{};

    error = gsx_backend_find_buffer_type(backend, type, &buffer_type);
    if(!gsx_error_is_success(error)) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return buffer_type;
}

static gsx_tensor_t make_rank1_tensor(gsx_backend_buffer_type_t buffer_type, gsx_index_t length, gsx_data_type data_type)
{
    gsx_arena_t arena = nullptr;
    gsx_tensor_t tensor = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_desc tensor_desc{};
    gsx_error error{};

    error = gsx_arena_init(&arena, buffer_type, &arena_desc);
    if(!gsx_error_is_success(error)) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    tensor_desc.rank = 1;
    tensor_desc.shape[0] = length;
    tensor_desc.data_type = data_type;
    tensor_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    tensor_desc.arena = arena;
    error = gsx_tensor_init(&tensor, &tensor_desc);
    if(!gsx_error_is_success(error)) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        (void)gsx_arena_free(arena);
        return nullptr;
    }
    return tensor;
}

static void free_tensor_and_arena(gsx_tensor_t tensor)
{
    gsx_arena_t arena = nullptr;
    gsx_tensor_desc desc{};

    ASSERT_NE(tensor, nullptr);
    ASSERT_GSX_SUCCESS(gsx_tensor_get_desc(tensor, &desc));
    arena = desc.arena;
    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
}

static std::vector<float> expected_rand_values(gsx_pcg32_state_t seed, gsx_index_t count)
{
    gsx_pcg32_t rng = nullptr;
    std::vector<float> values(static_cast<std::size_t>(count));

    EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
    for(gsx_index_t i = 0; i < count; ++i) {
        EXPECT_EQ(gsx_pcg32_next_float(rng, &values[static_cast<std::size_t>(i)]).code, GSX_ERROR_SUCCESS);
    }
    EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    return values;
}

static std::vector<float> expected_randn_values(gsx_pcg32_state_t seed, gsx_index_t count, float sigma)
{
    gsx_pcg32_t rng = nullptr;
    std::vector<float> values(static_cast<std::size_t>(count));

    EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
    for(gsx_index_t i = 0; i < count; i += 2) {
        float u1 = 0.0f;
        float u2 = 0.0f;
        float radius = 0.0f;
        float theta = 0.0f;

        EXPECT_EQ(gsx_pcg32_next_float(rng, &u1).code, GSX_ERROR_SUCCESS);
        EXPECT_EQ(gsx_pcg32_next_float(rng, &u2).code, GSX_ERROR_SUCCESS);
        if(u1 < 1e-7f) {
            u1 = 1e-7f;
        }
        radius = std::sqrt(-2.0f * std::log(u1));
        theta = 6.2831853071795864769f * u2;
        values[static_cast<std::size_t>(i)] = std::cos(theta) * radius * sigma;
        if(i + 1 < count) {
            values[static_cast<std::size_t>(i + 1)] = std::sin(theta) * radius * sigma;
        }
    }
    EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    return values;
}

static std::vector<int32_t> expected_randint_values(gsx_pcg32_state_t seed, gsx_index_t count, uint32_t bound)
{
    gsx_pcg32_t rng = nullptr;
    std::vector<int32_t> values(static_cast<std::size_t>(count));

    EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
    for(gsx_index_t i = 0; i < count; ++i) {
        uint32_t value = 0;

        EXPECT_EQ(gsx_pcg32_next_uint_bound(rng, &value, bound).code, GSX_ERROR_SUCCESS);
        values[static_cast<std::size_t>(i)] = static_cast<int32_t>(value);
    }
    EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    return values;
}

TEST(RandomTensorRuntime, CpuFillRandMatchesScalarSequence)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t tensor = make_rank1_tensor(buffer_type, 5, GSX_DATA_TYPE_F32);
    gsx_pcg32_t rng = nullptr;
    std::vector<float> values(5);
    std::vector<float> expected = expected_rand_values(1234, 5);

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 1234));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_rand(rng, tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, values.data(), values.size() * sizeof(float)));
    EXPECT_EQ(values, expected);

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, CpuFillRandnMatchesScalarSequenceAndAdvancesByPairs)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t tensor = make_rank1_tensor(buffer_type, 5, GSX_DATA_TYPE_F32);
    gsx_pcg32_t rng = nullptr;
    gsx_pcg32_t expected_rng = nullptr;
    bool equal = false;
    std::vector<float> values(5);
    std::vector<float> expected = expected_randn_values(55, 5, 0.25f);

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 55));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&expected_rng, 55));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randn(rng, tensor, 0.25f));
    ASSERT_GSX_SUCCESS(gsx_pcg32_advance(expected_rng, 6));
    ASSERT_GSX_SUCCESS(gsx_pcg32_equal(rng, expected_rng, &equal));
    EXPECT_TRUE(equal);
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, values.data(), values.size() * sizeof(float)));
    EXPECT_EQ(values, expected);

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(expected_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, CpuFillRandintMatchesScalarSequence)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t tensor = make_rank1_tensor(buffer_type, 8, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rng = nullptr;
    std::vector<int32_t> values(8);
    std::vector<int32_t> expected = expected_randint_values(9, 8, 17);

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 9));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randint(rng, tensor, 17));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, values.data(), values.size() * sizeof(int32_t)));
    EXPECT_EQ(values, expected);

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, InvalidTensorTypeAndParametersAreRejected)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t float_tensor = make_rank1_tensor(buffer_type, 4, GSX_DATA_TYPE_F32);
    gsx_tensor_t int_tensor = make_rank1_tensor(buffer_type, 4, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rng = nullptr;

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 77));
    EXPECT_GSX_CODE(gsx_pcg32_fill_randint(rng, float_tensor, 8), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_pcg32_fill_rand(rng, int_tensor), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_pcg32_fill_randn(rng, float_tensor, -1.0f), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_pcg32_fill_randint(rng, int_tensor, 0), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(int_tensor);
    free_tensor_and_arena(float_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, MetalDeviceFillOperationsMatchCpuSequence)
{
    if(!has_backend_device(GSX_BACKEND_TYPE_METAL)) {
        GTEST_SKIP() << "No Metal devices available";
    }

    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_METAL);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t rand_tensor = make_rank1_tensor(buffer_type, 4, GSX_DATA_TYPE_F32);
    gsx_tensor_t randn_tensor = make_rank1_tensor(buffer_type, 5, GSX_DATA_TYPE_F32);
    gsx_tensor_t randint_tensor = make_rank1_tensor(buffer_type, 6, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rand_rng = nullptr;
    gsx_pcg32_t randn_rng = nullptr;
    gsx_pcg32_t randint_rng = nullptr;
    std::vector<float> rand_values(4);
    std::vector<float> randn_values(5);
    std::vector<int32_t> randint_values(6);

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rand_rng, 11));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&randn_rng, 12));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&randint_rng, 13));

    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_rand(rand_rng, rand_tensor));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randn(randn_rng, randn_tensor, 0.5f));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randint(randint_rng, randint_tensor, 23));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(rand_tensor, rand_values.data(), rand_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(randn_tensor, randn_values.data(), randn_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(randint_tensor, randint_values.data(), randint_values.size() * sizeof(int32_t)));

    EXPECT_EQ(rand_values, expected_rand_values(11, 4));
    EXPECT_EQ(randn_values, expected_randn_values(12, 5, 0.5f));
    EXPECT_EQ(randint_values, expected_randint_values(13, 6, 23));

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(randint_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(randn_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rand_rng));
    free_tensor_and_arena(randint_tensor);
    free_tensor_and_arena(randn_tensor);
    free_tensor_and_arena(rand_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, CudaDeviceFillOperationsMatchCpuSequence)
{
    if(!has_backend_device(GSX_BACKEND_TYPE_CUDA)) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CUDA);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t rand_tensor = make_rank1_tensor(buffer_type, 256, GSX_DATA_TYPE_F32);
    gsx_tensor_t randn_tensor = make_rank1_tensor(buffer_type, 257, GSX_DATA_TYPE_F32);
    gsx_pcg32_t rand_rng = nullptr;
    gsx_pcg32_t randn_rng = nullptr;
    std::vector<float> rand_values(256);
    std::vector<float> randn_values(257);

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rand_rng, 21));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&randn_rng, 22));

    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_rand(rand_rng, rand_tensor));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randn(randn_rng, randn_tensor, 0.5f));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(rand_tensor, rand_values.data(), rand_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(randn_tensor, randn_values.data(), randn_values.size() * sizeof(float)));

    auto expected_rand = expected_rand_values(21, 256);
    auto expected_randn = expected_randn_values(22, 257, 0.5f);
    for(size_t i = 0; i < rand_values.size(); ++i) {
        EXPECT_NEAR(rand_values[i], expected_rand[i], 1e-6f) << "index=" << i;
    }
    for(size_t i = 0; i < randn_values.size(); ++i) {
        EXPECT_NEAR(randn_values[i], expected_randn[i], 1e-6f) << "index=" << i;
    }

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(randn_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rand_rng));
    free_tensor_and_arena(randn_tensor);
    free_tensor_and_arena(rand_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
