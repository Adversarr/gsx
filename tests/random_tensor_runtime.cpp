#include "gsx/gsx.h"
#include "gsx/gsx-random.h"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
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

static gsx_tensor_t make_rank2_tensor(gsx_backend_buffer_type_t buffer_type, gsx_index_t rows, gsx_index_t cols, gsx_data_type data_type)
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
    tensor_desc.rank = 2;
    tensor_desc.shape[0] = rows;
    tensor_desc.shape[1] = cols;
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

static testing::AssertionResult floats_near(const float *expected, const float *actual, gsx_index_t count, float tol)
{
    for(gsx_index_t i = 0; i < count; ++i) {
        float diff = fabsf(expected[i] - actual[i]);
        if(diff > tol) {
            return testing::AssertionFailure()
                << "float mismatch at index " << i << ": expected=" << expected[i] << ", actual=" << actual[i]
                << ", diff=" << diff << ", tol=" << tol;
        }
    }
    return testing::AssertionSuccess();
}

template <typename T>
static std::vector<T> download_tensor_values(gsx_tensor_t tensor, gsx_index_t count)
{
    std::vector<T> values(static_cast<std::size_t>(count));
    gsx_error error = gsx_tensor_download(tensor, values.data(), values.size() * sizeof(T));

    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    return values;
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

static std::vector<float> expected_metal_kernel_rand_values(gsx_pcg32_state_t seed, gsx_index_t count)
{
    constexpr gsx_index_t values_per_thread = 4;
    std::vector<float> values(static_cast<std::size_t>(count));
    gsx_index_t thread_count = (count + values_per_thread - 1) / values_per_thread;

    for(gsx_index_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        gsx_pcg32_t rng = nullptr;

        EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
        EXPECT_EQ(gsx_pcg32_advance(rng, thread_index * values_per_thread).code, GSX_ERROR_SUCCESS);
        for(gsx_index_t j = 0; j < values_per_thread; ++j) {
            float value = 0.0f;
            gsx_index_t idx = thread_index + thread_count * j;

            if(idx >= count) {
                break;
            }
            EXPECT_EQ(gsx_pcg32_next_float(rng, &value).code, GSX_ERROR_SUCCESS);
            values[static_cast<std::size_t>(idx)] = value;
        }
        EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    }

    return values;
}

static std::vector<int32_t> expected_multinomial_values(
    gsx_pcg32_state_t seed,
    const std::vector<float> &cdf,
    gsx_index_t sample_count
)
{
    gsx_pcg32_t rng = nullptr;
    std::vector<int32_t> values(static_cast<std::size_t>(sample_count));

    EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
    for(gsx_index_t i = 0; i < sample_count; ++i) {
        float draw = 0.0f;
        auto it = cdf.begin();

        EXPECT_EQ(gsx_pcg32_next_float(rng, &draw).code, GSX_ERROR_SUCCESS);
        draw *= cdf.back();
        it = std::upper_bound(cdf.begin(), cdf.end(), draw);
        if(it == cdf.end()) {
            it = cdf.end() - 1;
        }
        values[static_cast<std::size_t>(i)] = static_cast<int32_t>(it - cdf.begin());
    }
    EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    return values;
}

static std::vector<float> expected_metal_kernel_randn_values(gsx_pcg32_state_t seed, gsx_index_t count, float sigma)
{
    constexpr gsx_index_t values_per_thread = 4;
    constexpr gsx_index_t pairs_per_thread = values_per_thread / 2;
    std::vector<float> values(static_cast<std::size_t>(count));
    gsx_index_t thread_count = (count + values_per_thread - 1) / values_per_thread;

    for(gsx_index_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        gsx_pcg32_t rng = nullptr;

        EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
        EXPECT_EQ(gsx_pcg32_advance(rng, thread_index * values_per_thread).code, GSX_ERROR_SUCCESS);
        for(gsx_index_t pair_index = 0; pair_index < pairs_per_thread; ++pair_index) {
            float u1 = 0.0f;
            float u2 = 0.0f;
            float radius = 0.0f;
            float theta = 0.0f;
            gsx_index_t idx0 = thread_index + thread_count * (2 * pair_index);
            gsx_index_t idx1 = thread_index + thread_count * (2 * pair_index + 1);

            EXPECT_EQ(gsx_pcg32_next_float(rng, &u1).code, GSX_ERROR_SUCCESS);
            EXPECT_EQ(gsx_pcg32_next_float(rng, &u2).code, GSX_ERROR_SUCCESS);
            if(u1 < 1e-7f) {
                u1 = 1e-7f;
            }
            radius = std::sqrt(-2.0f * std::log(u1));
            theta = 6.2831853071795864769f * u2;
            if(idx0 < count) {
                values[static_cast<std::size_t>(idx0)] = std::cos(theta) * radius * sigma;
            }
            if(idx1 < count) {
                values[static_cast<std::size_t>(idx1)] = std::sin(theta) * radius * sigma;
            }
        }
        EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    }

    return values;
}

static std::vector<int32_t> expected_metal_kernel_randint_values(gsx_pcg32_state_t seed, gsx_index_t count, uint32_t bound)
{
    constexpr gsx_index_t values_per_thread = 4;
    std::vector<int32_t> values(static_cast<std::size_t>(count));
    gsx_index_t thread_count = (count + values_per_thread - 1) / values_per_thread;

    for(gsx_index_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        gsx_pcg32_t rng = nullptr;

        EXPECT_EQ(gsx_pcg32_init(&rng, seed).code, GSX_ERROR_SUCCESS);
        EXPECT_EQ(gsx_pcg32_advance(rng, thread_index * values_per_thread).code, GSX_ERROR_SUCCESS);
        for(gsx_index_t j = 0; j < values_per_thread; ++j) {
            uint32_t value = 0;
            gsx_index_t idx = thread_index + thread_count * j;

            if(idx >= count) {
                break;
            }
            EXPECT_EQ(gsx_pcg32_next_uint_bound(rng, &value, bound).code, GSX_ERROR_SUCCESS);
            values[static_cast<std::size_t>(idx)] = static_cast<int32_t>(value);
        }
        EXPECT_EQ(gsx_pcg32_free(rng).code, GSX_ERROR_SUCCESS);
    }

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

TEST(RandomTensorRuntime, CpuMultinomialMatchesScalarSequenceAndAdvances)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t cdf_tensor = make_rank1_tensor(buffer_type, 4, GSX_DATA_TYPE_F32);
    gsx_tensor_t out_tensor = make_rank1_tensor(buffer_type, 7, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rng = nullptr;
    gsx_pcg32_t expected_rng = nullptr;
    bool equal = false;
    std::vector<float> cdf{ 0.2f, 0.7f, 1.0f, 3.5f };

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(cdf_tensor, cdf.data(), cdf.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 123));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&expected_rng, 123));
    ASSERT_GSX_SUCCESS(gsx_pcg32_multinomial(rng, out_tensor, cdf_tensor));
    ASSERT_GSX_SUCCESS(gsx_pcg32_advance(expected_rng, 7));
    ASSERT_GSX_SUCCESS(gsx_pcg32_equal(rng, expected_rng, &equal));
    EXPECT_TRUE(equal);
    EXPECT_EQ(download_tensor_values<int32_t>(out_tensor, 7), expected_multinomial_values(123, cdf, 7));

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(expected_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(out_tensor);
    free_tensor_and_arena(cdf_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, MultinomialRejectsInvalidInputs)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t cdf_tensor = make_rank1_tensor(buffer_type, 3, GSX_DATA_TYPE_F32);
    gsx_tensor_t out_tensor = make_rank1_tensor(buffer_type, 2, GSX_DATA_TYPE_I32);
    gsx_tensor_t bad_out_tensor = make_rank2_tensor(buffer_type, 1, 2, GSX_DATA_TYPE_I32);
    gsx_tensor_t bad_cdf_tensor = make_rank2_tensor(buffer_type, 1, 3, GSX_DATA_TYPE_F32);
    gsx_pcg32_t rng = nullptr;
    std::vector<float> bad_cdf{ 0.5f, 0.3f, 1.0f };
    std::vector<float> valid_cdf{ 0.5f, 1.0f, 1.5f };

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 99));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(cdf_tensor, bad_cdf.data(), bad_cdf.size() * sizeof(float)));
    EXPECT_GSX_CODE(gsx_pcg32_multinomial(rng, out_tensor, cdf_tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(cdf_tensor, valid_cdf.data(), valid_cdf.size() * sizeof(float)));
    EXPECT_GSX_CODE(gsx_pcg32_multinomial(rng, bad_out_tensor, cdf_tensor), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_pcg32_multinomial(rng, out_tensor, bad_cdf_tensor), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_pcg32_multinomial(rng, out_tensor, cdf_tensor), GSX_ERROR_SUCCESS);

    free_tensor_and_arena(bad_cdf_tensor);
    free_tensor_and_arena(bad_out_tensor);

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(out_tensor);
    free_tensor_and_arena(cdf_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, MetalDeviceMultinomialMatchesCpuSequence)
{
    if(!has_backend_device(GSX_BACKEND_TYPE_METAL)) {
        GTEST_SKIP() << "No Metal devices available";
    }

    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_METAL);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t cdf_tensor = make_rank1_tensor(buffer_type, 4, GSX_DATA_TYPE_F32);
    gsx_tensor_t out_tensor = make_rank1_tensor(buffer_type, 9, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rng = nullptr;
    std::vector<float> cdf{ 0.1f, 0.5f, 1.25f, 2.0f };

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(cdf_tensor, cdf.data(), cdf.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 77));
    ASSERT_GSX_SUCCESS(gsx_pcg32_multinomial(rng, out_tensor, cdf_tensor));
    EXPECT_EQ(download_tensor_values<int32_t>(out_tensor, 9), expected_multinomial_values(77, cdf, 9));

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(out_tensor);
    free_tensor_and_arena(cdf_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, CudaDeviceMultinomialMatchesCpuSequence)
{
    if(!has_backend_device(GSX_BACKEND_TYPE_CUDA)) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CUDA);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t cdf_tensor = make_rank1_tensor(buffer_type, 4, GSX_DATA_TYPE_F32);
    gsx_tensor_t out_tensor = make_rank1_tensor(buffer_type, 9, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rng = nullptr;
    std::vector<float> cdf{ 0.1f, 0.5f, 1.25f, 2.0f };

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(cdf_tensor, cdf.data(), cdf.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rng, 77));
    ASSERT_GSX_SUCCESS(gsx_pcg32_multinomial(rng, out_tensor, cdf_tensor));
    EXPECT_EQ(download_tensor_values<int32_t>(out_tensor, 9), expected_multinomial_values(77, cdf, 9));

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rng));
    free_tensor_and_arena(out_tensor);
    free_tensor_and_arena(cdf_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RandomTensorRuntime, MetalDeviceRandKernelsGenerateExpectedStridedSequence)
{
    if(!has_backend_device(GSX_BACKEND_TYPE_METAL)) {
        GTEST_SKIP() << "No Metal devices available";
    }

    gsx_backend_t metal_backend = create_backend_by_type(GSX_BACKEND_TYPE_METAL);
    gsx_backend_buffer_type_t metal_buffer_type = find_buffer_type(metal_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t metal_rand_tensor = make_rank1_tensor(metal_buffer_type, 1024, GSX_DATA_TYPE_F32);
    gsx_tensor_t metal_randn_tensor = make_rank1_tensor(metal_buffer_type, 1025, GSX_DATA_TYPE_F32);
    gsx_tensor_t metal_randint_tensor = make_rank1_tensor(metal_buffer_type, 1026, GSX_DATA_TYPE_I32);
    gsx_pcg32_t rand_rng = nullptr;
    gsx_pcg32_t randn_rng = nullptr;
    gsx_pcg32_t randint_rng = nullptr;

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&rand_rng, 11));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&randn_rng, 12));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&randint_rng, 13));

    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_rand(rand_rng, metal_rand_tensor));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randn(randn_rng, metal_randn_tensor, 0.5f));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_randint(randint_rng, metal_randint_tensor, 23));

    EXPECT_EQ(download_tensor_values<float>(metal_rand_tensor, 1024), expected_metal_kernel_rand_values(11, 1024));
    {
        std::vector<float> metal_randn = download_tensor_values<float>(metal_randn_tensor, 1025);
        std::vector<float> expected_randn = expected_metal_kernel_randn_values(12, 1025, 0.5f);
        EXPECT_TRUE(floats_near(expected_randn.data(), metal_randn.data(), 1025, 1e-5f));
    }
    EXPECT_EQ(download_tensor_values<int32_t>(metal_randint_tensor, 1026), expected_metal_kernel_randint_values(13, 1026, 23));

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(randint_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(randn_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(rand_rng));
    free_tensor_and_arena(metal_randint_tensor);
    free_tensor_and_arena(metal_randn_tensor);
    free_tensor_and_arena(metal_rand_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(metal_backend));
}

TEST(RandomTensorRuntime, MetalManagedVsDeviceRandProduceDifferentSequences)
{
    if(!has_backend_device(GSX_BACKEND_TYPE_METAL)) {
        GTEST_SKIP() << "No Metal devices available";
    }

    constexpr gsx_index_t element_count = 16;
    constexpr gsx_pcg32_state_t seed = 42;

    gsx_backend_t metal_backend = create_backend_by_type(GSX_BACKEND_TYPE_METAL);
    gsx_backend_buffer_type_t managed_buffer_type = find_buffer_type(metal_backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED);
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(metal_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t managed_tensor = make_rank1_tensor(managed_buffer_type, element_count, GSX_DATA_TYPE_F32);
    gsx_tensor_t device_tensor = make_rank1_tensor(device_buffer_type, element_count, GSX_DATA_TYPE_F32);
    gsx_pcg32_t managed_rng = nullptr;
    gsx_pcg32_t device_rng = nullptr;

    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&managed_rng, seed));
    ASSERT_GSX_SUCCESS(gsx_pcg32_init(&device_rng, seed));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_rand(managed_rng, managed_tensor));
    ASSERT_GSX_SUCCESS(gsx_pcg32_fill_rand(device_rng, device_tensor));

    std::vector<float> managed_values = download_tensor_values<float>(managed_tensor, element_count);
    std::vector<float> device_values = download_tensor_values<float>(device_tensor, element_count);

    std::vector<float> expected_managed = expected_rand_values(seed, element_count);
    std::vector<float> expected_device = expected_metal_kernel_rand_values(seed, element_count);

    EXPECT_EQ(managed_values, expected_managed) << "Managed buffer should produce sequential sequence";
    EXPECT_EQ(device_values, expected_device) << "Device buffer should produce strided sequence";
    EXPECT_NE(managed_values, device_values) << "Managed and device buffers produce different sequences with same seed";

    ASSERT_GSX_SUCCESS(gsx_pcg32_free(device_rng));
    ASSERT_GSX_SUCCESS(gsx_pcg32_free(managed_rng));
    free_tensor_and_arena(device_tensor);
    free_tensor_and_arena(managed_tensor);
    ASSERT_GSX_SUCCESS(gsx_backend_free(metal_backend));
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
