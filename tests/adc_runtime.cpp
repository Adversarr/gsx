extern "C" {
#include "../gsx/src/gsx-impl.h"
}

#include <gtest/gtest.h>

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

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_desc backend_desc{};

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device), GSX_ERROR_SUCCESS);
    backend_desc.device = backend_device;
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, type, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static gsx_adc_desc make_default_adc_desc()
{
    gsx_adc_desc desc{};

    desc.algorithm = GSX_ADC_ALGORITHM_DEFAULT;
    return desc;
}

TEST(AdcRuntime, InitRejectsInvalidArguments)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_NE(backend, nullptr);
    EXPECT_GSX_CODE(gsx_adc_init(nullptr, backend, &desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_adc_init(&adc, nullptr, &desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, InitRejectsNonDefaultAlgorithms)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_NE(backend, nullptr);
    desc.algorithm = GSX_ADC_ALGORITHM_ABSGS;
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, &desc), GSX_ERROR_NOT_SUPPORTED);
    desc.algorithm = GSX_ADC_ALGORITHM_MCMC;
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, &desc), GSX_ERROR_NOT_SUPPORTED);
    desc.algorithm = GSX_ADC_ALGORITHM_FASTGS;
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, &desc), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, InitDefaultSucceedsOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    ASSERT_NE(adc, nullptr);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, GetSetDescRoundTripDefault)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();
    gsx_adc_desc queried{};

    desc.pruning_opacity_threshold = 0.1f;
    desc.opacity_clamp_value = 0.95f;
    desc.refine_every = 16;
    desc.max_num_gaussians = 4096;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_adc_get_desc(adc, &queried));
    EXPECT_EQ(queried.algorithm, GSX_ADC_ALGORITHM_DEFAULT);
    EXPECT_FLOAT_EQ(queried.pruning_opacity_threshold, 0.1f);
    EXPECT_FLOAT_EQ(queried.opacity_clamp_value, 0.95f);
    EXPECT_EQ(queried.refine_every, 16);
    EXPECT_EQ(queried.max_num_gaussians, 4096u);

    queried.max_num_gaussians = 8192;
    queried.reset_every = 128;
    ASSERT_GSX_SUCCESS(gsx_adc_set_desc(adc, &queried));

    gsx_adc_desc queried_again{};
    ASSERT_GSX_SUCCESS(gsx_adc_get_desc(adc, &queried_again));
    EXPECT_EQ(queried_again.max_num_gaussians, 8192u);
    EXPECT_EQ(queried_again.reset_every, 128);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, SetDescRejectsNonDefault)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    desc.algorithm = GSX_ADC_ALGORITHM_FASTGS;
    EXPECT_GSX_CODE(gsx_adc_set_desc(adc, &desc), GSX_ERROR_NOT_SUPPORTED);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, StepRejectsInvalidArguments)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();
    gsx_adc_result result{};

    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    EXPECT_GSX_CODE(gsx_adc_step(adc, nullptr, &result), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_adc_step(adc, nullptr, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    gsx_adc_request request{};
    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, StepDefaultSucceedsWithGsRuntimeNoMutation)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_optim fake_optim{};
    gsx_renderer fake_renderer{};
    gsx_adc_request request{};
    gsx_adc_result result{};

    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    gs_desc.arena = arena;
    gs_desc.count = 8;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 42;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, result.gaussians_after);
    EXPECT_EQ(result.gaussians_before, 8u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.reset_count, 0u);
    EXPECT_FALSE(result.mutated);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, BackendFreeFailsWhileAdcAlive)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
