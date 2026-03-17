extern "C" {
#include "../gsx/src/gsx-impl.h"
}

#include <gtest/gtest.h>

#include <cmath>
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

static float sigmoidf(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static float logitf(float p)
{
    const float eps = 1e-6f;
    float clamped = p;

    if(clamped < eps) {
        clamped = eps;
    }
    if(clamped > 1.0f - eps) {
        clamped = 1.0f - eps;
    }
    return std::log(clamped / (1.0f - clamped));
}

static void upload_gs_field_f32(gsx_gs_t gs, gsx_gs_field field, const std::vector<float> &values)
{
    gsx_tensor_t tensor = nullptr;
    gsx_size_t expected_bytes = values.size() * sizeof(float);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, field, &tensor));
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(tensor->data_type, GSX_DATA_TYPE_F32);
    ASSERT_EQ(tensor->size_bytes, expected_bytes);
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, values.data(), expected_bytes));
}

static std::vector<float> download_gs_field_f32(gsx_gs_t gs, gsx_gs_field field)
{
    gsx_tensor_t tensor = nullptr;

    EXPECT_GSX_CODE(gsx_gs_get_field(gs, field, &tensor), GSX_ERROR_SUCCESS);
    if(tensor == nullptr || tensor->data_type != GSX_DATA_TYPE_F32) {
        return {};
    }
    std::vector<float> values(tensor->size_bytes / sizeof(float));
    if(!values.empty()) {
        EXPECT_GSX_CODE(gsx_tensor_download(tensor, values.data(), tensor->size_bytes), GSX_ERROR_SUCCESS);
    }
    return values;
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

TEST(AdcRuntime, InitRejectsInvalidDescriptorRanges)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    desc.opacity_clamp_value = 1.5f;
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, &desc), GSX_ERROR_OUT_OF_RANGE);

    desc = make_default_adc_desc();
    desc.start_refine = 10;
    desc.end_refine = 3;
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, &desc), GSX_ERROR_OUT_OF_RANGE);

    desc = make_default_adc_desc();
    desc.max_num_gaussians = -1;
    EXPECT_GSX_CODE(gsx_adc_init(&adc, backend, &desc), GSX_ERROR_OUT_OF_RANGE);

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
    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
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
    request.scene_scale = 0.0f;
    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);
    request.scene_scale = 1.0f;

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

TEST(AdcRuntime, StepDefaultRefineDuplicatesAndPrunes)
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

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 8;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.4f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 4;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f, 0.9f, 0.2f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_OPACITY,
        {
            logitf(0.9f),
            logitf(0.1f),
            logitf(0.8f),
            logitf(0.2f),
        }
    );

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 4u);
    EXPECT_EQ(result.gaussians_after, 4u);
    EXPECT_EQ(result.duplicated_count, 2u);
    EXPECT_EQ(result.pruned_count, 2u);
    EXPECT_EQ(result.reset_count, 0u);
    EXPECT_TRUE(result.mutated);

    std::vector<float> opacity = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    ASSERT_EQ(opacity.size(), 4u);
    EXPECT_GT(sigmoidf(opacity[0]), 0.4f);
    EXPECT_GT(sigmoidf(opacity[1]), 0.4f);
    EXPECT_GT(sigmoidf(opacity[2]), 0.4f);
    EXPECT_GT(sigmoidf(opacity[3]), 0.4f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, StepDefaultResetClampsOpacity)
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

    desc.reset_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.pruning_opacity_threshold = 0.25f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 3;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { 2.0f, -1.0f, 3.0f });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 2;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 3u);
    EXPECT_EQ(result.gaussians_after, 3u);
    EXPECT_EQ(result.reset_count, 1u);
    EXPECT_TRUE(result.mutated);

    std::vector<float> opacity = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    ASSERT_EQ(opacity.size(), 3u);
    EXPECT_LE(opacity[0], logitf(0.5f) + 1e-6f);
    EXPECT_LE(opacity[1], logitf(0.5f) + 1e-6f);
    EXPECT_LE(opacity[2], logitf(0.5f) + 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, StepDefaultRefineSplitsWhenScaleIsLarge)
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
    std::vector<float> mean_before;
    std::vector<float> logscale_before;
    std::vector<float> opacity_before;
    std::vector<float> mean_after;
    std::vector<float> logscale_after;
    std::vector<float> opacity_after;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.seed = 1234;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.5f), std::log(0.5f), std::log(0.5f),
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.8f) });

    mean_before = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    logscale_before = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    opacity_before = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 3u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 1u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_TRUE(result.mutated);

    mean_after = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    logscale_after = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    opacity_after = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);

    ASSERT_EQ(mean_after.size(), 9u);
    ASSERT_EQ(logscale_after.size(), 9u);
    ASSERT_EQ(opacity_after.size(), 3u);
    EXPECT_NE(mean_after[0], mean_before[0]);
    EXPECT_NE(mean_after[1], mean_before[1]);
    EXPECT_NE(mean_after[2], mean_before[2]);
    EXPECT_NE(mean_after[6], mean_before[0]);
    EXPECT_NE(mean_after[7], mean_before[1]);
    EXPECT_NE(mean_after[8], mean_before[2]);
    EXPECT_NEAR(logscale_after[0], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[1], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[2], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[6], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[7], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[8], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_LT(sigmoidf(opacity_after[0]), sigmoidf(opacity_before[0]));
    EXPECT_LT(sigmoidf(opacity_after[2]), sigmoidf(opacity_before[0]));

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, StepDefaultRefineSceneScaleAffectsGrowDecision)
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

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.4f), std::log(0.4f), std::log(0.4f),
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.8f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 3.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 3u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_TRUE(result.mutated);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
