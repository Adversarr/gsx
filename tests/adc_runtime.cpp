extern "C" {
#include "../gsx/src/gsx-impl.h"
}

#include <gtest/gtest.h>

#include <algorithm>
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

static gsx_size_t get_gs_count(gsx_gs_t gs)
{
    gsx_gs_info info{};

    EXPECT_GSX_CODE(gsx_gs_get_info(gs, &info), GSX_ERROR_SUCCESS);
    return info.count;
}

static void expect_gs_fields_finite(gsx_gs_t gs)
{
    gsx_gs_finite_check_result finite{};

    ASSERT_GSX_SUCCESS(gsx_gs_check_finite(gs, &finite));
    EXPECT_TRUE(finite.is_finite);
}

struct AdamRefGroup {
    std::vector<float> params;
    std::vector<float> grads;
    std::vector<float> m;
    std::vector<float> v;
    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float weight_decay = 0.0f;
    float epsilon = 0.0f;
    float max_grad = 0.0f;
    gsx_size_t step = 0;
};

struct CpuOptimLayout {
    gsx_optim base;
    gsx_size_t *step_counts = nullptr;
    gsx_tensor_t *first_moments = nullptr;
    gsx_tensor_t *second_moments = nullptr;
    gsx_tensor_t *scratch_first_moments = nullptr;
    gsx_tensor_t *scratch_second_moments = nullptr;
    gsx_arena_t state_arena = nullptr;
    gsx_arena_t scratch_arena = nullptr;
};

static void init_mean3d_optimizer(gsx_backend_t backend, gsx_gs_t gs, gsx_optim_t *out_optim)
{
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t grad_mean3d = nullptr;
    gsx_optim_param_group_desc group{};
    gsx_optim_desc desc{};

    ASSERT_NE(out_optim, nullptr);
    *out_optim = nullptr;

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &grad_mean3d));
    group.role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    group.parameter = mean3d;
    group.gradient = grad_mean3d;
    group.learning_rate = 0.01f;
    group.beta1 = 0.9f;
    group.beta2 = 0.99f;
    group.weight_decay = 0.0f;
    group.epsilon = 1e-6f;
    group.max_grad = 0.0f;
    desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    desc.param_groups = &group;
    desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(out_optim, backend, &desc));
}

static std::vector<float> download_tensor_f32(gsx_tensor_t tensor)
{
    std::vector<float> values;

    EXPECT_NE(tensor, nullptr);
    if(tensor == nullptr) {
        return values;
    }
    EXPECT_EQ(tensor->data_type, GSX_DATA_TYPE_F32);
    if(tensor->data_type != GSX_DATA_TYPE_F32) {
        return values;
    }
    values.resize(tensor->size_bytes / sizeof(float));
    if(!values.empty()) {
        EXPECT_GSX_CODE(gsx_tensor_download(tensor, values.data(), tensor->size_bytes), GSX_ERROR_SUCCESS);
    }
    return values;
}

static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1e-5f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index=" << i;
    }
}

static void adam_step(AdamRefGroup *group)
{
    ASSERT_NE(group, nullptr);
    group->step += 1;

    {
        double beta1_correction = 1.0 - std::pow((double)group->beta1, (double)group->step);
        double beta2_correction = 1.0 - std::pow((double)group->beta2, (double)group->step);
        for(std::size_t i = 0; i < group->params.size(); ++i) {
            float first_moment = group->beta1 * group->m[i] + (1.0f - group->beta1) * group->grads[i];
            float second_moment = group->beta2 * group->v[i] + (1.0f - group->beta2) * group->grads[i] * group->grads[i];
            float parameter = group->params[i];

            group->m[i] = first_moment;
            group->v[i] = second_moment;
            parameter -=
                group->learning_rate
                * ((float)((double)first_moment / beta1_correction)
                    / (std::sqrt((float)((double)second_moment / beta2_correction)) + group->epsilon));
            group->params[i] = parameter;
        }
    }
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

TEST(AdcRuntime, RefinePruneOnlyByOpacity)
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
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 100.0f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.4f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 0.0f, 0.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
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
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_OPACITY,
        {
            logitf(0.9f),
            logitf(0.1f),
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
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.pruned_count, 1u);
    EXPECT_EQ(result.reset_count, 0u);
    EXPECT_TRUE(result.mutated);

    std::vector<float> opacity = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    ASSERT_EQ(opacity.size(), 1u);
    EXPECT_GT(sigmoidf(opacity[0]), 0.4f);

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

TEST(AdcRuntime, RefineSplitOnlyDisplacementOpacityAndScale)
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
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f, 4.0f, 1.0f, 2.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
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
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.7f) });

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
    EXPECT_NEAR(mean_after[3], mean_before[3], 1e-6f);
    EXPECT_NEAR(mean_after[4], mean_before[4], 1e-6f);
    EXPECT_NEAR(mean_after[5], mean_before[5], 1e-6f);
    EXPECT_NEAR(logscale_after[3], logscale_before[3], 1e-6f);
    EXPECT_NEAR(logscale_after[4], logscale_before[4], 1e-6f);
    EXPECT_NEAR(logscale_after[5], logscale_before[5], 1e-6f);
    EXPECT_NEAR(opacity_after[1], opacity_before[1], 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, RefineDuplicateOnlyExactCopyNoDisplacementOrOpacityScaleChange)
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
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 3.0f, -2.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(0.25f), std::log(0.25f), std::log(0.25f),
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.73f) });

    std::vector<float> mean_before = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    std::vector<float> logscale_before = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    std::vector<float> rotation_before = download_gs_field_f32(gs, GSX_GS_FIELD_ROTATION);
    std::vector<float> opacity_before = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 1u);
    EXPECT_EQ(result.gaussians_after, 2u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_TRUE(result.mutated);

    std::vector<float> mean_after = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    std::vector<float> logscale_after = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    std::vector<float> rotation_after = download_gs_field_f32(gs, GSX_GS_FIELD_ROTATION);
    std::vector<float> opacity_after = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);

    ASSERT_EQ(mean_after.size(), 6u);
    ASSERT_EQ(logscale_after.size(), 6u);
    ASSERT_EQ(rotation_after.size(), 8u);
    ASSERT_EQ(opacity_after.size(), 2u);
    EXPECT_NEAR(mean_after[0], mean_before[0], 1e-6f);
    EXPECT_NEAR(mean_after[1], mean_before[1], 1e-6f);
    EXPECT_NEAR(mean_after[2], mean_before[2], 1e-6f);
    EXPECT_NEAR(mean_after[3], mean_before[0], 1e-6f);
    EXPECT_NEAR(mean_after[4], mean_before[1], 1e-6f);
    EXPECT_NEAR(mean_after[5], mean_before[2], 1e-6f);
    EXPECT_NEAR(logscale_after[0], logscale_before[0], 1e-6f);
    EXPECT_NEAR(logscale_after[1], logscale_before[1], 1e-6f);
    EXPECT_NEAR(logscale_after[2], logscale_before[2], 1e-6f);
    EXPECT_NEAR(logscale_after[3], logscale_before[0], 1e-6f);
    EXPECT_NEAR(logscale_after[4], logscale_before[1], 1e-6f);
    EXPECT_NEAR(logscale_after[5], logscale_before[2], 1e-6f);
    EXPECT_NEAR(rotation_after[0], rotation_before[0], 1e-6f);
    EXPECT_NEAR(rotation_after[1], rotation_before[1], 1e-6f);
    EXPECT_NEAR(rotation_after[2], rotation_before[2], 1e-6f);
    EXPECT_NEAR(rotation_after[3], rotation_before[3], 1e-6f);
    EXPECT_NEAR(rotation_after[4], rotation_before[0], 1e-6f);
    EXPECT_NEAR(rotation_after[5], rotation_before[1], 1e-6f);
    EXPECT_NEAR(rotation_after[6], rotation_before[2], 1e-6f);
    EXPECT_NEAR(rotation_after[7], rotation_before[3], 1e-6f);
    EXPECT_NEAR(opacity_after[0], opacity_before[0], 1e-6f);
    EXPECT_NEAR(opacity_after[1], opacity_before[0], 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, RefineStressTriggersPruneSplitDuplicateAndStaysFinite)
{
    static const gsx_size_t kInitialCount = 500;
    static const gsx_size_t kMaxCount = 600;
    static const gsx_size_t kStepCount = 128;

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
    std::vector<float> mean3d(kInitialCount * 3u);
    std::vector<float> grad_acc(kInitialCount);
    std::vector<float> logscale(kInitialCount * 3u);
    std::vector<float> rotation(kInitialCount * 4u);
    std::vector<float> opacity(kInitialCount);
    gsx_size_t total_pruned = 0;
    gsx_size_t total_split = 0;
    gsx_size_t total_duplicate = 0;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10000;
    desc.max_num_gaussians = (gsx_index_t)kMaxCount;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.10f;
    desc.max_world_scale = 0.0f;
    desc.reset_every = 2;
    desc.seed = 1337;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 26;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = kInitialCount;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    for(gsx_size_t i = 0; i < kInitialCount; ++i) {
        const gsx_size_t off3 = i * 3u;
        const gsx_size_t off4 = i * 4u;
        const float scale = ((i % 6u) == 0u) ? 1.1f : 0.2f;
        const float opacity_prob = ((i % 5u) == 0u) ? 0.02f : 0.88f;

        mean3d[off3 + 0u] = -1.0f + 0.004f * (float)i;
        mean3d[off3 + 1u] = -0.5f + 0.002f * (float)(i % 97u);
        mean3d[off3 + 2u] = 2.0f + 0.001f * (float)(i % 31u);
        grad_acc[i] = ((i % 2u) == 0u) ? 1.2f : 0.1f;
        logscale[off3 + 0u] = std::log(scale);
        logscale[off3 + 1u] = std::log(scale);
        logscale[off3 + 2u] = std::log(scale);
        rotation[off4 + 0u] = 0.0f;
        rotation[off4 + 1u] = 0.0f;
        rotation[off4 + 2u] = 0.0f;
        rotation[off4 + 3u] = 1.0f;
        opacity[i] = logitf(opacity_prob);
    }

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, mean3d);
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, grad_acc);
    upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, logscale);
    upload_gs_field_f32(gs, GSX_GS_FIELD_ROTATION, rotation);
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, opacity);

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.scene_scale = 1.0f;

    for(gsx_size_t step = 1; step <= kStepCount; ++step) {
        const gsx_size_t count_before = get_gs_count(gs);

        request.global_step = step;
        ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
        EXPECT_EQ(result.gaussians_before, count_before);
        EXPECT_GE(result.gaussians_after, 1u);
        EXPECT_LE(result.gaussians_after, kMaxCount);
        total_pruned += result.pruned_count;
        total_split += result.grown_count;
        total_duplicate += result.duplicated_count;
        if((step % 8u) == 0u || step == kStepCount) {
            expect_gs_fields_finite(gs);
        }
    }

    EXPECT_GT(total_pruned, 0u);
    EXPECT_GT(total_split, 0u);
    EXPECT_GT(total_duplicate, 0u);
    expect_gs_fields_finite(gs);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, RefineSplitDuplicateThresholdBoundary)
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
    desc.seed = 42;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 1.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(1.0001f), std::log(1.0001f), std::log(1.0001f),
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
    std::vector<float> opacity_before = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 2.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 4u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.grown_count, 1u);
    EXPECT_EQ(result.pruned_count, 0u);

    std::vector<float> opacity_after = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    std::vector<float> logscale_after = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    ASSERT_EQ(opacity_after.size(), 4u);
    ASSERT_EQ(logscale_after.size(), 12u);
    EXPECT_NEAR(opacity_after[0], opacity_before[0], 1e-6f);
    EXPECT_NEAR(opacity_after[2], opacity_before[0], 1e-6f);
    EXPECT_LT(sigmoidf(opacity_after[1]), sigmoidf(opacity_before[1]));
    EXPECT_LT(sigmoidf(opacity_after[3]), sigmoidf(opacity_before[1]));
    EXPECT_NEAR(logscale_after[0], std::log(1.0f), 1e-6f);
    EXPECT_NEAR(logscale_after[6], std::log(1.0f), 1e-6f);
    EXPECT_NEAR(logscale_after[3], std::log(1.0001f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[9], std::log(1.0001f / 1.6f), 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, RefinePruneSplitDuplicateInSingleStep)
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
    desc.max_num_gaussians = 5;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.1f;
    desc.seed = 7;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 3;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, -3.0f, 1.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 1.0f, 0.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.85f), logitf(0.85f), logitf(0.02f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 3u);
    EXPECT_EQ(result.gaussians_after, 4u);
    EXPECT_EQ(result.grown_count, 1u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.pruned_count, 1u);
    EXPECT_TRUE(result.mutated);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, PrunesByMaxWorldScaleWhenPruneLargeIsEnabled)
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
    std::vector<float> logscale_after;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 100.0f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.max_world_scale = 0.5f;
    desc.reset_every = 1;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 0.0f, 0.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
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
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.9f), logitf(0.9f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 2;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.pruned_count, 1u);

    logscale_after = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    ASSERT_EQ(logscale_after.size(), 3u);
    EXPECT_NEAR(std::exp(logscale_after[0]), 0.2f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, PrunesByMaxScreenScaleWithAuxField)
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
    std::vector<float> max_screen_after;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 100.0f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.max_screen_scale = 0.5f;
    desc.reset_every = 1;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_MAX_SCREEN_RADIUS;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 0.0f, 0.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.9f), logitf(0.9f) });
    upload_gs_field_f32(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, { 1.0f, 0.1f });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 2;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.pruned_count, 1u);

    max_screen_after = download_gs_field_f32(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS);
    ASSERT_EQ(max_screen_after.size(), 1u);
    EXPECT_NEAR(max_screen_after[0], 0.1f, 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, PrunesDegenerateQuaternion)
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
    std::vector<float> rotation_after;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 100.0f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.reset_every = 100;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 0.0f, 0.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.9f), logitf(0.9f) });

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
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.pruned_count, 1u);

    rotation_after = download_gs_field_f32(gs, GSX_GS_FIELD_ROTATION);
    ASSERT_EQ(rotation_after.size(), 4u);
    EXPECT_NEAR(rotation_after[3], 1.0f, 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, PruneLargeGateActivatesOnlyAfterResetEvery)
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
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 100.0f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.max_world_scale = 0.5f;
    desc.reset_every = 5;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 0.0f, 0.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
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
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.9f), logitf(0.9f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.scene_scale = 1.0f;

    request.global_step = 5;
    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_EQ(result.gaussians_after, 2u);

    request.global_step = 6;
    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.pruned_count, 1u);
    EXPECT_EQ(result.gaussians_after, 1u);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, SplitAndDuplicateVerifyAllCopiedFields)
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
    std::vector<float> mean_after;
    std::vector<float> logscale_before;
    std::vector<float> logscale_after;
    std::vector<float> opacity_before;
    std::vector<float> opacity_after;
    std::vector<float> rotation_before;
    std::vector<float> rotation_after;
    std::vector<float> sh0_before;
    std::vector<float> sh0_after;
    std::vector<float> sh1_before;
    std::vector<float> sh1_after;
    std::vector<float> sh2_before;
    std::vector<float> sh2_after;
    std::vector<float> sh3_before;
    std::vector<float> sh3_after;
    const float split_expected_prob = 1.0f - std::sqrt(1.0f - 0.81f);

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.seed = 2026;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_SH1 | GSX_GS_AUX_SH2 | GSX_GS_AUX_SH3;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f, 3.0f, 1.0f, 2.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 1.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(1.0f), std::log(1.0f), std::log(1.0f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
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
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.81f), logitf(0.36f) });
    upload_gs_field_f32(gs, GSX_GS_FIELD_SH0, { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_SH1,
        {
            100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f,
            200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f, 206.0f, 207.0f, 208.0f,
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_SH2,
        {
            300.0f, 301.0f, 302.0f, 303.0f, 304.0f,
            305.0f, 306.0f, 307.0f, 308.0f, 309.0f,
            310.0f, 311.0f, 312.0f, 313.0f, 314.0f,
            400.0f, 401.0f, 402.0f, 403.0f, 404.0f,
            405.0f, 406.0f, 407.0f, 408.0f, 409.0f,
            410.0f, 411.0f, 412.0f, 413.0f, 414.0f,
        }
    );
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_SH3,
        {
            500.0f, 501.0f, 502.0f, 503.0f, 504.0f, 505.0f, 506.0f,
            507.0f, 508.0f, 509.0f, 510.0f, 511.0f, 512.0f, 513.0f,
            514.0f, 515.0f, 516.0f, 517.0f, 518.0f, 519.0f, 520.0f,
            600.0f, 601.0f, 602.0f, 603.0f, 604.0f, 605.0f, 606.0f,
            607.0f, 608.0f, 609.0f, 610.0f, 611.0f, 612.0f, 613.0f,
            614.0f, 615.0f, 616.0f, 617.0f, 618.0f, 619.0f, 620.0f,
        }
    );

    mean_before = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    logscale_before = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    opacity_before = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    rotation_before = download_gs_field_f32(gs, GSX_GS_FIELD_ROTATION);
    sh0_before = download_gs_field_f32(gs, GSX_GS_FIELD_SH0);
    sh1_before = download_gs_field_f32(gs, GSX_GS_FIELD_SH1);
    sh2_before = download_gs_field_f32(gs, GSX_GS_FIELD_SH2);
    sh3_before = download_gs_field_f32(gs, GSX_GS_FIELD_SH3);

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
    EXPECT_EQ(result.gaussians_after, 4u);
    EXPECT_EQ(result.grown_count, 1u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.pruned_count, 0u);

    mean_after = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    logscale_after = download_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    opacity_after = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    rotation_after = download_gs_field_f32(gs, GSX_GS_FIELD_ROTATION);
    sh0_after = download_gs_field_f32(gs, GSX_GS_FIELD_SH0);
    sh1_after = download_gs_field_f32(gs, GSX_GS_FIELD_SH1);
    sh2_after = download_gs_field_f32(gs, GSX_GS_FIELD_SH2);
    sh3_after = download_gs_field_f32(gs, GSX_GS_FIELD_SH3);

    ASSERT_EQ(mean_after.size(), 12u);
    ASSERT_EQ(logscale_after.size(), 12u);
    ASSERT_EQ(opacity_after.size(), 4u);
    ASSERT_EQ(rotation_after.size(), 16u);
    ASSERT_EQ(sh0_after.size(), 12u);
    ASSERT_EQ(sh1_after.size(), 36u);
    ASSERT_EQ(sh2_after.size(), 60u);
    ASSERT_EQ(sh3_after.size(), 84u);

    EXPECT_NE(mean_after[0], mean_before[0]);
    EXPECT_NE(mean_after[1], mean_before[1]);
    EXPECT_NE(mean_after[2], mean_before[2]);
    EXPECT_NE(mean_after[6], mean_before[0]);
    EXPECT_NE(mean_after[7], mean_before[1]);
    EXPECT_NE(mean_after[8], mean_before[2]);
    EXPECT_NE(mean_after[0], mean_after[6]);
    EXPECT_NE(mean_after[1], mean_after[7]);
    EXPECT_NE(mean_after[2], mean_after[8]);
    EXPECT_NEAR(logscale_after[0], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[1], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[2], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[6], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[7], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(logscale_after[8], std::log(1.0f / 1.6f), 1e-5f);
    EXPECT_NEAR(sigmoidf(opacity_after[0]), split_expected_prob, 1e-5f);
    EXPECT_NEAR(sigmoidf(opacity_after[2]), split_expected_prob, 1e-5f);

    EXPECT_NEAR(mean_after[9], mean_before[3], 1e-6f);
    EXPECT_NEAR(mean_after[10], mean_before[4], 1e-6f);
    EXPECT_NEAR(mean_after[11], mean_before[5], 1e-6f);
    EXPECT_NEAR(logscale_after[9], logscale_before[3], 1e-6f);
    EXPECT_NEAR(logscale_after[10], logscale_before[4], 1e-6f);
    EXPECT_NEAR(logscale_after[11], logscale_before[5], 1e-6f);
    EXPECT_NEAR(opacity_after[3], opacity_before[1], 1e-6f);
    EXPECT_NEAR(rotation_after[12], rotation_before[4], 1e-6f);
    EXPECT_NEAR(rotation_after[13], rotation_before[5], 1e-6f);
    EXPECT_NEAR(rotation_after[14], rotation_before[6], 1e-6f);
    EXPECT_NEAR(rotation_after[15], rotation_before[7], 1e-6f);
    for(gsx_size_t i = 0; i < 3u; ++i) {
        EXPECT_NEAR(sh0_after[9u + i], sh0_before[3u + i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < 9u; ++i) {
        EXPECT_NEAR(sh1_after[27u + i], sh1_before[9u + i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < 15u; ++i) {
        EXPECT_NEAR(sh2_after[45u + i], sh2_before[15u + i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < 21u; ++i) {
        EXPECT_NEAR(sh3_after[63u + i], sh3_before[21u + i], 1e-6f);
    }

    for(gsx_size_t i = 0; i < 3u; ++i) {
        EXPECT_NEAR(sh0_after[6u + i], sh0_before[i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < 9u; ++i) {
        EXPECT_NEAR(sh1_after[18u + i], sh1_before[i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < 15u; ++i) {
        EXPECT_NEAR(sh2_after[30u + i], sh2_before[i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < 21u; ++i) {
        EXPECT_NEAR(sh3_after[42u + i], sh3_before[i], 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, VisibleCounterSuppressesGrowthWhenCounterIsNonPositive)
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
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 3.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER, { 0.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_ROTATION, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, { 0.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 1u);
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, VisibleCounterNormalizesGradientForGrowthDecision)
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
    desc.max_num_gaussians = 3;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER, { 4.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_ROTATION, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, { 0.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 1u);
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, MissingVisibleCounterDefaultsToOneAndAllowsGrowth)
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
    desc.max_num_gaussians = 3;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_ROTATION, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, { 0.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_before, 1u);
    EXPECT_EQ(result.gaussians_after, 2u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.grown_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, GrowthZeroInitializesNewOptimizerRows)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_optim_t optim = nullptr;
    gsx_renderer fake_renderer{};
    gsx_adc_request request{};
    gsx_adc_result result{};
    CpuOptimLayout *cpu_optim = nullptr;
    AdamRefGroup ref{};
    std::vector<float> first_moments_after_growth;
    std::vector<float> second_moments_after_growth;
    std::vector<float> mean_after_growth;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 3;
    desc.duplicate_grad_threshold = 0.2f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    init_mean3d_optimizer(backend, gs, &optim);

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_MEAN3D, { 0.3f, -0.2f, 0.1f, -0.4f, 0.5f, -0.6f });
    ref.params = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    ref.grads = { 0.3f, -0.2f, 0.1f, -0.4f, 0.5f, -0.6f };
    ref.m.assign(6, 0.0f);
    ref.v.assign(6, 0.0f);
    ref.learning_rate = 0.01f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(0.4f), std::log(0.4f), std::log(0.4f),
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
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.9f), logitf(0.8f) });

    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_after, 3u);
    EXPECT_EQ(result.duplicated_count, 1u);
    cpu_optim = reinterpret_cast<CpuOptimLayout *>(optim);
    ASSERT_NE(cpu_optim, nullptr);
    first_moments_after_growth = download_tensor_f32(cpu_optim->first_moments[0]);
    second_moments_after_growth = download_tensor_f32(cpu_optim->second_moments[0]);
    ASSERT_EQ(first_moments_after_growth.size(), 9u);
    ASSERT_EQ(second_moments_after_growth.size(), 9u);
    EXPECT_NEAR(first_moments_after_growth[6], 0.0f, 1e-7f);
    EXPECT_NEAR(first_moments_after_growth[7], 0.0f, 1e-7f);
    EXPECT_NEAR(first_moments_after_growth[8], 0.0f, 1e-7f);
    EXPECT_NEAR(second_moments_after_growth[6], 0.0f, 1e-7f);
    EXPECT_NEAR(second_moments_after_growth[7], 0.0f, 1e-7f);
    EXPECT_NEAR(second_moments_after_growth[8], 0.0f, 1e-7f);

    mean_after_growth = download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    ASSERT_EQ(mean_after_growth.size(), 9u);
    ref.params = mean_after_growth;
    ref.grads = { 0.2f, -0.1f, 0.4f, -0.3f, 0.6f, -0.2f, 0.5f, 0.25f, -0.75f };
    ref.m.insert(ref.m.end(), 3, 0.0f);
    ref.v.insert(ref.v.end(), 3, 0.0f);
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_MEAN3D, ref.grads);

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D), ref.params);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(AdcRuntime, RefineResetsAuxStatistics)
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
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER | GSX_GS_AUX_MAX_SCREEN_RADIUS;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 3.0f, -2.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER, { 4.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, { 2.5f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, { std::log(0.25f), std::log(0.25f), std::log(0.25f) });
    upload_gs_field_f32(gs, GSX_GS_FIELD_ROTATION, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.73f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 1;
    request.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc, &request, &result));
    EXPECT_EQ(result.gaussians_after, 1u);
    EXPECT_EQ(result.duplicated_count, 0u);
    expect_near_vectors(download_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC), { 0.0f }, 1e-7f);
    expect_near_vectors(download_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER), { 0.0f }, 1e-7f);
    expect_near_vectors(download_gs_field_f32(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS), { 0.0f }, 1e-7f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
