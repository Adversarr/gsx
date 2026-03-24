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

static bool has_metal_device()
{
    gsx_index_t cuda_device_count = 0;
    gsx_error error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_METAL, &cuda_device_count);
    return error.code == GSX_ERROR_SUCCESS && cuda_device_count > 0;
}

static gsx_backend_t create_metal_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_desc backend_desc{};

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device), GSX_ERROR_SUCCESS);
    backend_desc.device = backend_device;
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

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

static float sigmoidf(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
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
    float epsilon = 0.0f;
    gsx_size_t step = 0;
};

struct MetalOptimLayout {
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

static std::vector<float> download_tensor_f32(gsx_backend_t backend, gsx_tensor_t tensor)
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
        EXPECT_GSX_CODE(gsx_backend_major_stream_sync(backend), GSX_ERROR_SUCCESS);
    }
    return values;
}

static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 2e-4f)
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

static void compute_dry_run_gs_required_bytes(
    gsx_backend_buffer_type_t buffer_type, gsx_size_t count, gsx_gs_aux_flags aux_flags, gsx_size_t *out_required_bytes)
{
    gsx_gs_t dry_run_gs = nullptr;
    gsx_gs_desc dry_run_desc{};
    gsx_gs_info dry_run_info{};

    ASSERT_NE(out_required_bytes, nullptr);
    *out_required_bytes = 0;

    dry_run_desc.buffer_type = buffer_type;
    dry_run_desc.arena_desc.initial_capacity_bytes = 0;
    dry_run_desc.arena_desc.dry_run = true;
    dry_run_desc.count = (gsx_index_t)count;
    dry_run_desc.aux_flags = aux_flags;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&dry_run_gs, &dry_run_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(dry_run_gs, &dry_run_info));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(dry_run_info.arena, out_required_bytes));
    ASSERT_GSX_SUCCESS(gsx_gs_free(dry_run_gs));
}

class MetalAdcRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if(!has_metal_device()) {
            GTEST_SKIP() << "No Metal devices available";
        }
    }
};

TEST_F(MetalAdcRuntimeTest, InitDefaultSucceedsOnMetal)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    ASSERT_NE(adc, nullptr);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, BackendFreeFailsWhileAdcAlive)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepRejectsOptimizerBackendMismatch)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
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
    gs_desc.count = 4;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    fake_optim.backend = cpu_backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 2;
    request.scene_scale = 1.0f;

    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(cpu_backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepRejectsRendererBackendMismatch)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
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
    gs_desc.count = 4;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    fake_optim.backend = backend;
    fake_renderer.backend = cpu_backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 2;
    request.scene_scale = 1.0f;

    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(cpu_backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepRejectsNonPositiveSceneScale)
{
    gsx_backend_t backend = create_metal_backend();
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
    gs_desc.count = 4;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request.gs = gs;
    request.optim = &fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &fake_renderer;
    request.global_step = 2;
    request.scene_scale = 0.0f;
    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);
    request.scene_scale = -1.0f;
    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepDefaultSucceedsWithGsRuntimeNoMutation)
{
    gsx_backend_t backend = create_metal_backend();
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
    EXPECT_EQ(result.gaussians_before, 8u);
    EXPECT_EQ(result.gaussians_after, 8u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.reset_count, 0u);
    EXPECT_FALSE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 8u);
    expect_gs_fields_finite(gs);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepDefaultRefineDuplicatesAndPrunes)
{
    gsx_backend_t backend = create_metal_backend();
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
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.pruned_count, 2u);
    EXPECT_EQ(result.reset_count, 0u);
    EXPECT_TRUE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 4u);
    expect_gs_fields_finite(gs);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepDefaultRefineSucceedsWhenGsArenaCoversSingleMaxLayout)
{
    gsx_backend_t backend = create_metal_backend();
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
    gsx_size_t single_max_layout_bytes = 0;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 8;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.4f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    compute_dry_run_gs_required_bytes(buffer_type, 8u, GSX_GS_AUX_GRAD_ACC, &single_max_layout_bytes);
    arena_desc.initial_capacity_bytes = single_max_layout_bytes;
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
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.pruned_count, 2u);
    EXPECT_TRUE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 4u);
    expect_gs_fields_finite(gs);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepDefaultResetClampsOpacity)
{
    gsx_backend_t backend = create_metal_backend();
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
    desc.opacity_clamp_value = 0.5f;
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
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.reset_count, 1u);
    EXPECT_TRUE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 3u);
    expect_gs_fields_finite(gs);

    std::vector<float> opacity = download_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    ASSERT_EQ(opacity.size(), 3u);
    EXPECT_LE(sigmoidf(opacity[0]), 0.5f + 1e-6f);
    EXPECT_LE(sigmoidf(opacity[1]), 0.5f + 1e-6f);
    EXPECT_LE(sigmoidf(opacity[2]), 0.5f + 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepWithLiveOptimizerSupportsGrowthGather)
{
    gsx_backend_t backend = create_metal_backend();
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
    gsx_optim_info optim_info{};

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
    EXPECT_EQ(result.gaussians_before, 2u);
    EXPECT_EQ(result.gaussians_after, 3u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_EQ(result.duplicated_count, 1u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_TRUE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 3u);
    expect_gs_fields_finite(gs);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &optim_info));
    EXPECT_EQ(optim_info.param_group_count, 1);
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, StepAcceptsRank3ShAuxFields)
{
    gsx_backend_t backend = create_metal_backend();
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
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT | GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 0.1f, 0.1f });
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
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(0.5f), std::log(0.5f), std::log(0.5f),
            std::log(0.5f), std::log(0.5f), std::log(0.5f),
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.7f) });

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
    EXPECT_EQ(result.gaussians_after, 2u);
    EXPECT_EQ(result.pruned_count, 0u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_EQ(result.reset_count, 0u);
    EXPECT_FALSE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 2u);
    expect_gs_fields_finite(gs);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, RefineSplitDuplicateThresholdBoundary)
{
    gsx_backend_t backend = create_metal_backend();
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

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, PruneLargeGateActivatesOnlyAfterResetEvery)
{
    gsx_backend_t backend = create_metal_backend();
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

TEST_F(MetalAdcRuntimeTest, VisibleCounterNormalizesGradientForGrowthDecision)
{
    gsx_backend_t backend = create_metal_backend();
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

TEST_F(MetalAdcRuntimeTest, DuplicateAndPruneStayDeterministicAcrossIdenticalRuns)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_adc_t adc_a = nullptr;
    gsx_adc_t adc_b = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();
    gsx_arena_t arena_a = nullptr;
    gsx_arena_t arena_b = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs_a = nullptr;
    gsx_gs_t gs_b = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_optim fake_optim{};
    gsx_renderer fake_renderer{};
    gsx_adc_request req_a{};
    gsx_adc_request req_b{};
    gsx_adc_result result_a{};
    gsx_adc_result result_b{};
    std::vector<float> mean_a;
    std::vector<float> mean_b;
    std::vector<float> opacity_a;
    std::vector<float> opacity_b;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.2f;
    desc.reset_every = 0;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc_a, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc_b, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena_a, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena_b, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 3;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs_a, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs_b, &gs_desc));

    upload_gs_field_f32(gs_a, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs_a, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f, 1.0f });
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 0.1f, 1.0f });
    upload_gs_field_f32(
        gs_a,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
        }
    );
    upload_gs_field_f32(
        gs_b,
        GSX_GS_FIELD_LOGSCALE,
        {
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
            std::log(0.2f), std::log(0.2f), std::log(0.2f),
        }
    );
    upload_gs_field_f32(
        gs_a,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(
        gs_b,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs_a, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.1f), logitf(0.9f) });
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.1f), logitf(0.9f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;

    req_a.gs = gs_a;
    req_a.optim = &fake_optim;
    req_a.dataloader = (gsx_dataloader_t)0x1;
    req_a.renderer = &fake_renderer;
    req_a.global_step = 1;
    req_a.scene_scale = 1.0f;

    req_b.gs = gs_b;
    req_b.optim = &fake_optim;
    req_b.dataloader = (gsx_dataloader_t)0x1;
    req_b.renderer = &fake_renderer;
    req_b.global_step = 1;
    req_b.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc_a, &req_a, &result_a));
    ASSERT_GSX_SUCCESS(gsx_adc_step(adc_b, &req_b, &result_b));

    EXPECT_EQ(result_a.gaussians_before, result_b.gaussians_before);
    EXPECT_EQ(result_a.gaussians_after, result_b.gaussians_after);
    EXPECT_EQ(result_a.duplicated_count, result_b.duplicated_count);
    EXPECT_EQ(result_a.grown_count, result_b.grown_count);
    EXPECT_EQ(result_a.pruned_count, result_b.pruned_count);
    EXPECT_EQ(result_a.reset_count, result_b.reset_count);
    EXPECT_EQ(result_a.mutated, result_b.mutated);

    mean_a = download_gs_field_f32(gs_a, GSX_GS_FIELD_MEAN3D);
    mean_b = download_gs_field_f32(gs_b, GSX_GS_FIELD_MEAN3D);
    opacity_a = download_gs_field_f32(gs_a, GSX_GS_FIELD_OPACITY);
    opacity_b = download_gs_field_f32(gs_b, GSX_GS_FIELD_OPACITY);
    ASSERT_EQ(mean_a.size(), mean_b.size());
    ASSERT_EQ(opacity_a.size(), opacity_b.size());
    for(gsx_size_t i = 0; i < mean_a.size(); ++i) {
        EXPECT_NEAR(mean_a[i], mean_b[i], 1e-6f);
    }
    for(gsx_size_t i = 0; i < opacity_a.size(); ++i) {
        EXPECT_NEAR(opacity_a[i], opacity_b[i], 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc_a));
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc_b));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs_a));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs_b));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena_a));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena_b));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, ConsecutiveSplitStepsAdvanceRngState)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_adc_t adc_a = nullptr;
    gsx_adc_t adc_b = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();
    gsx_arena_t arena_a = nullptr;
    gsx_arena_t arena_b = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs_a = nullptr;
    gsx_gs_t gs_b = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_optim fake_optim{};
    gsx_renderer fake_renderer{};
    gsx_adc_request request_a{};
    gsx_adc_request request_b{};
    gsx_adc_result result_a{};
    gsx_adc_result result_b{};
    std::vector<float> mean_after_first;
    std::vector<float> logscale_after_first;
    std::vector<float> opacity_after_first;
    std::vector<float> rotation_after_first;
    std::vector<float> mean_after_second_a;
    std::vector<float> mean_after_second_b;
    bool found_difference = false;

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 100;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.seed = 1234;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc_a, backend, &desc));
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc_b, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena_a, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena_b, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs_a, &gs_desc));

    upload_gs_field_f32(gs_a, GSX_GS_FIELD_MEAN3D, { 0.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs_a, GSX_GS_FIELD_GRAD_ACC, { 1.0f });
    upload_gs_field_f32(gs_a, GSX_GS_FIELD_LOGSCALE, { std::log(1.0f), std::log(1.0f), std::log(1.0f) });
    upload_gs_field_f32(gs_a, GSX_GS_FIELD_ROTATION, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_gs_field_f32(gs_a, GSX_GS_FIELD_OPACITY, { logitf(0.8f) });

    fake_optim.backend = backend;
    fake_renderer.backend = backend;
    request_a.gs = gs_a;
    request_a.optim = &fake_optim;
    request_a.dataloader = (gsx_dataloader_t)0x1;
    request_a.renderer = &fake_renderer;
    request_a.global_step = 1;
    request_a.scene_scale = 1.0f;

    ASSERT_GSX_SUCCESS(gsx_adc_step(adc_a, &request_a, &result_a));
    EXPECT_EQ(result_a.grown_count, 1u);
    EXPECT_EQ(result_a.gaussians_after, 2u);

    mean_after_first = download_gs_field_f32(gs_a, GSX_GS_FIELD_MEAN3D);
    logscale_after_first = download_gs_field_f32(gs_a, GSX_GS_FIELD_LOGSCALE);
    opacity_after_first = download_gs_field_f32(gs_a, GSX_GS_FIELD_OPACITY);
    rotation_after_first = download_gs_field_f32(gs_a, GSX_GS_FIELD_ROTATION);

    gs_desc.count = 2;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs_b, &gs_desc));
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_MEAN3D, mean_after_first);
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_LOGSCALE, logscale_after_first);
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_OPACITY, opacity_after_first);
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_ROTATION, rotation_after_first);

    upload_gs_field_f32(gs_a, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 1.0f });
    upload_gs_field_f32(gs_b, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 1.0f });

    request_a.global_step = 2;
    ASSERT_GSX_SUCCESS(gsx_adc_step(adc_a, &request_a, &result_a));
    EXPECT_EQ(result_a.grown_count, 2u);
    EXPECT_EQ(result_a.gaussians_after, 4u);

    request_b.gs = gs_b;
    request_b.optim = &fake_optim;
    request_b.dataloader = (gsx_dataloader_t)0x1;
    request_b.renderer = &fake_renderer;
    request_b.global_step = 2;
    request_b.scene_scale = 1.0f;
    ASSERT_GSX_SUCCESS(gsx_adc_step(adc_b, &request_b, &result_b));
    EXPECT_EQ(result_b.grown_count, 2u);
    EXPECT_EQ(result_b.gaussians_after, 4u);

    mean_after_second_a = download_gs_field_f32(gs_a, GSX_GS_FIELD_MEAN3D);
    mean_after_second_b = download_gs_field_f32(gs_b, GSX_GS_FIELD_MEAN3D);
    ASSERT_EQ(mean_after_second_a.size(), mean_after_second_b.size());
    for(std::size_t i = 0; i < mean_after_second_a.size(); ++i) {
        if(std::fabs(mean_after_second_a[i] - mean_after_second_b[i]) > 1e-5f) {
            found_difference = true;
            break;
        }
    }
    EXPECT_TRUE(found_difference);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc_a));
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc_b));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs_a));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs_b));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena_a));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena_b));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, VisibleCounterSuppressesGrowthWhenCounterIsNonPositive)
{
    gsx_backend_t backend = create_metal_backend();
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
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));

    arena_desc.initial_capacity_bytes = 1U << 20;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, { 1.0f, 1.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER, { 0.0f, -1.0f });
    upload_gs_field_f32(
        gs,
        GSX_GS_FIELD_ROTATION,
        {
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        }
    );
    upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { logitf(0.8f), logitf(0.8f) });

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
    EXPECT_EQ(result.gaussians_after, 2u);
    EXPECT_EQ(result.duplicated_count, 0u);
    EXPECT_EQ(result.grown_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, MissingVisibleCounterDefaultsToOneAndAllowsGrowth)
{
    gsx_backend_t backend = create_metal_backend();
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

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, PrunesByMaxScreenScaleWithAuxField)
{
    gsx_backend_t backend = create_metal_backend();
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

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalAdcRuntimeTest, RefineSplitOnlyDisplacementOpacityAndScale)
{
    gsx_backend_t backend = create_metal_backend();
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

TEST_F(MetalAdcRuntimeTest, RefineDuplicateOnlyExactCopyNoDisplacementOrOpacityScaleChange)
{
    gsx_backend_t backend = create_metal_backend();
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

TEST_F(MetalAdcRuntimeTest, GrowthZeroInitializesNewOptimizerRows)
{
    gsx_backend_t backend = create_metal_backend();
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
    MetalOptimLayout *metal_optim = nullptr;
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
    metal_optim = reinterpret_cast<MetalOptimLayout *>(optim);
    ASSERT_NE(metal_optim, nullptr);
    first_moments_after_growth = download_tensor_f32(backend, metal_optim->first_moments[0]);
    second_moments_after_growth = download_tensor_f32(backend, metal_optim->second_moments[0]);
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

TEST_F(MetalAdcRuntimeTest, RefineResetsAuxStatistics)
{
    gsx_backend_t backend = create_metal_backend();
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
