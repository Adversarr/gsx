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
    dry_run_desc.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
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
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
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
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
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
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
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

TEST_F(MetalAdcRuntimeTest, StepDefaultRefineFailsWhenGsArenaOnlyCoversSingleMaxLayout)
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

    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_OUT_OF_RANGE);

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
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
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
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
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

} // namespace
