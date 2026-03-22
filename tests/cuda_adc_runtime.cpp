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

static bool has_cuda_device()
{
    gsx_index_t cuda_device_count = 0;
    gsx_error error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CUDA, &cuda_device_count);
    return error.code == GSX_ERROR_SUCCESS && cuda_device_count > 0;
}

static gsx_backend_t create_cuda_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_desc backend_desc{};

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CUDA, 0, &backend_device), GSX_ERROR_SUCCESS);
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
    dry_run_desc.arena_desc.dry_run = true;
    dry_run_desc.count = (gsx_index_t)count;
    dry_run_desc.aux_flags = aux_flags;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&dry_run_gs, &dry_run_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(dry_run_gs, &dry_run_info));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(dry_run_info.arena, out_required_bytes));
    ASSERT_GSX_SUCCESS(gsx_gs_free(dry_run_gs));
}

class CudaAdcRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if(!has_cuda_device()) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CudaAdcRuntimeTest, InitDefaultSucceedsOnCuda)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    ASSERT_NE(adc, nullptr);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaAdcRuntimeTest, BackendFreeFailsWhileAdcAlive)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_adc_t adc = nullptr;
    gsx_adc_desc desc = make_default_adc_desc();

    ASSERT_GSX_SUCCESS(gsx_adc_init(&adc, backend, &desc));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaAdcRuntimeTest, StepRejectsOptimizerBackendMismatch)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, StepRejectsRendererBackendMismatch)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, StepRejectsNonPositiveSceneScale)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, StepDefaultSucceedsWithGsRuntimeNoMutation)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, StepDefaultRefineDuplicatesAndPrunes)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, StepDefaultRefineAndPruneRebuildsGsStorageWithExactLayoutArena)
{
    gsx_backend_t backend = create_cuda_backend();
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
    EXPECT_EQ(result.pruned_count, 2u);
    EXPECT_EQ(result.duplicated_count, 2u);
    EXPECT_EQ(result.grown_count, 0u);
    EXPECT_TRUE(result.mutated);
    EXPECT_EQ(get_gs_count(gs), 4u);
    expect_gs_fields_finite(gs);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaAdcRuntimeTest, StepDefaultResetClampsOpacity)
{
    gsx_backend_t backend = create_cuda_backend();
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
    EXPECT_NEAR(sigmoidf(opacity[0]), 0.5f, 1e-6f);
    EXPECT_GT(sigmoidf(opacity[1]), 0.25f);
    EXPECT_LT(sigmoidf(opacity[1]), 0.5f + 1e-6f);
    EXPECT_NEAR(sigmoidf(opacity[2]), 0.5f, 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaAdcRuntimeTest, StepWithLiveOptimizerSupportsGrowthGather)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, StepAcceptsRank3ShAuxFields)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, RefineStressTriggersPruneSplitDuplicateAndStaysFinite)
{
    static const gsx_size_t kInitialCount = 500;
    static const gsx_size_t kMaxCount = 600;
    static const gsx_size_t kStepCount = 128;

    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, SplitAndDuplicateVerifyAllCopiedFields)
{
    gsx_backend_t backend = create_cuda_backend();
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

TEST_F(CudaAdcRuntimeTest, VisibleCounterSuppressesGrowthWhenCounterIsNonPositive)
{
    gsx_backend_t backend = create_cuda_backend();
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

} // namespace
