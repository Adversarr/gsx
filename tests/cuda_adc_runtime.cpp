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

TEST_F(CudaAdcRuntimeTest, StepDefaultSucceedsWithGsRuntimeNoMutation)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(cpu_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
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

    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(cpu_backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaAdcRuntimeTest, StepDefaultRefineDuplicatesAndPrunes)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(cpu_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
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

    gs_desc.arena = arena;
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

    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(cpu_backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaAdcRuntimeTest, StepDefaultResetClampsOpacity)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_t cpu_backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(cpu_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
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

    gs_desc.arena = arena;
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

    EXPECT_GSX_CODE(gsx_adc_step(adc, &request, &result), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_adc_free(adc));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(cpu_backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
