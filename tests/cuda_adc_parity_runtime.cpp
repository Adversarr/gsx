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

enum class BackendKind {
    Cpu,
    Cuda,
};

struct AdcFieldData {
    std::vector<float> mean3d;
    std::vector<float> grad_acc;
    std::vector<float> absgrad_acc;
    std::vector<float> visible_counter;
    std::vector<float> logscale;
    std::vector<float> opacity;
    std::vector<float> rotation;
    std::vector<float> sh0;
    std::vector<float> sh1;
    std::vector<float> sh2;
    std::vector<float> sh3;
    std::vector<float> max_screen_radius;
};

struct AdcSnapshot {
    gsx_size_t count = 0;
    std::vector<float> mean3d;
    std::vector<float> grad_acc;
    std::vector<float> absgrad_acc;
    std::vector<float> visible_counter;
    std::vector<float> logscale;
    std::vector<float> opacity;
    std::vector<float> rotation;
    std::vector<float> sh0;
    std::vector<float> sh1;
    std::vector<float> sh2;
    std::vector<float> sh3;
    std::vector<float> max_screen_radius;
};

struct AdcRunOutput {
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_adc_result result{};
    AdcSnapshot snapshot{};
};

struct BackendHarness {
    gsx_backend_t backend = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_adc_t adc = nullptr;
    gsx_optim fake_optim{};
    gsx_renderer fake_renderer{};
};

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

static gsx_backend_t create_backend(BackendKind kind)
{
    gsx_backend_t backend = nullptr;
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_desc backend_desc{};
    const gsx_backend_type backend_type = kind == BackendKind::Cuda ? GSX_BACKEND_TYPE_CUDA : GSX_BACKEND_TYPE_CPU;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(backend_type, 0, &backend_device), GSX_ERROR_SUCCESS);
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

static std::vector<float> download_optional_gs_field_f32(gsx_gs_t gs, gsx_gs_field field)
{
    gsx_tensor_t tensor = nullptr;
    gsx_error error = gsx_gs_get_field(gs, field, &tensor);

    if(error.code != GSX_ERROR_SUCCESS || tensor == nullptr || tensor->data_type != GSX_DATA_TYPE_F32) {
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

static void upload_case_fields(gsx_gs_t gs, const AdcFieldData &data)
{
    if(!data.mean3d.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, data.mean3d);
    }
    if(!data.grad_acc.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC, data.grad_acc);
    }
    if(!data.absgrad_acc.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_ABSGRAD_ACC, data.absgrad_acc);
    }
    if(!data.visible_counter.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER, data.visible_counter);
    }
    if(!data.logscale.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE, data.logscale);
    }
    if(!data.opacity.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, data.opacity);
    }
    if(!data.rotation.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_ROTATION, data.rotation);
    }
    if(!data.sh0.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_SH0, data.sh0);
    }
    if(!data.sh1.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_SH1, data.sh1);
    }
    if(!data.sh2.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_SH2, data.sh2);
    }
    if(!data.sh3.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_SH3, data.sh3);
    }
    if(!data.max_screen_radius.empty()) {
        upload_gs_field_f32(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS, data.max_screen_radius);
    }
}

static AdcSnapshot snapshot_gs(gsx_gs_t gs)
{
    AdcSnapshot snapshot{};

    snapshot.count = get_gs_count(gs);
    snapshot.mean3d = download_optional_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D);
    snapshot.grad_acc = download_optional_gs_field_f32(gs, GSX_GS_FIELD_GRAD_ACC);
    snapshot.absgrad_acc = download_optional_gs_field_f32(gs, GSX_GS_FIELD_ABSGRAD_ACC);
    snapshot.visible_counter = download_optional_gs_field_f32(gs, GSX_GS_FIELD_VISIBLE_COUNTER);
    snapshot.logscale = download_optional_gs_field_f32(gs, GSX_GS_FIELD_LOGSCALE);
    snapshot.opacity = download_optional_gs_field_f32(gs, GSX_GS_FIELD_OPACITY);
    snapshot.rotation = download_optional_gs_field_f32(gs, GSX_GS_FIELD_ROTATION);
    snapshot.sh0 = download_optional_gs_field_f32(gs, GSX_GS_FIELD_SH0);
    snapshot.sh1 = download_optional_gs_field_f32(gs, GSX_GS_FIELD_SH1);
    snapshot.sh2 = download_optional_gs_field_f32(gs, GSX_GS_FIELD_SH2);
    snapshot.sh3 = download_optional_gs_field_f32(gs, GSX_GS_FIELD_SH3);
    snapshot.max_screen_radius = download_optional_gs_field_f32(gs, GSX_GS_FIELD_MAX_SCREEN_RADIUS);
    return snapshot;
}

static BackendHarness make_harness(BackendKind kind, const gsx_adc_desc &desc, gsx_size_t count, gsx_gs_aux_flags aux_flags)
{
    BackendHarness harness{};
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_desc gs_desc{};

    harness.backend = create_backend(kind);
    buffer_type = find_buffer_type(harness.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    arena_desc.initial_capacity_bytes = 1U << 20;
    EXPECT_GSX_CODE(gsx_arena_init(&harness.arena, buffer_type, &arena_desc), GSX_ERROR_SUCCESS);
    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = (gsx_index_t)count;
    gs_desc.aux_flags = aux_flags;
    EXPECT_GSX_CODE(gsx_gs_init(&harness.gs, &gs_desc), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_adc_init(&harness.adc, harness.backend, &desc), GSX_ERROR_SUCCESS);
    harness.fake_optim.backend = harness.backend;
    harness.fake_renderer.backend = harness.backend;
    return harness;
}

static void free_harness(BackendHarness *harness)
{
    if(harness == nullptr) {
        return;
    }
    if(harness->adc != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_adc_free(harness->adc));
    }
    if(harness->gs != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_gs_free(harness->gs));
    }
    if(harness->arena != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_arena_free(harness->arena));
    }
    if(harness->backend != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_backend_free(harness->backend));
    }
}

static AdcRunOutput run_case(
    BackendKind kind,
    const gsx_adc_desc &desc,
    gsx_size_t count,
    gsx_gs_aux_flags aux_flags,
    const AdcFieldData &data,
    gsx_size_t global_step,
    float scene_scale)
{
    BackendHarness harness = make_harness(kind, desc, count, aux_flags);
    gsx_adc_request request{};
    AdcRunOutput output{};

    upload_case_fields(harness.gs, data);
    request.gs = harness.gs;
    request.optim = &harness.fake_optim;
    request.dataloader = (gsx_dataloader_t)0x1;
    request.renderer = &harness.fake_renderer;
    request.global_step = global_step;
    request.scene_scale = scene_scale;
    output.error = gsx_adc_step(harness.adc, &request, &output.result);
    output.snapshot = snapshot_gs(harness.gs);
    free_harness(&harness);
    return output;
}

static void expect_result_eq(const gsx_adc_result &actual, const gsx_adc_result &expected)
{
    EXPECT_EQ(actual.gaussians_before, expected.gaussians_before);
    EXPECT_EQ(actual.gaussians_after, expected.gaussians_after);
    EXPECT_EQ(actual.pruned_count, expected.pruned_count);
    EXPECT_EQ(actual.duplicated_count, expected.duplicated_count);
    EXPECT_EQ(actual.grown_count, expected.grown_count);
    EXPECT_EQ(actual.reset_count, expected.reset_count);
    EXPECT_EQ(actual.mutated, expected.mutated);
}

static void expect_vector_near(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
    }
}

static void expect_snapshot_eq(const AdcSnapshot &actual, const AdcSnapshot &expected, bool compare_mean3d)
{
    EXPECT_EQ(actual.count, expected.count);
    if(compare_mean3d) {
        expect_vector_near(actual.mean3d, expected.mean3d, 1e-6f);
    } else {
        EXPECT_EQ(actual.mean3d.size(), expected.mean3d.size());
    }
    expect_vector_near(actual.grad_acc, expected.grad_acc, 1e-6f);
    expect_vector_near(actual.absgrad_acc, expected.absgrad_acc, 1e-6f);
    expect_vector_near(actual.visible_counter, expected.visible_counter, 1e-6f);
    expect_vector_near(actual.logscale, expected.logscale, 1e-6f);
    expect_vector_near(actual.opacity, expected.opacity, 1e-6f);
    expect_vector_near(actual.rotation, expected.rotation, 1e-6f);
    expect_vector_near(actual.sh0, expected.sh0, 1e-6f);
    expect_vector_near(actual.sh1, expected.sh1, 1e-6f);
    expect_vector_near(actual.sh2, expected.sh2, 1e-6f);
    expect_vector_near(actual.sh3, expected.sh3, 1e-6f);
    expect_vector_near(actual.max_screen_radius, expected.max_screen_radius, 1e-6f);
}

static void expect_success_parity(
    const gsx_adc_desc &desc,
    gsx_size_t count,
    gsx_gs_aux_flags aux_flags,
    const AdcFieldData &data,
    gsx_size_t global_step,
    float scene_scale,
    bool compare_mean3d)
{
    AdcRunOutput cpu = run_case(BackendKind::Cpu, desc, count, aux_flags, data, global_step, scene_scale);
    AdcRunOutput cuda = run_case(BackendKind::Cuda, desc, count, aux_flags, data, global_step, scene_scale);

    ASSERT_EQ(cpu.error.code, GSX_ERROR_SUCCESS) << (cpu.error.message != nullptr ? cpu.error.message : "");
    ASSERT_EQ(cuda.error.code, GSX_ERROR_SUCCESS) << (cuda.error.message != nullptr ? cuda.error.message : "");
    expect_result_eq(cuda.result, cpu.result);
    expect_snapshot_eq(cuda.snapshot, cpu.snapshot, compare_mean3d);
}

class CudaAdcParityRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if(!has_cuda_device()) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CudaAdcParityRuntimeTest, MissingGradAccMatchesCpuEvenWithAbsgradField)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};
    AdcRunOutput cpu{};
    AdcRunOutput cuda{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 0.1f;
    desc.duplicate_absgrad_threshold = 0.1f;
    desc.pruning_opacity_threshold = 0.01f;
    data.mean3d = { 0.0f, 0.0f, 0.0f };
    data.absgrad_acc = { 3.0f };
    data.logscale = { std::log(0.25f), std::log(0.25f), std::log(0.25f) };
    data.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    data.opacity = { logitf(0.7f) };

    cpu = run_case(BackendKind::Cpu, desc, 1, GSX_GS_AUX_ABSGRAD_ACC, data, 1, 1.0f);
    cuda = run_case(BackendKind::Cuda, desc, 1, GSX_GS_AUX_ABSGRAD_ACC, data, 1, 1.0f);

    EXPECT_EQ(cpu.error.code, GSX_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(cuda.error.code, GSX_ERROR_NOT_SUPPORTED);
}

TEST_F(CudaAdcParityRuntimeTest, DuplicateAbsgradThresholdIsIgnoredLikeCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_absgrad_threshold = 0.01f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    data.mean3d = { 1.0f, 2.0f, 3.0f };
    data.grad_acc = { 0.1f };
    data.absgrad_acc = { 100.0f };
    data.logscale = { std::log(0.25f), std::log(0.25f), std::log(0.25f) };
    data.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    data.opacity = { logitf(0.7f) };

    expect_success_parity(desc, 1, GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_ABSGRAD_ACC, data, 1, 1.0f, true);
}

TEST_F(CudaAdcParityRuntimeTest, DuplicateGradThresholdBoundaryMatchesCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    data.mean3d = { 0.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f };
    data.grad_acc = { 0.5f, 0.5001f };
    data.logscale = {
        std::log(0.25f), std::log(0.25f), std::log(0.25f),
        std::log(0.25f), std::log(0.25f), std::log(0.25f),
    };
    data.rotation = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.opacity = { logitf(0.8f), logitf(0.6f) };
    data.sh0 = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };

    expect_success_parity(desc, 2, GSX_GS_AUX_GRAD_ACC, data, 1, 1.0f, true);
}

TEST_F(CudaAdcParityRuntimeTest, VisibleCounterNormalizationMatchesCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 5;
    desc.duplicate_grad_threshold = 1.0f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    data.mean3d = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
    };
    data.grad_acc = { 3.0f, 3.0f, 3.0f };
    data.visible_counter = { 0.0f, 6.0f, 2.0f };
    data.logscale = {
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
    };
    data.rotation = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.opacity = { logitf(0.7f), logitf(0.7f), logitf(0.7f) };

    expect_success_parity(desc, 3, GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER, data, 1, 1.0f, true);
}

TEST_F(CudaAdcParityRuntimeTest, GrowthBudgetSelectionOrderMatchesCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 6;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    data.mean3d = {
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f,
    };
    data.grad_acc = { 2.0f, 2.0f, 2.0f, 2.0f };
    data.logscale = {
        std::log(0.3f), std::log(0.3f), std::log(0.3f),
        std::log(0.3f), std::log(0.3f), std::log(0.3f),
        std::log(0.3f), std::log(0.3f), std::log(0.3f),
        std::log(0.3f), std::log(0.3f), std::log(0.3f),
    };
    data.rotation = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.opacity = { logitf(0.7f), logitf(0.7f), logitf(0.7f), logitf(0.7f) };

    expect_success_parity(desc, 4, GSX_GS_AUX_GRAD_ACC, data, 1, 1.0f, true);
}

TEST_F(CudaAdcParityRuntimeTest, SplitScaleThresholdBoundaryMatchesCpuExceptMeanOffsets)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 4;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 0.5f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.seed = 2026;
    data.mean3d = { 0.0f, 0.0f, 0.0f, 3.0f, 1.0f, 2.0f };
    data.grad_acc = { 1.0f, 1.0f };
    data.absgrad_acc = { 9.0f, 9.0f };
    data.logscale = {
        std::log(1.0f), std::log(1.0f), std::log(1.0f),
        std::log(1.0001f), std::log(1.0001f), std::log(1.0001f),
    };
    data.rotation = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.opacity = { logitf(0.81f), logitf(0.36f) };
    data.sh0 = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };
    data.sh1 = {
        100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f,
        200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f, 206.0f, 207.0f, 208.0f,
    };
    data.max_screen_radius = { 0.1f, 0.1f };

    expect_success_parity(
        desc,
        2,
        GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_ABSGRAD_ACC | GSX_GS_AUX_SH1 | GSX_GS_AUX_MAX_SCREEN_RADIUS,
        data,
        1,
        2.0f,
        false);
}

TEST_F(CudaAdcParityRuntimeTest, PruningOpacityAndWorldScaleMatchCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 3;
    desc.duplicate_grad_threshold = 100.0f;
    desc.pruning_opacity_threshold = 0.2f;
    desc.max_world_scale = 0.5f;
    desc.reset_every = 1;
    data.mean3d = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
    };
    data.grad_acc = { 0.0f, 0.0f, 0.0f };
    data.logscale = {
        std::log(1.0f), std::log(1.0f), std::log(1.0f),
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
    };
    data.rotation = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.opacity = { logitf(0.8f), logitf(0.8f), logitf(0.05f) };

    expect_success_parity(desc, 3, GSX_GS_AUX_GRAD_ACC, data, 2, 1.0f, true);
}

TEST_F(CudaAdcParityRuntimeTest, MaxScreenRadiusPruningMatchesCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 1;
    desc.start_refine = 0;
    desc.end_refine = 10;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 100.0f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.max_screen_scale = 0.5f;
    desc.reset_every = 1;
    data.mean3d = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
    data.grad_acc = { 0.0f, 0.0f };
    data.logscale = {
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
        std::log(0.2f), std::log(0.2f), std::log(0.2f),
    };
    data.rotation = {
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.opacity = { logitf(0.8f), logitf(0.8f) };
    data.max_screen_radius = { 1.0f, 0.1f };

    expect_success_parity(desc, 2, GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_MAX_SCREEN_RADIUS, data, 2, 1.0f, true);
}

TEST_F(CudaAdcParityRuntimeTest, RefineAndResetWindowsMatchCpu)
{
    gsx_adc_desc desc = make_default_adc_desc();
    AdcFieldData data{};

    desc.refine_every = 4;
    desc.reset_every = 3;
    desc.start_refine = 4;
    desc.end_refine = 9;
    desc.max_num_gaussians = 2;
    desc.duplicate_grad_threshold = 0.5f;
    desc.duplicate_scale_threshold = 10.0f;
    desc.pruning_opacity_threshold = 0.01f;
    desc.opacity_clamp_value = 0.4f;
    data.mean3d = { 0.0f, 0.0f, 0.0f };
    data.grad_acc = { 1.0f };
    data.logscale = { std::log(0.25f), std::log(0.25f), std::log(0.25f) };
    data.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    data.opacity = { logitf(0.8f) };

    expect_success_parity(desc, 1, GSX_GS_AUX_GRAD_ACC, data, 3, 1.0f, true);
    expect_success_parity(desc, 1, GSX_GS_AUX_GRAD_ACC, data, 4, 1.0f, true);
    expect_success_parity(desc, 1, GSX_GS_AUX_GRAD_ACC, data, 6, 1.0f, true);
}

} // namespace
