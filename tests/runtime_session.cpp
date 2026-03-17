#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
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

struct SessionTestDataset {
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    std::vector<float> rgb;
};

struct SessionStepDatasetSample {
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    std::vector<float> rgb;
    void *release_token;
};

struct SessionStepDataset {
    std::vector<SessionStepDatasetSample> samples;
    gsx_size_t get_length_calls;
    gsx_size_t get_sample_calls;
    gsx_size_t release_calls;
    std::vector<gsx_size_t> fetched_indices;
    std::vector<void *> released_tokens;
};

static std::vector<float> make_rgb_pattern(gsx_index_t width, gsx_index_t height, gsx_float_t seed)
{
    std::vector<float> rgb((size_t)width * (size_t)height * 3u);

    for(gsx_index_t y = 0; y < height; ++y) {
        for(gsx_index_t x = 0; x < width; ++x) {
            size_t pixel_offset = ((size_t)y * (size_t)width + (size_t)x) * 3u;
            gsx_float_t base = seed + (gsx_float_t)(x % 17) * 0.01f + (gsx_float_t)(y % 13) * 0.02f;

            rgb[pixel_offset + 0u] = base;
            rgb[pixel_offset + 1u] = base + 0.1f;
            rgb[pixel_offset + 2u] = base + 0.2f;
        }
    }
    return rgb;
}

static gsx_error session_test_dataset_get_length(void *object, gsx_size_t *out_length)
{
    (void)object;
    if(out_length == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "out_length must be non-null" };
    }
    *out_length = 1;
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error session_test_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    SessionTestDataset *dataset = static_cast<SessionTestDataset *>(object);

    if(out_sample == nullptr || dataset == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_sample must be non-null" };
    }
    if(sample_index != 0) {
        return gsx_error{ GSX_ERROR_OUT_OF_RANGE, "sample_index must be 0 for this fixture" };
    }

    std::memset(out_sample, 0, sizeof(*out_sample));
    out_sample->intrinsics = dataset->intrinsics;
    out_sample->pose = dataset->pose;
    out_sample->rgb.data = dataset->rgb.data();
    out_sample->rgb.data_type = GSX_DATA_TYPE_F32;
    out_sample->rgb.width = 1;
    out_sample->rgb.height = 1;
    out_sample->rgb.channel_count = 3;
    out_sample->rgb.row_stride_bytes = 3 * sizeof(float);
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static void session_test_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    (void)object;
    (void)sample;
}

static gsx_error session_step_dataset_get_length(void *object, gsx_size_t *out_length)
{
    SessionStepDataset *dataset = static_cast<SessionStepDataset *>(object);

    if(out_length == nullptr || dataset == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_length must be non-null" };
    }
    dataset->get_length_calls += 1;
    *out_length = (gsx_size_t)dataset->samples.size();
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error session_step_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    SessionStepDataset *dataset = static_cast<SessionStepDataset *>(object);
    SessionStepDatasetSample *sample = nullptr;

    if(out_sample == nullptr || dataset == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_sample must be non-null" };
    }
    if(sample_index >= dataset->samples.size()) {
        return gsx_error{ GSX_ERROR_OUT_OF_RANGE, "sample_index out of range" };
    }
    sample = &dataset->samples[(size_t)sample_index];
    dataset->get_sample_calls += 1;
    dataset->fetched_indices.push_back(sample_index);

    std::memset(out_sample, 0, sizeof(*out_sample));
    out_sample->intrinsics = sample->intrinsics;
    out_sample->pose = sample->pose;
    out_sample->rgb.data = sample->rgb.data();
    out_sample->rgb.data_type = GSX_DATA_TYPE_F32;
    out_sample->rgb.width = sample->intrinsics.width;
    out_sample->rgb.height = sample->intrinsics.height;
    out_sample->rgb.channel_count = 3;
    out_sample->rgb.row_stride_bytes = (gsx_size_t)sample->intrinsics.width * 3u * sizeof(float);
    out_sample->release_token = sample->release_token;
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static void session_step_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    SessionStepDataset *dataset = static_cast<SessionStepDataset *>(object);

    if(dataset == nullptr || sample == nullptr) {
        return;
    }
    dataset->release_calls += 1;
    dataset->released_tokens.push_back(sample->release_token);
}

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_desc backend_desc{};
    gsx_backend_t backend = nullptr;
    gsx_error error{ GSX_ERROR_SUCCESS, nullptr };

    error = gsx_backend_registry_init();
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return nullptr;
    }
    error = gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_desc.device);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return nullptr;
    }
    error = gsx_backend_init(&backend, &backend_desc);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return nullptr;
    }
    return backend;
}

static gsx_backend_buffer_type_t find_device_buffer_type(gsx_backend_t backend)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static gsx_arena_t create_arena(gsx_backend_buffer_type_t buffer_type)
{
    gsx_arena_desc desc{};
    gsx_arena_t arena = nullptr;

    desc.initial_capacity_bytes = 1U << 20;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    EXPECT_GSX_CODE(gsx_arena_init(&arena, buffer_type, &desc), GSX_ERROR_SUCCESS);
    return arena;
}

static void upload_tensor_f32(gsx_tensor_t tensor, const std::vector<float> &values)
{
    gsx_tensor_info info{};

    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    ASSERT_EQ(info.data_type, GSX_DATA_TYPE_F32);
    ASSERT_EQ((gsx_size_t)values.size() * sizeof(float), info.size_bytes);
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, values.data(), info.size_bytes));
}

static gsx_optim_t create_optim_for_gs(gsx_backend_t backend, gsx_gs_t gs)
{
    gsx_optim_t optim = nullptr;
    gsx_optim_desc desc{};
    gsx_optim_param_group_desc groups[8]{};
    gsx_error error{ GSX_ERROR_SUCCESS, nullptr };

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &groups[0].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &groups[0].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[0].role = GSX_OPTIM_PARAM_ROLE_MEAN3D;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &groups[1].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_LOGSCALE, &groups[1].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[1].role = GSX_OPTIM_PARAM_ROLE_LOGSCALE;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &groups[2].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_ROTATION, &groups[2].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[2].role = GSX_OPTIM_PARAM_ROLE_ROTATION;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &groups[3].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_OPACITY, &groups[3].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[3].role = GSX_OPTIM_PARAM_ROLE_OPACITY;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &groups[4].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH0, &groups[4].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[4].role = GSX_OPTIM_PARAM_ROLE_SH0;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &groups[5].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH1, &groups[5].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[5].role = GSX_OPTIM_PARAM_ROLE_SH1;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &groups[6].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH2, &groups[6].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[6].role = GSX_OPTIM_PARAM_ROLE_SH2;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &groups[7].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH3, &groups[7].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    groups[7].role = GSX_OPTIM_PARAM_ROLE_SH3;

    for(int i = 0; i < 8; ++i) {
        groups[i].learning_rate = 0.1f;
        groups[i].beta1 = 0.9f;
        groups[i].beta2 = 0.99f;
        groups[i].weight_decay = 0.0f;
        groups[i].epsilon = 1e-8f;
        groups[i].max_grad = 0.0f;
    }

    desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    desc.param_groups = groups;
    desc.param_group_count = 8;
    EXPECT_GSX_CODE(gsx_optim_init(&optim, backend, &desc), GSX_ERROR_SUCCESS);
    return optim;
}

TEST(SessionRuntime, InitRejectsNullAndMissingDependencies)
{
    gsx_session_t session = nullptr;
    gsx_session_desc desc = {};

    EXPECT_GSX_CODE(gsx_session_init(nullptr, &desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_session_init(&session, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_session_init(&session, &desc), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(SessionRuntime, InitStepStateAndFreeRoundTrip)
{
    SessionTestDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc = {};
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_scheduler_t scheduler = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_gs_desc gs_desc = {};
    gsx_renderer_desc renderer_desc = {};
    gsx_loss_desc loss_desc = {};
    gsx_scheduler_desc scheduler_desc = {};
    gsx_dataset_desc dataset_desc = {};
    gsx_dataloader_desc dataloader_desc = {};
    gsx_session_desc session_desc = {};
    gsx_session_state state = {};
    gsx_float_t lr = 0.0f;
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t sh0 = nullptr;

    dataset_object.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    dataset_object.intrinsics.fx = 1.0f;
    dataset_object.intrinsics.fy = 1.0f;
    dataset_object.intrinsics.cx = 0.5f;
    dataset_object.intrinsics.cy = 0.5f;
    dataset_object.intrinsics.width = 1;
    dataset_object.intrinsics.height = 1;
    dataset_object.pose.rot.w = 1.0f;
    dataset_object.pose.camera_id = 0;
    dataset_object.rgb = { 0.2f, 0.1f, 0.05f };

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    arena = create_arena(buffer_type);
    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &rotation));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &sh0));
    upload_tensor_f32(mean3d, { 0.0f, 0.0f, 2.0f });
    upload_tensor_f32(rotation, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_tensor_f32(opacity, { 1.0f });
    upload_tensor_f32(sh0, { 0.2f, 0.1f, 0.05f });

    optim = create_optim_for_gs(backend, gs);

    renderer_desc.width = 1;
    renderer_desc.height = 1;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_renderer_init(&renderer, backend, &renderer_desc));

    loss_desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    loss_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &loss_desc));

    scheduler_desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
    scheduler_desc.initial_learning_rate = 0.03f;
    scheduler_desc.final_learning_rate = 0.03f;
    scheduler_desc.delay_multiplier = 1.0f;
    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &scheduler_desc));

    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = session_test_dataset_get_length;
    dataset_desc.get_sample = session_test_dataset_get_sample;
    dataset_desc.release_sample = session_test_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataloader_desc.output_width = 1;
    dataloader_desc.output_height = 1;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    session_desc.backend = backend;
    session_desc.gs = gs;
    session_desc.optim = optim;
    session_desc.dataloader = dataloader;
    session_desc.validation_dataloader = nullptr;
    session_desc.scheduler = scheduler;
    session_desc.renderer = renderer;
    session_desc.loss = loss;
    session_desc.initial_global_step = 7;
    session_desc.initial_epoch_index = 3;
    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &session_desc));

    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
    EXPECT_EQ(state.global_step, 7u);
    EXPECT_EQ(state.epoch_index, 3u);
    EXPECT_EQ(state.successful_step_count, 0u);
    EXPECT_EQ(state.failed_step_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_session_step(session));
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
    EXPECT_EQ(state.global_step, 8u);
    EXPECT_EQ(state.epoch_index, 4u);
    EXPECT_EQ(state.successful_step_count, 1u);
    EXPECT_EQ(state.failed_step_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_optim_get_learning_rate_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &lr));
    EXPECT_NEAR(lr, 0.03f, 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(SessionRuntime, StepsSixTimesAcrossFourSampleDatasetOnCpu)
{
    SessionStepDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_scheduler_t scheduler = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_gs_desc gs_desc = {};
    gsx_arena_desc arena_desc = {};
    gsx_renderer_desc renderer_desc = {};
    gsx_loss_desc loss_desc = {};
    gsx_scheduler_desc scheduler_desc = {};
    gsx_dataset_desc dataset_desc = {};
    gsx_dataloader_desc dataloader_desc = {};
    gsx_session_desc session_desc = {};
    gsx_session_state state = {};
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t sh0 = nullptr;
    std::array<gsx_size_t, 4> sample_hit_count{ 0, 0, 0, 0 };
    std::array<gsx_index_t, 4> sample_widths{ 128, 160, 192, 224 };
    std::array<gsx_index_t, 4> sample_heights{ 128, 144, 160, 192 };
    gsx_size_t distinct_sample_count = 0;
    gsx_size_t total_hit_count = 0;
    gsx_size_t step_index = 0;

    for(gsx_size_t i = 0; i < 4; ++i) {
        SessionStepDatasetSample sample{};
        sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
        sample.intrinsics.fx = 100.0f + 5.0f * (gsx_float_t)i;
        sample.intrinsics.fy = 102.0f + 7.0f * (gsx_float_t)i;
        sample.intrinsics.cx = (gsx_float_t)sample_widths[(size_t)i] * 0.5f;
        sample.intrinsics.cy = (gsx_float_t)sample_heights[(size_t)i] * 0.5f;
        sample.intrinsics.camera_id = (gsx_id_t)i;
        sample.intrinsics.width = sample_widths[(size_t)i];
        sample.intrinsics.height = sample_heights[(size_t)i];
        sample.pose.rot.w = 1.0f;
        sample.pose.camera_id = (gsx_id_t)i;
        sample.pose.frame_id = (gsx_id_t)(100 + i);
        sample.rgb = make_rgb_pattern(sample.intrinsics.width, sample.intrinsics.height, 0.05f * (gsx_float_t)i);
        sample.release_token = reinterpret_cast<void *>(uintptr_t{ 0x1000 + i });
        dataset_object.samples.push_back(sample);
    }

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    arena = create_arena(buffer_type);

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = 1;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &rotation));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &sh0));
    upload_tensor_f32(mean3d, { 0.0f, 0.0f, 2.0f });
    upload_tensor_f32(rotation, { 0.0f, 0.0f, 0.0f, 1.0f });
    upload_tensor_f32(opacity, { 1.0f });
    upload_tensor_f32(sh0, { 0.2f, 0.1f, 0.05f });

    optim = create_optim_for_gs(backend, gs);

    renderer_desc.width = 128;
    renderer_desc.height = 128;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_renderer_init(&renderer, backend, &renderer_desc));

    loss_desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    loss_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &loss_desc));

    scheduler_desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
    scheduler_desc.initial_learning_rate = 0.03f;
    scheduler_desc.final_learning_rate = 0.03f;
    scheduler_desc.delay_multiplier = 1.0f;
    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &scheduler_desc));

    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = session_step_dataset_get_length;
    dataset_desc.get_sample = session_step_dataset_get_sample;
    dataset_desc.release_sample = session_step_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataloader_desc.output_width = 128;
    dataloader_desc.output_height = 128;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    session_desc.backend = backend;
    session_desc.gs = gs;
    session_desc.optim = optim;
    session_desc.dataloader = dataloader;
    session_desc.validation_dataloader = nullptr;
    session_desc.scheduler = scheduler;
    session_desc.renderer = renderer;
    session_desc.loss = loss;
    session_desc.initial_global_step = 0;
    session_desc.initial_epoch_index = 0;
    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &session_desc));

    for(step_index = 0; step_index < 6; ++step_index) {
        gsx_size_t expected_epoch = step_index >= 4 ? 2 : 1;

        ASSERT_GSX_SUCCESS(gsx_session_step(session));
        ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
        EXPECT_EQ(state.global_step, step_index + 1);
        EXPECT_EQ(state.successful_step_count, step_index + 1);
        EXPECT_EQ(state.failed_step_count, 0u);
        EXPECT_EQ(state.epoch_index, expected_epoch);
    }

    EXPECT_EQ(dataset_object.get_sample_calls, 6u);
    EXPECT_EQ(dataset_object.release_calls, 6u);
    EXPECT_EQ(dataset_object.fetched_indices.size(), 6u);
    EXPECT_EQ(dataset_object.released_tokens.size(), 6u);
    EXPECT_GE(dataset_object.get_length_calls, 1u);
    for(gsx_size_t i = 0; i < dataset_object.fetched_indices.size(); ++i) {
        gsx_size_t sample_index = dataset_object.fetched_indices[i];

        ASSERT_LT(sample_index, sample_hit_count.size());
        sample_hit_count[(size_t)sample_index] += 1;
        total_hit_count += 1;
    }
    EXPECT_EQ(total_hit_count, 6u);
    for(gsx_size_t i = 0; i < sample_hit_count.size(); ++i) {
        if(sample_hit_count[i] > 0u) {
            distinct_sample_count += 1;
        }
    }
    EXPECT_EQ(distinct_sample_count, 4u);

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(SessionRuntime, StressTest64GaussiansAcrossFourViewsOnCpu)
{
    /*
     * Stress test: 64 gaussians in an 8×8 grid, 4 training views, 128 steps.
     * Verifies that the session advances without failure, all samples are
     * fetched and released in pairs, and gaussian parameters remain finite.
     */
    static const gsx_size_t kGaussianCount = 64;
    static const gsx_size_t kStepCount = 128;
    static const gsx_index_t kImageWidth = 32;
    static const gsx_index_t kImageHeight = 32;

    SessionStepDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_scheduler_t scheduler = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_gs_desc gs_desc = {};
    gsx_arena_desc arena_desc = {};
    gsx_renderer_desc renderer_desc = {};
    gsx_loss_desc loss_desc = {};
    gsx_scheduler_desc scheduler_desc = {};
    gsx_dataset_desc dataset_desc = {};
    gsx_dataloader_desc dataloader_desc = {};
    gsx_session_desc session_desc = {};
    gsx_session_state state = {};
    gsx_gs_finite_check_result finite_check = {};
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t logscale = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t sh0 = nullptr;
    gsx_tensor_t sh1 = nullptr;
    gsx_tensor_t sh2 = nullptr;
    gsx_tensor_t sh3 = nullptr;
    /* 8×8 grid positions, log-scale=0 (unit sphere), identity quaternion */
    std::vector<float> mean3d_data(kGaussianCount * 3u);
    std::vector<float> logscale_data(kGaussianCount * 3u, 0.0f);
    std::vector<float> rotation_data(kGaussianCount * 4u);
    std::vector<float> opacity_data(kGaussianCount, 0.8f);
    std::vector<float> sh0_data(kGaussianCount * 3u, 0.5f);
    std::vector<float> sh1_data(kGaussianCount * 3u * 3u, 0.0f);
    std::vector<float> sh2_data(kGaussianCount * 5u * 3u, 0.0f);
    std::vector<float> sh3_data(kGaussianCount * 7u * 3u, 0.0f);
    std::array<gsx_size_t, 4> sample_hit_count{ 0, 0, 0, 0 };
    gsx_size_t distinct_sample_count = 0;

    for(gsx_size_t i = 0; i < kGaussianCount; ++i) {
        gsx_size_t row = i / 8u;
        gsx_size_t col = i % 8u;
        mean3d_data[i * 3u + 0u] = -0.35f + 0.10f * (gsx_float_t)col;
        mean3d_data[i * 3u + 1u] = -0.35f + 0.10f * (gsx_float_t)row;
        mean3d_data[i * 3u + 2u] = 3.0f;
        /* quaternion stored as (x, y, z, w) — identity rotation */
        rotation_data[i * 4u + 0u] = 0.0f;
        rotation_data[i * 4u + 1u] = 0.0f;
        rotation_data[i * 4u + 2u] = 0.0f;
        rotation_data[i * 4u + 3u] = 1.0f;
    }

    for(gsx_size_t i = 0; i < 4; ++i) {
        SessionStepDatasetSample sample{};
        sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
        sample.intrinsics.fx = 40.0f;
        sample.intrinsics.fy = 40.0f;
        sample.intrinsics.cx = (gsx_float_t)kImageWidth * 0.5f;
        sample.intrinsics.cy = (gsx_float_t)kImageHeight * 0.5f;
        sample.intrinsics.camera_id = (gsx_id_t)i;
        sample.intrinsics.width = kImageWidth;
        sample.intrinsics.height = kImageHeight;
        sample.pose.rot.w = 1.0f;
        sample.pose.camera_id = (gsx_id_t)i;
        sample.pose.frame_id = (gsx_id_t)(200u + i);
        sample.rgb = make_rgb_pattern(kImageWidth, kImageHeight, 0.1f * (gsx_float_t)i);
        sample.release_token = reinterpret_cast<void *>(uintptr_t{ 0x3000u + i });
        dataset_object.samples.push_back(sample);
    }

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    arena = create_arena(buffer_type);

    arena_desc.initial_capacity_bytes = 1U << 22;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = kGaussianCount;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &logscale));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &rotation));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &sh0));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &sh1));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &sh2));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &sh3));
    upload_tensor_f32(mean3d, mean3d_data);
    upload_tensor_f32(logscale, logscale_data);
    upload_tensor_f32(rotation, rotation_data);
    upload_tensor_f32(opacity, opacity_data);
    upload_tensor_f32(sh0, sh0_data);
    upload_tensor_f32(sh1, sh1_data);
    upload_tensor_f32(sh2, sh2_data);
    upload_tensor_f32(sh3, sh3_data);

    optim = create_optim_for_gs(backend, gs);

    renderer_desc.width = kImageWidth;
    renderer_desc.height = kImageHeight;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_renderer_init(&renderer, backend, &renderer_desc));

    loss_desc.algorithm = GSX_LOSS_ALGORITHM_MSE;
    loss_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    ASSERT_GSX_SUCCESS(gsx_loss_init(&loss, backend, &loss_desc));

    scheduler_desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
    scheduler_desc.initial_learning_rate = 0.001f;
    scheduler_desc.final_learning_rate = 0.001f;
    scheduler_desc.delay_multiplier = 1.0f;
    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &scheduler_desc));

    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = session_step_dataset_get_length;
    dataset_desc.get_sample = session_step_dataset_get_sample;
    dataset_desc.release_sample = session_step_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataloader_desc.output_width = kImageWidth;
    dataloader_desc.output_height = kImageHeight;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    session_desc.backend = backend;
    session_desc.gs = gs;
    session_desc.optim = optim;
    session_desc.dataloader = dataloader;
    session_desc.validation_dataloader = nullptr;
    session_desc.scheduler = scheduler;
    session_desc.renderer = renderer;
    session_desc.loss = loss;
    session_desc.initial_global_step = 0;
    session_desc.initial_epoch_index = 0;
    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &session_desc));

    for(gsx_size_t step_index = 0; step_index < kStepCount; ++step_index) {
        ASSERT_GSX_SUCCESS(gsx_session_step(session));
    }

    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
    EXPECT_EQ(state.global_step, kStepCount);
    EXPECT_EQ(state.successful_step_count, kStepCount);
    EXPECT_EQ(state.failed_step_count, 0u);

    ASSERT_GSX_SUCCESS(gsx_gs_check_finite(gs, &finite_check));
    EXPECT_TRUE(finite_check.is_finite);

    EXPECT_EQ(dataset_object.get_sample_calls, kStepCount);
    EXPECT_EQ(dataset_object.release_calls, dataset_object.get_sample_calls);
    for(gsx_size_t i = 0; i < dataset_object.fetched_indices.size(); ++i) {
        gsx_size_t sample_index = dataset_object.fetched_indices[i];
        ASSERT_LT(sample_index, sample_hit_count.size());
        sample_hit_count[(size_t)sample_index] += 1;
    }
    for(gsx_size_t i = 0; i < sample_hit_count.size(); ++i) {
        if(sample_hit_count[i] > 0u) {
            distinct_sample_count += 1;
        }
    }
    EXPECT_EQ(distinct_sample_count, 4u);

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
