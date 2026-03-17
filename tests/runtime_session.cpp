#include "gsx/gsx.h"

#include <gtest/gtest.h>

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

}  // namespace
