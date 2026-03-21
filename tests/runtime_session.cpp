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

struct SessionDatasetSample {
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    std::vector<float> rgb;
    void *release_token;
};

struct SessionDataset {
    std::vector<SessionDatasetSample> samples;
    gsx_size_t get_length_calls;
    gsx_size_t get_sample_calls;
    gsx_size_t release_calls;
    std::vector<gsx_size_t> fetched_indices;
    std::vector<void *> released_tokens;
};

struct MemoryCheckpoint {
    std::vector<uint8_t> bytes;
    gsx_size_t read_offset;
};

static std::vector<float> make_rgb_pattern(gsx_index_t width, gsx_index_t height, gsx_float_t seed)
{
    std::vector<float> rgb((size_t)width * (size_t)height * 3u);

    for(gsx_index_t y = 0; y < height; ++y) {
        for(gsx_index_t x = 0; x < width; ++x) {
            size_t pixel_offset = ((size_t)y * (size_t)width + (size_t)x) * 3u;
            gsx_float_t base = seed + (gsx_float_t)(x % 11) * 0.01f + (gsx_float_t)(y % 7) * 0.02f;

            rgb[pixel_offset + 0u] = base;
            rgb[pixel_offset + 1u] = base + 0.1f;
            rgb[pixel_offset + 2u] = base + 0.2f;
        }
    }
    return rgb;
}

static gsx_error session_dataset_get_length(void *object, gsx_size_t *out_length)
{
    SessionDataset *dataset = static_cast<SessionDataset *>(object);

    if(dataset == nullptr || out_length == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_length must be non-null" };
    }
    dataset->get_length_calls += 1;
    *out_length = (gsx_size_t)dataset->samples.size();
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error session_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    SessionDataset *dataset = static_cast<SessionDataset *>(object);
    SessionDatasetSample *sample = nullptr;

    if(dataset == nullptr || out_sample == nullptr) {
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

static void session_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    SessionDataset *dataset = static_cast<SessionDataset *>(object);

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

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_desc.device), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

static gsx_backend_buffer_type_t find_device_buffer_type(gsx_backend_t backend)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static void upload_tensor_f32(gsx_tensor_t tensor, const std::vector<float> &values)
{
    gsx_tensor_info info{};

    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    ASSERT_EQ(info.data_type, GSX_DATA_TYPE_F32);
    ASSERT_EQ((gsx_size_t)values.size() * sizeof(float), info.size_bytes);
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, values.data(), info.size_bytes));
}

static gsx_gs_t create_initialized_gs(gsx_backend_buffer_type_t buffer_type, gsx_size_t count)
{
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_arena_desc arena_desc{};
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t logscale = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t sh0 = nullptr;
    gsx_tensor_t sh1 = nullptr;
    gsx_tensor_t sh2 = nullptr;
    gsx_tensor_t sh3 = nullptr;
    std::vector<float> mean3d_data(count * 3u);
    std::vector<float> logscale_data(count * 3u, 0.0f);
    std::vector<float> rotation_data(count * 4u, 0.0f);
    std::vector<float> opacity_data(count, 0.8f);
    std::vector<float> sh0_data(count * 3u, 0.2f);
    std::vector<float> sh1_data(count * 9u, 0.0f);
    std::vector<float> sh2_data(count * 15u, 0.0f);
    std::vector<float> sh3_data(count * 21u, 0.0f);

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = count;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
    EXPECT_GSX_CODE(gsx_gs_init(&gs, &gs_desc), GSX_ERROR_SUCCESS);
    if(gs == nullptr) {
        return nullptr;
    }

    for(gsx_size_t i = 0; i < count; ++i) {
        mean3d_data[i * 3u + 0u] = 0.0f;
        mean3d_data[i * 3u + 1u] = 0.0f;
        mean3d_data[i * 3u + 2u] = 2.0f + 0.1f * (gsx_float_t)i;
        rotation_data[i * 4u + 3u] = 1.0f;
    }

    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &logscale), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &rotation), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &sh0), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &sh1), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &sh2), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &sh3), GSX_ERROR_SUCCESS);
    if(mean3d == nullptr || logscale == nullptr || rotation == nullptr || opacity == nullptr || sh0 == nullptr || sh1 == nullptr
        || sh2 == nullptr || sh3 == nullptr) {
        return gs;
    }

    upload_tensor_f32(mean3d, mean3d_data);
    upload_tensor_f32(logscale, logscale_data);
    upload_tensor_f32(rotation, rotation_data);
    upload_tensor_f32(opacity, opacity_data);
    upload_tensor_f32(sh0, sh0_data);
    upload_tensor_f32(sh1, sh1_data);
    upload_tensor_f32(sh2, sh2_data);
    upload_tensor_f32(sh3, sh3_data);
    return gs;
}

static gsx_optim_t create_optim_for_gs(gsx_backend_t backend, gsx_gs_t gs)
{
    gsx_optim_t optim = nullptr;
    gsx_optim_desc desc{};
    gsx_optim_param_group_desc groups[8]{};
    gsx_error error{ GSX_ERROR_SUCCESS, nullptr };

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &groups[0].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &groups[0].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[0].role = GSX_OPTIM_PARAM_ROLE_MEAN3D;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &groups[1].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_LOGSCALE, &groups[1].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[1].role = GSX_OPTIM_PARAM_ROLE_LOGSCALE;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &groups[2].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_ROTATION, &groups[2].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[2].role = GSX_OPTIM_PARAM_ROLE_ROTATION;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &groups[3].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_OPACITY, &groups[3].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[3].role = GSX_OPTIM_PARAM_ROLE_OPACITY;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &groups[4].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH0, &groups[4].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[4].role = GSX_OPTIM_PARAM_ROLE_SH0;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &groups[5].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH1, &groups[5].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[5].role = GSX_OPTIM_PARAM_ROLE_SH1;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &groups[6].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH2, &groups[6].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    groups[6].role = GSX_OPTIM_PARAM_ROLE_SH2;

    error = gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &groups[7].parameter);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
    error = gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_SH3, &groups[7].gradient);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS);
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

static gsx_renderer_t create_renderer(gsx_backend_t backend, gsx_index_t width, gsx_index_t height, gsx_data_type output_data_type)
{
    gsx_renderer_t renderer = nullptr;
    gsx_renderer_desc desc{};

    desc.width = width;
    desc.height = height;
    desc.output_data_type = output_data_type;
    EXPECT_GSX_CODE(gsx_renderer_init(&renderer, backend, &desc), GSX_ERROR_SUCCESS);
    return renderer;
}

static gsx_loss_t create_loss(gsx_backend_t backend, gsx_loss_algorithm algorithm)
{
    gsx_loss_t loss = nullptr;
    gsx_loss_desc desc{};

    desc.algorithm = algorithm;
    desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    EXPECT_GSX_CODE(gsx_loss_init(&loss, backend, &desc), GSX_ERROR_SUCCESS);
    return loss;
}

static gsx_loss_context_t create_loss_context(gsx_loss_t loss)
{
    gsx_loss_context_t context = nullptr;

    EXPECT_GSX_CODE(gsx_loss_context_init(&context, loss), GSX_ERROR_SUCCESS);
    return context;
}

static gsx_scheduler_t create_constant_scheduler(gsx_float_t learning_rate)
{
    gsx_scheduler_t scheduler = nullptr;
    gsx_scheduler_desc desc{};

    desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
    desc.initial_learning_rate = learning_rate;
    desc.final_learning_rate = learning_rate;
    desc.delay_multiplier = 1.0f;
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_SUCCESS);
    return scheduler;
}

static gsx_dataset_t create_dataset(SessionDataset *dataset_object)
{
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc desc{};

    desc.object = dataset_object;
    desc.get_length = session_dataset_get_length;
    desc.get_sample = session_dataset_get_sample;
    desc.release_sample = session_dataset_release_sample;
    EXPECT_GSX_CODE(gsx_dataset_init(&dataset, &desc), GSX_ERROR_SUCCESS);
    return dataset;
}

static gsx_dataloader_t create_dataloader(
    gsx_backend_t backend,
    gsx_dataset_t dataset,
    gsx_index_t width,
    gsx_index_t height,
    gsx_data_type data_type)
{
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc desc{};

    desc.image_data_type = data_type;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.output_width = width;
    desc.output_height = height;
    EXPECT_GSX_CODE(gsx_dataloader_init(&dataloader, backend, dataset, &desc), GSX_ERROR_SUCCESS);
    return dataloader;
}

static gsx_error checkpoint_writer(void *user_data, const void *src_bytes, gsx_size_t byte_count)
{
    MemoryCheckpoint *checkpoint = static_cast<MemoryCheckpoint *>(user_data);
    const uint8_t *src = static_cast<const uint8_t *>(src_bytes);

    if(checkpoint == nullptr || src == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "checkpoint and src must be non-null" };
    }
    checkpoint->bytes.insert(checkpoint->bytes.end(), src, src + byte_count);
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error checkpoint_reader(void *user_data, void *dst_bytes, gsx_size_t requested_bytes, gsx_size_t *out_bytes_read)
{
    MemoryCheckpoint *checkpoint = static_cast<MemoryCheckpoint *>(user_data);
    gsx_size_t remaining_bytes = 0;
    gsx_size_t bytes_to_read = 0;

    if(checkpoint == nullptr || dst_bytes == nullptr || out_bytes_read == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "checkpoint, dst_bytes and out_bytes_read must be non-null" };
    }

    remaining_bytes = (gsx_size_t)checkpoint->bytes.size() - checkpoint->read_offset;
    bytes_to_read = requested_bytes < remaining_bytes ? requested_bytes : remaining_bytes;
    if(bytes_to_read > 0) {
        std::memcpy(dst_bytes, checkpoint->bytes.data() + checkpoint->read_offset, (size_t)bytes_to_read);
        checkpoint->read_offset += bytes_to_read;
    }
    *out_bytes_read = bytes_to_read;
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

TEST(SessionRuntime, InitRejectsInvalidConfig)
{
    SessionDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t loss_context = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_loss_item loss_item{};
    gsx_session_desc desc{};

    SessionDatasetSample sample{};
    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.fx = 1.0f;
    sample.intrinsics.fy = 1.0f;
    sample.intrinsics.cx = 0.5f;
    sample.intrinsics.cy = 0.5f;
    sample.intrinsics.camera_id = 7;
    sample.intrinsics.width = 1;
    sample.intrinsics.height = 1;
    sample.pose.rot.w = 1.0f;
    sample.rgb = { 0.2f, 0.1f, 0.05f };
    dataset_object.samples.push_back(sample);

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    gs = create_initialized_gs(buffer_type, 1);
    optim = create_optim_for_gs(backend, gs);
    renderer = create_renderer(backend, 1, 1, GSX_DATA_TYPE_F32);
    loss = create_loss(backend, GSX_LOSS_ALGORITHM_MSE);
    loss_context = create_loss_context(loss);
    dataset = create_dataset(&dataset_object);
    dataloader = create_dataloader(backend, dataset, 1, 1, GSX_DATA_TYPE_F32);

    EXPECT_GSX_CODE(gsx_session_init(nullptr, &desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_session_init(&session, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    desc.backend = backend;
    desc.gs = gs;
    desc.optim = optim;
    desc.renderer = renderer;
    desc.train_dataloader = dataloader;
    desc.render.near_plane = 0.01f;
    desc.render.far_plane = 10.0f;
    desc.render.precision = GSX_RENDER_PRECISION_FLOAT32;
    desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    desc.render.sh_degree = 0;
    desc.workspace.buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    desc.workspace.auto_plan = true;
    desc.workspace.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    desc.optim_step.force_all = true;

    EXPECT_GSX_CODE(gsx_session_init(&session, &desc), GSX_ERROR_INVALID_ARGUMENT);

    loss_item.loss = loss;
    loss_item.context = loss_context;
    loss_item.scale = 1.0f;
    desc.loss_count = 1;
    desc.loss_items = &loss_item;
    desc.optim_step.force_all = false;
    EXPECT_GSX_CODE(gsx_session_init(&session, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc.optim_step.force_all = true;
    desc.workspace.arena_desc.dry_run = true;
    EXPECT_GSX_CODE(gsx_session_init(&session, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc.workspace.arena_desc.dry_run = false;
    desc.renderer = create_renderer(backend, 2, 2, GSX_DATA_TYPE_F32);
    EXPECT_GSX_CODE(gsx_session_init(&session, &desc), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_renderer_free(desc.renderer));

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_loss_context_free(loss_context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(SessionRuntime, StepReportOutputsDescAndResetRoundTrip)
{
    SessionDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t loss_context = nullptr;
    gsx_scheduler_t scheduler = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_loss_item loss_item{};
    gsx_session_desc desc{};
    gsx_session_desc effective_desc{};
    gsx_session_state state{};
    gsx_session_step_report report{};
    gsx_session_outputs outputs{};
    gsx_float_t learning_rate = 0.0f;

    SessionDatasetSample sample{};
    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.fx = 1.0f;
    sample.intrinsics.fy = 1.0f;
    sample.intrinsics.cx = 0.5f;
    sample.intrinsics.cy = 0.5f;
    sample.intrinsics.camera_id = 7;
    sample.intrinsics.width = 1;
    sample.intrinsics.height = 1;
    sample.pose.rot.w = 1.0f;
    sample.pose.camera_id = 7;
    sample.pose.frame_id = 11;
    sample.rgb = { 0.2f, 0.1f, 0.05f };
    sample.release_token = reinterpret_cast<void *>(uintptr_t{ 0x1111u });
    dataset_object.samples.push_back(sample);

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    gs = create_initialized_gs(buffer_type, 1);
    optim = create_optim_for_gs(backend, gs);
    renderer = create_renderer(backend, 1, 1, GSX_DATA_TYPE_F32);
    loss = create_loss(backend, GSX_LOSS_ALGORITHM_MSE);
    loss_context = create_loss_context(loss);
    scheduler = create_constant_scheduler(0.03f);
    dataset = create_dataset(&dataset_object);
    dataloader = create_dataloader(backend, dataset, 1, 1, GSX_DATA_TYPE_F32);

    loss_item.loss = loss;
    loss_item.context = loss_context;
    loss_item.scale = 1.0f;

    desc.backend = backend;
    desc.gs = gs;
    desc.optim = optim;
    desc.renderer = renderer;
    desc.train_dataloader = dataloader;
    desc.scheduler = scheduler;
    desc.loss_count = 1;
    desc.loss_items = &loss_item;
    desc.render.near_plane = 0.01f;
    desc.render.far_plane = 10.0f;
    desc.render.precision = GSX_RENDER_PRECISION_FLOAT32;
    desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    desc.render.sh_degree = 0;
    desc.render.borrow_train_state = true;
    desc.optim_step.force_all = true;
    desc.workspace.buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    desc.workspace.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    desc.workspace.auto_plan = true;
    desc.reporting.retain_prediction = true;
    desc.reporting.retain_target = true;
    desc.reporting.retain_loss_map = true;
    desc.reporting.retain_grad_prediction = true;
    desc.reporting.collect_timings = true;
    desc.initial_global_step = 7;
    desc.initial_epoch_index = 3;

    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &desc));
    ASSERT_GSX_SUCCESS(gsx_session_get_desc(session, &effective_desc));
    EXPECT_EQ(effective_desc.loss_count, 1u);
    ASSERT_NE(effective_desc.loss_items, nullptr);
    EXPECT_GT(effective_desc.workspace.arena_desc.initial_capacity_bytes, 0u);

    ASSERT_GSX_SUCCESS(gsx_session_step(session));
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
    ASSERT_GSX_SUCCESS(gsx_session_get_last_step_report(session, &report));
    ASSERT_GSX_SUCCESS(gsx_session_get_last_outputs(session, &outputs));
    ASSERT_GSX_SUCCESS(gsx_optim_get_learning_rate_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &learning_rate));

    EXPECT_EQ(state.global_step, 8u);
    EXPECT_EQ(state.epoch_index, 4u);
    EXPECT_EQ(state.successful_step_count, 1u);
    EXPECT_EQ(state.failed_step_count, 0u);
    EXPECT_EQ(report.global_step_before, 7u);
    EXPECT_EQ(report.global_step_after, 8u);
    EXPECT_EQ(report.epoch_index_before, 3u);
    EXPECT_EQ(report.epoch_index_after, 4u);
    EXPECT_TRUE(report.has_applied_learning_rate);
    EXPECT_NEAR(report.applied_learning_rate, 0.03f, 1e-6f);
    EXPECT_NEAR(learning_rate, 0.03f, 1e-6f);
    EXPECT_TRUE(report.outputs_available);
    EXPECT_TRUE(report.has_timings);
    EXPECT_NE(outputs.prediction, nullptr);
    EXPECT_NE(outputs.target, nullptr);
    EXPECT_NE(outputs.loss_map, nullptr);
    EXPECT_NE(outputs.grad_prediction, nullptr);

    ASSERT_GSX_SUCCESS(gsx_session_reset(session));
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
    EXPECT_EQ(state.global_step, 7u);
    EXPECT_EQ(state.epoch_index, 3u);
    EXPECT_EQ(state.successful_step_count, 0u);
    EXPECT_EQ(state.failed_step_count, 0u);
    EXPECT_GSX_CODE(gsx_session_get_last_step_report(session, &report), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_session_get_last_outputs(session, &outputs), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_learning_rate(scheduler, &learning_rate));
    EXPECT_NEAR(learning_rate, 0.03f, 1e-6f);

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
    ASSERT_GSX_SUCCESS(gsx_loss_context_free(loss_context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(SessionRuntime, StepsAcrossDatasetTrackEpochsAndCanSkipRetainedOutputs)
{
    SessionDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t loss_context = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_loss_item loss_item{};
    gsx_session_desc desc{};
    gsx_session_state state{};
    gsx_session_step_report report{};
    gsx_session_outputs outputs{};
    std::array<gsx_size_t, 4> sample_hit_count{ 0, 0, 0, 0 };

    for(gsx_size_t i = 0; i < 4; ++i) {
        SessionDatasetSample sample{};
        sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
        sample.intrinsics.fx = 100.0f + 5.0f * (gsx_float_t)i;
        sample.intrinsics.fy = 102.0f + 7.0f * (gsx_float_t)i;
        sample.intrinsics.cx = 64.0f;
        sample.intrinsics.cy = 64.0f;
        sample.intrinsics.camera_id = (gsx_id_t)i;
        sample.intrinsics.width = 64;
        sample.intrinsics.height = 64;
        sample.pose.rot.w = 1.0f;
        sample.pose.camera_id = (gsx_id_t)i;
        sample.pose.frame_id = (gsx_id_t)(100 + i);
        sample.rgb = make_rgb_pattern(64, 64, 0.05f * (gsx_float_t)i);
        sample.release_token = reinterpret_cast<void *>(uintptr_t{ 0x2000u + i });
        dataset_object.samples.push_back(sample);
    }

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    gs = create_initialized_gs(buffer_type, 1);
    optim = create_optim_for_gs(backend, gs);
    renderer = create_renderer(backend, 64, 64, GSX_DATA_TYPE_F32);
    loss = create_loss(backend, GSX_LOSS_ALGORITHM_MSE);
    loss_context = create_loss_context(loss);
    dataset = create_dataset(&dataset_object);
    dataloader = create_dataloader(backend, dataset, 64, 64, GSX_DATA_TYPE_F32);

    loss_item.loss = loss;
    loss_item.context = loss_context;
    loss_item.scale = 1.0f;

    desc.backend = backend;
    desc.gs = gs;
    desc.optim = optim;
    desc.renderer = renderer;
    desc.train_dataloader = dataloader;
    desc.loss_count = 1;
    desc.loss_items = &loss_item;
    desc.render.near_plane = 0.01f;
    desc.render.far_plane = 10.0f;
    desc.render.precision = GSX_RENDER_PRECISION_FLOAT32;
    desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    desc.render.sh_degree = 0;
    desc.optim_step.force_all = true;
    desc.workspace.buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    desc.workspace.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    desc.workspace.auto_plan = true;
    desc.reporting.collect_timings = false;

    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &desc));

    for(gsx_size_t step_index = 0; step_index < 6; ++step_index) {
        ASSERT_GSX_SUCCESS(gsx_session_step(session));
        ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state));
        ASSERT_GSX_SUCCESS(gsx_session_get_last_step_report(session, &report));
        sample_hit_count[(size_t)report.stable_sample_index] += 1;

        EXPECT_EQ(state.global_step, step_index + 1);
        EXPECT_EQ(state.successful_step_count, step_index + 1);
        EXPECT_EQ(state.failed_step_count, 0u);
        EXPECT_FALSE(report.has_applied_learning_rate);
        EXPECT_FALSE(report.outputs_available);
        EXPECT_FALSE(report.has_timings);
        EXPECT_EQ(state.epoch_index, step_index >= 4 ? 2u : 1u);
    }

    EXPECT_GSX_CODE(gsx_session_get_last_outputs(session, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_session_get_last_outputs(session, &outputs), GSX_ERROR_INVALID_STATE);

    EXPECT_EQ(dataset_object.get_sample_calls, 6u);
    EXPECT_EQ(dataset_object.release_calls, 6u);
    EXPECT_EQ(dataset_object.fetched_indices.size(), 6u);
    EXPECT_EQ(dataset_object.released_tokens.size(), 6u);
    for(gsx_size_t i = 0; i < sample_hit_count.size(); ++i) {
        EXPECT_GT(sample_hit_count[i], 0u);
    }

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_loss_context_free(loss_context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(SessionRuntime, CheckpointRoundTripRestoresStateAndScheduler)
{
    SessionDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t loss_context = nullptr;
    gsx_scheduler_t scheduler = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_loss_item loss_item{};
    gsx_session_desc desc{};
    gsx_session_state saved_state{};
    gsx_session_state restored_state{};
    gsx_checkpoint_info checkpoint_info{};
    gsx_scheduler_state saved_scheduler_state{};
    gsx_scheduler_state restored_scheduler_state{};
    MemoryCheckpoint checkpoint{};
    gsx_io_writer writer{ &checkpoint, checkpoint_writer };
    gsx_io_reader reader{ &checkpoint, checkpoint_reader };

    SessionDatasetSample sample{};
    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.fx = 1.0f;
    sample.intrinsics.fy = 1.0f;
    sample.intrinsics.cx = 0.5f;
    sample.intrinsics.cy = 0.5f;
    sample.intrinsics.width = 1;
    sample.intrinsics.height = 1;
    sample.pose.rot.w = 1.0f;
    sample.rgb = { 0.2f, 0.1f, 0.05f };
    dataset_object.samples.push_back(sample);

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    gs = create_initialized_gs(buffer_type, 1);
    optim = create_optim_for_gs(backend, gs);
    renderer = create_renderer(backend, 1, 1, GSX_DATA_TYPE_F32);
    loss = create_loss(backend, GSX_LOSS_ALGORITHM_MSE);
    loss_context = create_loss_context(loss);
    scheduler = create_constant_scheduler(0.02f);
    dataset = create_dataset(&dataset_object);
    dataloader = create_dataloader(backend, dataset, 1, 1, GSX_DATA_TYPE_F32);

    loss_item.loss = loss;
    loss_item.context = loss_context;
    loss_item.scale = 1.0f;

    desc.backend = backend;
    desc.gs = gs;
    desc.optim = optim;
    desc.renderer = renderer;
    desc.train_dataloader = dataloader;
    desc.scheduler = scheduler;
    desc.loss_count = 1;
    desc.loss_items = &loss_item;
    desc.render.near_plane = 0.01f;
    desc.render.far_plane = 10.0f;
    desc.render.precision = GSX_RENDER_PRECISION_FLOAT32;
    desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    desc.render.sh_degree = 0;
    desc.optim_step.force_all = true;
    desc.workspace.buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    desc.workspace.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    desc.workspace.auto_plan = true;

    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &desc));
    ASSERT_GSX_SUCCESS(gsx_session_step(session));
    ASSERT_GSX_SUCCESS(gsx_session_step(session));
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &saved_state));
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_state(scheduler, &saved_scheduler_state));
    ASSERT_GSX_SUCCESS(gsx_session_save_checkpoint(session, &writer, nullptr));

    ASSERT_GSX_SUCCESS(gsx_session_step(session));
    ASSERT_GSX_SUCCESS(gsx_session_step(session));

    checkpoint.read_offset = 0;
    ASSERT_GSX_SUCCESS(gsx_session_load_checkpoint(session, &reader, &checkpoint_info));
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &restored_state));
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_state(scheduler, &restored_scheduler_state));

    EXPECT_EQ(restored_state.global_step, saved_state.global_step);
    EXPECT_EQ(restored_state.epoch_index, saved_state.epoch_index);
    EXPECT_EQ(restored_state.successful_step_count, saved_state.successful_step_count);
    EXPECT_EQ(restored_state.failed_step_count, saved_state.failed_step_count);
    EXPECT_EQ(restored_scheduler_state.current_step, saved_scheduler_state.current_step);
    EXPECT_NEAR(restored_scheduler_state.current_learning_rate, saved_scheduler_state.current_learning_rate, 1e-6f);
    EXPECT_EQ(checkpoint_info.global_step, saved_state.global_step);
    EXPECT_EQ(checkpoint_info.epoch_index, saved_state.epoch_index);
    EXPECT_EQ(checkpoint_info.format_version, 2u);

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
    ASSERT_GSX_SUCCESS(gsx_loss_context_free(loss_context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(SessionRuntime, FailedStepPreservesPreviousSuccessfulReport)
{
    SessionDataset dataset_object{};
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_gs_t gs = nullptr;
    gsx_optim_t optim = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t loss_context = nullptr;
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_session_t session = nullptr;
    gsx_loss_item loss_item{};
    gsx_session_desc desc{};
    gsx_session_state state_before_failure{};
    gsx_session_state state_after_failure{};
    gsx_session_step_report report_before_failure{};
    gsx_session_step_report report_after_failure{};

    SessionDatasetSample sample{};
    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.fx = 1.0f;
    sample.intrinsics.fy = 1.0f;
    sample.intrinsics.cx = 0.5f;
    sample.intrinsics.cy = 0.5f;
    sample.intrinsics.width = 1;
    sample.intrinsics.height = 1;
    sample.pose.rot.w = 1.0f;
    sample.rgb = { 0.2f, 0.1f, 0.05f };
    dataset_object.samples.push_back(sample);

    backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(backend);
    gs = create_initialized_gs(buffer_type, 1);
    optim = create_optim_for_gs(backend, gs);
    renderer = create_renderer(backend, 1, 1, GSX_DATA_TYPE_F32);
    loss = create_loss(backend, GSX_LOSS_ALGORITHM_MSE);
    loss_context = create_loss_context(loss);
    dataset = create_dataset(&dataset_object);
    dataloader = create_dataloader(backend, dataset, 1, 1, GSX_DATA_TYPE_F32);

    loss_item.loss = loss;
    loss_item.context = loss_context;
    loss_item.scale = 1.0f;

    desc.backend = backend;
    desc.gs = gs;
    desc.optim = optim;
    desc.renderer = renderer;
    desc.train_dataloader = dataloader;
    desc.loss_count = 1;
    desc.loss_items = &loss_item;
    desc.render.near_plane = 0.01f;
    desc.render.far_plane = 10.0f;
    desc.render.precision = GSX_RENDER_PRECISION_FLOAT32;
    desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    desc.render.sh_degree = 0;
    desc.optim_step.force_all = true;
    desc.workspace.buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    desc.workspace.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    desc.workspace.auto_plan = true;
    desc.reporting.retain_prediction = true;
    desc.reporting.retain_target = true;

    ASSERT_GSX_SUCCESS(gsx_session_init(&session, &desc));
    ASSERT_GSX_SUCCESS(gsx_session_step(session));
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state_before_failure));
    ASSERT_GSX_SUCCESS(gsx_session_get_last_step_report(session, &report_before_failure));

    ASSERT_GSX_SUCCESS(gsx_dataloader_set_output_shape(dataloader, 2, 2));
    {
        const gsx_error error = gsx_session_step(session);
        EXPECT_NE(error.code, GSX_ERROR_SUCCESS);
    }
    ASSERT_GSX_SUCCESS(gsx_session_get_state(session, &state_after_failure));
    ASSERT_GSX_SUCCESS(gsx_session_get_last_step_report(session, &report_after_failure));

    EXPECT_EQ(state_after_failure.global_step, state_before_failure.global_step);
    EXPECT_EQ(state_after_failure.epoch_index, state_before_failure.epoch_index);
    EXPECT_EQ(state_after_failure.successful_step_count, state_before_failure.successful_step_count);
    EXPECT_EQ(state_after_failure.failed_step_count, state_before_failure.failed_step_count + 1u);
    EXPECT_EQ(report_after_failure.global_step_after, report_before_failure.global_step_after);
    EXPECT_EQ(report_after_failure.epoch_index_after, report_before_failure.epoch_index_after);

    ASSERT_GSX_SUCCESS(gsx_session_free(session));
    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_loss_context_free(loss_context));
    ASSERT_GSX_SUCCESS(gsx_loss_free(loss));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
