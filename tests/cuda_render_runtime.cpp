#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
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

static gsx_backend_device_t get_cuda_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CUDA, 0, &backend_device), GSX_ERROR_SUCCESS);
    return backend_device;
}

static gsx_backend_t create_cuda_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc desc{};

    desc.device = get_cuda_backend_device();
    EXPECT_NE(desc.device, nullptr);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &desc), GSX_ERROR_SUCCESS);
    return backend;
}

static void sync_cuda_backend(gsx_backend_t backend)
{
    void *stream = nullptr;

    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    ASSERT_EQ(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), cudaSuccess);
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type_class)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, type_class, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static gsx_arena_t create_arena(gsx_backend_buffer_type_t buffer_type)
{
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};

    desc.initial_capacity_bytes = 1u << 20;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    EXPECT_GSX_CODE(gsx_arena_init(&arena, buffer_type, &desc), GSX_ERROR_SUCCESS);
    return arena;
}

static gsx_size_t product(const std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> &shape, gsx_index_t rank)
{
    gsx_size_t total = 1;

    for(gsx_index_t dim = 0; dim < rank; ++dim) {
        total *= (gsx_size_t)shape[(std::size_t)dim];
    }
    return total;
}

static gsx_tensor_t make_f32_tensor(
    gsx_arena_t arena,
    const std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> &shape,
    gsx_index_t rank,
    const std::vector<float> &values,
    gsx_storage_format storage_format = GSX_STORAGE_FORMAT_CHW)
{
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc desc{};

    desc.rank = rank;
    for(std::size_t i = 0; i < shape.size(); ++i) {
        desc.shape[i] = shape[i];
    }
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = storage_format;
    desc.arena = arena;
    EXPECT_GSX_CODE(gsx_tensor_init(&tensor, &desc), GSX_ERROR_SUCCESS);
    if(tensor != nullptr && !values.empty()) {
        EXPECT_EQ(values.size(), (std::size_t)product(shape, rank));
        EXPECT_GSX_CODE(gsx_tensor_upload(tensor, values.data(), (gsx_size_t)values.size() * sizeof(float)), GSX_ERROR_SUCCESS);
    }
    return tensor;
}

static std::vector<float> download_f32_tensor(gsx_backend_t backend, gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    std::vector<float> values;

    EXPECT_GSX_CODE(gsx_tensor_get_info(tensor, &info), GSX_ERROR_SUCCESS);
    values.resize((std::size_t)(info.size_bytes / sizeof(float)));
    EXPECT_GSX_CODE(gsx_tensor_download(tensor, values.data(), info.size_bytes), GSX_ERROR_SUCCESS);
    sync_cuda_backend(backend);
    return values;
}

static void upload_f32_tensor(gsx_backend_t backend, gsx_tensor_t tensor, const std::vector<float> &values)
{
    EXPECT_GSX_CODE(gsx_tensor_upload(tensor, values.data(), (gsx_size_t)values.size() * sizeof(float)), GSX_ERROR_SUCCESS);
    sync_cuda_backend(backend);
}

static gsx_renderer_t create_renderer(gsx_backend_t backend, gsx_index_t width, gsx_index_t height, gsx_renderer_feature_flags flags = 0)
{
    gsx_renderer_t renderer = nullptr;
    gsx_renderer_desc desc{};

    desc.width = width;
    desc.height = height;
    desc.output_data_type = GSX_DATA_TYPE_F32;
    desc.feature_flags = flags;
    EXPECT_GSX_CODE(gsx_renderer_init(&renderer, backend, &desc), GSX_ERROR_SUCCESS);
    return renderer;
}

static gsx_render_context_t create_context(gsx_renderer_t renderer)
{
    gsx_render_context_t context = nullptr;

    EXPECT_GSX_CODE(gsx_render_context_init(&context, renderer), GSX_ERROR_SUCCESS);
    return context;
}

static gsx_camera_intrinsics make_intrinsics(gsx_index_t width, gsx_index_t height, float fx, float fy, float cx, float cy)
{
    gsx_camera_intrinsics intrinsics{};

    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.width = width;
    intrinsics.height = height;
    intrinsics.fx = fx;
    intrinsics.fy = fy;
    intrinsics.cx = cx;
    intrinsics.cy = cy;
    return intrinsics;
}

static gsx_camera_pose make_identity_pose()
{
    gsx_camera_pose pose{};

    pose.rot.w = 1.0f;
    return pose;
}

static float dot_product(const std::vector<float> &lhs, const std::vector<float> &rhs)
{
    float sum = 0.0f;

    EXPECT_EQ(lhs.size(), rhs.size());
    for(std::size_t i = 0; i < lhs.size(); ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

static bool all_finite(const std::vector<float> &values)
{
    for(float value : values) {
        if(!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static bool any_abs_greater_than(const std::vector<float> &values, float threshold)
{
    for(float value : values) {
        if(std::fabs(value) > threshold) {
            return true;
        }
    }
    return false;
}

struct CudaRenderScene {
    gsx_index_t width = 0;
    gsx_index_t height = 0;
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_type_t host_pinned_buffer_type = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_render_context_t context = nullptr;
    gsx_arena_t device_arena = nullptr;
    gsx_arena_t host_arena = nullptr;
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t logscale = nullptr;
    gsx_tensor_t sh0 = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t out_rgb = nullptr;
    gsx_tensor_t grad_rgb = nullptr;
    gsx_tensor_t grad_mean3d = nullptr;
    gsx_tensor_t grad_rotation = nullptr;
    gsx_tensor_t grad_logscale = nullptr;
    gsx_tensor_t grad_sh0 = nullptr;
    gsx_tensor_t grad_opacity = nullptr;
    gsx_camera_intrinsics intrinsics{};
    gsx_camera_pose pose{};
    gsx_vec3 background_color{};
    gsx_render_forward_request forward{};
    gsx_render_backward_request backward{};
    std::vector<float> grad_rgb_values;
};

static void destroy_scene(CudaRenderScene *scene)
{
    if(scene->grad_opacity != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_opacity));
    }
    if(scene->grad_sh0 != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_sh0));
    }
    if(scene->grad_logscale != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_logscale));
    }
    if(scene->grad_rotation != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_rotation));
    }
    if(scene->grad_mean3d != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_mean3d));
    }
    if(scene->grad_rgb != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_rgb));
    }
    if(scene->out_rgb != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->out_rgb));
    }
    if(scene->opacity != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->opacity));
    }
    if(scene->sh0 != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->sh0));
    }
    if(scene->logscale != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->logscale));
    }
    if(scene->rotation != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->rotation));
    }
    if(scene->mean3d != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->mean3d));
    }
    if(scene->host_arena != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_arena_free(scene->host_arena));
    }
    if(scene->device_arena != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_arena_free(scene->device_arena));
    }
    if(scene->context != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_render_context_free(scene->context));
    }
    if(scene->renderer != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_renderer_free(scene->renderer));
    }
    if(scene->backend != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_backend_free(scene->backend));
    }
}

static CudaRenderScene make_scene()
{
    CudaRenderScene scene{};
    const gsx_index_t width = 1;
    const gsx_index_t height = 1;

    scene.width = width;
    scene.height = height;
    scene.backend = create_cuda_backend();
    scene.device_buffer_type = find_buffer_type(scene.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    scene.host_pinned_buffer_type = find_buffer_type(scene.backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED);
    scene.device_arena = create_arena(scene.device_buffer_type);
    scene.host_arena = create_arena(scene.host_pinned_buffer_type);
    scene.renderer = create_renderer(scene.backend, width, height);
    scene.context = create_context(scene.renderer);
    scene.intrinsics = make_intrinsics(width, height, 1.0f, 1.0f, 0.5f, 0.5f);
    scene.pose = make_identity_pose();
    scene.background_color = gsx_vec3{ 0.10f, 0.10f, 0.10f };

    scene.mean3d = make_f32_tensor(scene.device_arena, { 1, 3, 0, 0 }, 2, { 0.4f, -0.3f, 4.0f });
    scene.rotation = make_f32_tensor(scene.device_arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f, 1.0f });
    scene.logscale = make_f32_tensor(scene.device_arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f });
    scene.sh0 = make_f32_tensor(scene.device_arena, { 1, 3, 0, 0 }, 2, { 1.0f, 0.0f, 0.0f });
    scene.opacity = make_f32_tensor(scene.device_arena, { 1, 0, 0, 0 }, 1, { 0.0f });
    scene.out_rgb = make_f32_tensor(scene.device_arena, { 3, height, width, 0 }, 3, std::vector<float>(3 * width * height, 0.0f));

    scene.grad_rgb_values.resize((std::size_t)(3 * width * height));
    scene.grad_rgb_values = { 30.0f, -10.0f, 20.0f };
    scene.grad_rgb = make_f32_tensor(scene.device_arena, { 3, height, width, 0 }, 3, scene.grad_rgb_values);
    scene.grad_mean3d = make_f32_tensor(scene.device_arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_rotation = make_f32_tensor(scene.device_arena, { 1, 4, 0, 0 }, 2, std::vector<float>(4, 0.0f));
    scene.grad_logscale = make_f32_tensor(scene.device_arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_sh0 = make_f32_tensor(scene.device_arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_opacity = make_f32_tensor(scene.device_arena, { 1, 0, 0, 0 }, 1, std::vector<float>(1, 0.0f));
    sync_cuda_backend(scene.backend);

    scene.forward.intrinsics = &scene.intrinsics;
    scene.forward.pose = &scene.pose;
    scene.forward.near_plane = 0.1f;
    scene.forward.far_plane = 20.0f;
    scene.forward.background_color = scene.background_color;
    scene.forward.precision = GSX_RENDER_PRECISION_FLOAT32;
    scene.forward.sh_degree = 0;
    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    scene.forward.borrow_train_state = false;
    scene.forward.gs_mean3d = scene.mean3d;
    scene.forward.gs_rotation = scene.rotation;
    scene.forward.gs_logscale = scene.logscale;
    scene.forward.gs_sh0 = scene.sh0;
    scene.forward.gs_opacity = scene.opacity;
    scene.forward.out_rgb = scene.out_rgb;

    scene.backward.grad_rgb = scene.grad_rgb;
    scene.backward.grad_gs_mean3d = scene.grad_mean3d;
    scene.backward.grad_gs_rotation = scene.grad_rotation;
    scene.backward.grad_gs_logscale = scene.grad_logscale;
    scene.backward.grad_gs_sh0 = scene.grad_sh0;
    scene.backward.grad_gs_opacity = scene.grad_opacity;
    return scene;
}

static float numerical_gradient(
    CudaRenderScene *scene,
    gsx_tensor_t parameter,
    std::vector<float> values,
    std::size_t index,
    float epsilon)
{
    std::vector<float> output_values;
    float loss_plus = 0.0f;
    float loss_minus = 0.0f;
    gsx_error error{};

    values[index] += epsilon;
    upload_f32_tensor(scene->backend, parameter, values);
    scene->forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    error = gsx_renderer_render(scene->renderer, scene->context, &scene->forward);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return 0.0f;
    }
    output_values = download_f32_tensor(scene->backend, scene->out_rgb);
    loss_plus = dot_product(output_values, scene->grad_rgb_values);

    values[index] -= 2.0f * epsilon;
    upload_f32_tensor(scene->backend, parameter, values);
    error = gsx_renderer_render(scene->renderer, scene->context, &scene->forward);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return 0.0f;
    }
    output_values = download_f32_tensor(scene->backend, scene->out_rgb);
    loss_minus = dot_product(output_values, scene->grad_rgb_values);

    values[index] += epsilon;
    upload_f32_tensor(scene->backend, parameter, values);
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

static void expect_vectors_near(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "mismatch at index " << i;
    }
}

class CudaRenderRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override {
        if(!has_cuda_device()) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CudaRenderRuntimeTest, RendererLifecycleAndCapabilities)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_renderer_t renderer = nullptr;
    gsx_render_context_t context = nullptr;
    gsx_renderer_desc desc{};
    gsx_renderer_info info{};
    gsx_renderer_capabilities capabilities{};
    gsx_data_type output_data_type = GSX_DATA_TYPE_U8;

    desc.width = 4;
    desc.height = 3;
    desc.output_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_renderer_init(&renderer, backend, &desc));
    ASSERT_NE(renderer, nullptr);

    ASSERT_GSX_SUCCESS(gsx_renderer_get_info(renderer, &info));
    EXPECT_EQ(info.width, 4);
    EXPECT_EQ(info.height, 3);
    EXPECT_EQ(info.output_data_type, GSX_DATA_TYPE_F32);
    EXPECT_FALSE(info.enable_alpha_output);
    EXPECT_FALSE(info.enable_invdepth_output);

    ASSERT_GSX_SUCCESS(gsx_renderer_get_capabilities(renderer, &capabilities));
    EXPECT_EQ(capabilities.supported_precisions, GSX_RENDER_PRECISION_FLAG_FLOAT32);
    EXPECT_FALSE(capabilities.supports_alpha_output);
    EXPECT_FALSE(capabilities.supports_invdepth_output);
    EXPECT_FALSE(capabilities.supports_cov3d_input);

    ASSERT_GSX_SUCCESS(gsx_renderer_get_output_data_type(renderer, GSX_RENDER_PRECISION_FLOAT32, &output_data_type));
    EXPECT_EQ(output_data_type, GSX_DATA_TYPE_F32);
    EXPECT_GSX_CODE(gsx_renderer_get_output_data_type(renderer, GSX_RENDER_PRECISION_FLOAT16, &output_data_type), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_render_context_init(&context, renderer));
    EXPECT_GSX_CODE(gsx_renderer_free(renderer), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaRenderRuntimeTest, RendererRejectsUnsupportedConfigurationAndFields)
{
    CudaRenderScene scene = make_scene();
    gsx_renderer_t bad_renderer = nullptr;
    gsx_renderer_desc bad_desc{};
    gsx_tensor_t cov3d = make_f32_tensor(scene.device_arena, { 1, 6, 0, 0 }, 2, std::vector<float>(6, 0.0f));
    gsx_tensor_t out_alpha = make_f32_tensor(
        scene.device_arena,
        { 1, scene.height, scene.width, 0 },
        3,
        std::vector<float>((std::size_t)(scene.width * scene.height), 0.0f));
    gsx_tensor_t out_invdepth = make_f32_tensor(
        scene.device_arena,
        { 1, scene.height, scene.width, 0 },
        3,
        std::vector<float>((std::size_t)(scene.width * scene.height), 0.0f));
    gsx_tensor_t metric_map = make_f32_tensor(
        scene.device_arena,
        { 1, scene.height, scene.width, 0 },
        3,
        std::vector<float>((std::size_t)(scene.width * scene.height), 0.0f));

    bad_desc.width = scene.width;
    bad_desc.height = scene.height;
    bad_desc.output_data_type = GSX_DATA_TYPE_F16;
    EXPECT_GSX_CODE(gsx_renderer_init(&bad_renderer, scene.backend, &bad_desc), GSX_ERROR_NOT_SUPPORTED);

    bad_desc.output_data_type = GSX_DATA_TYPE_F32;
    bad_desc.enable_alpha_output = true;
    EXPECT_GSX_CODE(gsx_renderer_init(&bad_renderer, scene.backend, &bad_desc), GSX_ERROR_NOT_SUPPORTED);

    bad_desc.enable_alpha_output = false;
    bad_desc.enable_invdepth_output = true;
    EXPECT_GSX_CODE(gsx_renderer_init(&bad_renderer, scene.backend, &bad_desc), GSX_ERROR_NOT_SUPPORTED);

    bad_desc.enable_invdepth_output = false;
    bad_desc.feature_flags = GSX_RENDERER_FEATURE_ANTIALIASING;
    EXPECT_GSX_CODE(gsx_renderer_init(&bad_renderer, scene.backend, &bad_desc), GSX_ERROR_NOT_SUPPORTED);

    scene.forward.precision = GSX_RENDER_PRECISION_FLOAT16;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);

    scene.forward.precision = GSX_RENDER_PRECISION_FLOAT32;
    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_METRIC;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    scene.forward.gs_cov3d = cov3d;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);

    scene.forward.gs_cov3d = nullptr;
    scene.forward.out_alpha = out_alpha;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);

    scene.forward.out_alpha = nullptr;
    scene.forward.out_invdepth = out_invdepth;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);

    scene.forward.out_invdepth = nullptr;
    scene.forward.metric_map = metric_map;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(metric_map));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_invdepth));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_alpha));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(cov3d));
    destroy_scene(&scene);
}

TEST_F(CudaRenderRuntimeTest, RendererRejectsHostPinnedTensorBindings)
{
    CudaRenderScene scene = make_scene();
    gsx_tensor_t host_out_rgb = make_f32_tensor(
        scene.host_arena,
        { 3, scene.height, scene.width, 0 },
        3,
        std::vector<float>((std::size_t)(3 * scene.width * scene.height), 0.0f));

    scene.forward.out_rgb = host_out_rgb;
    EXPECT_GSX_CODE(gsx_renderer_render(scene.renderer, scene.context, &scene.forward), GSX_ERROR_NOT_SUPPORTED);
    scene.forward.out_rgb = scene.out_rgb;

    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    scene.backward.grad_gs_mean3d = make_f32_tensor(scene.host_arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_NOT_SUPPORTED);
    ASSERT_GSX_SUCCESS(gsx_tensor_free(scene.backward.grad_gs_mean3d));
    scene.backward.grad_gs_mean3d = scene.grad_mean3d;

    ASSERT_GSX_SUCCESS(gsx_tensor_free(host_out_rgb));
    destroy_scene(&scene);
}

TEST_F(CudaRenderRuntimeTest, ForwardProducesFiniteImage)
{
    CudaRenderScene scene = make_scene();
    std::vector<float> output = {};

    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    output = download_f32_tensor(scene.backend, scene.out_rgb);

    ASSERT_TRUE(all_finite(output));
    ASSERT_TRUE(any_abs_greater_than(output, 1.0e-4f));

    const float background_sum = scene.background_color.x + scene.background_color.y + scene.background_color.z;
    const std::size_t plane_stride = (std::size_t)(scene.width * scene.height);
    const float pixel_sum = output[0] + output[plane_stride] + output[2 * plane_stride];
    EXPECT_NEAR(pixel_sum, background_sum, 1.0f);
    EXPECT_GT(std::fabs(pixel_sum - background_sum), 1.0e-3f);

    destroy_scene(&scene);
}

TEST_F(CudaRenderRuntimeTest, TrainForwardAndBackwardProduceFiniteGradients)
{
    CudaRenderScene scene = make_scene();
    std::vector<float> grad_mean;
    std::vector<float> grad_rotation;
    std::vector<float> grad_logscale;
    std::vector<float> grad_sh0;
    std::vector<float> grad_opacity;

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));

    grad_mean = download_f32_tensor(scene.backend, scene.grad_mean3d);
    grad_rotation = download_f32_tensor(scene.backend, scene.grad_rotation);
    grad_logscale = download_f32_tensor(scene.backend, scene.grad_logscale);
    grad_sh0 = download_f32_tensor(scene.backend, scene.grad_sh0);
    grad_opacity = download_f32_tensor(scene.backend, scene.grad_opacity);

    EXPECT_TRUE(all_finite(grad_mean));
    EXPECT_TRUE(all_finite(grad_rotation));
    EXPECT_TRUE(all_finite(grad_logscale));
    EXPECT_TRUE(all_finite(grad_sh0));
    EXPECT_TRUE(all_finite(grad_opacity));
    EXPECT_TRUE(any_abs_greater_than(grad_mean, 1.0e-5f));
    EXPECT_TRUE(any_abs_greater_than(grad_logscale, 1.0e-5f));
    EXPECT_TRUE(any_abs_greater_than(grad_sh0, 1.0e-5f));
    EXPECT_TRUE(any_abs_greater_than(grad_opacity, 1.0e-6f));

    destroy_scene(&scene);
}

TEST_F(CudaRenderRuntimeTest, BackwardUsesRetainedTrainState)
{
    CudaRenderScene reference = make_scene();
    CudaRenderScene retained = make_scene();
    std::vector<float> reference_grad_mean;
    std::vector<float> reference_grad_logscale;
    std::vector<float> reference_grad_sh0;
    std::vector<float> reference_grad_opacity;

    reference.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(reference.renderer, reference.context, &reference.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(reference.renderer, reference.context, &reference.backward));
    reference_grad_mean = download_f32_tensor(reference.backend, reference.grad_mean3d);
    reference_grad_logscale = download_f32_tensor(reference.backend, reference.grad_logscale);
    reference_grad_sh0 = download_f32_tensor(reference.backend, reference.grad_sh0);
    reference_grad_opacity = download_f32_tensor(reference.backend, reference.grad_opacity);

    retained.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(retained.renderer, retained.context, &retained.forward));

    upload_f32_tensor(retained.backend, retained.mean3d, { -1.2f, 0.8f, 9.0f });
    upload_f32_tensor(retained.backend, retained.logscale, { 0.7f, 0.6f, 0.5f });
    upload_f32_tensor(retained.backend, retained.sh0, { 0.0f, 0.0f, 0.0f });
    upload_f32_tensor(retained.backend, retained.opacity, { -5.0f });
    upload_f32_tensor(
        retained.backend,
        retained.out_rgb,
        std::vector<float>((std::size_t)(3 * retained.width * retained.height), 0.0f));

    ASSERT_GSX_SUCCESS(gsx_renderer_backward(retained.renderer, retained.context, &retained.backward));

    expect_vectors_near(download_f32_tensor(retained.backend, retained.grad_mean3d), reference_grad_mean, 1.0e-5f);
    expect_vectors_near(download_f32_tensor(retained.backend, retained.grad_logscale), reference_grad_logscale, 1.0e-5f);
    expect_vectors_near(download_f32_tensor(retained.backend, retained.grad_sh0), reference_grad_sh0, 1.0e-5f);
    expect_vectors_near(download_f32_tensor(retained.backend, retained.grad_opacity), reference_grad_opacity, 1.0e-5f);

    destroy_scene(&retained);
    destroy_scene(&reference);
}

TEST_F(CudaRenderRuntimeTest, BackwardBorrowedTrainStateTracksInputMutations)
{
    CudaRenderScene reference = make_scene();
    CudaRenderScene borrowed = make_scene();
    std::vector<float> reference_grad_mean;
    std::vector<float> reference_grad_logscale;
    std::vector<float> reference_grad_sh0;
    std::vector<float> reference_grad_opacity;
    const std::vector<float> mutated_mean = { -1.2f, 0.8f, 9.0f };
    const std::vector<float> mutated_logscale = { 0.7f, 0.6f, 0.5f };
    const std::vector<float> mutated_sh0 = { 0.0f, 0.0f, 0.0f };
    const std::vector<float> mutated_opacity = { -5.0f };
    auto has_difference = [](const std::vector<float> &lhs, const std::vector<float> &rhs, float tolerance) {
        if(lhs.size() != rhs.size()) {
            return true;
        }
        for(std::size_t i = 0; i < lhs.size(); ++i) {
            if(std::fabs(lhs[i] - rhs[i]) > tolerance) {
                return true;
            }
        }
        return false;
    };

    reference.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    reference.forward.borrow_train_state = false;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(reference.renderer, reference.context, &reference.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(reference.renderer, reference.context, &reference.backward));
    reference_grad_mean = download_f32_tensor(reference.backend, reference.grad_mean3d);
    reference_grad_logscale = download_f32_tensor(reference.backend, reference.grad_logscale);
    reference_grad_sh0 = download_f32_tensor(reference.backend, reference.grad_sh0);
    reference_grad_opacity = download_f32_tensor(reference.backend, reference.grad_opacity);

    borrowed.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    borrowed.forward.borrow_train_state = true;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(borrowed.renderer, borrowed.context, &borrowed.forward));
    upload_f32_tensor(borrowed.backend, borrowed.mean3d, mutated_mean);
    upload_f32_tensor(borrowed.backend, borrowed.logscale, mutated_logscale);
    upload_f32_tensor(borrowed.backend, borrowed.sh0, mutated_sh0);
    upload_f32_tensor(borrowed.backend, borrowed.opacity, mutated_opacity);
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(borrowed.renderer, borrowed.context, &borrowed.backward));

    EXPECT_TRUE(has_difference(download_f32_tensor(borrowed.backend, borrowed.grad_mean3d), reference_grad_mean, 1.0e-3f));
    EXPECT_TRUE(has_difference(download_f32_tensor(borrowed.backend, borrowed.grad_logscale), reference_grad_logscale, 1.0e-4f));
    EXPECT_TRUE(has_difference(download_f32_tensor(borrowed.backend, borrowed.grad_opacity), reference_grad_opacity, 1.0e-3f));

    destroy_scene(&borrowed);
    destroy_scene(&reference);
}

TEST_F(CudaRenderRuntimeTest, BackwardRejectsUnsupportedOptionalSinks)
{
    CudaRenderScene scene = make_scene();
    gsx_tensor_t grad_alpha = make_f32_tensor(
        scene.device_arena,
        { 1, scene.height, scene.width, 0 },
        3,
        std::vector<float>((std::size_t)(scene.width * scene.height), 0.0f));
    gsx_tensor_t grad_cov3d = make_f32_tensor(scene.device_arena, { 1, 6, 0, 0 }, 2, std::vector<float>(6, 0.0f));

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));

    scene.backward.grad_alpha = grad_alpha;
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_NOT_SUPPORTED);

    scene.backward.grad_alpha = nullptr;
    scene.backward.grad_gs_cov3d = grad_cov3d;
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_cov3d));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_alpha));
    destroy_scene(&scene);
}

TEST_F(CudaRenderRuntimeTest, BackwardRoughlyMatchesFiniteDifferences)
{
    CudaRenderScene scene = make_scene();
    std::vector<float> analytic_mean;
    std::vector<float> analytic_logscale;
    std::vector<float> analytic_sh0;
    std::vector<float> analytic_opacity;
    std::vector<float> mean_values = download_f32_tensor(scene.backend, scene.mean3d);
    std::vector<float> logscale_values = download_f32_tensor(scene.backend, scene.logscale);
    std::vector<float> sh0_values = download_f32_tensor(scene.backend, scene.sh0);
    std::vector<float> opacity_values = download_f32_tensor(scene.backend, scene.opacity);
    float numeric_mean_z = 0.0f;
    float numeric_logscale_x = 0.0f;
    float numeric_sh0_r = 0.0f;
    float numeric_opacity = 0.0f;

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));

    analytic_mean = download_f32_tensor(scene.backend, scene.grad_mean3d);
    analytic_logscale = download_f32_tensor(scene.backend, scene.grad_logscale);
    analytic_sh0 = download_f32_tensor(scene.backend, scene.grad_sh0);
    analytic_opacity = download_f32_tensor(scene.backend, scene.grad_opacity);

    numeric_mean_z = numerical_gradient(&scene, scene.mean3d, mean_values, 2, 1.0e-3f);
    numeric_logscale_x = numerical_gradient(&scene, scene.logscale, logscale_values, 0, 1.0e-3f);
    numeric_sh0_r = numerical_gradient(&scene, scene.sh0, sh0_values, 0, 1.0e-3f);
    numeric_opacity = numerical_gradient(&scene, scene.opacity, opacity_values, 0, 1.0e-3f);

    EXPECT_NEAR(analytic_mean[2], numeric_mean_z, 1.0e-1f);
    EXPECT_NEAR(analytic_logscale[0], numeric_logscale_x, 1.0e-1f);
    EXPECT_NEAR(analytic_sh0[0], numeric_sh0_r, 1.0e-1f);
    EXPECT_NEAR(analytic_opacity[0], numeric_opacity, 1.25f);

    destroy_scene(&scene);
}

} /* namespace */
