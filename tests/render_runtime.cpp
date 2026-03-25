#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

static gsx_backend_device_t get_cpu_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device), GSX_ERROR_SUCCESS);
    return backend_device;
}

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc desc{};

    desc.device = get_cpu_backend_device();
    EXPECT_NE(desc.device, nullptr);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &desc), GSX_ERROR_SUCCESS);
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
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};

    desc.initial_capacity_bytes = 4096;
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

static std::vector<float> download_f32_tensor(gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    std::vector<float> values;

    EXPECT_GSX_CODE(gsx_tensor_get_info(tensor, &info), GSX_ERROR_SUCCESS);
    values.resize((std::size_t)(info.size_bytes / sizeof(float)));
    EXPECT_GSX_CODE(gsx_tensor_download(tensor, values.data(), info.size_bytes), GSX_ERROR_SUCCESS);
    return values;
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

struct RenderScene {
    gsx_backend_t backend = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_render_context_t context = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t logscale = nullptr;
    gsx_tensor_t sh0 = nullptr;
    gsx_tensor_t sh1 = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t out_rgb = nullptr;
    gsx_tensor_t grad_rgb = nullptr;
    gsx_tensor_t grad_mean3d = nullptr;
    gsx_tensor_t grad_rotation = nullptr;
    gsx_tensor_t grad_logscale = nullptr;
    gsx_tensor_t grad_sh0 = nullptr;
    gsx_tensor_t grad_sh1 = nullptr;
    gsx_tensor_t grad_opacity = nullptr;
    gsx_camera_intrinsics intrinsics{};
    gsx_camera_pose pose{};
    gsx_render_forward_request forward{};
    gsx_render_backward_request backward{};
    std::vector<float> grad_rgb_values;
};

static void destroy_scene(RenderScene *scene)
{
    if(scene->grad_opacity != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_opacity));
    }
    if(scene->grad_sh1 != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->grad_sh1));
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
    if(scene->sh1 != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_tensor_free(scene->sh1));
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
    if(scene->arena != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_arena_free(scene->arena));
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

static RenderScene make_gradient_scene()
{
    RenderScene scene{};
    gsx_backend_buffer_type_t buffer_type = nullptr;

    scene.backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(scene.backend);
    scene.arena = create_arena(buffer_type);
    scene.renderer = create_renderer(scene.backend, 2, 2);
    scene.context = create_context(scene.renderer);
    scene.intrinsics = make_intrinsics(2, 2, 3.0f, 2.5f, 0.8f, 1.1f);
    scene.pose = make_identity_pose();
    scene.mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.2f, -0.1f, 3.5f });
    scene.rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, { 0.1f, -0.05f, 0.02f, 0.99f });
    scene.logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { -0.2f, 0.1f, -0.05f });
    scene.sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.1f, -0.2f, 0.3f });
    scene.sh1 = make_f32_tensor(
        scene.arena,
        { 1, 3, 3, 0 },
        3,
        { 0.05f, -0.02f, 0.01f, -0.03f, 0.04f, -0.05f, 0.02f, 0.01f, -0.04f });
    scene.opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, { 0.4f });
    scene.out_rgb = make_f32_tensor(scene.arena, { 3, 2, 2, 0 }, 3, std::vector<float>(12, 0.0f));
    scene.grad_rgb_values = { 0.3f, -0.1f, 0.2f, 0.05f, -0.2f, 0.4f, -0.3f, 0.1f, 0.25f, -0.15f, 0.35f, -0.05f };
    scene.grad_rgb = make_f32_tensor(scene.arena, { 3, 2, 2, 0 }, 3, scene.grad_rgb_values);
    scene.grad_mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, std::vector<float>(4, 0.0f));
    scene.grad_logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_sh1 = make_f32_tensor(scene.arena, { 1, 3, 3, 0 }, 3, std::vector<float>(9, 0.0f));
    scene.grad_opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, std::vector<float>(1, 0.0f));

    scene.forward.intrinsics = &scene.intrinsics;
    scene.forward.pose = &scene.pose;
    scene.forward.near_plane = 0.1f;
    scene.forward.far_plane = 10.0f;
    scene.forward.background_color = gsx_vec3{ 0.05f, 0.1f, 0.2f };
    scene.forward.precision = GSX_RENDER_PRECISION_FLOAT32;
    scene.forward.sh_degree = 1;
    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    scene.forward.borrow_train_state = false;
    scene.forward.gs_mean3d = scene.mean3d;
    scene.forward.gs_rotation = scene.rotation;
    scene.forward.gs_logscale = scene.logscale;
    scene.forward.gs_sh0 = scene.sh0;
    scene.forward.gs_sh1 = scene.sh1;
    scene.forward.gs_opacity = scene.opacity;
    scene.forward.out_rgb = scene.out_rgb;

    scene.backward.grad_rgb = scene.grad_rgb;
    scene.backward.grad_gs_mean3d = scene.grad_mean3d;
    scene.backward.grad_gs_rotation = scene.grad_rotation;
    scene.backward.grad_gs_logscale = scene.grad_logscale;
    scene.backward.grad_gs_sh0 = scene.grad_sh0;
    scene.backward.grad_gs_sh1 = scene.grad_sh1;
    scene.backward.grad_gs_opacity = scene.grad_opacity;
    return scene;
}

static RenderScene make_near_alpha_saturation_scene()
{
    RenderScene scene{};
    gsx_backend_buffer_type_t buffer_type = nullptr;

    scene.backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(scene.backend);
    scene.arena = create_arena(buffer_type);
    scene.renderer = create_renderer(scene.backend, 1, 1);
    scene.context = create_context(scene.renderer);
    scene.intrinsics = make_intrinsics(1, 1, 1.0f, 1.0f, 0.5f, 0.5f);
    scene.pose = make_identity_pose();
    scene.mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 2.0f });
    scene.rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f, 1.0f });
    scene.logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f });
    scene.sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 1.2f, -0.3f, 0.5f });
    scene.opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, { 5.3f });
    scene.out_rgb = make_f32_tensor(scene.arena, { 3, 1, 1, 0 }, 3, std::vector<float>(3, 0.0f));
    scene.grad_rgb_values = { 0.7f, -0.4f, 0.2f };
    scene.grad_rgb = make_f32_tensor(scene.arena, { 3, 1, 1, 0 }, 3, scene.grad_rgb_values);
    scene.grad_mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, std::vector<float>(4, 0.0f));
    scene.grad_logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, std::vector<float>(1, 0.0f));

    scene.forward.intrinsics = &scene.intrinsics;
    scene.forward.pose = &scene.pose;
    scene.forward.near_plane = 0.1f;
    scene.forward.far_plane = 10.0f;
    scene.forward.background_color = gsx_vec3{ 0.05f, 0.1f, 0.2f };
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

static RenderScene make_rotated_anisotropic_scene()
{
    RenderScene scene{};
    gsx_backend_buffer_type_t buffer_type = nullptr;

    scene.backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(scene.backend);
    scene.arena = create_arena(buffer_type);
    scene.renderer = create_renderer(scene.backend, 8, 8);
    scene.context = create_context(scene.renderer);
    scene.intrinsics = make_intrinsics(8, 8, 12.0f, 11.0f, 4.0f, 4.0f);
    scene.pose = make_identity_pose();
    scene.mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.35f, 0.1f, 3.6f });
    scene.rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.1305262f, 0.9914449f });
    scene.logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { -0.2f, -0.6f, -0.4f });
    scene.sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.1f, 0.4f, 0.9f });
    scene.opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, { -0.1f });
    scene.out_rgb = make_f32_tensor(scene.arena, { 3, 8, 8, 0 }, 3, std::vector<float>(3 * 8 * 8, 0.0f));
    scene.grad_rgb_values.resize(3 * 8 * 8);
    for(std::size_t i = 0; i < scene.grad_rgb_values.size(); ++i) {
        scene.grad_rgb_values[i] = (float)((int)(i % 19) - 9) * 0.07f;
    }
    scene.grad_rgb = make_f32_tensor(scene.arena, { 3, 8, 8, 0 }, 3, scene.grad_rgb_values);
    scene.grad_mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, std::vector<float>(4, 0.0f));
    scene.grad_logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, std::vector<float>(1, 0.0f));

    scene.forward.intrinsics = &scene.intrinsics;
    scene.forward.pose = &scene.pose;
    scene.forward.near_plane = 0.1f;
    scene.forward.far_plane = 10.0f;
    scene.forward.background_color = gsx_vec3{ 0.05f, 0.1f, 0.2f };
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

static RenderScene make_stable_hard_culling_scene()
{
    RenderScene scene{};
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_arena_desc arena_desc{};

    scene.backend = create_cpu_backend();
    buffer_type = find_device_buffer_type(scene.backend);
    arena_desc.initial_capacity_bytes = 2u << 20;
    EXPECT_GSX_CODE(gsx_arena_init(&scene.arena, buffer_type, &arena_desc), GSX_ERROR_SUCCESS);
    scene.renderer = create_renderer(scene.backend, 64, 64);
    scene.context = create_context(scene.renderer);
    scene.intrinsics = make_intrinsics(64, 64, 50.0f, 66.6666718f, 32.0f, 32.0f);
    scene.pose = make_identity_pose();
    scene.mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 3.5f });
    scene.rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.1305262f, 0.9914449f });
    scene.logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { -1.8f, -2.0f, -1.9f });
    scene.sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, { 0.1f, 0.4f, 0.9f });
    scene.opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, { -0.1f });
    scene.out_rgb = make_f32_tensor(scene.arena, { 3, 64, 64, 0 }, 3, std::vector<float>(3 * 64 * 64, 0.0f));
    scene.grad_rgb_values.resize(3 * 64 * 64);
    for(std::size_t i = 0; i < scene.grad_rgb_values.size(); ++i) {
        scene.grad_rgb_values[i] = (float)((int)(i % 29) - 14) * 0.03f;
    }
    scene.grad_rgb = make_f32_tensor(scene.arena, { 3, 64, 64, 0 }, 3, scene.grad_rgb_values);
    scene.grad_mean3d = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_rotation = make_f32_tensor(scene.arena, { 1, 4, 0, 0 }, 2, std::vector<float>(4, 0.0f));
    scene.grad_logscale = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_sh0 = make_f32_tensor(scene.arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    scene.grad_opacity = make_f32_tensor(scene.arena, { 1, 0, 0, 0 }, 1, std::vector<float>(1, 0.0f));

    scene.forward.intrinsics = &scene.intrinsics;
    scene.forward.pose = &scene.pose;
    scene.forward.near_plane = 0.1f;
    scene.forward.far_plane = 10.0f;
    scene.forward.background_color = gsx_vec3{ 0.02f, 0.02f, 0.03f };
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
    RenderScene *scene,
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
    error = gsx_tensor_upload(parameter, values.data(), (gsx_size_t)values.size() * sizeof(float));
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    scene->forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    error = gsx_renderer_render(scene->renderer, scene->context, &scene->forward);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    output_values = download_f32_tensor(scene->out_rgb);
    loss_plus = dot_product(output_values, scene->grad_rgb_values);

    values[index] -= 2.0f * epsilon;
    error = gsx_tensor_upload(parameter, values.data(), (gsx_size_t)values.size() * sizeof(float));
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    error = gsx_renderer_render(scene->renderer, scene->context, &scene->forward);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    output_values = download_f32_tensor(scene->out_rgb);
    loss_minus = dot_product(output_values, scene->grad_rgb_values);

    values[index] += epsilon;
    error = gsx_tensor_upload(parameter, values.data(), (gsx_size_t)values.size() * sizeof(float));
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

TEST(RenderRuntime, CpuRendererLifecycleAndCapabilities)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_renderer_t renderer = nullptr;
    gsx_render_context_t context = nullptr;
    gsx_renderer_desc desc{};
    gsx_renderer_info info{};
    gsx_renderer_capabilities capabilities{};
    gsx_data_type output_data_type = GSX_DATA_TYPE_U8;

    desc.width = 4;
    desc.height = 3;
    desc.output_data_type = GSX_DATA_TYPE_F32;
    desc.feature_flags = GSX_RENDERER_FEATURE_DEBUG;
    ASSERT_GSX_SUCCESS(gsx_renderer_init(&renderer, backend, &desc));
    ASSERT_NE(renderer, nullptr);
    ASSERT_GSX_SUCCESS(gsx_renderer_get_info(renderer, &info));
    EXPECT_EQ(info.width, 4);
    EXPECT_EQ(info.height, 3);
    EXPECT_EQ(info.output_data_type, GSX_DATA_TYPE_F32);
    EXPECT_EQ(info.feature_flags, GSX_RENDERER_FEATURE_DEBUG);
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
    ASSERT_NE(context, nullptr);
    EXPECT_GSX_CODE(gsx_renderer_free(renderer), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, BackendFreeRejectsLiveRenderer)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_renderer_t renderer = create_renderer(backend, 2, 2);

    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, CpuRendererRejectsUnsupportedFlagsAndLayouts)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_device_buffer_type(backend);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_renderer_t renderer = create_renderer(backend, 1, 1);
    gsx_render_context_t context = create_context(renderer);
    gsx_camera_intrinsics intrinsics = make_intrinsics(1, 1, 1.0f, 1.0f, 0.5f, 0.5f);
    gsx_camera_pose pose = make_identity_pose();
    gsx_tensor_t mean3d = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 4.0f });
    gsx_tensor_t rotation = make_f32_tensor(arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f, 1.0f });
    gsx_tensor_t logscale = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f });
    gsx_tensor_t sh0 = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 1.0f, 0.0f, 0.0f });
    gsx_tensor_t opacity = make_f32_tensor(arena, { 1, 0, 0, 0 }, 1, { 0.0f });
    gsx_tensor_t out_rgb = make_f32_tensor(arena, { 3, 1, 1, 0 }, 3, std::vector<float>(3, 0.0f), GSX_STORAGE_FORMAT_CHW);
    gsx_tensor_t tiled_out = make_f32_tensor(arena, { 3, 1, 1, 0 }, 3, std::vector<float>(3, 0.0f), GSX_STORAGE_FORMAT_TILED_CHW);
    gsx_tensor_t alpha_out = make_f32_tensor(arena, { 1, 1, 1, 0 }, 3, std::vector<float>(1, 0.0f), GSX_STORAGE_FORMAT_CHW);
    gsx_render_forward_request request{};
    gsx_renderer_desc bad_desc{};
    gsx_renderer_t bad_renderer = nullptr;

    request.intrinsics = &intrinsics;
    request.pose = &pose;
    request.near_plane = 0.1f;
    request.far_plane = 10.0f;
    request.precision = GSX_RENDER_PRECISION_FLOAT32;
    request.sh_degree = 0;
    request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    request.borrow_train_state = false;
    request.gs_mean3d = mean3d;
    request.gs_rotation = rotation;
    request.gs_logscale = logscale;
    request.gs_sh0 = sh0;
    request.gs_opacity = opacity;
    request.out_rgb = tiled_out;
    EXPECT_GSX_CODE(gsx_renderer_render(renderer, context, &request), GSX_ERROR_INVALID_ARGUMENT);

    request.out_rgb = out_rgb;
    request.precision = GSX_RENDER_PRECISION_FLOAT16;
    EXPECT_GSX_CODE(gsx_renderer_render(renderer, context, &request), GSX_ERROR_NOT_SUPPORTED);

    request.precision = GSX_RENDER_PRECISION_FLOAT32;
    request.forward_type = GSX_RENDER_FORWARD_TYPE_METRIC;
    EXPECT_GSX_CODE(gsx_renderer_render(renderer, context, &request), GSX_ERROR_NOT_SUPPORTED);

    request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    request.out_alpha = alpha_out;
    EXPECT_GSX_CODE(gsx_renderer_render(renderer, context, &request), GSX_ERROR_NOT_SUPPORTED);

    bad_desc.width = 1;
    bad_desc.height = 1;
    bad_desc.output_data_type = GSX_DATA_TYPE_F32;
    bad_desc.feature_flags = GSX_RENDERER_FEATURE_ANTIALIASING;
    EXPECT_GSX_CODE(gsx_renderer_init(&bad_renderer, backend, &bad_desc), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(alpha_out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(tiled_out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_rgb));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(sh0));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(logscale));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(rotation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mean3d));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, CpuRendererForwardRejectsInvalidGaussianShapes)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_device_buffer_type(backend);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_renderer_t renderer = create_renderer(backend, 1, 1);
    gsx_render_context_t context = create_context(renderer);
    gsx_camera_intrinsics intrinsics = make_intrinsics(1, 1, 1.0f, 1.0f, 0.5f, 0.5f);
    gsx_camera_pose pose = make_identity_pose();
    gsx_tensor_t mean3d = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 4.0f });
    gsx_tensor_t rotation = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 1.0f });
    gsx_tensor_t logscale = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f });
    gsx_tensor_t sh0 = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 1.0f, 0.0f, 0.0f });
    gsx_tensor_t opacity = make_f32_tensor(arena, { 1, 0, 0, 0 }, 1, { 0.0f });
    gsx_tensor_t out_rgb = make_f32_tensor(arena, { 3, 1, 1, 0 }, 3, std::vector<float>(3, 0.0f));
    gsx_render_forward_request request{};

    request.intrinsics = &intrinsics;
    request.pose = &pose;
    request.near_plane = 0.1f;
    request.far_plane = 10.0f;
    request.precision = GSX_RENDER_PRECISION_FLOAT32;
    request.sh_degree = 0;
    request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    request.gs_mean3d = mean3d;
    request.gs_rotation = rotation;
    request.gs_logscale = logscale;
    request.gs_sh0 = sh0;
    request.gs_opacity = opacity;
    request.out_rgb = out_rgb;

    EXPECT_GSX_CODE(gsx_renderer_render(renderer, context, &request), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_rgb));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(sh0));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(logscale));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(rotation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mean3d));
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, CpuRendererForwardMatchesSimpleReference)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_device_buffer_type(backend);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_renderer_t renderer = create_renderer(backend, 1, 1);
    gsx_render_context_t context = create_context(renderer);
    gsx_camera_intrinsics intrinsics = make_intrinsics(1, 1, 1.0f, 1.0f, 0.5f, 0.5f);
    gsx_camera_pose pose = make_identity_pose();
    gsx_tensor_t mean3d = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 4.0f });
    gsx_tensor_t rotation = make_f32_tensor(arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f, 1.0f });
    gsx_tensor_t logscale = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f });
    gsx_tensor_t sh0 = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 1.0f, 0.0f, 0.0f });
    gsx_tensor_t opacity = make_f32_tensor(arena, { 1, 0, 0, 0 }, 1, { 0.0f });
    gsx_tensor_t out_rgb = make_f32_tensor(arena, { 3, 1, 1, 0 }, 3, std::vector<float>(3, 0.0f));
    gsx_render_forward_request request{};
    std::vector<float> output_values;

    request.intrinsics = &intrinsics;
    request.pose = &pose;
    request.near_plane = 0.1f;
    request.far_plane = 10.0f;
    request.background_color = gsx_vec3{ 0.0f, 0.0f, 0.0f };
    request.precision = GSX_RENDER_PRECISION_FLOAT32;
    request.sh_degree = 0;
    request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    request.borrow_train_state = false;
    request.gs_mean3d = mean3d;
    request.gs_rotation = rotation;
    request.gs_logscale = logscale;
    request.gs_sh0 = sh0;
    request.gs_opacity = opacity;
    request.out_rgb = out_rgb;

    ASSERT_GSX_SUCCESS(gsx_renderer_render(renderer, context, &request));
    output_values = download_f32_tensor(out_rgb);
    ASSERT_EQ(output_values.size(), 3U);
    EXPECT_NEAR(output_values[0], 0.3910474f, 1.0e-5f);
    EXPECT_NEAR(output_values[1], 0.25f, 1.0e-5f);
    EXPECT_NEAR(output_values[2], 0.25f, 1.0e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_rgb));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(sh0));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(logscale));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(rotation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mean3d));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, CpuRendererBackwardConsumesTrainState)
{
    RenderScene scene = make_gradient_scene();

    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_INVALID_STATE);

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_INVALID_STATE);

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_INVALID_STATE);

    destroy_scene(&scene);
}

TEST(RenderRuntime, CpuRendererBackwardBorrowedTrainStateTracksInputMutations)
{
    RenderScene expected = make_gradient_scene();
    RenderScene borrowed = make_gradient_scene();
    std::vector<float> expected_grad_mean;
    std::vector<float> expected_grad_logscale;
    std::vector<float> expected_grad_sh0;
    std::vector<float> expected_grad_opacity;
    const std::vector<float> mutated_mean = { -0.4f, 0.3f, 4.8f };
    const std::vector<float> mutated_logscale = { 0.25f, -0.35f, 0.15f };
    const std::vector<float> mutated_sh0 = { -0.2f, 0.6f, 0.1f };
    const std::vector<float> mutated_opacity = { -1.2f };

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(expected.mean3d, mutated_mean.data(), (gsx_size_t)mutated_mean.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(expected.logscale, mutated_logscale.data(), (gsx_size_t)mutated_logscale.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(expected.sh0, mutated_sh0.data(), (gsx_size_t)mutated_sh0.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(expected.opacity, mutated_opacity.data(), (gsx_size_t)mutated_opacity.size() * sizeof(float)));
    expected.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    expected.forward.borrow_train_state = false;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(expected.renderer, expected.context, &expected.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(expected.renderer, expected.context, &expected.backward));
    expected_grad_mean = download_f32_tensor(expected.grad_mean3d);
    expected_grad_logscale = download_f32_tensor(expected.grad_logscale);
    expected_grad_sh0 = download_f32_tensor(expected.grad_sh0);
    expected_grad_opacity = download_f32_tensor(expected.grad_opacity);

    borrowed.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    borrowed.forward.borrow_train_state = true;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(borrowed.renderer, borrowed.context, &borrowed.forward));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(borrowed.mean3d, mutated_mean.data(), (gsx_size_t)mutated_mean.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(borrowed.logscale, mutated_logscale.data(), (gsx_size_t)mutated_logscale.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(borrowed.sh0, mutated_sh0.data(), (gsx_size_t)mutated_sh0.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(borrowed.opacity, mutated_opacity.data(), (gsx_size_t)mutated_opacity.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(borrowed.renderer, borrowed.context, &borrowed.backward));

    {
        const std::vector<float> grad = download_f32_tensor(borrowed.grad_mean3d);
        ASSERT_EQ(grad.size(), expected_grad_mean.size());
        for(std::size_t i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad[i], expected_grad_mean[i], 1.0e-5f);
        }
    }
    {
        const std::vector<float> grad = download_f32_tensor(borrowed.grad_logscale);
        ASSERT_EQ(grad.size(), expected_grad_logscale.size());
        for(std::size_t i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad[i], expected_grad_logscale[i], 1.0e-5f);
        }
    }
    {
        const std::vector<float> grad = download_f32_tensor(borrowed.grad_sh0);
        ASSERT_EQ(grad.size(), expected_grad_sh0.size());
        for(std::size_t i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad[i], expected_grad_sh0[i], 1.0e-5f);
        }
    }
    {
        const std::vector<float> grad = download_f32_tensor(borrowed.grad_opacity);
        ASSERT_EQ(grad.size(), expected_grad_opacity.size());
        for(std::size_t i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad[i], expected_grad_opacity[i], 1.0e-5f);
        }
    }

    destroy_scene(&borrowed);
    destroy_scene(&expected);
}

TEST(RenderRuntime, CpuRendererBackwardValidatesCoreRequestBeforeTrainState)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_device_buffer_type(backend);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_renderer_t renderer = create_renderer(backend, 2, 2);
    gsx_render_context_t context = create_context(renderer);
    gsx_tensor_t grad_rgb_bad_shape = make_f32_tensor(arena, { 1, 2, 2, 0 }, 3, std::vector<float>(4, 0.0f));
    gsx_tensor_t grad_alpha = make_f32_tensor(arena, { 1, 2, 2, 0 }, 3, std::vector<float>(4, 0.0f));
    gsx_tensor_t grad_mean3d = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    gsx_tensor_t grad_rotation = make_f32_tensor(arena, { 1, 4, 0, 0 }, 2, std::vector<float>(4, 0.0f));
    gsx_tensor_t grad_logscale = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    gsx_tensor_t grad_sh0 = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, std::vector<float>(3, 0.0f));
    gsx_tensor_t grad_opacity = make_f32_tensor(arena, { 1, 0, 0, 0 }, 1, std::vector<float>(1, 0.0f));
    gsx_render_backward_request request{};

    request.grad_rgb = grad_rgb_bad_shape;
    request.grad_alpha = grad_alpha;
    request.grad_gs_mean3d = grad_mean3d;
    request.grad_gs_rotation = grad_rotation;
    request.grad_gs_logscale = grad_logscale;
    request.grad_gs_sh0 = grad_sh0;
    request.grad_gs_opacity = grad_opacity;
    EXPECT_GSX_CODE(gsx_renderer_backward(renderer, context, &request), GSX_ERROR_NOT_SUPPORTED);

    request.grad_alpha = nullptr;
    EXPECT_GSX_CODE(gsx_renderer_backward(renderer, context, &request), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_sh0));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_logscale));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_rotation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_mean3d));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_alpha));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(grad_rgb_bad_shape));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, CpuRendererBackwardRejectsInactiveOptionalShSinks)
{
    RenderScene scene = make_gradient_scene();

    scene.forward.sh_degree = 0;
    scene.forward.gs_sh1 = nullptr;
    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));

    scene.backward.grad_gs_sh1 = scene.grad_sh1;
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_INVALID_ARGUMENT);

    destroy_scene(&scene);
}

TEST(RenderRuntime, CpuRendererBackwardRejectsAliasingGradientSinks)
{
    RenderScene scene = make_gradient_scene();

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));

    scene.backward.grad_gs_logscale = scene.backward.grad_gs_mean3d;
    EXPECT_GSX_CODE(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward), GSX_ERROR_INVALID_ARGUMENT);

    destroy_scene(&scene);
}

TEST(RenderRuntime, CpuRendererDebugRejectsNonFiniteActiveOptionalSh)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_device_buffer_type(backend);
    gsx_arena_t arena = create_arena(buffer_type);
    gsx_renderer_t renderer = create_renderer(backend, 1, 1, GSX_RENDERER_FEATURE_DEBUG);
    gsx_render_context_t context = create_context(renderer);
    gsx_camera_intrinsics intrinsics = make_intrinsics(1, 1, 1.0f, 1.0f, 0.5f, 0.5f);
    gsx_camera_pose pose = make_identity_pose();
    gsx_tensor_t mean3d = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 4.0f });
    gsx_tensor_t rotation = make_f32_tensor(arena, { 1, 4, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f, 1.0f });
    gsx_tensor_t logscale = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.0f, 0.0f, 0.0f });
    gsx_tensor_t sh0 = make_f32_tensor(arena, { 1, 3, 0, 0 }, 2, { 0.1f, 0.2f, 0.3f });
    gsx_tensor_t sh1 = make_f32_tensor(
        arena,
        { 1, 3, 3, 0 },
        3,
        { 0.0f, 0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
    gsx_tensor_t opacity = make_f32_tensor(arena, { 1, 0, 0, 0 }, 1, { 0.0f });
    gsx_tensor_t out_rgb = make_f32_tensor(arena, { 3, 1, 1, 0 }, 3, std::vector<float>(3, 0.0f));
    gsx_render_forward_request request{};

    request.intrinsics = &intrinsics;
    request.pose = &pose;
    request.near_plane = 0.1f;
    request.far_plane = 10.0f;
    request.precision = GSX_RENDER_PRECISION_FLOAT32;
    request.sh_degree = 1;
    request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    request.borrow_train_state = false;
    request.gs_mean3d = mean3d;
    request.gs_rotation = rotation;
    request.gs_logscale = logscale;
    request.gs_sh0 = sh0;
    request.gs_sh1 = sh1;
    request.gs_opacity = opacity;
    request.out_rgb = out_rgb;

    EXPECT_GSX_CODE(gsx_renderer_render(renderer, context, &request), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_rgb));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(sh1));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(sh0));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(logscale));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(rotation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mean3d));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(RenderRuntime, CpuRendererBackwardMatchesFiniteDifferences)
{
    RenderScene scene = make_gradient_scene();
    std::vector<float> analytic_mean;
    std::vector<float> analytic_rotation;
    std::vector<float> analytic_logscale;
    std::vector<float> analytic_sh0;
    std::vector<float> analytic_sh1;
    std::vector<float> analytic_opacity;
    std::vector<float> mean_values = download_f32_tensor(scene.mean3d);
    std::vector<float> rotation_values = download_f32_tensor(scene.rotation);
    std::vector<float> logscale_values = download_f32_tensor(scene.logscale);
    std::vector<float> sh0_values = download_f32_tensor(scene.sh0);
    std::vector<float> sh1_values = download_f32_tensor(scene.sh1);
    std::vector<float> opacity_values = download_f32_tensor(scene.opacity);
    float numeric_mean_x = 0.0f;
    float numeric_mean_y = 0.0f;
    float numeric_mean_z = 0.0f;
    float numeric_rotation = 0.0f;
    float numeric_logscale = 0.0f;
    float numeric_sh0 = 0.0f;
    float numeric_sh1 = 0.0f;
    float numeric_opacity = 0.0f;

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));

    analytic_mean = download_f32_tensor(scene.grad_mean3d);
    analytic_rotation = download_f32_tensor(scene.grad_rotation);
    analytic_logscale = download_f32_tensor(scene.grad_logscale);
    analytic_sh0 = download_f32_tensor(scene.grad_sh0);
    analytic_sh1 = download_f32_tensor(scene.grad_sh1);
    analytic_opacity = download_f32_tensor(scene.grad_opacity);

    numeric_mean_x = numerical_gradient(&scene, scene.mean3d, mean_values, 0, 1.0e-3f);
    numeric_mean_y = numerical_gradient(&scene, scene.mean3d, mean_values, 1, 1.0e-3f);
    numeric_mean_z = numerical_gradient(&scene, scene.mean3d, mean_values, 2, 1.0e-3f);
    numeric_rotation = numerical_gradient(&scene, scene.rotation, rotation_values, 2, 1.0e-3f);
    numeric_logscale = numerical_gradient(&scene, scene.logscale, logscale_values, 1, 1.0e-3f);
    numeric_sh0 = numerical_gradient(&scene, scene.sh0, sh0_values, 1, 1.0e-3f);
    numeric_sh1 = numerical_gradient(&scene, scene.sh1, sh1_values, 5, 1.0e-3f);
    numeric_opacity = numerical_gradient(&scene, scene.opacity, opacity_values, 0, 1.0e-3f);

    EXPECT_NEAR(analytic_mean[0], numeric_mean_x, 3.0e-2f);
    EXPECT_NEAR(analytic_mean[1], numeric_mean_y, 3.0e-2f);
    EXPECT_NEAR(analytic_mean[2], numeric_mean_z, 3.0e-2f);
    EXPECT_NEAR(analytic_rotation[2], numeric_rotation, 5.0e-2f);
    EXPECT_NEAR(analytic_logscale[1], numeric_logscale, 3.0e-2f);
    EXPECT_NEAR(analytic_sh0[1], numeric_sh0, 2.0e-2f);
    EXPECT_NEAR(analytic_sh1[5], numeric_sh1, 2.0e-2f);
    EXPECT_NEAR(analytic_opacity[0], numeric_opacity, 2.0e-2f);

    destroy_scene(&scene);
}

TEST(RenderRuntime, CpuRendererBackwardMatchesFiniteDifferencesNearAlphaSaturationThreshold)
{
    RenderScene scene = make_near_alpha_saturation_scene();
    std::vector<float> analytic_opacity;
    std::vector<float> opacity_values = download_f32_tensor(scene.opacity);
    float numeric_opacity = 0.0f;
    float activated_opacity = 1.0f / (1.0f + std::exp(-opacity_values[0]));

    EXPECT_GT(activated_opacity, 0.99f);
    EXPECT_LT(activated_opacity, 0.999f);

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));

    analytic_opacity = download_f32_tensor(scene.grad_opacity);
    numeric_opacity = numerical_gradient(&scene, scene.opacity, opacity_values, 0, 1.0e-3f);

    EXPECT_NEAR(analytic_opacity[0], numeric_opacity, 1.0e-2f);

    destroy_scene(&scene);
}

TEST(RenderRuntime, CpuRendererBackwardMatchesFiniteDifferencesForRotatedAnisotropicGaussian)
{
    RenderScene scene = make_rotated_anisotropic_scene();
    std::vector<float> analytic_mean;
    std::vector<float> analytic_rotation;
    std::vector<float> analytic_logscale;
    std::vector<float> mean_values = download_f32_tensor(scene.mean3d);
    std::vector<float> rotation_values = download_f32_tensor(scene.rotation);
    std::vector<float> logscale_values = download_f32_tensor(scene.logscale);
    float numeric_mean_z = 0.0f;
    float numeric_rotation_z = 0.0f;
    float numeric_rotation_w = 0.0f;
    float numeric_logscale_x = 0.0f;
    float numeric_logscale_y = 0.0f;

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));

    analytic_mean = download_f32_tensor(scene.grad_mean3d);
    analytic_rotation = download_f32_tensor(scene.grad_rotation);
    analytic_logscale = download_f32_tensor(scene.grad_logscale);

    numeric_mean_z = numerical_gradient(&scene, scene.mean3d, mean_values, 2, 1.0e-3f);
    numeric_rotation_z = numerical_gradient(&scene, scene.rotation, rotation_values, 2, 1.0e-3f);
    numeric_rotation_w = numerical_gradient(&scene, scene.rotation, rotation_values, 3, 1.0e-3f);
    numeric_logscale_x = numerical_gradient(&scene, scene.logscale, logscale_values, 0, 1.0e-3f);
    numeric_logscale_y = numerical_gradient(&scene, scene.logscale, logscale_values, 1, 1.0e-3f);

    EXPECT_NEAR(analytic_mean[2], numeric_mean_z, 5.0e-2f);
    EXPECT_NEAR(analytic_rotation[2], numeric_rotation_z, 5.0e-2f);
    EXPECT_NEAR(analytic_rotation[3], numeric_rotation_w, 5.0e-2f);
    EXPECT_NEAR(analytic_logscale[0], numeric_logscale_x, 5.0e-2f);
    EXPECT_NEAR(analytic_logscale[1], numeric_logscale_y, 5.0e-2f);

    destroy_scene(&scene);
}

TEST(RenderRuntime, CpuRendererBackwardMatchesFiniteDifferencesForStableHardCullingScene)
{
    RenderScene scene = make_stable_hard_culling_scene();
    std::vector<float> analytic_mean;
    std::vector<float> analytic_rotation;
    std::vector<float> analytic_logscale;
    std::vector<float> mean_values = download_f32_tensor(scene.mean3d);
    std::vector<float> rotation_values = download_f32_tensor(scene.rotation);
    std::vector<float> logscale_values = download_f32_tensor(scene.logscale);
    float numeric_mean_y = 0.0f;
    float numeric_mean_z = 0.0f;
    float numeric_rotation_z = 0.0f;
    float numeric_logscale_x = 0.0f;
    float numeric_logscale_y = 0.0f;

    scene.forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    ASSERT_GSX_SUCCESS(gsx_renderer_render(scene.renderer, scene.context, &scene.forward));
    ASSERT_GSX_SUCCESS(gsx_renderer_backward(scene.renderer, scene.context, &scene.backward));

    analytic_mean = download_f32_tensor(scene.grad_mean3d);
    analytic_rotation = download_f32_tensor(scene.grad_rotation);
    analytic_logscale = download_f32_tensor(scene.grad_logscale);

    numeric_mean_y = numerical_gradient(&scene, scene.mean3d, mean_values, 1, 1.0e-3f);
    numeric_mean_z = numerical_gradient(&scene, scene.mean3d, mean_values, 2, 1.0e-3f);
    numeric_rotation_z = numerical_gradient(&scene, scene.rotation, rotation_values, 2, 1.0e-3f);
    numeric_logscale_x = numerical_gradient(&scene, scene.logscale, logscale_values, 0, 1.0e-3f);
    numeric_logscale_y = numerical_gradient(&scene, scene.logscale, logscale_values, 1, 1.0e-3f);

    EXPECT_NEAR(analytic_mean[1], numeric_mean_y, 5.0e-2f);
    EXPECT_NEAR(analytic_mean[2], numeric_mean_z, 5.0e-2f);
    EXPECT_NEAR(analytic_rotation[2], numeric_rotation_z, 5.0e-2f);
    EXPECT_NEAR(analytic_logscale[0], numeric_logscale_x, 5.0e-2f);
    EXPECT_NEAR(analytic_logscale[1], numeric_logscale_y, 5.0e-2f);

    destroy_scene(&scene);
}

}  // namespace
