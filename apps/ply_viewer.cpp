#include <gsx/extra/gsx-image.h>
#include <gsx/extra/gsx-io-ply.h>
#include <gsx/gsx.h>

#include <SDL3/SDL.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_sdlrenderer3.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct vec3 {
    float x;
    float y;
    float z;
};

struct quat {
    float x;
    float y;
    float z;
    float w;
};

struct app_options {
    const char* input_ply_path = nullptr;
    gsx_backend_type backend_type =
#if defined(__APPLE__)
        GSX_BACKEND_TYPE_METAL;
#else
        GSX_BACKEND_TYPE_CPU;
#endif
    gsx_index_t device_index = 0;

    gsx_index_t width = 740;
    gsx_index_t height = 480;
    float fx = 740.0f;
    float fy = 740.0f;
    float cx = 370.0f;
    float cy = 240.0f;

    float move_speed = 2.0f;
    float mouse_sensitivity = 0.003f;
};

struct camera_state {
    vec3 position = { 0.0f, 0.0f, 4.0f };
    float yaw = 3.1415926535f;
    float pitch = 0.0f;
};

struct app_state {
    gsx_backend_t backend = nullptr;
    gsx_renderer_t renderer = nullptr;
    gsx_render_context_t render_context = nullptr;
    gsx_arena_t render_arena = nullptr;
    gsx_tensor_t out_rgb = nullptr;
    gsx_tensor_t rgb_f32_srgb = nullptr;
    gsx_tensor_t rgb_u8_chw = nullptr;
    gsx_tensor_t rgb_u8_hwc = nullptr;
    gsx_gs_t gs = nullptr;

    gsx_tensor_t gs_mean3d = nullptr;
    gsx_tensor_t gs_rotation = nullptr;
    gsx_tensor_t gs_logscale = nullptr;
    gsx_tensor_t gs_sh0 = nullptr;
    gsx_tensor_t gs_opacity = nullptr;

    std::vector<float> host_rgb_chw;
    std::vector<std::uint8_t> host_rgba;

    SDL_Window* window = nullptr;
    SDL_Renderer* sdl_renderer = nullptr;
    SDL_Texture* frame_texture = nullptr;

    camera_state camera;

    bool running = true;
    bool camera_dirty = true;
    bool request_resize = false;
    float last_render_ms = 0.0f;
};

static bool gsx_check(const gsx_error error, const char* context)
{
    if(gsx_error_is_success(error)) {
        return true;
    }

    std::fprintf(stderr, "error: %s failed (%d)", context, error.code);
    if(error.message != nullptr) {
        std::fprintf(stderr, ": %s", error.message);
    }
    std::fprintf(stderr, "\n");
    return false;
}

static float clampf(const float value, const float min_value, const float max_value)
{
    return std::max(min_value, std::min(max_value, value));
}

static float wrap_pi(float a)
{
    const float pi = 3.1415926535f;
    const float two_pi = 2.0f * pi;
    a = std::fmod(a + pi, two_pi);
    if(a < 0.0f) {
        a += two_pi;
    }
    return a - pi;
}

static vec3 vec3_add(const vec3 a, const vec3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

static vec3 vec3_sub(const vec3 a, const vec3 b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

static vec3 vec3_scale(const vec3 v, const float s)
{
    return { v.x * s, v.y * s, v.z * s };
}

static float vec3_dot(const vec3 a, const vec3 b)
{
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

static vec3 vec3_cross(const vec3 a, const vec3 b)
{
    return {
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.x),
    };
}

static float vec3_length(const vec3 v)
{
    return std::sqrt(vec3_dot(v, v));
}

static vec3 vec3_normalize(const vec3 v)
{
    const float len = vec3_length(v);
    if(len <= 1.0e-8f) {
        return { 0.0f, 0.0f, 0.0f };
    }
    return vec3_scale(v, 1.0f / len);
}

static void camera_basis(const camera_state& camera, vec3* out_forward, vec3* out_right, vec3* out_up)
{
    const vec3 forward = vec3_normalize({
        std::cos(camera.pitch) * std::sin(camera.yaw),
        std::sin(camera.pitch),
        std::cos(camera.pitch) * std::cos(camera.yaw),
    });
    const vec3 world_up = { 0.0f, 1.0f, 0.0f };
    vec3 right = vec3_normalize(vec3_cross(forward, world_up));
    if(vec3_length(right) <= 1.0e-6f) {
        right = { 1.0f, 0.0f, 0.0f };
    }
    const vec3 up = vec3_normalize(vec3_cross(right, forward));

    *out_forward = forward;
    *out_right = right;
    *out_up = up;
}

static quat quat_from_rotation_matrix(const std::array<float, 9>& r)
{
    const float trace = r[0] + r[4] + r[8];
    quat q = { 0.0f, 0.0f, 0.0f, 1.0f };

    if(trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (r[7] - r[5]) / s;
        q.y = (r[2] - r[6]) / s;
        q.z = (r[3] - r[1]) / s;
    } else if((r[0] > r[4]) && (r[0] > r[8])) {
        const float s = std::sqrt(1.0f + r[0] - r[4] - r[8]) * 2.0f;
        q.w = (r[7] - r[5]) / s;
        q.x = 0.25f * s;
        q.y = (r[1] + r[3]) / s;
        q.z = (r[2] + r[6]) / s;
    } else if(r[4] > r[8]) {
        const float s = std::sqrt(1.0f + r[4] - r[0] - r[8]) * 2.0f;
        q.w = (r[2] - r[6]) / s;
        q.x = (r[1] + r[3]) / s;
        q.y = 0.25f * s;
        q.z = (r[5] + r[7]) / s;
    } else {
        const float s = std::sqrt(1.0f + r[8] - r[0] - r[4]) * 2.0f;
        q.w = (r[3] - r[1]) / s;
        q.x = (r[2] + r[6]) / s;
        q.y = (r[5] + r[7]) / s;
        q.z = 0.25f * s;
    }

    const float norm = std::sqrt((q.x * q.x) + (q.y * q.y) + (q.z * q.z) + (q.w * q.w));
    if(norm > 1.0e-8f) {
        q.x /= norm;
        q.y /= norm;
        q.z /= norm;
        q.w /= norm;
    }
    return q;
}

static void camera_to_gsx_pose(const camera_state& camera, gsx_camera_pose* out_pose)
{
    vec3 forward = { 0.0f, 0.0f, 1.0f };
    vec3 right = { 1.0f, 0.0f, 0.0f };
    vec3 up = { 0.0f, 1.0f, 0.0f };
    camera_basis(camera, &forward, &right, &up);

    const std::array<float, 9> r_cw = {
        right.x,  up.x,  -forward.x,
        right.y,  up.y,  -forward.y,
        right.z,  up.z,  -forward.z,
    };

    const std::array<float, 9> r_wc = {
        r_cw[0], r_cw[3], r_cw[6],
        r_cw[1], r_cw[4], r_cw[7],
        r_cw[2], r_cw[5], r_cw[8],
    };

    const vec3 t_wc = {
        -((r_wc[0] * camera.position.x) + (r_wc[1] * camera.position.y) + (r_wc[2] * camera.position.z)),
        -((r_wc[3] * camera.position.x) + (r_wc[4] * camera.position.y) + (r_wc[5] * camera.position.z)),
        -((r_wc[6] * camera.position.x) + (r_wc[7] * camera.position.y) + (r_wc[8] * camera.position.z)),
    };

    const quat q_wc = quat_from_rotation_matrix(r_wc);
    out_pose->rot.x = q_wc.x;
    out_pose->rot.y = q_wc.y;
    out_pose->rot.z = q_wc.z;
    out_pose->rot.w = q_wc.w;
    out_pose->transl.x = t_wc.x;
    out_pose->transl.y = t_wc.y;
    out_pose->transl.z = t_wc.z;
    out_pose->camera_id = 0;
    out_pose->frame_id = 0;
}

static bool parse_i64(const char* value, long long* out_value)
{
    if(value == nullptr || out_value == nullptr) {
        return false;
    }
    char* end_ptr = nullptr;
    const long long parsed = std::strtoll(value, &end_ptr, 10);
    if(end_ptr == value || *end_ptr != '\0') {
        return false;
    }
    *out_value = parsed;
    return true;
}

static bool parse_f32(const char* value, float* out_value)
{
    if(value == nullptr || out_value == nullptr) {
        return false;
    }
    char* end_ptr = nullptr;
    const float parsed = std::strtof(value, &end_ptr);
    if(end_ptr == value || *end_ptr != '\0') {
        return false;
    }
    *out_value = parsed;
    return true;
}

static bool parse_backend_type(const char* value, gsx_backend_type* out_backend)
{
    if(value == nullptr || out_backend == nullptr) {
        return false;
    }
    if(std::strcmp(value, "cpu") == 0) {
        *out_backend = GSX_BACKEND_TYPE_CPU;
        return true;
    }
    if(std::strcmp(value, "cuda") == 0) {
        *out_backend = GSX_BACKEND_TYPE_CUDA;
        return true;
    }
    if(std::strcmp(value, "metal") == 0) {
        *out_backend = GSX_BACKEND_TYPE_METAL;
        return true;
    }
    return false;
}

static void print_usage(const char* program)
{
    std::fprintf(stderr, "usage: %s --input <pointcloud.ply> [options]\n", program);
    std::fprintf(stderr, "options:\n");
    std::fprintf(stderr, "  --backend <cpu|cuda|metal>   backend type\n");
    std::fprintf(stderr, "  --device <index>             backend device index\n");
    std::fprintf(stderr, "  --width <int>                output width\n");
    std::fprintf(stderr, "  --height <int>               output height\n");
    std::fprintf(stderr, "  --fx <float>                 focal length x\n");
    std::fprintf(stderr, "  --fy <float>                 focal length y\n");
    std::fprintf(stderr, "  --cx <float>                 principal point x\n");
    std::fprintf(stderr, "  --cy <float>                 principal point y\n");
    std::fprintf(stderr, "  --move-speed <float>         movement speed in world units/s\n");
    std::fprintf(stderr, "  --mouse-sensitivity <float>  radians per pixel while dragging\n");
}

static bool parse_args(const int argc, char** argv, app_options* options)
{
    for(int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if(std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if(std::strcmp(arg, "--input") == 0) {
            if(i + 1 >= argc) {
                std::fprintf(stderr, "error: --input requires a value\n");
                return false;
            }
            options->input_ply_path = argv[++i];
            continue;
        }
        if(std::strcmp(arg, "--backend") == 0) {
            if(i + 1 >= argc || !parse_backend_type(argv[++i], &options->backend_type)) {
                std::fprintf(stderr, "error: invalid --backend, expected cpu|cuda|metal\n");
                return false;
            }
            continue;
        }
        if(std::strcmp(arg, "--device") == 0) {
            long long value = 0;
            if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value < 0) {
                std::fprintf(stderr, "error: invalid --device\n");
                return false;
            }
            options->device_index = static_cast<gsx_index_t>(value);
            continue;
        }
        if(std::strcmp(arg, "--width") == 0) {
            long long value = 0;
            if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value <= 0) {
                std::fprintf(stderr, "error: invalid --width\n");
                return false;
            }
            options->width = static_cast<gsx_index_t>(value);
            continue;
        }
        if(std::strcmp(arg, "--height") == 0) {
            long long value = 0;
            if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value <= 0) {
                std::fprintf(stderr, "error: invalid --height\n");
                return false;
            }
            options->height = static_cast<gsx_index_t>(value);
            continue;
        }
        if(std::strcmp(arg, "--fx") == 0) {
            if(i + 1 >= argc || !parse_f32(argv[++i], &options->fx) || options->fx <= 0.0f) {
                std::fprintf(stderr, "error: invalid --fx\n");
                return false;
            }
            continue;
        }
        if(std::strcmp(arg, "--fy") == 0) {
            if(i + 1 >= argc || !parse_f32(argv[++i], &options->fy) || options->fy <= 0.0f) {
                std::fprintf(stderr, "error: invalid --fy\n");
                return false;
            }
            continue;
        }
        if(std::strcmp(arg, "--cx") == 0) {
            if(i + 1 >= argc || !parse_f32(argv[++i], &options->cx)) {
                std::fprintf(stderr, "error: invalid --cx\n");
                return false;
            }
            continue;
        }
        if(std::strcmp(arg, "--cy") == 0) {
            if(i + 1 >= argc || !parse_f32(argv[++i], &options->cy)) {
                std::fprintf(stderr, "error: invalid --cy\n");
                return false;
            }
            continue;
        }
        if(std::strcmp(arg, "--move-speed") == 0) {
            if(i + 1 >= argc || !parse_f32(argv[++i], &options->move_speed) || options->move_speed <= 0.0f) {
                std::fprintf(stderr, "error: invalid --move-speed\n");
                return false;
            }
            continue;
        }
        if(std::strcmp(arg, "--mouse-sensitivity") == 0) {
            if(i + 1 >= argc || !parse_f32(argv[++i], &options->mouse_sensitivity) || options->mouse_sensitivity <= 0.0f) {
                std::fprintf(stderr, "error: invalid --mouse-sensitivity\n");
                return false;
            }
            continue;
        }

        if(arg[0] != '-') {
            if(options->input_ply_path != nullptr) {
                std::fprintf(stderr, "error: multiple input paths provided\n");
                return false;
            }
            options->input_ply_path = arg;
            continue;
        }

        std::fprintf(stderr, "error: unknown argument: %s\n", arg);
        return false;
    }

    if(options->input_ply_path == nullptr) {
        std::fprintf(stderr, "error: --input <pointcloud.ply> is required\n");
        return false;
    }
    return true;
}

static bool create_output_texture(app_state* state, const gsx_index_t width, const gsx_index_t height)
{
    if(state->frame_texture != nullptr) {
        SDL_DestroyTexture(state->frame_texture);
        state->frame_texture = nullptr;
    }

    state->frame_texture = SDL_CreateTexture(
        state->sdl_renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        static_cast<int>(width),
        static_cast<int>(height));

    if(state->frame_texture == nullptr) {
        std::fprintf(stderr, "error: SDL_CreateTexture failed: %s\n", SDL_GetError());
        return false;
    }
    return true;
}

static bool init_render_targets(app_state* state, const app_options& options)
{
    if(state->render_context != nullptr) {
        gsx_check(gsx_render_context_free(state->render_context), "gsx_render_context_free");
        state->render_context = nullptr;
    }
    if(state->renderer != nullptr) {
        gsx_check(gsx_renderer_free(state->renderer), "gsx_renderer_free");
        state->renderer = nullptr;
    }
    if(state->out_rgb != nullptr) {
        gsx_check(gsx_tensor_free(state->out_rgb), "gsx_tensor_free(out_rgb)");
        state->out_rgb = nullptr;
    }
    if(state->render_arena != nullptr) {
        gsx_check(gsx_arena_free(state->render_arena), "gsx_arena_free(render_arena)");
        state->render_arena = nullptr;
    }

    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    if(!gsx_check(
           gsx_backend_find_buffer_type(state->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type),
           "gsx_backend_find_buffer_type(device)")) {
        return false;
    }

    gsx_renderer_desc renderer_desc = {};
    renderer_desc.width = options.width;
    renderer_desc.height = options.height;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    renderer_desc.feature_flags = 0u;
    renderer_desc.enable_alpha_output = false;
    renderer_desc.enable_invdepth_output = false;

    if(!gsx_check(gsx_renderer_init(&state->renderer, state->backend, &renderer_desc), "gsx_renderer_init")) {
        return false;
    }
    if(!gsx_check(gsx_render_context_init(&state->render_context, state->renderer), "gsx_render_context_init")) {
        return false;
    }

    gsx_arena_desc arena_desc = {};
    const gsx_size_t frame_f32_bytes = static_cast<gsx_size_t>(options.width) * static_cast<gsx_size_t>(options.height) * 3u * sizeof(float);
    const gsx_size_t frame_u8_bytes = static_cast<gsx_size_t>(options.width) * static_cast<gsx_size_t>(options.height) * 3u;
    arena_desc.initial_capacity_bytes = frame_f32_bytes * 2 + frame_u8_bytes * 2;
    if(!gsx_check(gsx_arena_init(&state->render_arena, device_buffer_type, &arena_desc), "gsx_arena_init(render_arena)")) {
        return false;
    }

    gsx_tensor_desc tensor_desc = {};
    tensor_desc.rank = 3;
    tensor_desc.shape[0] = 3;
    tensor_desc.shape[1] = options.height;
    tensor_desc.shape[2] = options.width;
    tensor_desc.data_type = GSX_DATA_TYPE_F32;
    tensor_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    tensor_desc.arena = state->render_arena;

    if(!gsx_check(gsx_tensor_init(&state->out_rgb, &tensor_desc), "gsx_tensor_init(out_rgb)")) {
        return false;
    }

    if(!gsx_check(gsx_tensor_init(&state->rgb_f32_srgb, &tensor_desc), "gsx_tensor_init(rgb_f32_srgb)")) {
        return false;
    }

    tensor_desc.data_type = GSX_DATA_TYPE_U8;
    if(!gsx_check(gsx_tensor_init(&state->rgb_u8_chw, &tensor_desc), "gsx_tensor_init(rgb_u8_chw)")) {
        return false;
    }

    tensor_desc.storage_format = GSX_STORAGE_FORMAT_HWC;
    if(!gsx_check(gsx_tensor_init(&state->rgb_u8_hwc, &tensor_desc), "gsx_tensor_init(rgb_u8_hwc)")) {
        return false;
    }

    state->host_rgb_chw.resize(static_cast<size_t>(frame_f32_bytes));
    state->host_rgba.resize(static_cast<size_t>(options.width) * static_cast<size_t>(options.height) * 4u);

    return create_output_texture(state, options.width, options.height);
}

static bool init_gsx_pipeline(app_state* state, const app_options& options)
{
    if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
        return false;
    }

    gsx_index_t visible_device_count = 0;
    if(!gsx_check(
           gsx_count_backend_devices_by_type(options.backend_type, &visible_device_count),
           "gsx_count_backend_devices_by_type")) {
        return false;
    }

    if(visible_device_count <= 0) {
        std::fprintf(stderr, "error: no visible devices for backend type %d\n", options.backend_type);
        return false;
    }
    if(options.device_index < 0 || options.device_index >= visible_device_count) {
        std::fprintf(stderr, "error: device index %d out of range [0, %d)\n", options.device_index, visible_device_count);
        return false;
    }

    gsx_backend_device_t device = nullptr;
    if(!gsx_check(
           gsx_get_backend_device_by_type(options.backend_type, options.device_index, &device),
           "gsx_get_backend_device_by_type")) {
        return false;
    }

    gsx_backend_desc backend_desc = {};
    backend_desc.device = device;
    if(!gsx_check(gsx_backend_init(&state->backend, &backend_desc), "gsx_backend_init")) {
        return false;
    }

    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    if(!gsx_check(
           gsx_backend_find_buffer_type(state->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type),
           "gsx_backend_find_buffer_type(device)")) {
        return false;
    }

    gsx_gs_desc gs_desc = {};
    gs_desc.buffer_type = device_buffer_type;
    gs_desc.arena_desc.initial_capacity_bytes = static_cast<gsx_size_t>(64u << 20);
    gs_desc.count = 0;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
    if(!gsx_check(gsx_gs_init(&state->gs, &gs_desc), "gsx_gs_init")) {
        return false;
    }

    if(!gsx_check(gsx_read_ply(&state->gs, options.input_ply_path), "gsx_read_ply")) {
        return false;
    }

    if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_MEAN3D, &state->gs_mean3d), "gsx_gs_get_field(mean3d)")) {
        return false;
    }
    if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_ROTATION, &state->gs_rotation), "gsx_gs_get_field(rotation)")) {
        return false;
    }
    if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_LOGSCALE, &state->gs_logscale), "gsx_gs_get_field(logscale)")) {
        return false;
    }
    if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_SH0, &state->gs_sh0), "gsx_gs_get_field(sh0)")) {
        return false;
    }
    if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_OPACITY, &state->gs_opacity), "gsx_gs_get_field(opacity)")) {
        return false;
    }

    return init_render_targets(state, options);
}

static void cleanup(app_state* state)
{
    if(state->frame_texture != nullptr) {
        SDL_DestroyTexture(state->frame_texture);
        state->frame_texture = nullptr;
    }

    if(state->out_rgb != nullptr) {
        gsx_check(gsx_tensor_free(state->out_rgb), "gsx_tensor_free(out_rgb)");
        state->out_rgb = nullptr;
    }
    if(state->rgb_f32_srgb != nullptr) {
        gsx_check(gsx_tensor_free(state->rgb_f32_srgb), "gsx_tensor_free(rgb_f32_srgb)");
        state->rgb_f32_srgb = nullptr;
    }
    if(state->rgb_u8_chw != nullptr) {
        gsx_check(gsx_tensor_free(state->rgb_u8_chw), "gsx_tensor_free(rgb_u8_chw)");
        state->rgb_u8_chw = nullptr;
    }
    if(state->rgb_u8_hwc != nullptr) {
        gsx_check(gsx_tensor_free(state->rgb_u8_hwc), "gsx_tensor_free(rgb_u8_hwc)");
        state->rgb_u8_hwc = nullptr;
    }
    if(state->render_arena != nullptr) {
        gsx_check(gsx_arena_free(state->render_arena), "gsx_arena_free(render_arena)");
        state->render_arena = nullptr;
    }
    if(state->render_context != nullptr) {
        gsx_check(gsx_render_context_free(state->render_context), "gsx_render_context_free");
        state->render_context = nullptr;
    }
    if(state->renderer != nullptr) {
        gsx_check(gsx_renderer_free(state->renderer), "gsx_renderer_free");
        state->renderer = nullptr;
    }
    if(state->gs != nullptr) {
        gsx_check(gsx_gs_free(state->gs), "gsx_gs_free");
        state->gs = nullptr;
    }
    if(state->backend != nullptr) {
        gsx_check(gsx_backend_free(state->backend), "gsx_backend_free");
        state->backend = nullptr;
    }

    if(state->sdl_renderer != nullptr) {
        SDL_DestroyRenderer(state->sdl_renderer);
        state->sdl_renderer = nullptr;
    }
    if(state->window != nullptr) {
        SDL_DestroyWindow(state->window);
        state->window = nullptr;
    }
}

static bool render_gsx_frame(app_state* state, const app_options& options)
{
    gsx_camera_intrinsics intrinsics = {};
    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.width = options.width;
    intrinsics.height = options.height;
    intrinsics.fx = options.fx;
    intrinsics.fy = options.fy;
    intrinsics.cx = options.cx;
    intrinsics.cy = options.cy;
    intrinsics.camera_id = 0;

    gsx_camera_pose pose = {};
    camera_to_gsx_pose(state->camera, &pose);

    gsx_render_forward_request request = {};
    request.intrinsics = &intrinsics;
    request.pose = &pose;
    request.near_plane = 0.01f;
    request.far_plane = 1000.0f;
    request.background_color.x = 0.0f;
    request.background_color.y = 0.0f;
    request.background_color.z = 0.0f;
    request.precision = GSX_RENDER_PRECISION_FLOAT32;
    request.sh_degree = 0;
    request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    request.borrow_train_state = false;
    request.gs_mean3d = state->gs_mean3d;
    request.gs_rotation = state->gs_rotation;
    request.gs_logscale = state->gs_logscale;
    request.gs_sh0 = state->gs_sh0;
    request.gs_opacity = state->gs_opacity;
    request.out_rgb = state->out_rgb;

    const auto start = std::chrono::steady_clock::now();

    if(!gsx_check(gsx_renderer_render(state->renderer, state->render_context, &request), "gsx_renderer_render")) {
        return false;
    }
    if(!gsx_check(gsx_backend_major_stream_sync(state->backend), "gsx_backend_major_stream_sync")) {
        return false;
    }

    const auto end = std::chrono::steady_clock::now();
    state->last_render_ms = std::chrono::duration<float, std::milli>(end - start).count();

    if(!gsx_check(
           gsx_tensor_image_convert_colorspace(
               state->rgb_f32_srgb,
               GSX_IMAGE_COLOR_SPACE_SRGB,
               state->out_rgb,
               GSX_IMAGE_COLOR_SPACE_LINEAR),
           "gsx_tensor_image_convert_colorspace")) {
        return false;
    }

    if(!gsx_check(
           gsx_tensor_image_convert_data_type(state->rgb_u8_chw, state->rgb_f32_srgb),
           "gsx_tensor_image_convert_data_type")) {
        return false;
    }

    if(!gsx_check(
           gsx_tensor_image_convert_storage_format(state->rgb_u8_hwc, state->rgb_u8_chw),
           "gsx_tensor_image_convert_storage_format")) {
        return false;
    }

    const gsx_size_t rgba_byte_count = static_cast<gsx_size_t>(options.width) * static_cast<gsx_size_t>(options.height) * 4u;
    if(!gsx_check(
           gsx_tensor_download(state->rgb_u8_hwc, state->host_rgba.data(), rgba_byte_count),
           "gsx_tensor_download(rgb_u8_hwc)")) {
        return false;
    }
    if(!gsx_check(gsx_backend_major_stream_sync(state->backend), "gsx_backend_major_stream_sync")) {
        return false;
    }

    if(!SDL_UpdateTexture(state->frame_texture, nullptr, state->host_rgba.data(), static_cast<int>(options.width * 4))) {
        std::fprintf(stderr, "error: SDL_UpdateTexture failed: %s\n", SDL_GetError());
        return false;
    }

    state->camera_dirty = false;
    return true;
}

static bool update_keyboard_camera(
    app_state* state,
    const app_options& options,
    const float delta_seconds,
    const bool capture_keyboard)
{
    if(capture_keyboard) {
        return false;
    }

    const bool* keyboard = SDL_GetKeyboardState(nullptr);
    if(keyboard == nullptr) {
        return false;
    }

    vec3 forward = { 0.0f, 0.0f, 1.0f };
    vec3 right = { 1.0f, 0.0f, 0.0f };
    vec3 up = { 0.0f, 1.0f, 0.0f };
    camera_basis(state->camera, &forward, &right, &up);

    forward = vec3_normalize({ forward.x, 0.0f, forward.z });
    if(vec3_length(forward) < 1.0e-6f) {
        forward = { 0.0f, 0.0f, 1.0f };
    }
    right = vec3_normalize({ right.x, 0.0f, right.z });
    if(vec3_length(right) < 1.0e-6f) {
        right = { 1.0f, 0.0f, 0.0f };
    }

    vec3 move = { 0.0f, 0.0f, 0.0f };
    if(keyboard[SDL_SCANCODE_W]) {
        move = vec3_add(move, forward);
    }
    if(keyboard[SDL_SCANCODE_S]) {
        move = vec3_sub(move, forward);
    }
    if(keyboard[SDL_SCANCODE_D]) {
        move = vec3_add(move, right);
    }
    if(keyboard[SDL_SCANCODE_A]) {
        move = vec3_sub(move, right);
    }

    if(vec3_length(move) < 1.0e-8f) {
        return false;
    }

    move = vec3_normalize(move);
    state->camera.position = vec3_add(state->camera.position, vec3_scale(move, options.move_speed * delta_seconds));
    return true;
}

static bool update_mouse_pan_camera(
    app_state* state,
    const app_options& options,
    const ImVec2 drag_delta,
    const bool capture_mouse)
{
    if(capture_mouse) {
        return false;
    }

    if(std::fabs(drag_delta.x) <= 1.0e-6f && std::fabs(drag_delta.y) <= 1.0e-6f) {
        return false;
    }

    vec3 forward = { 0.0f, 0.0f, 1.0f };
    vec3 right = { 1.0f, 0.0f, 0.0f };
    vec3 up = { 0.0f, 1.0f, 0.0f };
    camera_basis(state->camera, &forward, &right, &up);

    const float pan_speed = options.move_speed * 0.01f;
    vec3 pan_move = { 0.0f, 0.0f, 0.0f };
    pan_move = vec3_add(pan_move, vec3_scale(right, -drag_delta.x * pan_speed));
    pan_move = vec3_add(pan_move, vec3_scale(up, drag_delta.y * pan_speed));
    state->camera.position = vec3_add(state->camera.position, pan_move);
    return true;
}

static bool update_wheel_dolly_camera(
    app_state* state,
    const app_options& options,
    const float wheel_delta,
    const bool capture_mouse)
{
    if(capture_mouse) {
        return false;
    }
    if(std::fabs(wheel_delta) <= 1.0e-6f) {
        return false;
    }

    vec3 forward = { 0.0f, 0.0f, 1.0f };
    vec3 right = { 1.0f, 0.0f, 0.0f };
    vec3 up = { 0.0f, 1.0f, 0.0f };
    camera_basis(state->camera, &forward, &right, &up);

    const float dolly_step = options.move_speed * 0.25f;
    state->camera.position = vec3_add(state->camera.position, vec3_scale(forward, wheel_delta * dolly_step));
    return true;
}

static bool draw_ui(app_state* state, app_options* options)
{
    bool changed = false;

    ImGui::Begin("Camera");
    ImGui::Text("Mouse drag: rotate");
    ImGui::Text("Right drag: pan");
    ImGui::Text("Wheel: dolly");
    ImGui::Text("WASD: move");
    ImGui::Separator();
    ImGui::Text("Last render: %.3f ms", state->last_render_ms);

    int width_i = static_cast<int>(options->width);
    int height_i = static_cast<int>(options->height);
    if(ImGui::InputInt("Width", &width_i) && width_i > 0) {
        options->width = static_cast<gsx_index_t>(width_i);
        state->request_resize = true;
        changed = true;
    }
    if(ImGui::InputInt("Height", &height_i) && height_i > 0) {
        options->height = static_cast<gsx_index_t>(height_i);
        state->request_resize = true;
        changed = true;
    }

    if(ImGui::InputFloat("fx", &options->fx, 1.0f, 10.0f, "%.3f") && options->fx > 0.0f) {
        changed = true;
    }
    if(ImGui::InputFloat("fy", &options->fy, 1.0f, 10.0f, "%.3f") && options->fy > 0.0f) {
        changed = true;
    }
    if(ImGui::InputFloat("cx", &options->cx, 1.0f, 10.0f, "%.3f")) {
        changed = true;
    }
    if(ImGui::InputFloat("cy", &options->cy, 1.0f, 10.0f, "%.3f")) {
        changed = true;
    }

    changed |= ImGui::DragFloat3("Position", &state->camera.position.x, 0.01f);

    float yaw_deg = state->camera.yaw * (180.0f / 3.1415926535f);
    float pitch_deg = state->camera.pitch * (180.0f / 3.1415926535f);
    if(ImGui::SliderFloat("Yaw (deg)", &yaw_deg, -180.0f, 180.0f, "%.2f")) {
        state->camera.yaw = yaw_deg * (3.1415926535f / 180.0f);
        changed = true;
    }
    if(ImGui::SliderFloat("Pitch (deg)", &pitch_deg, -89.0f, 89.0f, "%.2f")) {
        state->camera.pitch = pitch_deg * (3.1415926535f / 180.0f);
        changed = true;
    }
    state->camera.yaw = wrap_pi(state->camera.yaw);
    state->camera.pitch = clampf(state->camera.pitch, -1.553343f, 1.553343f);

    if(ImGui::Button("Reset Camera")) {
        state->camera = camera_state{};
        changed = true;
    }

    ImGui::End();

    ImGui::Begin("Viewport");
    if(state->frame_texture != nullptr) {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float tex_w = static_cast<float>(options->width);
        const float tex_h = static_cast<float>(options->height);
        const float sx = (tex_w > 0.0f) ? (avail.x / tex_w) : 1.0f;
        const float sy = (tex_h > 0.0f) ? (avail.y / tex_h) : 1.0f;
        const float scale = std::max(0.01f, std::min(sx, sy));
        const ImVec2 image_size = { tex_w * scale, tex_h * scale };
        ImGui::Image(reinterpret_cast<ImTextureID>(state->frame_texture), image_size);
    }
    ImGui::End();

    return changed;
}

} // namespace

int main(int argc, char** argv)
{
    app_options options;
    options.cx = static_cast<float>(options.width) * 0.5f;
    options.cy = static_cast<float>(options.height) * 0.5f;
    if(!parse_args(argc, argv, &options)) {
        return EXIT_FAILURE;
    }

    if(!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        std::fprintf(stderr, "error: SDL_Init failed: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    app_state state;
    state.window = SDL_CreateWindow("GSX PLY Viewer", 1280, 900, SDL_WINDOW_RESIZABLE);
    if(state.window == nullptr) {
        std::fprintf(stderr, "error: SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    state.sdl_renderer = SDL_CreateRenderer(state.window, nullptr);
    if(state.sdl_renderer == nullptr) {
        std::fprintf(stderr, "error: SDL_CreateRenderer failed: %s\n", SDL_GetError());
        cleanup(&state);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    if(!init_gsx_pipeline(&state, options)) {
        cleanup(&state);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    if(!ImGui_ImplSDL3_InitForSDLRenderer(state.window, state.sdl_renderer)) {
        std::fprintf(stderr, "error: ImGui_ImplSDL3_InitForSDLRenderer failed\n");
        ImGui::DestroyContext();
        cleanup(&state);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    if(!ImGui_ImplSDLRenderer3_Init(state.sdl_renderer)) {
        std::fprintf(stderr, "error: ImGui_ImplSDLRenderer3_Init failed\n");
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        cleanup(&state);
        SDL_Quit();
        return EXIT_FAILURE;
    }

    auto previous_time = std::chrono::steady_clock::now();
    while(state.running) {
        const auto now = std::chrono::steady_clock::now();
        const float delta_seconds = std::chrono::duration<float>(now - previous_time).count();
        previous_time = now;

        float wheel_delta = 0.0f;
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if(event.type == SDL_EVENT_QUIT) {
                state.running = false;
            } else if(event.type == SDL_EVENT_MOUSE_WHEEL) {
                wheel_delta += event.wheel.y;
            }
        }

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        const ImGuiIO& frame_io = ImGui::GetIO();
        bool camera_changed = false;
        if(!frame_io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);

            state.camera.yaw -= drag_delta.x * options.mouse_sensitivity;
            state.camera.pitch -= drag_delta.y * options.mouse_sensitivity;
            state.camera.yaw = wrap_pi(state.camera.yaw);
            state.camera.pitch = clampf(state.camera.pitch, -1.553343f, 1.553343f);
            camera_changed = true;
        }
        if(ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            const ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.0f);
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
            camera_changed |= update_mouse_pan_camera(&state, options, drag_delta, frame_io.WantCaptureMouse);
        }
        camera_changed |= update_wheel_dolly_camera(&state, options, wheel_delta, frame_io.WantCaptureMouse);

        camera_changed |= update_keyboard_camera(&state, options, delta_seconds, frame_io.WantCaptureKeyboard);
        camera_changed |= draw_ui(&state, &options);
        state.camera_dirty = state.camera_dirty || camera_changed;

        if(state.request_resize) {
            if(!init_render_targets(&state, options)) {
                state.running = false;
            }
            state.request_resize = false;
            state.camera_dirty = true;
        }

        if(state.running && state.camera_dirty) {
            if(!render_gsx_frame(&state, options)) {
                state.running = false;
            }
        }

        ImGui::Render();
        if(!SDL_SetRenderDrawColor(state.sdl_renderer, 25, 28, 34, 255)) {
            std::fprintf(stderr, "error: SDL_SetRenderDrawColor failed: %s\n", SDL_GetError());
            break;
        }
        if(!SDL_RenderClear(state.sdl_renderer)) {
            std::fprintf(stderr, "error: SDL_RenderClear failed: %s\n", SDL_GetError());
            break;
        }
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), state.sdl_renderer);
        SDL_RenderPresent(state.sdl_renderer);
    }

    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    cleanup(&state);
    SDL_Quit();
    return EXIT_SUCCESS;
}