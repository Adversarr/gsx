#include <gsx/extra/gsx-stbi.h>
#include <gsx/gsx.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct app_adc_dummy_dataset {
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    float rgb[3];
} app_adc_dummy_dataset;

typedef struct app_options {
    const char *input_path;
    const char *output_path;

    gsx_backend_type backend_type;
    gsx_index_t device_index;
    gsx_backend_buffer_type_class buffer_type_class;

    gsx_size_t gaussian_count;
    gsx_index_t train_steps;
    gsx_index_t log_interval;
    uint32_t seed;
    gsx_index_t train_width;
    gsx_index_t train_height;

    gsx_float_t learning_rate_mean3d;
    gsx_float_t learning_rate_logscale;
    gsx_float_t learning_rate_rotation;
    gsx_float_t learning_rate_opacity;
    gsx_float_t learning_rate_sh0;
    gsx_float_t learning_rate_sh1;
    gsx_float_t learning_rate_sh2;
    gsx_float_t learning_rate_sh3;
    gsx_float_t beta1;
    gsx_float_t beta2;
    gsx_float_t epsilon;
    gsx_float_t weight_decay;
    gsx_float_t max_grad;

    gsx_float_t l1_scale;
    gsx_float_t ssim_scale;

    gsx_render_precision render_precision;
    gsx_index_t sh_degree;
    gsx_float_t near_plane;
    gsx_float_t far_plane;
    gsx_float_t fx;
    gsx_float_t fy;
    gsx_float_t cx;
    gsx_float_t cy;
    gsx_float_t camera_z;
    gsx_vec3 background_color;

    gsx_float_t init_scale_xy;
    gsx_float_t init_scale_z;
    gsx_float_t init_opacity;
    gsx_float_t init_position_jitter;
    gsx_float_t init_rotation_jitter;
    gsx_float_t init_sh0_jitter;
} app_options;

typedef struct app_timing {
    double total_render_forward_us;
    double total_loss_forward_us;
    double total_loss_backward_us;
    double total_render_backward_us;
    double total_optim_step_us;
    double total_adc_step_us;
    gsx_index_t step_count;
} app_timing;

typedef struct app_state {
    gsx_backend_t backend;
    gsx_arena_t arena;

    gsx_adc_t adc;
    gsx_dataset_t adc_dataset;
    gsx_dataloader_t adc_dataloader;
    app_adc_dummy_dataset adc_dataset_object;

    gsx_gs_t gs;
    gsx_renderer_t renderer;
    gsx_render_context_t render_context;

    gsx_loss_t l1_loss;
    gsx_loss_context_t l1_context;
    gsx_loss_t ssim_loss;
    gsx_loss_context_t ssim_context;

    gsx_optim_t optim;

    gsx_tensor_t target_rgb;
    gsx_tensor_t out_rgb;
    gsx_tensor_t loss_map;
    gsx_tensor_t grad_out_rgb;

    gsx_tensor_t gs_mean3d;
    gsx_tensor_t gs_rotation;
    gsx_tensor_t gs_logscale;
    gsx_tensor_t gs_opacity;
    gsx_tensor_t gs_sh0;
    gsx_tensor_t gs_sh1;
    gsx_tensor_t gs_sh2;
    gsx_tensor_t gs_sh3;

    gsx_tensor_t gs_grad_mean3d;
    gsx_tensor_t gs_grad_rotation;
    gsx_tensor_t gs_grad_logscale;
    gsx_tensor_t gs_grad_opacity;
    gsx_tensor_t gs_grad_sh0;
    gsx_tensor_t gs_grad_sh1;
    gsx_tensor_t gs_grad_sh2;
    gsx_tensor_t gs_grad_sh3;
    gsx_tensor_t gs_grad_acc;

    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;

    gsx_index_t width;
    gsx_index_t height;
    gsx_size_t image_element_count;
    float *target_host;
    float *render_host;

    app_timing timing;
} app_state;

static gsx_error app_adc_dummy_dataset_get_length(void *object, gsx_size_t *out_length)
{
    (void)object;
    if(out_length == NULL) {
        return (gsx_error){ GSX_ERROR_INVALID_ARGUMENT, "out_length must be non-null" };
    }
    *out_length = 1;
    return (gsx_error){ GSX_ERROR_SUCCESS, NULL };
}

static gsx_error app_adc_dummy_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    app_adc_dummy_dataset *dataset = (app_adc_dummy_dataset *)object;

    if(dataset == NULL || out_sample == NULL) {
        return (gsx_error){ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_sample must be non-null" };
    }
    if(sample_index != 0) {
        return (gsx_error){ GSX_ERROR_OUT_OF_RANGE, "sample_index must be 0 for dummy dataset" };
    }

    memset(out_sample, 0, sizeof(*out_sample));
    out_sample->intrinsics = dataset->intrinsics;
    out_sample->pose = dataset->pose;
    out_sample->rgb.data = dataset->rgb;
    out_sample->rgb.data_type = GSX_DATA_TYPE_F32;
    out_sample->rgb.width = 1;
    out_sample->rgb.height = 1;
    out_sample->rgb.channel_count = 3;
    out_sample->rgb.row_stride_bytes = 3u * sizeof(float);
    return (gsx_error){ GSX_ERROR_SUCCESS, NULL };
}

static void app_adc_dummy_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    (void)object;
    (void)sample;
}

static bool gsx_check(gsx_error err, const char *context)
{
    if(gsx_error_is_success(err)) {
        return true;
    }
    fprintf(stderr, "error: %s failed (%d)", context, (int)err.code);
    if(err.message != NULL) {
        fprintf(stderr, ": %s", err.message);
    }
    fprintf(stderr, "\n");
    return false;
}

static void set_default_options(app_options *opt)
{
    memset(opt, 0, sizeof(*opt));

    opt->input_path = NULL;
    opt->output_path = "fit_output.png";

    opt->backend_type = GSX_BACKEND_TYPE_CPU;
    opt->device_index = 0;
    opt->buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;

    opt->gaussian_count = 16;
    opt->train_steps = 400;
    opt->log_interval = 20;
    opt->seed = 1234u;
    opt->train_width = 0;
    opt->train_height = 0;

    opt->learning_rate_mean3d = 0.02f;
    opt->learning_rate_logscale = 0.01f;
    opt->learning_rate_rotation = 0.003f;
    opt->learning_rate_opacity = 0.01f;
    opt->learning_rate_sh0 = 0.02f;
    opt->learning_rate_sh1 = 0.005f;
    opt->learning_rate_sh2 = 0.0025f;
    opt->learning_rate_sh3 = 0.0015f;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;
    opt->weight_decay = 0.0f;
    opt->max_grad = 0.0f;

    opt->l1_scale = 0.8f;
    opt->ssim_scale = 0.2f;

    opt->render_precision = GSX_RENDER_PRECISION_FLOAT32;
    opt->sh_degree = 3;
    opt->near_plane = 0.01f;
    opt->far_plane = 10.0f;
    opt->fx = 0.0f;
    opt->fy = 0.0f;
    opt->cx = 0.0f;
    opt->cy = 0.0f;
    opt->camera_z = 1.0f;
    opt->background_color.x = 0.0f;
    opt->background_color.y = 0.0f;
    opt->background_color.z = 0.0f;

    opt->init_scale_xy = 0.02f;
    opt->init_scale_z = 0.02f;
    opt->init_opacity = 0.25f;
    opt->init_position_jitter = 0.0f;
    opt->init_rotation_jitter = 0.0f;
    opt->init_sh0_jitter = 0.05f;
}

static bool parse_i64(const char *value, int64_t *out)
{
    char *end = NULL;
    long long parsed = 0;

    if(value == NULL || out == NULL) {
        return false;
    }
    errno = 0;
    parsed = strtoll(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = (int64_t)parsed;
    return true;
}

static bool parse_u32(const char *value, uint32_t *out)
{
    char *end = NULL;
    unsigned long parsed = 0;

    if(value == NULL || out == NULL) {
        return false;
    }
    errno = 0;
    parsed = strtoul(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return false;
    }
    *out = (uint32_t)parsed;
    return true;
}

static bool parse_f32(const char *value, float *out)
{
    char *end = NULL;
    float parsed = 0.0f;

    if(value == NULL || out == NULL) {
        return false;
    }
    errno = 0;
    parsed = strtof(value, &end);
    if(errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out = parsed;
    return true;
}

static bool parse_backend_type(const char *value, gsx_backend_type *out)
{
    if(value == NULL || out == NULL) {
        return false;
    }
    if(strcmp(value, "cpu") == 0) {
        *out = GSX_BACKEND_TYPE_CPU;
        return true;
    }
    if(strcmp(value, "cuda") == 0) {
        *out = GSX_BACKEND_TYPE_CUDA;
        return true;
    }
    if(strcmp(value, "metal") == 0) {
        *out = GSX_BACKEND_TYPE_METAL;
        return true;
    }
    return false;
}

static bool parse_buffer_type_class(const char *value, gsx_backend_buffer_type_class *out)
{
    if(value == NULL || out == NULL) {
        return false;
    }
    if(strcmp(value, "host") == 0) {
        *out = GSX_BACKEND_BUFFER_TYPE_HOST;
        return true;
    }
    if(strcmp(value, "host_pinned") == 0) {
        *out = GSX_BACKEND_BUFFER_TYPE_HOST_PINNED;
        return true;
    }
    if(strcmp(value, "device") == 0) {
        *out = GSX_BACKEND_BUFFER_TYPE_DEVICE;
        return true;
    }
    if(strcmp(value, "unified") == 0) {
        *out = GSX_BACKEND_BUFFER_TYPE_UNIFIED;
        return true;
    }
    return false;
}

static const char *backend_name(gsx_backend_type type)
{
    switch(type) {
    case GSX_BACKEND_TYPE_CPU:
        return "cpu";
    case GSX_BACKEND_TYPE_CUDA:
        return "cuda";
    case GSX_BACKEND_TYPE_METAL:
        return "metal";
    default:
        return "unknown";
    }
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s --input <image> [options]\n"
        "\n"
        "core:\n"
        "  --output <path>                 output image path (default: fit_output.png)\n"
        "  --backend cpu|cuda|metal        backend (default: cpu)\n"
        "  --device <index>                backend device index (default: 0)\n"
        "  --buffer-type host|host_pinned|device|unified (default: device)\n"
        "\n"
        "training:\n"
        "  --gaussians <count>             gaussian count (default: 16)\n"
        "  --steps <count>                 optimization steps (default: 400)\n"
        "  --log-interval <n>              print every n steps (default: 20)\n"
        "  --seed <u32>                    random seed\n"
        "  --train-width <px>              training image width (default: input width)\n"
        "  --train-height <px>             training image height (default: input height)\n"
        "\n"
        "loss:\n"
        "  --l1-scale <f32>                L1 scale (default: 0.8)\n"
        "  --ssim-scale <f32>              SSIM scale (default: 0.2)\n"
        "\n"
        "optimizer (adam):\n"
        "  --lr-mean3d <f32>\n"
        "  --lr-logscale <f32>\n"
        "  --lr-rotation <f32>\n"
        "  --lr-opacity <f32>\n"
        "  --lr-sh0 <f32>\n"
        "  --lr-sh1 <f32>\n"
        "  --lr-sh2 <f32>\n"
        "  --lr-sh3 <f32>\n"
        "  --beta1 <f32> --beta2 <f32> --epsilon <f32>\n"
        "  --weight-decay <f32> --max-grad <f32>\n"
        "\n"
        "render/camera:\n"
        "  --sh-degree 0|1|2|3             (default: 3)\n"
        "  --near <f32> --far <f32>\n"
        "  --fx <f32> --fy <f32>           <= 0 uses image width/height\n"
        "  --cx <f32> --cy <f32>           <= 0 uses centered principal point\n"
        "  --camera-z <f32>                camera z in world frame (default: 1.0)\n"
        "  --bg-r <f32> --bg-g <f32> --bg-b <f32>\n"
        "\n"
        "initialization:\n"
        "  --init-scale-xy <f32>\n"
        "  --init-scale-z <f32>\n"
        "  --init-opacity <f32>            sigmoid-space value in (0,1)\n"
        "  --init-position-jitter <f32>\n"
        "  --init-rotation-jitter <f32>\n"
        "  --init-sh0-jitter <f32>\n",
        prog);
}

static bool parse_options(int argc, char **argv, app_options *opt)
{
    int i = 0;

    for(i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if(strcmp(arg, "--input") == 0 && i + 1 < argc) {
            opt->input_path = argv[++i];
            continue;
        }
        if(strcmp(arg, "--output") == 0 && i + 1 < argc) {
            opt->output_path = argv[++i];
            continue;
        }

        if(strcmp(arg, "--backend") == 0 && i + 1 < argc) {
            if(!parse_backend_type(argv[++i], &opt->backend_type)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--device") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value < 0) {
                return false;
            }
            opt->device_index = (gsx_index_t)value;
            continue;
        }
        if(strcmp(arg, "--buffer-type") == 0 && i + 1 < argc) {
            if(!parse_buffer_type_class(argv[++i], &opt->buffer_type_class)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--gaussians") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value <= 0) {
                return false;
            }
            opt->gaussian_count = (gsx_size_t)value;
            continue;
        }
        if(strcmp(arg, "--steps") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value <= 0) {
                return false;
            }
            opt->train_steps = (gsx_index_t)value;
            continue;
        }
        if(strcmp(arg, "--log-interval") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value <= 0) {
                return false;
            }
            opt->log_interval = (gsx_index_t)value;
            continue;
        }
        if(strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            if(!parse_u32(argv[++i], &opt->seed)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--train-width") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value <= 0) {
                return false;
            }
            opt->train_width = (gsx_index_t)value;
            continue;
        }
        if(strcmp(arg, "--train-height") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value <= 0) {
                return false;
            }
            opt->train_height = (gsx_index_t)value;
            continue;
        }

        if(strcmp(arg, "--lr-mean3d") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_mean3d)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-logscale") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_logscale)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-rotation") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_rotation)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-opacity") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_opacity)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-sh0") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_sh0)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-sh1") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_sh1)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-sh2") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_sh2)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--lr-sh3") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->learning_rate_sh3)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--beta1") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->beta1)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--beta2") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->beta2)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--epsilon") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->epsilon)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--weight-decay") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->weight_decay)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--max-grad") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->max_grad)) {
                return false;
            }
            continue;
        }

        if(strcmp(arg, "--l1-scale") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->l1_scale)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--ssim-scale") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->ssim_scale)) {
                return false;
            }
            continue;
        }

        if(strcmp(arg, "--sh-degree") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if(!parse_i64(argv[++i], &value) || value < 0 || value > 3) {
                return false;
            }
            opt->sh_degree = (gsx_index_t)value;
            continue;
        }
        if(strcmp(arg, "--near") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->near_plane)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--far") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->far_plane)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--fx") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->fx)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--fy") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->fy)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--cx") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->cx)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--cy") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->cy)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--camera-z") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->camera_z)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--bg-r") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->background_color.x)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--bg-g") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->background_color.y)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--bg-b") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->background_color.z)) {
                return false;
            }
            continue;
        }

        if(strcmp(arg, "--init-scale-xy") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->init_scale_xy)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--init-scale-z") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->init_scale_z)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--init-opacity") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->init_opacity)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--init-position-jitter") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->init_position_jitter)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--init-rotation-jitter") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->init_rotation_jitter)) {
                return false;
            }
            continue;
        }
        if(strcmp(arg, "--init-sh0-jitter") == 0 && i + 1 < argc) {
            if(!parse_f32(argv[++i], &opt->init_sh0_jitter)) {
                return false;
            }
            continue;
        }

        fprintf(stderr, "error: unknown argument '%s'\n", arg);
        return false;
    }

    if(opt->input_path == NULL) {
        fprintf(stderr, "error: --input is required\n");
        return false;
    }
    if(opt->gaussian_count == 0 || opt->train_steps <= 0) {
        return false;
    }
    if(opt->near_plane <= 0.0f || opt->far_plane <= opt->near_plane) {
        return false;
    }
    if(opt->camera_z <= 0.0f) {
        return false;
    }
    return true;
}

static float clamp_f32(float x, float lo, float hi)
{
    if(x < lo) {
        return lo;
    }
    if(x > hi) {
        return hi;
    }
    return x;
}

static uint32_t lcg_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float random_uniform01(uint32_t *state)
{
    const uint32_t value = lcg_next(state);
    return (float)(value & 0x00FFFFFFu) / 16777216.0f;
}

static float random_normal(uint32_t *state)
{
    const float u1 = clamp_f32(random_uniform01(state), 1e-7f, 1.0f);
    const float u2 = random_uniform01(state);
    const float radius = sqrtf(-2.0f * logf(u1));
    const float theta = 6.28318530718f * u2;
    return radius * cosf(theta);
}

static double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

static bool sync_backend_if_needed(gsx_backend_t backend)
{
    gsx_backend_info info = {0};

    if(!gsx_check(gsx_backend_get_info(backend, &info), "gsx_backend_get_info")) {
        return false;
    }
    if(info.backend_type == GSX_BACKEND_TYPE_CPU) {
        return true;
    }
    return gsx_check(gsx_backend_major_stream_sync(backend), "gsx_backend_major_stream_sync");
}

static bool compute_runtime_arena_required_bytes(
    gsx_backend_buffer_type_t buffer_type,
    gsx_index_t width,
    gsx_index_t height,
    gsx_size_t *out_required_bytes)
{
    gsx_arena_desc dry_run_arena_desc = {0};
    gsx_tensor_desc tensor_descs[4] = {0};

    if(out_required_bytes == NULL) {
        return false;
    }
    *out_required_bytes = 0;

    dry_run_arena_desc.initial_capacity_bytes = 0;
    dry_run_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    dry_run_arena_desc.dry_run = true;
    tensor_descs[0].rank = 3;
    tensor_descs[0].shape[0] = 3;
    tensor_descs[0].shape[1] = height;
    tensor_descs[0].shape[2] = width;
    tensor_descs[0].data_type = GSX_DATA_TYPE_F32;
    tensor_descs[0].storage_format = GSX_STORAGE_FORMAT_CHW;
    tensor_descs[1] = tensor_descs[0];
    tensor_descs[2] = tensor_descs[0];
    tensor_descs[3] = tensor_descs[0];

    return gsx_check(
        gsx_tensor_plan_required_bytes(buffer_type, &dry_run_arena_desc, tensor_descs, 4, out_required_bytes),
        "gsx_tensor_plan_required_bytes(runtime)");
}

static bool compute_initial_gs_required_bytes(
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t gaussian_count,
    gsx_gs_aux_flags aux_flags,
    gsx_size_t *out_required_bytes)
{
    gsx_gs_t dry_run_gs = NULL;
    gsx_gs_desc dry_run_desc = {0};
    gsx_gs_info gs_info = {0};

    if(buffer_type == NULL || out_required_bytes == NULL || gaussian_count == 0) {
        return false;
    }
    *out_required_bytes = 0;

    dry_run_desc.buffer_type = buffer_type;
    dry_run_desc.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    dry_run_desc.arena_desc.dry_run = true;
    dry_run_desc.count = gaussian_count;
    dry_run_desc.aux_flags = aux_flags;
    if(!gsx_check(gsx_gs_init(&dry_run_gs, &dry_run_desc), "gsx_gs_init(gs dry run)")) {
        return false;
    }
    if(!gsx_check(gsx_gs_get_info(dry_run_gs, &gs_info), "gsx_gs_get_info(gs dry run)")) {
        (void)gsx_gs_free(dry_run_gs);
        return false;
    }
    if(!gsx_check(gsx_arena_get_required_bytes(gs_info.arena, out_required_bytes), "gsx_arena_get_required_bytes(gs dry run)")) {
        (void)gsx_gs_free(dry_run_gs);
        return false;
    }
    return gsx_check(gsx_gs_free(dry_run_gs), "gsx_gs_free(gs dry run)");
}

static bool fetch_gs_fields(app_state *s)
{
    return gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_MEAN3D, &s->gs_mean3d), "gsx_gs_get_field(mean3d)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_ROTATION, &s->gs_rotation), "gsx_gs_get_field(rotation)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_LOGSCALE, &s->gs_logscale), "gsx_gs_get_field(logscale)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_OPACITY, &s->gs_opacity), "gsx_gs_get_field(opacity)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_SH0, &s->gs_sh0), "gsx_gs_get_field(sh0)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_SH1, &s->gs_sh1), "gsx_gs_get_field(sh1)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_SH2, &s->gs_sh2), "gsx_gs_get_field(sh2)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_SH3, &s->gs_sh3), "gsx_gs_get_field(sh3)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_MEAN3D, &s->gs_grad_mean3d), "gsx_gs_get_field(grad_mean3d)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_ROTATION, &s->gs_grad_rotation), "gsx_gs_get_field(grad_rotation)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_LOGSCALE, &s->gs_grad_logscale), "gsx_gs_get_field(grad_logscale)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_OPACITY, &s->gs_grad_opacity), "gsx_gs_get_field(grad_opacity)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_SH0, &s->gs_grad_sh0), "gsx_gs_get_field(grad_sh0)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_SH1, &s->gs_grad_sh1), "gsx_gs_get_field(grad_sh1)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_SH2, &s->gs_grad_sh2), "gsx_gs_get_field(grad_sh2)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_SH3, &s->gs_grad_sh3), "gsx_gs_get_field(grad_sh3)")
        && gsx_check(gsx_gs_get_field(s->gs, GSX_GS_FIELD_GRAD_ACC, &s->gs_grad_acc), "gsx_gs_get_field(grad_acc)");
}

static bool accumulate_grad_acc(app_state *s)
{
    gsx_tensor_info grad_mean_info = {0};
    gsx_tensor_info grad_acc_info = {0};
    gsx_size_t count = 0;
    gsx_size_t i = 0;
    float *grad_mean = NULL;
    float *grad_acc = NULL;
    bool ok = false;

    if(s->gs_grad_acc == NULL || s->gs_grad_mean3d == NULL) {
        return true;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_grad_mean3d, &grad_mean_info), "gsx_tensor_get_info(grad_mean3d)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_grad_acc, &grad_acc_info), "gsx_tensor_get_info(grad_acc)")) {
        return false;
    }
    if(grad_mean_info.data_type != GSX_DATA_TYPE_F32 || grad_acc_info.data_type != GSX_DATA_TYPE_F32) {
        fprintf(stderr, "error: grad_mean3d/grad_acc must be float32\n");
        return false;
    }
    count = (gsx_size_t)(grad_acc_info.size_bytes / sizeof(float));
    if(count == 0) {
        return true;
    }
    if((gsx_size_t)(grad_mean_info.size_bytes / sizeof(float)) != count * 3u) {
        fprintf(stderr, "error: grad_mean3d and grad_acc shapes are inconsistent\n");
        return false;
    }

    grad_mean = (float *)malloc((size_t)grad_mean_info.size_bytes);
    grad_acc = (float *)malloc((size_t)grad_acc_info.size_bytes);
    if(grad_mean == NULL || grad_acc == NULL) {
        fprintf(stderr, "error: out of memory while updating grad_acc\n");
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_download(s->gs_grad_mean3d, grad_mean, grad_mean_info.size_bytes), "gsx_tensor_download(grad_mean3d)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_download(s->gs_grad_acc, grad_acc, grad_acc_info.size_bytes), "gsx_tensor_download(grad_acc)")) {
        goto cleanup;
    }

    for(i = 0; i < count; ++i) {
        const float gx = grad_mean[i * 3u + 0u];
        const float gy = grad_mean[i * 3u + 1u];
        const float gz = grad_mean[i * 3u + 2u];
        grad_acc[i] += sqrtf(gx * gx + gy * gy + gz * gz);
    }

    if(!gsx_check(gsx_tensor_upload(s->gs_grad_acc, grad_acc, grad_acc_info.size_bytes), "gsx_tensor_upload(grad_acc)")) {
        goto cleanup;
    }
    ok = true;

cleanup:
    free(grad_mean);
    free(grad_acc);
    return ok;
}

static void print_timing_stats(const app_timing *t)
{
    if(t->step_count == 0) {
        printf("timing: no steps recorded\n");
        return;
    }
    printf("timing (avg per step, us): render_fwd=%.2f loss_fwd=%.2f loss_bwd=%.2f render_bwd=%.2f optim=%.2f adc=%.2f\n",
        t->total_render_forward_us / (double)t->step_count,
        t->total_loss_forward_us / (double)t->step_count,
        t->total_loss_backward_us / (double)t->step_count,
        t->total_render_backward_us / (double)t->step_count,
        t->total_optim_step_us / (double)t->step_count,
        t->total_adc_step_us / (double)t->step_count);
    printf("timing (total ms): render_fwd=%.2f loss_fwd=%.2f loss_bwd=%.2f render_bwd=%.2f optim=%.2f adc=%.2f\n",
        t->total_render_forward_us / 1000.0,
        t->total_loss_forward_us / 1000.0,
        t->total_loss_backward_us / 1000.0,
        t->total_render_backward_us / 1000.0,
        t->total_optim_step_us / 1000.0,
        t->total_adc_step_us / 1000.0);
}

static float compute_mse(const float *lhs, const float *rhs, gsx_size_t count)
{
    gsx_size_t i = 0;
    double sum = 0.0;

    for(i = 0; i < count; ++i) {
        const double d = (double)lhs[i] - (double)rhs[i];
        sum += d * d;
    }
    return (float)(sum / (double)count);
}

static void resize_chw3_nearest(
    const float *src,
    gsx_index_t src_w,
    gsx_index_t src_h,
    float *dst,
    gsx_index_t dst_w,
    gsx_index_t dst_h)
{
    const gsx_size_t src_plane = (gsx_size_t)src_w * (gsx_size_t)src_h;
    const gsx_size_t dst_plane = (gsx_size_t)dst_w * (gsx_size_t)dst_h;
    gsx_index_t y = 0;

    for(y = 0; y < dst_h; ++y) {
        gsx_index_t x = 0;
        const gsx_index_t src_y = (gsx_index_t)(((int64_t)y * (int64_t)src_h) / (int64_t)dst_h);
        for(x = 0; x < dst_w; ++x) {
            const gsx_index_t src_x = (gsx_index_t)(((int64_t)x * (int64_t)src_w) / (int64_t)dst_w);
            const gsx_size_t src_idx = (gsx_size_t)src_y * (gsx_size_t)src_w + (gsx_size_t)src_x;
            const gsx_size_t dst_idx = (gsx_size_t)y * (gsx_size_t)dst_w + (gsx_size_t)x;

            dst[0 * dst_plane + dst_idx] = src[0 * src_plane + src_idx];
            dst[1 * dst_plane + dst_idx] = src[1 * src_plane + src_idx];
            dst[2 * dst_plane + dst_idx] = src[2 * src_plane + src_idx];
        }
    }
}

static bool upload_gs_initial_values(const app_options *opt, app_state *s)
{
    gsx_tensor_info mean_info = {0};
    gsx_tensor_info rot_info = {0};
    gsx_tensor_info logscale_info = {0};
    gsx_tensor_info opacity_info = {0};
    gsx_tensor_info sh0_info = {0};
    gsx_tensor_info sh1_info = {0};
    gsx_tensor_info sh2_info = {0};
    gsx_tensor_info sh3_info = {0};

    float *mean = NULL;
    float *rotation = NULL;
    float *logscale = NULL;
    float *opacity = NULL;
    float *sh0 = NULL;
    float *sh1 = NULL;
    float *sh2 = NULL;
    float *sh3 = NULL;

    gsx_size_t i = 0;
    uint32_t rng = opt->seed;
    const float safe_opacity = clamp_f32(opt->init_opacity, 1e-4f, 1.0f - 1e-4f);
    const float opacity_logit = logf(safe_opacity / (1.0f - safe_opacity));

    if(!gsx_check(gsx_tensor_get_info(s->gs_mean3d, &mean_info), "gsx_tensor_get_info(mean3d)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_rotation, &rot_info), "gsx_tensor_get_info(rotation)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_logscale, &logscale_info), "gsx_tensor_get_info(logscale)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_opacity, &opacity_info), "gsx_tensor_get_info(opacity)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_sh0, &sh0_info), "gsx_tensor_get_info(sh0)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_sh1, &sh1_info), "gsx_tensor_get_info(sh1)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_sh2, &sh2_info), "gsx_tensor_get_info(sh2)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_get_info(s->gs_sh3, &sh3_info), "gsx_tensor_get_info(sh3)")) {
        return false;
    }

    mean = (float *)malloc((size_t)mean_info.size_bytes);
    rotation = (float *)malloc((size_t)rot_info.size_bytes);
    logscale = (float *)malloc((size_t)logscale_info.size_bytes);
    opacity = (float *)malloc((size_t)opacity_info.size_bytes);
    sh0 = (float *)malloc((size_t)sh0_info.size_bytes);
    sh1 = (float *)malloc((size_t)sh1_info.size_bytes);
    sh2 = (float *)malloc((size_t)sh2_info.size_bytes);
    sh3 = (float *)malloc((size_t)sh3_info.size_bytes);

    if(mean == NULL || rotation == NULL || logscale == NULL || opacity == NULL || sh0 == NULL || sh1 == NULL || sh2 == NULL || sh3 == NULL) {
        fprintf(stderr, "error: out of memory during GS initialization\n");
        free(mean);
        free(rotation);
        free(logscale);
        free(opacity);
        free(sh0);
        free(sh1);
        free(sh2);
        free(sh3);
        return false;
    }

    memset(sh1, 0, (size_t)sh1_info.size_bytes);
    memset(sh2, 0, (size_t)sh2_info.size_bytes);
    memset(sh3, 0, (size_t)sh3_info.size_bytes);

    for(i = 0; i < opt->gaussian_count; ++i) {
        const float jitter_x = opt->init_position_jitter * random_normal(&rng);
        const float jitter_y = opt->init_position_jitter * random_normal(&rng);
        const float x = clamp_f32(random_uniform01(&rng) + jitter_x, 0.0f, 1.0f);
        const float y = clamp_f32(random_uniform01(&rng) + jitter_y, 0.0f, 1.0f);
        const gsx_index_t ix = (gsx_index_t)clamp_f32(x * (float)(s->width - 1), 0.0f, (float)(s->width - 1));
        const gsx_index_t iy = (gsx_index_t)clamp_f32(y * (float)(s->height - 1), 0.0f, (float)(s->height - 1));
        const gsx_size_t image_idx = (gsx_size_t)iy * (gsx_size_t)s->width + (gsx_size_t)ix;

        float qx = opt->init_rotation_jitter * random_normal(&rng);
        float qy = opt->init_rotation_jitter * random_normal(&rng);
        float qz = opt->init_rotation_jitter * random_normal(&rng);
        float qw = 1.0f;
        const float qnorm = sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);

        if(qnorm > 0.0f) {
            qx /= qnorm;
            qy /= qnorm;
            qz /= qnorm;
            qw /= qnorm;
        }

        mean[i * 3 + 0] = x;
        mean[i * 3 + 1] = y;
        mean[i * 3 + 2] = 0.0f;

        rotation[i * 4 + 0] = qx;
        rotation[i * 4 + 1] = qy;
        rotation[i * 4 + 2] = qz;
        rotation[i * 4 + 3] = qw;

        logscale[i * 3 + 0] = logf(clamp_f32(opt->init_scale_xy, 1e-6f, 1.0f));
        logscale[i * 3 + 1] = logf(clamp_f32(opt->init_scale_xy, 1e-6f, 1.0f));
        logscale[i * 3 + 2] = logf(clamp_f32(opt->init_scale_z, 1e-6f, 1.0f));

        opacity[i] = opacity_logit;

        sh0[i * 3 + 0] = clamp_f32(s->target_host[0 * ((gsx_size_t)s->width * (gsx_size_t)s->height) + image_idx]
                                       + opt->init_sh0_jitter * random_normal(&rng),
            0.0f,
            1.0f);
        sh0[i * 3 + 1] = clamp_f32(s->target_host[1 * ((gsx_size_t)s->width * (gsx_size_t)s->height) + image_idx]
                                       + opt->init_sh0_jitter * random_normal(&rng),
            0.0f,
            1.0f);
        sh0[i * 3 + 2] = clamp_f32(s->target_host[2 * ((gsx_size_t)s->width * (gsx_size_t)s->height) + image_idx]
                                       + opt->init_sh0_jitter * random_normal(&rng),
            0.0f,
            1.0f);
    }

    if(!gsx_check(gsx_tensor_upload(s->gs_mean3d, mean, mean_info.size_bytes), "gsx_tensor_upload(mean3d)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_rotation, rotation, rot_info.size_bytes), "gsx_tensor_upload(rotation)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_logscale, logscale, logscale_info.size_bytes), "gsx_tensor_upload(logscale)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_opacity, opacity, opacity_info.size_bytes), "gsx_tensor_upload(opacity)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_sh0, sh0, sh0_info.size_bytes), "gsx_tensor_upload(sh0)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_sh1, sh1, sh1_info.size_bytes), "gsx_tensor_upload(sh1)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_sh2, sh2, sh2_info.size_bytes), "gsx_tensor_upload(sh2)")) {
        goto fail;
    }
    if(!gsx_check(gsx_tensor_upload(s->gs_sh3, sh3, sh3_info.size_bytes), "gsx_tensor_upload(sh3)")) {
        goto fail;
    }

    free(mean);
    free(rotation);
    free(logscale);
    free(opacity);
    free(sh0);
    free(sh1);
    free(sh2);
    free(sh3);
    return true;

fail:
    free(mean);
    free(rotation);
    free(logscale);
    free(opacity);
    free(sh0);
    free(sh1);
    free(sh2);
    free(sh3);
    return false;
}

static bool init_training_pipeline(const app_options *opt, app_state *s)
{
    gsx_backend_device_t device = NULL;
    gsx_backend_desc backend_desc = {0};
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_desc arena_desc = {0};

    gsx_renderer_desc renderer_desc = {0};
    gsx_gs_desc gs_desc = {0};
    gsx_loss_desc l1_desc = {0};
    gsx_loss_desc ssim_desc = {0};
    gsx_adc_desc adc_desc = {0};
    gsx_gs_aux_flags adc_aux_flags = GSX_GS_AUX_DEFAULT;
    gsx_dataset_desc adc_dataset_desc = {0};
    gsx_dataloader_desc adc_dataloader_desc = {0};

    gsx_optim_param_group_desc groups[8] = {0};
    gsx_optim_desc optim_desc = {0};
    gsx_tensor_desc runtime_descs[4] = {0};
    gsx_tensor_t runtime_tensors[4] = {NULL};

    gsx_size_t device_count = 0;
    gsx_size_t runtime_required_bytes = 0;
    gsx_size_t gs_required_bytes = 0;

    if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
        return false;
    }
    if(!gsx_check(gsx_count_backend_devices_by_type(opt->backend_type, (gsx_index_t *)&device_count), "gsx_count_backend_devices_by_type")) {
        return false;
    }
    if(device_count == 0) {
        fprintf(stderr, "error: no device available for backend=%s\n", backend_name(opt->backend_type));
        return false;
    }
    if((gsx_size_t)opt->device_index >= device_count) {
        fprintf(stderr, "error: device index out of range for backend=%s\n", backend_name(opt->backend_type));
        return false;
    }

    if(!gsx_check(gsx_get_backend_device_by_type(opt->backend_type, opt->device_index, &device), "gsx_get_backend_device_by_type")) {
        return false;
    }
    backend_desc.device = device;
    if(!gsx_check(gsx_backend_init(&s->backend, &backend_desc), "gsx_backend_init")) {
        return false;
    }

    if(!gsx_check(gsx_backend_find_buffer_type(s->backend, opt->buffer_type_class, &buffer_type), "gsx_backend_find_buffer_type")) {
        return false;
    }

    memset(&s->adc_dataset_object, 0, sizeof(s->adc_dataset_object));
    s->adc_dataset_object.intrinsics = s->intrinsics;
    s->adc_dataset_object.intrinsics.width = 1;
    s->adc_dataset_object.intrinsics.height = 1;
    s->adc_dataset_object.intrinsics.fx = 1.0f;
    s->adc_dataset_object.intrinsics.fy = 1.0f;
    s->adc_dataset_object.intrinsics.cx = 0.5f;
    s->adc_dataset_object.intrinsics.cy = 0.5f;
    s->adc_dataset_object.pose = s->pose;
    s->adc_dataset_object.rgb[0] = 0.0f;
    s->adc_dataset_object.rgb[1] = 0.0f;
    s->adc_dataset_object.rgb[2] = 0.0f;

    adc_dataset_desc.object = &s->adc_dataset_object;
    adc_dataset_desc.get_length = app_adc_dummy_dataset_get_length;
    adc_dataset_desc.get_sample = app_adc_dummy_dataset_get_sample;
    adc_dataset_desc.release_sample = app_adc_dummy_dataset_release_sample;
    if(!gsx_check(gsx_dataset_init(&s->adc_dataset, &adc_dataset_desc), "gsx_dataset_init(adc dummy dataset)")) {
        return false;
    }

    adc_dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    adc_dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    adc_dataloader_desc.output_width = s->width;
    adc_dataloader_desc.output_height = s->height;
    if(!gsx_check(gsx_dataloader_init(&s->adc_dataloader, s->backend, s->adc_dataset, &adc_dataloader_desc), "gsx_dataloader_init(adc dummy dataloader)")) {
        return false;
    }

    adc_desc.algorithm = GSX_ADC_ALGORITHM_DEFAULT;
    adc_desc.pruning_opacity_threshold = 0.01f;
    adc_desc.opacity_clamp_value = 1.0f;
    adc_desc.max_world_scale = 0.0f;
    adc_desc.max_screen_scale = 0.0f;
    adc_desc.duplicate_grad_threshold = 0.0005f;
    adc_desc.duplicate_scale_threshold = 0.05f;
    adc_desc.refine_every = 20;
    adc_desc.start_refine = 20;
    adc_desc.end_refine = opt->train_steps;
    adc_desc.max_num_gaussians = (gsx_index_t)(opt->gaussian_count * 2u);
    adc_desc.reset_every = 100;
    adc_desc.seed = opt->seed;
    adc_desc.prune_degenerate_rotation = true;
    if(!gsx_check(gsx_adc_init(&s->adc, s->backend, &adc_desc), "gsx_adc_init")) {
        return false;
    }
    if(!gsx_check(gsx_adc_get_gs_aux_fields(s->adc, &adc_aux_flags), "gsx_adc_get_gs_aux_fields")) {
        return false;
    }

    if(!compute_runtime_arena_required_bytes(buffer_type, s->width, s->height, &runtime_required_bytes)) {
        return false;
    }
    if(!compute_initial_gs_required_bytes(buffer_type, opt->gaussian_count, adc_aux_flags, &gs_required_bytes)) {
        return false;
    }

    printf("arena dry-run runtime required=%llu bytes (%.2f MiB)\n",
        (unsigned long long)runtime_required_bytes,
        (double)runtime_required_bytes / (1024.0 * 1024.0));
    printf("arena dry-run gs required=%llu bytes (%.2f MiB)\n",
        (unsigned long long)gs_required_bytes,
        (double)gs_required_bytes / (1024.0 * 1024.0));

    arena_desc.initial_capacity_bytes = runtime_required_bytes;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    if(!gsx_check(gsx_arena_init(&s->arena, buffer_type, &arena_desc), "gsx_arena_init")) {
        return false;
    }

    gs_desc.buffer_type = buffer_type;
    gs_desc.count = opt->gaussian_count;
    gs_desc.aux_flags = adc_aux_flags;
    if(!gsx_check(gsx_gs_init(&s->gs, &gs_desc), "gsx_gs_init")) {
        return false;
    }
    if(!fetch_gs_fields(s)) {
        return false;
    }

    renderer_desc.width = s->width;
    renderer_desc.height = s->height;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    renderer_desc.feature_flags = 0;
    renderer_desc.enable_alpha_output = false;
    renderer_desc.enable_invdepth_output = false;
    if(!gsx_check(gsx_renderer_init(&s->renderer, s->backend, &renderer_desc), "gsx_renderer_init")) {
        return false;
    }
    if(!gsx_check(gsx_render_context_init(&s->render_context, s->renderer), "gsx_render_context_init")) {
        return false;
    }

    runtime_descs[0].rank = 3;
    runtime_descs[0].shape[0] = 3;
    runtime_descs[0].shape[1] = s->height;
    runtime_descs[0].shape[2] = s->width;
    runtime_descs[0].data_type = GSX_DATA_TYPE_F32;
    runtime_descs[0].storage_format = GSX_STORAGE_FORMAT_CHW;
    runtime_descs[1] = runtime_descs[0];
    runtime_descs[2] = runtime_descs[0];
    runtime_descs[3] = runtime_descs[0];
    if(!gsx_check(gsx_tensor_init_many(runtime_tensors, s->arena, runtime_descs, 4), "gsx_tensor_init_many(runtime tensors)")) {
        return false;
    }
    s->target_rgb = runtime_tensors[0];
    s->out_rgb = runtime_tensors[1];
    s->loss_map = runtime_tensors[2];
    s->grad_out_rgb = runtime_tensors[3];

    if(!gsx_check(gsx_tensor_upload(s->target_rgb, s->target_host, s->image_element_count * sizeof(float)), "gsx_tensor_upload(target_rgb)")) {
        return false;
    }

    l1_desc.algorithm = GSX_LOSS_ALGORITHM_L1;
    l1_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    if(!gsx_check(gsx_loss_init(&s->l1_loss, s->backend, &l1_desc), "gsx_loss_init(l1)")) {
        return false;
    }
    if(!gsx_check(gsx_loss_context_init(&s->l1_context, s->l1_loss), "gsx_loss_context_init(l1)")) {
        return false;
    }

    ssim_desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    ssim_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    if(!gsx_check(gsx_loss_init(&s->ssim_loss, s->backend, &ssim_desc), "gsx_loss_init(ssim)")) {
        return false;
    }
    if(!gsx_check(gsx_loss_context_init(&s->ssim_context, s->ssim_loss), "gsx_loss_context_init(ssim)")) {
        return false;
    }

    groups[0].role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    groups[0].parameter = s->gs_mean3d;
    groups[0].gradient = s->gs_grad_mean3d;
    groups[0].learning_rate = opt->learning_rate_mean3d;

    groups[1].role = GSX_OPTIM_PARAM_ROLE_LOGSCALE;
    groups[1].parameter = s->gs_logscale;
    groups[1].gradient = s->gs_grad_logscale;
    groups[1].learning_rate = opt->learning_rate_logscale;

    groups[2].role = GSX_OPTIM_PARAM_ROLE_ROTATION;
    groups[2].parameter = s->gs_rotation;
    groups[2].gradient = s->gs_grad_rotation;
    groups[2].learning_rate = opt->learning_rate_rotation;

    groups[3].role = GSX_OPTIM_PARAM_ROLE_OPACITY;
    groups[3].parameter = s->gs_opacity;
    groups[3].gradient = s->gs_grad_opacity;
    groups[3].learning_rate = opt->learning_rate_opacity;

    groups[4].role = GSX_OPTIM_PARAM_ROLE_SH0;
    groups[4].parameter = s->gs_sh0;
    groups[4].gradient = s->gs_grad_sh0;
    groups[4].learning_rate = opt->learning_rate_sh0;

    groups[5].role = GSX_OPTIM_PARAM_ROLE_SH1;
    groups[5].parameter = s->gs_sh1;
    groups[5].gradient = s->gs_grad_sh1;
    groups[5].learning_rate = opt->learning_rate_sh1;

    groups[6].role = GSX_OPTIM_PARAM_ROLE_SH2;
    groups[6].parameter = s->gs_sh2;
    groups[6].gradient = s->gs_grad_sh2;
    groups[6].learning_rate = opt->learning_rate_sh2;

    groups[7].role = GSX_OPTIM_PARAM_ROLE_SH3;
    groups[7].parameter = s->gs_sh3;
    groups[7].gradient = s->gs_grad_sh3;
    groups[7].learning_rate = opt->learning_rate_sh3;

    {
        int i;
        for(i = 0; i < 8; ++i) {
            groups[i].beta1 = opt->beta1;
            groups[i].beta2 = opt->beta2;
            groups[i].epsilon = opt->epsilon;
            groups[i].weight_decay = opt->weight_decay;
            groups[i].max_grad = opt->max_grad;
        }
    }

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 8;
    if(!gsx_check(gsx_optim_init(&s->optim, s->backend, &optim_desc), "gsx_optim_init")) {
        return false;
    }

    if(!upload_gs_initial_values(opt, s)) {
        return false;
    }
    return gsx_check(gsx_gs_zero_aux_tensors(s->gs, adc_aux_flags), "gsx_gs_zero_aux_tensors");
}

static bool run_one_step(const app_options *opt, app_state *s, gsx_size_t global_step, gsx_adc_result *out_adc_result)
{
    gsx_render_forward_request forward = {0};
    gsx_render_backward_request backward = {0};
    gsx_loss_forward_request loss_forward = {0};
    gsx_loss_backward_request loss_backward = {0};
    gsx_optim_step_request step_request = {0};
    gsx_adc_request adc_request = {0};
    double t0 = 0.0, t1 = 0.0;

    if(out_adc_result != NULL) {
        memset(out_adc_result, 0, sizeof(*out_adc_result));
    }

    forward.intrinsics = &s->intrinsics;
    forward.pose = &s->pose;
    forward.near_plane = opt->near_plane;
    forward.far_plane = opt->far_plane;
    forward.background_color = opt->background_color;
    forward.precision = opt->render_precision;
    forward.sh_degree = opt->sh_degree;
    forward.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    forward.borrow_train_state = true;
    forward.gs_mean3d = s->gs_mean3d;
    forward.gs_rotation = s->gs_rotation;
    forward.gs_logscale = s->gs_logscale;
    forward.gs_sh0 = s->gs_sh0;
    forward.gs_sh1 = s->gs_sh1;
    forward.gs_sh2 = s->gs_sh2;
    forward.gs_sh3 = s->gs_sh3;
    forward.gs_opacity = s->gs_opacity;
    forward.out_rgb = s->out_rgb;

    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t0 = get_time_us();
    if(!gsx_check(gsx_renderer_render(s->renderer, s->render_context, &forward), "gsx_renderer_render(train)")) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t1 = get_time_us();
    s->timing.total_render_forward_us += (t1 - t0);

    if(!gsx_check(gsx_tensor_set_zero(s->loss_map), "gsx_tensor_set_zero(loss_map)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_set_zero(s->grad_out_rgb), "gsx_tensor_set_zero(grad_out_rgb)")) {
        return false;
    }

    loss_forward.prediction = s->out_rgb;
    loss_forward.target = s->target_rgb;
    loss_forward.loss_map_accumulator = s->loss_map;
    loss_forward.train = true;

    loss_backward.grad_prediction_accumulator = s->grad_out_rgb;

    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t0 = get_time_us();
    loss_forward.scale = opt->l1_scale;
    if(!gsx_check(gsx_loss_forward(s->l1_loss, s->l1_context, &loss_forward), "gsx_loss_forward(l1)")) {
        return false;
    }
    loss_forward.scale = opt->ssim_scale;
    if(!gsx_check(gsx_loss_forward(s->ssim_loss, s->ssim_context, &loss_forward), "gsx_loss_forward(ssim)")) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t1 = get_time_us();
    s->timing.total_loss_forward_us += (t1 - t0);

    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t0 = get_time_us();
    loss_backward.scale = opt->l1_scale;
    if(!gsx_check(gsx_loss_backward(s->l1_loss, s->l1_context, &loss_backward), "gsx_loss_backward(l1)")) {
        return false;
    }
    loss_backward.scale = opt->ssim_scale;
    if(!gsx_check(gsx_loss_backward(s->ssim_loss, s->ssim_context, &loss_backward), "gsx_loss_backward(ssim)")) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t1 = get_time_us();
    s->timing.total_loss_backward_us += (t1 - t0);

    if(!gsx_check(gsx_gs_zero_gradients(s->gs), "gsx_gs_zero_gradients")) {
        return false;
    }

    backward.grad_rgb = s->grad_out_rgb;
    backward.grad_gs_mean3d = s->gs_grad_mean3d;
    backward.grad_gs_rotation = s->gs_grad_rotation;
    backward.grad_gs_logscale = s->gs_grad_logscale;
    backward.grad_gs_sh0 = s->gs_grad_sh0;
    backward.grad_gs_sh1 = s->gs_grad_sh1;
    backward.grad_gs_sh2 = s->gs_grad_sh2;
    backward.grad_gs_sh3 = s->gs_grad_sh3;
    backward.grad_gs_opacity = s->gs_grad_opacity;

    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t0 = get_time_us();
    if(!gsx_check(gsx_renderer_backward(s->renderer, s->render_context, &backward), "gsx_renderer_backward")) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t1 = get_time_us();
    s->timing.total_render_backward_us += (t1 - t0);

    step_request.force_all = true;
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t0 = get_time_us();
    if(!gsx_check(gsx_optim_step(s->optim, &step_request), "gsx_optim_step")) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    t1 = get_time_us();
    s->timing.total_optim_step_us += (t1 - t0);

    if(s->adc != NULL) {
        gsx_adc_result adc_result = {0};

        if(!accumulate_grad_acc(s)) {
            return false;
        }

        adc_request.gs = s->gs;
        adc_request.optim = s->optim;
        adc_request.dataloader = s->adc_dataloader;
        adc_request.renderer = s->renderer;
        adc_request.global_step = global_step;
        adc_request.scene_scale = 1.0f;
        if(!sync_backend_if_needed(s->backend)) {
            return false;
        }
        t0 = get_time_us();
        if(!gsx_check(gsx_adc_step(s->adc, &adc_request, &adc_result), "gsx_adc_step")) {
            return false;
        }
        if(!sync_backend_if_needed(s->backend)) {
            return false;
        }
        t1 = get_time_us();
        s->timing.total_adc_step_us += (t1 - t0);
        if(adc_result.mutated) {
            if(!fetch_gs_fields(s)) {
                return false;
            }
        }
        if(out_adc_result != NULL) {
            *out_adc_result = adc_result;
        }
    }

    s->timing.step_count++;
    return true;
}

static bool run_final_inference(const app_options *opt, app_state *s)
{
    gsx_render_forward_request forward = {0};

    forward.intrinsics = &s->intrinsics;
    forward.pose = &s->pose;
    forward.near_plane = opt->near_plane;
    forward.far_plane = opt->far_plane;
    forward.background_color = opt->background_color;
    forward.precision = opt->render_precision;
    forward.sh_degree = opt->sh_degree;
    forward.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    forward.borrow_train_state = false;
    forward.gs_mean3d = s->gs_mean3d;
    forward.gs_rotation = s->gs_rotation;
    forward.gs_logscale = s->gs_logscale;
    forward.gs_sh0 = s->gs_sh0;
    forward.gs_sh1 = s->gs_sh1;
    forward.gs_sh2 = s->gs_sh2;
    forward.gs_sh3 = s->gs_sh3;
    forward.gs_opacity = s->gs_opacity;
    forward.out_rgb = s->out_rgb;

    return gsx_check(gsx_renderer_render(s->renderer, s->render_context, &forward), "gsx_renderer_render(inference)");
}

static bool save_output(const app_state *s, const char *path)
{
    const char *ext = strrchr(path, '.');

    if(ext != NULL && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPG") == 0 || strcmp(ext, ".JPEG") == 0)) {
        return gsx_check(
            gsx_image_write_jpg(path, s->render_host, s->width, s->height, 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 95),
            "gsx_image_write_jpg");
    }
    return gsx_check(
        gsx_image_write_png(path, s->render_host, s->width, s->height, 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW),
        "gsx_image_write_png");
}

static void cleanup_state(app_state *s)
{
    if(s->adc != NULL) {
        (void)gsx_adc_free(s->adc);
    }
    if(s->optim != NULL) {
        (void)gsx_optim_free(s->optim);
    }
    if(s->ssim_context != NULL) {
        (void)gsx_loss_context_free(s->ssim_context);
    }
    if(s->ssim_loss != NULL) {
        (void)gsx_loss_free(s->ssim_loss);
    }
    if(s->l1_context != NULL) {
        (void)gsx_loss_context_free(s->l1_context);
    }
    if(s->l1_loss != NULL) {
        (void)gsx_loss_free(s->l1_loss);
    }
    {
        gsx_tensor_t runtime_tensors[4] = {
            s->target_rgb,
            s->out_rgb,
            s->loss_map,
            s->grad_out_rgb,
        };
        (void)gsx_tensor_free_many(runtime_tensors, 4);
        s->target_rgb = runtime_tensors[0];
        s->out_rgb = runtime_tensors[1];
        s->loss_map = runtime_tensors[2];
        s->grad_out_rgb = runtime_tensors[3];
    }
    if(s->render_context != NULL) {
        (void)gsx_render_context_free(s->render_context);
    }
    if(s->renderer != NULL) {
        (void)gsx_renderer_free(s->renderer);
    }
    if(s->gs != NULL) {
        (void)gsx_gs_free(s->gs);
    }
    if(s->adc_dataloader != NULL) {
        (void)gsx_dataloader_free(s->adc_dataloader);
    }
    if(s->adc_dataset != NULL) {
        (void)gsx_dataset_free(s->adc_dataset);
    }
    if(s->arena != NULL) {
        (void)gsx_arena_free(s->arena);
    }
    if(s->backend != NULL) {
        (void)gsx_backend_free(s->backend);
    }

    free(s->target_host);
    free(s->render_host);
    memset(s, 0, sizeof(*s));
}

int main(int argc, char **argv)
{
    app_options opt;
    app_state state;
    gsx_image image = {0};
    gsx_index_t step = 0;
    int exit_code = EXIT_FAILURE;

    set_default_options(&opt);
    memset(&state, 0, sizeof(state));

    if(!parse_options(argc, argv, &opt)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if(!gsx_check(gsx_image_load(&image, opt.input_path, 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW), "gsx_image_load")) {
        goto cleanup;
    }

    state.width = opt.train_width > 0 ? opt.train_width : image.width;
    state.height = opt.train_height > 0 ? opt.train_height : image.height;
    state.image_element_count = (gsx_size_t)3 * (gsx_size_t)state.width * (gsx_size_t)state.height;
    state.target_host = (float *)malloc((size_t)(state.image_element_count * sizeof(float)));
    state.render_host = (float *)malloc((size_t)(state.image_element_count * sizeof(float)));
    if(state.target_host == NULL || state.render_host == NULL) {
        fprintf(stderr, "error: host memory allocation failed\n");
        goto cleanup;
    }
    if(state.width == image.width && state.height == image.height) {
        memcpy(state.target_host, image.pixels, (size_t)(state.image_element_count * sizeof(float)));
    } else {
        resize_chw3_nearest((const float *)image.pixels, image.width, image.height, state.target_host, state.width, state.height);
    }

    state.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    state.intrinsics.width = state.width;
    state.intrinsics.height = state.height;
    state.intrinsics.fx = opt.fx > 0.0f ? opt.fx : (gsx_float_t)state.width;
    state.intrinsics.fy = opt.fy > 0.0f ? opt.fy : (gsx_float_t)state.height;
    state.intrinsics.cx = opt.cx > 0.0f ? opt.cx : ((gsx_float_t)state.width - 1.0f) * 0.5f;
    state.intrinsics.cy = opt.cy > 0.0f ? opt.cy : ((gsx_float_t)state.height - 1.0f) * 0.5f;

    state.pose.rot.x = 0.0f;
    state.pose.rot.y = 1.0f;
    state.pose.rot.z = 0.0f;
    state.pose.rot.w = 0.0f;
    state.pose.transl.x = 0.5f;
    state.pose.transl.y = -0.5f;
    state.pose.transl.z = opt.camera_z;
    state.pose.camera_id = 0;
    state.pose.frame_id = 0;

    printf("image=%s train_size=%lldx%lld (src=%lldx%lld) gaussians=%llu backend=%s device=%lld\n",
        opt.input_path,
        (long long)state.width,
        (long long)state.height,
        (long long)image.width,
        (long long)image.height,
        (unsigned long long)opt.gaussian_count,
        backend_name(opt.backend_type),
        (long long)opt.device_index);
    printf("camera: facing -z, centered at world (0.5, 0.5, 0.0), camera_z=%.4f\n", opt.camera_z);

    if(!init_training_pipeline(&opt, &state)) {
        goto cleanup;
    }

    for(step = 1; step <= opt.train_steps; ++step) {
        gsx_gs_finite_check_result finite_result = {0};
        gsx_adc_result adc_result = {0};

        if(!run_one_step(&opt, &state, (gsx_size_t)step, &adc_result)) {
            goto cleanup;
        }

        if(step == 1 || step == opt.train_steps || (step % opt.log_interval) == 0) {
            float mse = 0.0f;

            if(!sync_backend_if_needed(state.backend)) {
                goto cleanup;
            }
            if(!gsx_check(
                   gsx_tensor_download(state.out_rgb, state.render_host, state.image_element_count * sizeof(float)),
                   "gsx_tensor_download(out_rgb)")) {
                goto cleanup;
            }

            mse = compute_mse(state.render_host, state.target_host, state.image_element_count);
            if(!gsx_check(gsx_gs_check_finite(state.gs, &finite_result), "gsx_gs_check_finite")) {
                goto cleanup;
            }
            printf("step=%lld mse=%.8f finite=%s adc(before=%llu after=%llu prune=%llu dup=%llu split=%llu)\n",
                (long long)step,
                mse,
                finite_result.is_finite ? "yes" : "no",
                (unsigned long long)adc_result.gaussians_before,
                (unsigned long long)adc_result.gaussians_after,
                (unsigned long long)adc_result.pruned_count,
                (unsigned long long)adc_result.duplicated_count,
                (unsigned long long)adc_result.grown_count);
            print_timing_stats(&state.timing);
            if(!finite_result.is_finite) {
                fprintf(stderr, "error: non-finite GS parameters detected at step %lld\n", (long long)step);
                goto cleanup;
            }
        }
    }

    printf("=== final timing summary ===\n");
    print_timing_stats(&state.timing);

    if(!run_final_inference(&opt, &state)) {
        goto cleanup;
    }
    if(!sync_backend_if_needed(state.backend)) {
        goto cleanup;
    }
    if(!gsx_check(
           gsx_tensor_download(state.out_rgb, state.render_host, state.image_element_count * sizeof(float)),
           "gsx_tensor_download(final out_rgb)")) {
        goto cleanup;
    }

    printf("final mse=%.8f\n", compute_mse(state.render_host, state.target_host, state.image_element_count));
    if(!save_output(&state, opt.output_path)) {
        goto cleanup;
    }
    printf("saved output to %s\n", opt.output_path);

    exit_code = EXIT_SUCCESS;

cleanup:
    if(image.pixels != NULL) {
        (void)gsx_image_free(&image);
    }
    cleanup_state(&state);
    return exit_code;
}
