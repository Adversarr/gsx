#include <gsx/extra/gsx-stbi.h>
#include <gsx/extra/gsx-flann.h>
#include <gsx/gsx.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct app_image_dataset {
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    const float *rgb;
} app_image_dataset;

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
    gsx_backend_buffer_type_t metric_buffer_type;
    gsx_dataset_t train_dataset;
    gsx_dataloader_t train_dataloader;
    app_image_dataset train_dataset_object;

    gsx_adc_t adc;
    gsx_gs_t gs;
    gsx_renderer_t renderer;
    gsx_loss_t l1_loss;
    gsx_loss_context_t l1_context;
    gsx_loss_t ssim_loss;
    gsx_loss_context_t ssim_context;
    gsx_optim_t optim;
    gsx_session_t session;
    gsx_arena_t metric_arena;
    gsx_tensor_t mse_tensor;

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
    float *target_hwc;
    float *target_chw;
    float *render_host;

    app_timing timing;
} app_state;

static gsx_error app_image_dataset_get_length(void *object, gsx_size_t *out_length)
{
    (void)object;
    if(out_length == NULL) {
        return (gsx_error){ GSX_ERROR_INVALID_ARGUMENT, "out_length must be non-null" };
    }
    *out_length = 1;
    return (gsx_error){ GSX_ERROR_SUCCESS, NULL };
}

static gsx_error app_image_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    app_image_dataset *dataset = (app_image_dataset *)object;

    if(dataset == NULL || out_sample == NULL) {
        return (gsx_error){ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_sample must be non-null" };
    }
    if(sample_index != 0) {
        return (gsx_error){ GSX_ERROR_OUT_OF_RANGE, "sample_index must be 0 for image-fit dataset" };
    }

    memset(out_sample, 0, sizeof(*out_sample));
    out_sample->intrinsics = dataset->intrinsics;
    out_sample->pose = dataset->pose;
    out_sample->rgb.data = dataset->rgb;
    out_sample->rgb.data_type = GSX_DATA_TYPE_F32;
    out_sample->rgb.width = dataset->intrinsics.width;
    out_sample->rgb.height = dataset->intrinsics.height;
    out_sample->rgb.channel_count = 3;
    out_sample->rgb.row_stride_bytes = (gsx_size_t)dataset->intrinsics.width * 3u * sizeof(float);
    return (gsx_error){ GSX_ERROR_SUCCESS, NULL };
}

static void app_image_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
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

    opt->learning_rate_mean3d = 0.005f;
    opt->learning_rate_logscale = 0.001f;
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

static gsx_data_type app_render_precision_to_data_type(gsx_render_precision precision)
{
    if(precision == GSX_RENDER_PRECISION_FLOAT16) {
        return GSX_DATA_TYPE_F16;
    }
    return GSX_DATA_TYPE_F32;
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

static void accumulate_step_timing(app_timing *timing, const gsx_session_step_report *report)
{
    if(timing == NULL || report == NULL || !report->has_timings) {
        return;
    }

    timing->total_render_forward_us += report->timings.render_forward_us;
    timing->total_loss_forward_us += report->timings.loss_forward_us;
    timing->total_loss_backward_us += report->timings.loss_backward_us;
    timing->total_render_backward_us += report->timings.render_backward_us;
    timing->total_optim_step_us += report->timings.optim_step_us;
    timing->total_adc_step_us += report->timings.adc_step_us;
    timing->step_count += 1;
}

static void convert_chw3_to_hwc(const float *src, gsx_index_t w, gsx_index_t h, float *dst)
{
    const gsx_size_t plane = (gsx_size_t)w * (gsx_size_t)h;
    gsx_index_t y = 0;

    for(y = 0; y < h; ++y) {
        gsx_index_t x = 0;
        for(x = 0; x < w; ++x) {
            const gsx_size_t chw_idx = (gsx_size_t)y * (gsx_size_t)w + (gsx_size_t)x;
            const gsx_size_t hwc_idx = ((gsx_size_t)y * (gsx_size_t)w + (gsx_size_t)x) * 3u;
            dst[hwc_idx + 0] = src[chw_idx];
            dst[hwc_idx + 1] = src[plane + chw_idx];
            dst[hwc_idx + 2] = src[2 * plane + chw_idx];
        }
    }
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

        sh0[i * 3 + 0] = clamp_f32(s->target_chw[0 * ((gsx_size_t)s->width * (gsx_size_t)s->height) + image_idx]
                                       + opt->init_sh0_jitter * random_normal(&rng),
            0.0f,
            1.0f);
        sh0[i * 3 + 1] = clamp_f32(s->target_chw[1 * ((gsx_size_t)s->width * (gsx_size_t)s->height) + image_idx]
                                       + opt->init_sh0_jitter * random_normal(&rng),
            0.0f,
            1.0f);
        sh0[i * 3 + 2] = clamp_f32(s->target_chw[2 * ((gsx_size_t)s->width * (gsx_size_t)s->height) + image_idx]
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
    gsx_renderer_desc renderer_desc = {0};
    gsx_gs_desc gs_desc = {0};
    gsx_loss_desc l1_desc = {0};
    gsx_loss_desc ssim_desc = {0};
    gsx_adc_desc adc_desc = {0};
    gsx_gs_aux_flags adc_aux_flags = GSX_GS_AUX_DEFAULT;
    gsx_dataset_desc train_dataset_desc = {0};
    gsx_dataloader_desc train_dataloader_desc = {0};
    gsx_optim_param_group_desc groups[8] = {0};
    gsx_optim_desc optim_desc = {0};
    gsx_loss_item loss_items[2] = {0};
    gsx_session_desc session_desc = {0};
    gsx_size_t device_count = 0;

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
    s->metric_buffer_type = buffer_type;

    memset(&s->train_dataset_object, 0, sizeof(s->train_dataset_object));
    s->train_dataset_object.intrinsics = s->intrinsics;
    s->train_dataset_object.pose = s->pose;
    s->train_dataset_object.rgb = s->target_hwc;

    train_dataset_desc.object = &s->train_dataset_object;
    train_dataset_desc.get_length = app_image_dataset_get_length;
    train_dataset_desc.get_sample = app_image_dataset_get_sample;
    train_dataset_desc.release_sample = app_image_dataset_release_sample;
    if(!gsx_check(gsx_dataset_init(&s->train_dataset, &train_dataset_desc), "gsx_dataset_init(train dataset)")) {
        return false;
    }

    train_dataloader_desc.shuffle_each_epoch = false;
    train_dataloader_desc.enable_async_prefetch = false;
    train_dataloader_desc.prefetch_count = 0;
    train_dataloader_desc.seed = opt->seed;
    train_dataloader_desc.image_data_type = app_render_precision_to_data_type(opt->render_precision);
    train_dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    train_dataloader_desc.output_width = s->width;
    train_dataloader_desc.output_height = s->height;
    if(!gsx_check(
           gsx_dataloader_init(&s->train_dataloader, s->backend, s->train_dataset, &train_dataloader_desc),
           "gsx_dataloader_init(train dataloader)")) {
        return false;
    }

    adc_desc.algorithm = GSX_ADC_ALGORITHM_DEFAULT;
    adc_desc.pruning_opacity_threshold = 0.01f;
    adc_desc.opacity_clamp_value = 0.1f;
    adc_desc.max_world_scale = 0.0f;
    adc_desc.max_screen_scale = 0.0f;
    adc_desc.duplicate_grad_threshold = 0.001f;
    adc_desc.duplicate_scale_threshold = 0.05f;
    adc_desc.refine_every = 20;
    adc_desc.start_refine = 20;
    adc_desc.end_refine = opt->train_steps;
    adc_desc.max_num_gaussians = (gsx_index_t)(opt->gaussian_count * 2u);
    adc_desc.reset_every = 150;
    adc_desc.seed = opt->seed;
    adc_desc.prune_degenerate_rotation = true;
    if(!gsx_check(gsx_adc_init(&s->adc, s->backend, &adc_desc), "gsx_adc_init")) {
        return false;
    }
    if(!gsx_check(gsx_adc_get_gs_aux_fields(s->adc, &adc_aux_flags), "gsx_adc_get_gs_aux_fields")) {
        return false;
    }

    gs_desc.buffer_type = buffer_type;
    gs_desc.count = opt->gaussian_count;
    gs_desc.aux_flags = adc_aux_flags;
    if(!gsx_check(gsx_gs_init(&s->gs, &gs_desc), "gsx_gs_init")) {
        return false;
    }
    {
        gsx_gs_info gs_info = {0};
        gsx_size_t gs_allocated_bytes = 0;
        if(!gsx_check(gsx_gs_get_info(s->gs, &gs_info), "gsx_gs_get_info")) {
            return false;
        }
        if(!gsx_check(gsx_arena_get_required_bytes(gs_info.arena, &gs_allocated_bytes), "gsx_arena_get_required_bytes(gs)")) {
            return false;
        }
        printf("gs allocated=%llu bytes (%.2f MiB)\n",
            (unsigned long long)gs_allocated_bytes,
            (double)gs_allocated_bytes / (1024.0 * 1024.0));
    }
    if(!fetch_gs_fields(s)) {
        return false;
    }

    renderer_desc.width = s->width;
    renderer_desc.height = s->height;
    renderer_desc.output_data_type = app_render_precision_to_data_type(opt->render_precision);
    renderer_desc.feature_flags = 0;
    renderer_desc.enable_alpha_output = false;
    renderer_desc.enable_invdepth_output = false;
    if(!gsx_check(gsx_renderer_init(&s->renderer, s->backend, &renderer_desc), "gsx_renderer_init")) {
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
    if(!gsx_check(gsx_gs_zero_aux_tensors(s->gs, adc_aux_flags), "gsx_gs_zero_aux_tensors")) {
        return false;
    }
    if (!gsx_check(gsx_gs_recompute_scale_rotation_flann(s->gs, 16, 1.0f, 0.001f, 0.1f, 0.01f, 0.2f, false),
        "gsx_gs_recompute_scale_rotation_flann")) {
        return false;
    }

    loss_items[0].loss = s->l1_loss;
    loss_items[0].context = s->l1_context;
    loss_items[0].scale = opt->l1_scale;
    loss_items[1].loss = s->ssim_loss;
    loss_items[1].context = s->ssim_context;
    loss_items[1].scale = opt->ssim_scale;

    session_desc.backend = s->backend;
    session_desc.gs = s->gs;
    session_desc.optim = s->optim;
    session_desc.renderer = s->renderer;
    session_desc.train_dataloader = s->train_dataloader;
    session_desc.adc = s->adc;
    session_desc.scheduler = NULL;
    session_desc.loss_count = 2;
    session_desc.loss_items = loss_items;
    session_desc.render.near_plane = opt->near_plane;
    session_desc.render.far_plane = opt->far_plane;
    session_desc.render.background_color = opt->background_color;
    session_desc.render.precision = opt->render_precision;
    session_desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    session_desc.render.sh_degree = opt->sh_degree;
    session_desc.render.borrow_train_state = true;
    session_desc.optim_step.role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D
        | GSX_OPTIM_PARAM_ROLE_FLAG_LOGSCALE
        | GSX_OPTIM_PARAM_ROLE_FLAG_ROTATION
        | GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH0
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH1
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH2
        | GSX_OPTIM_PARAM_ROLE_FLAG_SH3;
    session_desc.optim_step.force_all = true;
    session_desc.adc_step.enabled = s->adc != NULL;
    session_desc.adc_step.dataloader = NULL;
    session_desc.adc_step.scene_scale = 1.0f;
    session_desc.workspace.buffer_type_class = opt->buffer_type_class;
    session_desc.workspace.arena_desc.initial_capacity_bytes = 0;
    session_desc.workspace.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    session_desc.workspace.auto_plan = true;
    session_desc.reporting.retain_prediction = true;
    session_desc.reporting.retain_target = true;
    session_desc.reporting.retain_loss_map = true;
    session_desc.reporting.retain_grad_prediction = true;
    session_desc.reporting.collect_timings = true;
    session_desc.initial_global_step = 0;
    session_desc.initial_epoch_index = 0;

    return gsx_check(gsx_session_init(&s->session, &session_desc), "gsx_session_init");
}

typedef struct app_mse_plan_context {
    gsx_tensor_t prediction;
    gsx_tensor_t target;
} app_mse_plan_context;

static gsx_error app_plan_mse_required_bytes(gsx_arena_t dry_run_arena, void *user_data)
{
    app_mse_plan_context *context = (app_mse_plan_context *)user_data;
    gsx_tensor_desc mse_desc = {0};
    gsx_tensor_t mse_tensor = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(context == NULL || context->prediction == NULL || context->target == NULL) {
        return (gsx_error){ GSX_ERROR_INVALID_ARGUMENT, "mse plan context must provide prediction and target" };
    }

    mse_desc.rank = 1;
    mse_desc.shape[0] = 1;
    mse_desc.data_type = GSX_DATA_TYPE_F32;
    mse_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    mse_desc.arena = dry_run_arena;
    error = gsx_tensor_init(&mse_tensor, &mse_desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_tensor_mse(dry_run_arena, context->prediction, context->target, mse_tensor, 0);
    if(!gsx_error_is_success(error)) {
        (void)gsx_tensor_free(mse_tensor);
        return error;
    }
    return gsx_tensor_free(mse_tensor);
}

static bool ensure_metric_mse_tensor(app_state *s, gsx_tensor_t prediction, gsx_tensor_t target)
{
    gsx_arena_desc sizing_desc = {0};
    gsx_arena_desc metric_arena_desc = {0};
    gsx_tensor_desc mse_desc = {0};
    app_mse_plan_context plan_context = {0};
    gsx_size_t required_bytes = 0;

    if(s == NULL || prediction == NULL || target == NULL || s->metric_buffer_type == NULL) {
        fprintf(stderr, "error: metric mse setup requires backend buffer type and valid tensors\n");
        return false;
    }
    if(s->metric_arena != NULL && s->mse_tensor != NULL) {
        return true;
    }

    plan_context.prediction = prediction;
    plan_context.target = target;
    if(!gsx_check(
           gsx_arena_plan_required_bytes(s->metric_buffer_type, &sizing_desc, app_plan_mse_required_bytes, &plan_context, &required_bytes),
           "gsx_arena_plan_required_bytes(mse)")) {
        return false;
    }

    metric_arena_desc.initial_capacity_bytes = required_bytes;
    metric_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    if(!gsx_check(gsx_arena_init(&s->metric_arena, s->metric_buffer_type, &metric_arena_desc), "gsx_arena_init(metric)")) {
        return false;
    }

    mse_desc.rank = 1;
    mse_desc.shape[0] = 1;
    mse_desc.data_type = GSX_DATA_TYPE_F32;
    mse_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    mse_desc.arena = s->metric_arena;
    if(!gsx_check(gsx_tensor_init(&s->mse_tensor, &mse_desc), "gsx_tensor_init(mse_tensor)")) {
        return false;
    }
    return true;
}

static bool compute_last_step_mse(app_state *s, float *out_mse)
{
    gsx_session_outputs outputs = {0};

    if(out_mse == NULL) {
        return false;
    }
    *out_mse = 0.0f;

    if(!gsx_check(gsx_session_get_last_outputs(s->session, &outputs), "gsx_session_get_last_outputs")) {
        return false;
    }
    if(outputs.prediction == NULL || outputs.target == NULL) {
        fprintf(stderr, "error: session did not retain prediction/target tensors\n");
        return false;
    }
    if(!ensure_metric_mse_tensor(s, outputs.prediction, outputs.target)) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    if(!gsx_check(gsx_tensor_mse(s->metric_arena, outputs.prediction, outputs.target, s->mse_tensor, 0), "gsx_tensor_mse")) {
        return false;
    }
    if(!sync_backend_if_needed(s->backend)) {
        return false;
    }
    return gsx_check(gsx_tensor_download(s->mse_tensor, out_mse, sizeof(*out_mse)), "gsx_tensor_download(mse_tensor)");
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
    if(s->session != NULL) {
        (void)gsx_session_free(s->session);
    }
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
    if(s->mse_tensor != NULL) {
        (void)gsx_tensor_free(s->mse_tensor);
    }
    if(s->renderer != NULL) {
        (void)gsx_renderer_free(s->renderer);
    }
    if(s->gs != NULL) {
        (void)gsx_gs_free(s->gs);
    }
    if(s->train_dataloader != NULL) {
        (void)gsx_dataloader_free(s->train_dataloader);
    }
    if(s->train_dataset != NULL) {
        (void)gsx_dataset_free(s->train_dataset);
    }
    if(s->metric_arena != NULL) {
        (void)gsx_arena_free(s->metric_arena);
    }
    if(s->backend != NULL) {
        (void)gsx_backend_free(s->backend);
    }

    free(s->target_hwc);
    free(s->target_chw);
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
    state.target_chw = (float *)malloc((size_t)(state.image_element_count * sizeof(float)));
    state.target_hwc = (float *)malloc((size_t)(state.image_element_count * sizeof(float)));
    state.render_host = (float *)malloc((size_t)(state.image_element_count * sizeof(float)));
    if(state.target_chw == NULL || state.target_hwc == NULL || state.render_host == NULL) {
        fprintf(stderr, "error: host memory allocation failed\n");
        goto cleanup;
    }
    if(state.width == image.width && state.height == image.height) {
        memcpy(state.target_chw, image.pixels, (size_t)(state.image_element_count * sizeof(float)));
    } else {
        resize_chw3_nearest((const float *)image.pixels, image.width, image.height, state.target_chw, state.width, state.height);
    }
    convert_chw3_to_hwc(state.target_chw, state.width, state.height, state.target_hwc);

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
        gsx_session_step_report step_report = {0};
        gsx_gs_finite_check_result finite_result = {0};

        if(!gsx_check(gsx_session_step(state.session), "gsx_session_step")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_session_get_last_step_report(state.session, &step_report), "gsx_session_get_last_step_report")) {
            goto cleanup;
        }
        accumulate_step_timing(&state.timing, &step_report);

        if(step == 1 || step == opt.train_steps || (step % opt.log_interval) == 0) {
            float mse = 0.0f;
            const gsx_adc_result *adc_result = NULL;

            if(!compute_last_step_mse(&state, &mse)) {
                goto cleanup;
            }
            if(!gsx_check(gsx_gs_check_finite(state.gs, &finite_result), "gsx_gs_check_finite")) {
                goto cleanup;
            }
            if(step_report.adc_result_available) {
                adc_result = &step_report.adc_result;
            }
            printf("step=%lld mse=%.8f finite=%s adc(before=%llu after=%llu prune=%llu dup=%llu split=%llu)\n",
                (long long)step_report.global_step_after,
                mse,
                finite_result.is_finite ? "yes" : "no",
                (unsigned long long)(adc_result != NULL ? adc_result->gaussians_before : 0u),
                (unsigned long long)(adc_result != NULL ? adc_result->gaussians_after : 0u),
                (unsigned long long)(adc_result != NULL ? adc_result->pruned_count : 0u),
                (unsigned long long)(adc_result != NULL ? adc_result->duplicated_count : 0u),
                (unsigned long long)(adc_result != NULL ? adc_result->grown_count : 0u));
            print_timing_stats(&state.timing);
            if(!finite_result.is_finite) {
                fprintf(stderr, "error: non-finite GS parameters detected at step %lld\n", (long long)step);
                goto cleanup;
            }
        }
    }

    printf("=== final timing summary ===\n");
    print_timing_stats(&state.timing);

    {
        gsx_session_outputs outputs = {0};
        float final_mse = 0.0f;

        if(!compute_last_step_mse(&state, &final_mse)) {
            goto cleanup;
        }
        printf("final mse=%.8f\n", final_mse);

        if(!gsx_check(gsx_session_get_last_outputs(state.session, &outputs), "gsx_session_get_last_outputs(final)")) {
            goto cleanup;
        }
        if(outputs.prediction == NULL) {
            fprintf(stderr, "error: session did not retain the final prediction tensor\n");
            goto cleanup;
        }
        if(!sync_backend_if_needed(state.backend)) {
            goto cleanup;
        }
        if(!gsx_check(
               gsx_tensor_download(outputs.prediction, state.render_host, state.image_element_count * sizeof(float)),
               "gsx_tensor_download(final prediction)")) {
            goto cleanup;
        }
    }
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
