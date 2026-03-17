#include <gsx/extra/gsx-stbi.h>
#include <gsx/gsx.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum image_format {
    IMAGE_FORMAT_PNG = 0,
    IMAGE_FORMAT_JPG = 1
} image_format;

typedef struct app_options {
    gsx_backend_type backend_type;
    gsx_index_t device_index;
    gsx_backend_buffer_type_class buffer_type_class;
    gsx_index_t width;
    gsx_index_t height;
    gsx_data_type renderer_output_data_type;
    gsx_renderer_feature_flags renderer_feature_flags;
    bool renderer_enable_alpha_output;
    bool renderer_enable_invdepth_output;
    gsx_render_precision render_precision;
    gsx_render_forward_type render_forward_type;
    gsx_index_t sh_degree;
    gsx_float_t fx;
    gsx_float_t fy;
    gsx_float_t cx;
    gsx_float_t cy;
    gsx_float_t near_plane;
    gsx_float_t far_plane;
    gsx_vec3 background_color;
    gsx_quat pose_rotation_xyzw;
    gsx_vec3 pose_translation;
    gsx_index_t camera_id;
    gsx_index_t frame_id;
    gsx_index_t gaussian_count;
    uint32_t gaussian_seed;
    const char *gs_mean3d_override;
    const char *gs_rotation_override;
    const char *gs_logscale_override;
    const char *gs_sh0_override;
    const char *gs_sh1_override;
    const char *gs_sh2_override;
    const char *gs_sh3_override;
    const char *gs_opacity_override;
    float *gs_mean3d;
    float *gs_rotation;
    float *gs_logscale;
    float *gs_sh0;
    float *gs_sh1;
    float *gs_sh2;
    float *gs_sh3;
    float *gs_opacity;
    const char *output_path;
    image_format output_format;
    gsx_index_t jpg_quality;
    const char *reference_image_path;
    const char *cpu_reference_output_path;
    bool compare_with_cpu;
    float compare_max_abs_tol;
    float compare_rmse_tol;
    bool numerical_diff_enable;
    float numerical_diff_eps;
    float numerical_diff_tol;
    float numerical_diff_rel_tol;
    uint32_t numerical_diff_seed;
} app_options;

typedef struct app_state {
    gsx_backend_t backend;
    gsx_arena_t arena;
    gsx_renderer_t renderer;
    gsx_render_context_t context;
    gsx_tensor_t mean3d;
    gsx_tensor_t rotation;
    gsx_tensor_t logscale;
    gsx_tensor_t sh0;
    gsx_tensor_t sh1;
    gsx_tensor_t sh2;
    gsx_tensor_t sh3;
    gsx_tensor_t opacity;
    gsx_tensor_t out_rgb;
    void *host_rgb;
    gsx_size_t host_rgb_size_bytes;
} app_state;

typedef struct gaussian_params {
    gsx_index_t count;
    float *mean3d;
    float *rotation;
    float *logscale;
    float *sh0;
    float *sh1;
    float *sh2;
    float *sh3;
    float *opacity;
} gaussian_params;

typedef struct image_compare_stats {
    gsx_size_t count;
    double max_abs;
    double mean_abs;
    double rmse;
    gsx_size_t max_index;
} image_compare_stats;

static bool init_tensor_f32(gsx_tensor_t *out_tensor, gsx_arena_t arena, gsx_index_t rank, const gsx_index_t *shape, const float *values, gsx_size_t value_count);
static uint32_t lcg_next(uint32_t *state_ptr);
static float uniform01(uint32_t *state_ptr);
static float randn(uint32_t *state_ptr);
static double dot_f32(const float *lhs, const float *rhs, gsx_size_t count);
static bool write_render_output(const app_options *options, const app_state *state, const char *path);
static void compute_image_compare_stats(const float *actual, const float *reference, gsx_size_t count, image_compare_stats *out_stats);
static void image_index_to_chw(gsx_size_t index, gsx_index_t width, gsx_index_t height, gsx_index_t *out_channel, gsx_index_t *out_y, gsx_index_t *out_x);
static bool compare_host_rgb_buffers(const app_options *options, const float *actual, const float *reference, const char *label);
static bool compare_against_reference_image(const app_options *options, const app_state *state);
static bool compare_against_cpu_reference(const app_options *options, const app_state *state);
static bool run_render(const app_options *options, app_state *state);
static void free_options(app_options *options);
static bool initialize_gaussian_params(app_options *options);
static void cleanup_state(app_state *state);
static void configure_numerical_diff_options(const app_options *options, app_options *out_options, bool *out_adjusted);
static void configure_numerical_diff_params(const app_options *options, gaussian_params *out_params);
static bool gaussian_params_alloc(gaussian_params *params, gsx_index_t count);
static void gaussian_params_free(gaussian_params *params);
static bool gaussian_params_copy_from_options(gaussian_params *params, const app_options *options);
static bool evaluate_objective(
    const app_options *opt,
    app_state *app,
    const gaussian_params *params,
    const float *grad_rgb_host,
    gsx_size_t rgb_count,
    double *out_obj);

static bool gsx_check(gsx_error err, const char *context)
{
    if(gsx_error_is_success(err)) {
        return true;
    }
    fprintf(stderr, "error: %s failed (%d)", context, err.code);
    if(err.message != NULL) {
        fprintf(stderr, ": %s", err.message);
    }
    fprintf(stderr, "\n");
    return false;
}

static bool parse_i64(const char *value, int64_t *out_value)
{
    char *end = NULL;
    long long parsed = 0;

    if(value == NULL || out_value == NULL) {
        return false;
    }
    errno = 0;
    parsed = strtoll(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out_value = (int64_t)parsed;
    return true;
}

static bool parse_u32(const char *value, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long parsed = 0;

    if(value == NULL || out_value == NULL) {
        return false;
    }
    errno = 0;
    parsed = strtoul(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return false;
    }
    *out_value = (uint32_t)parsed;
    return true;
}

static bool parse_f32(const char *value, float *out_value)
{
    char *end = NULL;
    float parsed = 0.0f;

    if(value == NULL || out_value == NULL) {
        return false;
    }
    errno = 0;
    parsed = strtof(value, &end);
    if(errno != 0 || end == value || *end != '\0') {
        return false;
    }
    *out_value = parsed;
    return true;
}

static bool parse_bool_value(const char *value, bool *out_value)
{
    if(value == NULL || out_value == NULL) {
        return false;
    }
    if(strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "yes") == 0) {
        *out_value = true;
        return true;
    }
    if(strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "no") == 0) {
        *out_value = false;
        return true;
    }
    return false;
}

static bool parse_backend_type(const char *value, gsx_backend_type *out_type)
{
    if(value == NULL || out_type == NULL) {
        return false;
    }
    if(strcmp(value, "cpu") == 0) {
        *out_type = GSX_BACKEND_TYPE_CPU;
        return true;
    }
    if(strcmp(value, "cuda") == 0) {
        *out_type = GSX_BACKEND_TYPE_CUDA;
        return true;
    }
    if(strcmp(value, "metal") == 0) {
        *out_type = GSX_BACKEND_TYPE_METAL;
        return true;
    }
    return false;
}

static const char *backend_type_name(gsx_backend_type type)
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

static bool parse_buffer_type_class(const char *value, gsx_backend_buffer_type_class *out_class)
{
    if(value == NULL || out_class == NULL) {
        return false;
    }
    if(strcmp(value, "host") == 0) {
        *out_class = GSX_BACKEND_BUFFER_TYPE_HOST;
        return true;
    }
    if(strcmp(value, "host_pinned") == 0) {
        *out_class = GSX_BACKEND_BUFFER_TYPE_HOST_PINNED;
        return true;
    }
    if(strcmp(value, "device") == 0) {
        *out_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
        return true;
    }
    if(strcmp(value, "unified") == 0) {
        *out_class = GSX_BACKEND_BUFFER_TYPE_UNIFIED;
        return true;
    }
    return false;
}

static bool parse_data_type(const char *value, gsx_data_type *out_data_type)
{
    if(value == NULL || out_data_type == NULL) {
        return false;
    }
    if(strcmp(value, "f32") == 0) {
        *out_data_type = GSX_DATA_TYPE_F32;
        return true;
    }
    if(strcmp(value, "f16") == 0) {
        *out_data_type = GSX_DATA_TYPE_F16;
        return true;
    }
    if(strcmp(value, "bf16") == 0) {
        *out_data_type = GSX_DATA_TYPE_BF16;
        return true;
    }
    if(strcmp(value, "u8") == 0) {
        *out_data_type = GSX_DATA_TYPE_U8;
        return true;
    }
    return false;
}

static bool parse_render_precision(const char *value, gsx_render_precision *out_precision)
{
    if(value == NULL || out_precision == NULL) {
        return false;
    }
    if(strcmp(value, "f32") == 0) {
        *out_precision = GSX_RENDER_PRECISION_FLOAT32;
        return true;
    }
    if(strcmp(value, "f16") == 0) {
        *out_precision = GSX_RENDER_PRECISION_FLOAT16;
        return true;
    }
    return false;
}

static bool parse_render_forward_type(const char *value, gsx_render_forward_type *out_forward_type)
{
    if(value == NULL || out_forward_type == NULL) {
        return false;
    }
    if(strcmp(value, "inference") == 0) {
        *out_forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
        return true;
    }
    if(strcmp(value, "train") == 0) {
        *out_forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
        return true;
    }
    if(strcmp(value, "metric") == 0) {
        *out_forward_type = GSX_RENDER_FORWARD_TYPE_METRIC;
        return true;
    }
    return false;
}

static bool parse_image_format(const char *value, image_format *out_format)
{
    if(value == NULL || out_format == NULL) {
        return false;
    }
    if(strcmp(value, "png") == 0) {
        *out_format = IMAGE_FORMAT_PNG;
        return true;
    }
    if(strcmp(value, "jpg") == 0 || strcmp(value, "jpeg") == 0) {
        *out_format = IMAGE_FORMAT_JPG;
        return true;
    }
    return false;
}

static bool parse_f32_list(const char *value, float *out_values, size_t expected_count)
{
    char *buffer = NULL;
    char *token = NULL;
    size_t parsed_count = 0;

    if(value == NULL || out_values == NULL || expected_count == 0) {
        return false;
    }
    buffer = (char *)malloc(strlen(value) + 1u);
    if(buffer == NULL) {
        return false;
    }
    strcpy(buffer, value);
    token = strtok(buffer, ",");
    while(token != NULL) {
        if(parsed_count >= expected_count || !parse_f32(token, &out_values[parsed_count])) {
            free(buffer);
            return false;
        }
        parsed_count += 1u;
        token = strtok(NULL, ",");
    }
    free(buffer);
    return parsed_count == expected_count;
}

static bool gaussian_params_alloc(gaussian_params *params, gsx_index_t count)
{
    gsx_size_t n = (gsx_size_t)count;

    memset(params, 0, sizeof(*params));
    params->count = count;
    params->mean3d = (float *)malloc((size_t)(n * 3u) * sizeof(float));
    params->rotation = (float *)malloc((size_t)(n * 4u) * sizeof(float));
    params->logscale = (float *)malloc((size_t)(n * 3u) * sizeof(float));
    params->sh0 = (float *)malloc((size_t)(n * 3u) * sizeof(float));
    params->sh1 = (float *)malloc((size_t)(n * 9u) * sizeof(float));
    params->sh2 = (float *)malloc((size_t)(n * 15u) * sizeof(float));
    params->sh3 = (float *)malloc((size_t)(n * 21u) * sizeof(float));
    params->opacity = (float *)malloc((size_t)n * sizeof(float));
    if(params->mean3d == NULL || params->rotation == NULL || params->logscale == NULL || params->sh0 == NULL || params->sh1 == NULL || params->sh2 == NULL
        || params->sh3 == NULL || params->opacity == NULL) {
        gaussian_params_free(params);
        return false;
    }
    return true;
}

static void gaussian_params_free(gaussian_params *params)
{
    free(params->mean3d);
    free(params->rotation);
    free(params->logscale);
    free(params->sh0);
    free(params->sh1);
    free(params->sh2);
    free(params->sh3);
    free(params->opacity);
    memset(params, 0, sizeof(*params));
}

static bool gaussian_params_copy_from_options(gaussian_params *params, const app_options *options)
{
    const gsx_size_t n = (gsx_size_t)options->gaussian_count;

    if(!gaussian_params_alloc(params, options->gaussian_count)) {
        return false;
    }
    memcpy(params->mean3d, options->gs_mean3d, (size_t)(n * 3u) * sizeof(float));
    memcpy(params->rotation, options->gs_rotation, (size_t)(n * 4u) * sizeof(float));
    memcpy(params->logscale, options->gs_logscale, (size_t)(n * 3u) * sizeof(float));
    memcpy(params->sh0, options->gs_sh0, (size_t)(n * 3u) * sizeof(float));
    memcpy(params->sh1, options->gs_sh1, (size_t)(n * 9u) * sizeof(float));
    memcpy(params->sh2, options->gs_sh2, (size_t)(n * 15u) * sizeof(float));
    memcpy(params->sh3, options->gs_sh3, (size_t)(n * 21u) * sizeof(float));
    memcpy(params->opacity, options->gs_opacity, (size_t)n * sizeof(float));
    return true;
}

static void free_options(app_options *options)
{
    free(options->gs_mean3d);
    free(options->gs_rotation);
    free(options->gs_logscale);
    free(options->gs_sh0);
    free(options->gs_sh1);
    free(options->gs_sh2);
    free(options->gs_sh3);
    free(options->gs_opacity);
    options->gs_mean3d = NULL;
    options->gs_rotation = NULL;
    options->gs_logscale = NULL;
    options->gs_sh0 = NULL;
    options->gs_sh1 = NULL;
    options->gs_sh2 = NULL;
    options->gs_sh3 = NULL;
    options->gs_opacity = NULL;
}

static bool initialize_gaussian_params(app_options *options)
{
    const float defaults_mean3d[6] = { -0.2f, 0.0f, 4.0f, 0.35f, 0.1f, 3.6f };
    const float defaults_rotation[8] = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.1305262f, 0.9914449f };
    const float defaults_logscale[6] = { -0.1f, -0.3f, -0.3f, -0.2f, -0.6f, -0.4f };
    const float defaults_sh0[6] = { 0.8f, 0.2f, 0.1f, 0.1f, 0.4f, 0.9f };
    const float defaults_sh1[18] = {
        0.02f, 0.01f, -0.01f, 0.00f, 0.02f, -0.02f, 0.01f, -0.01f, 0.00f,
        -0.01f, 0.02f, 0.01f, 0.02f, -0.02f, 0.00f, -0.01f, 0.00f, 0.01f
    };
    const float defaults_sh2[30] = {
        0.00f, 0.01f, -0.01f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f,
        0.01f, -0.01f, 0.00f, 0.01f, 0.00f, -0.01f, 0.01f, 0.00f, 0.01f, -0.01f, 0.00f, 0.00f, 0.01f, 0.00f, -0.01f
    };
    const float defaults_sh3[42] = {
        0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f,
        0.01f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f, 0.00f, 0.00f, 0.01f, 0.00f, -0.01f, 0.00f, 0.01f
    };
    const float defaults_opacity[2] = { 0.3f, -0.1f };
    const gsx_size_t n = (gsx_size_t)options->gaussian_count;
    uint32_t rng_state = options->gaussian_seed;

    free_options(options);
    options->gs_mean3d = (float *)malloc((size_t)(n * 3u) * sizeof(float));
    options->gs_rotation = (float *)malloc((size_t)(n * 4u) * sizeof(float));
    options->gs_logscale = (float *)malloc((size_t)(n * 3u) * sizeof(float));
    options->gs_sh0 = (float *)malloc((size_t)(n * 3u) * sizeof(float));
    options->gs_sh1 = (float *)malloc((size_t)(n * 9u) * sizeof(float));
    options->gs_sh2 = (float *)malloc((size_t)(n * 15u) * sizeof(float));
    options->gs_sh3 = (float *)malloc((size_t)(n * 21u) * sizeof(float));
    options->gs_opacity = (float *)malloc((size_t)n * sizeof(float));
    if(options->gs_mean3d == NULL || options->gs_rotation == NULL || options->gs_logscale == NULL || options->gs_sh0 == NULL || options->gs_sh1 == NULL
        || options->gs_sh2 == NULL || options->gs_sh3 == NULL || options->gs_opacity == NULL) {
        fprintf(stderr, "error: allocation failed for gaussian parameter buffers\n");
        free_options(options);
        return false;
    }

    for(gsx_size_t i = 0; i < n; ++i) {
        const gsx_size_t base3 = i * 3u;
        const gsx_size_t base4 = i * 4u;
        const gsx_size_t base9 = i * 9u;
        const gsx_size_t base15 = i * 15u;
        const gsx_size_t base21 = i * 21u;
        float qx = randn(&rng_state);
        float qy = randn(&rng_state);
        float qz = randn(&rng_state);
        float qw = randn(&rng_state);
        float qnorm = sqrtf((qx * qx) + (qy * qy) + (qz * qz) + (qw * qw));

        options->gs_mean3d[base3 + 0u] = -1.2f + 2.4f * uniform01(&rng_state);
        options->gs_mean3d[base3 + 1u] = -1.2f + 2.4f * uniform01(&rng_state);
        options->gs_mean3d[base3 + 2u] = 2.2f + 4.3f * uniform01(&rng_state);
        if(qnorm < 1.0e-8f) {
            qx = 0.0f;
            qy = 0.0f;
            qz = 0.0f;
            qw = 1.0f;
            qnorm = 1.0f;
        }
        options->gs_rotation[base4 + 0u] = qx / qnorm;
        options->gs_rotation[base4 + 1u] = qy / qnorm;
        options->gs_rotation[base4 + 2u] = qz / qnorm;
        options->gs_rotation[base4 + 3u] = qw / qnorm;
        options->gs_logscale[base3 + 0u] = -2.2f + 1.9f * uniform01(&rng_state);
        options->gs_logscale[base3 + 1u] = -2.2f + 1.9f * uniform01(&rng_state);
        options->gs_logscale[base3 + 2u] = -2.2f + 1.9f * uniform01(&rng_state);
        options->gs_sh0[base3 + 0u] = uniform01(&rng_state);
        options->gs_sh0[base3 + 1u] = uniform01(&rng_state);
        options->gs_sh0[base3 + 2u] = uniform01(&rng_state);
        for(gsx_size_t j = 0; j < 9u; ++j) {
            options->gs_sh1[base9 + j] = -0.03f + 0.06f * uniform01(&rng_state);
        }
        for(gsx_size_t j = 0; j < 15u; ++j) {
            options->gs_sh2[base15 + j] = -0.02f + 0.04f * uniform01(&rng_state);
        }
        for(gsx_size_t j = 0; j < 21u; ++j) {
            options->gs_sh3[base21 + j] = -0.015f + 0.03f * uniform01(&rng_state);
        }
        options->gs_opacity[i] = -1.2f + 2.0f * uniform01(&rng_state);
    }

    if(n >= 1u) {
        memcpy(&options->gs_mean3d[0], &defaults_mean3d[0], 3u * sizeof(float));
        memcpy(&options->gs_rotation[0], &defaults_rotation[0], 4u * sizeof(float));
        memcpy(&options->gs_logscale[0], &defaults_logscale[0], 3u * sizeof(float));
        memcpy(&options->gs_sh0[0], &defaults_sh0[0], 3u * sizeof(float));
        memcpy(&options->gs_sh1[0], &defaults_sh1[0], 9u * sizeof(float));
        memcpy(&options->gs_sh2[0], &defaults_sh2[0], 15u * sizeof(float));
        memcpy(&options->gs_sh3[0], &defaults_sh3[0], 21u * sizeof(float));
        options->gs_opacity[0] = defaults_opacity[0];
    }
    if(n >= 2u) {
        memcpy(&options->gs_mean3d[3], &defaults_mean3d[3], 3u * sizeof(float));
        memcpy(&options->gs_rotation[4], &defaults_rotation[4], 4u * sizeof(float));
        memcpy(&options->gs_logscale[3], &defaults_logscale[3], 3u * sizeof(float));
        memcpy(&options->gs_sh0[3], &defaults_sh0[3], 3u * sizeof(float));
        memcpy(&options->gs_sh1[9], &defaults_sh1[9], 9u * sizeof(float));
        memcpy(&options->gs_sh2[15], &defaults_sh2[15], 15u * sizeof(float));
        memcpy(&options->gs_sh3[21], &defaults_sh3[21], 21u * sizeof(float));
        options->gs_opacity[1] = defaults_opacity[1];
    }

    if(options->gs_mean3d_override != NULL && !parse_f32_list(options->gs_mean3d_override, options->gs_mean3d, (size_t)(n * 3u))) {
        fprintf(stderr, "error: invalid gs mean3d list '%s' (expected %llu floats)\n", options->gs_mean3d_override, (unsigned long long)(n * 3u));
        return false;
    }
    if(options->gs_rotation_override != NULL && !parse_f32_list(options->gs_rotation_override, options->gs_rotation, (size_t)(n * 4u))) {
        fprintf(stderr, "error: invalid gs rotation list '%s' (expected %llu floats)\n", options->gs_rotation_override, (unsigned long long)(n * 4u));
        return false;
    }
    if(options->gs_logscale_override != NULL && !parse_f32_list(options->gs_logscale_override, options->gs_logscale, (size_t)(n * 3u))) {
        fprintf(stderr, "error: invalid gs logscale list '%s' (expected %llu floats)\n", options->gs_logscale_override, (unsigned long long)(n * 3u));
        return false;
    }
    if(options->gs_sh0_override != NULL && !parse_f32_list(options->gs_sh0_override, options->gs_sh0, (size_t)(n * 3u))) {
        fprintf(stderr, "error: invalid gs sh0 list '%s' (expected %llu floats)\n", options->gs_sh0_override, (unsigned long long)(n * 3u));
        return false;
    }
    if(options->gs_sh1_override != NULL && !parse_f32_list(options->gs_sh1_override, options->gs_sh1, (size_t)(n * 9u))) {
        fprintf(stderr, "error: invalid gs sh1 list '%s' (expected %llu floats)\n", options->gs_sh1_override, (unsigned long long)(n * 9u));
        return false;
    }
    if(options->gs_sh2_override != NULL && !parse_f32_list(options->gs_sh2_override, options->gs_sh2, (size_t)(n * 15u))) {
        fprintf(stderr, "error: invalid gs sh2 list '%s' (expected %llu floats)\n", options->gs_sh2_override, (unsigned long long)(n * 15u));
        return false;
    }
    if(options->gs_sh3_override != NULL && !parse_f32_list(options->gs_sh3_override, options->gs_sh3, (size_t)(n * 21u))) {
        fprintf(stderr, "error: invalid gs sh3 list '%s' (expected %llu floats)\n", options->gs_sh3_override, (unsigned long long)(n * 21u));
        return false;
    }
    if(options->gs_opacity_override != NULL && !parse_f32_list(options->gs_opacity_override, options->gs_opacity, (size_t)n)) {
        fprintf(stderr, "error: invalid gs opacity list '%s' (expected %llu floats)\n", options->gs_opacity_override, (unsigned long long)n);
        return false;
    }
    return true;
}

static void set_default_options(app_options *options)
{
    memset(options, 0, sizeof(*options));
    options->backend_type = GSX_BACKEND_TYPE_CPU;
    options->device_index = 0;
    options->buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    options->width = 640;
    options->height = 480;
    options->renderer_output_data_type = GSX_DATA_TYPE_F32;
    options->renderer_feature_flags = 0u;
    options->renderer_enable_alpha_output = false;
    options->renderer_enable_invdepth_output = false;
    options->render_precision = GSX_RENDER_PRECISION_FLOAT32;
    options->render_forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    options->sh_degree = 0;
    options->fx = 500.0f;
    options->fy = 500.0f;
    options->cx = 320.0f;
    options->cy = 240.0f;
    options->near_plane = 0.1f;
    options->far_plane = 20.0f;
    options->background_color.x = 0.02f;
    options->background_color.y = 0.02f;
    options->background_color.z = 0.03f;
    options->pose_rotation_xyzw.x = 0.0f;
    options->pose_rotation_xyzw.y = 0.0f;
    options->pose_rotation_xyzw.z = 0.0f;
    options->pose_rotation_xyzw.w = 1.0f;
    options->pose_translation.x = 0.0f;
    options->pose_translation.y = 0.0f;
    options->pose_translation.z = 0.0f;
    options->camera_id = 0;
    options->frame_id = 0;
    options->gaussian_count = 512;
    options->gaussian_seed = 7u;
    options->gs_mean3d_override = NULL;
    options->gs_rotation_override = NULL;
    options->gs_logscale_override = NULL;
    options->gs_sh0_override = NULL;
    options->gs_sh1_override = NULL;
    options->gs_sh2_override = NULL;
    options->gs_sh3_override = NULL;
    options->gs_opacity_override = NULL;
    options->output_path = "multi-gaussian.png";
    options->output_format = IMAGE_FORMAT_PNG;
    options->jpg_quality = 95;
    options->reference_image_path = NULL;
    options->cpu_reference_output_path = NULL;
    options->compare_with_cpu = false;
    options->compare_max_abs_tol = 5.0f / 255.0f;
    options->compare_rmse_tol = 2.0f / 255.0f;
    options->numerical_diff_enable = false;
    options->numerical_diff_eps = 1e-3f;
    options->numerical_diff_tol = 5e-2f;
    options->numerical_diff_rel_tol = 0.0f;
    options->numerical_diff_seed = 12345u;
}

static void print_usage(const char *program_name)
{
    fprintf(stderr, "usage: %s [options]\n", program_name);
    fprintf(stderr, "backend options:\n");
    fprintf(stderr, "  --backend <cpu|cuda|metal>\n");
    fprintf(stderr, "  --device <index>\n");
    fprintf(stderr, "  --buffer-type <host|host_pinned|device|unified>\n");
    fprintf(stderr, "renderer options:\n");
    fprintf(stderr, "  --width <int> --height <int>\n");
    fprintf(stderr, "  --renderer-output-type <f32|f16|bf16|u8>\n");
    fprintf(stderr, "  --renderer-feature-flags <uint32_bitmask>\n");
    fprintf(stderr, "  --renderer-enable-alpha <bool>\n");
    fprintf(stderr, "  --renderer-enable-invdepth <bool>\n");
    fprintf(stderr, "render request options:\n");
    fprintf(stderr, "  --precision <f32|f16>\n");
    fprintf(stderr, "  --forward-type <inference|train|metric>\n");
    fprintf(stderr, "  --sh-degree <0|1|2|3>\n");
    fprintf(stderr, "scene options:\n");
    fprintf(stderr, "  --fx <float> --fy <float> --cx <float> --cy <float>\n");
    fprintf(stderr, "  --near <float> --far <float>\n");
    fprintf(stderr, "  --bg <r,g,b>\n");
    fprintf(stderr, "  --pose-rot <x,y,z,w>\n");
    fprintf(stderr, "  --pose-trans <x,y,z>\n");
    fprintf(stderr, "  --camera-id <int> --frame-id <int>\n");
    fprintf(stderr, "gaussian options:\n");
    fprintf(stderr, "  --gaussian-count <int>\n");
    fprintf(stderr, "  --gaussian-seed <uint32>\n");
    fprintf(stderr, "  --gs-mean3d <N*3 floats: xyz per gaussian>\n");
    fprintf(stderr, "  --gs-rotation <N*4 floats: xyzw per gaussian>\n");
    fprintf(stderr, "  --gs-logscale <N*3 floats>\n");
    fprintf(stderr, "  --gs-sh0 <N*3 floats>\n");
    fprintf(stderr, "  --gs-sh1 <N*9 floats>\n");
    fprintf(stderr, "  --gs-sh2 <N*15 floats>\n");
    fprintf(stderr, "  --gs-sh3 <N*21 floats>\n");
    fprintf(stderr, "  --gs-opacity <N floats>\n");
    fprintf(stderr, "output options:\n");
    fprintf(stderr, "  --output <path>\n");
    fprintf(stderr, "  --format <png|jpg>\n");
    fprintf(stderr, "  --jpg-quality <1..100>\n");
    fprintf(stderr, "reference compare options:\n");
    fprintf(stderr, "  --reference-image <path>\n");
    fprintf(stderr, "  --compare-with-cpu <bool>\n");
    fprintf(stderr, "  --cpu-reference-output <path>\n");
    fprintf(stderr, "  --compare-max-abs-tol <float>\n");
    fprintf(stderr, "  --compare-rmse-tol <float>\n");
    fprintf(stderr, "numerical diff test options:\n");
    fprintf(stderr, "  --numerical-diff <bool>\n");
    fprintf(stderr, "  --diff-eps <float>\n");
    fprintf(stderr, "  --diff-tol <float>\n");
    fprintf(stderr, "  --diff-rel-tol <float>\n");
    fprintf(stderr, "  --diff-seed <uint32>\n");
}

static bool parse_args(int argc, char **argv, app_options *options)
{
    for(int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *value = (i + 1 < argc) ? argv[i + 1] : NULL;
        int64_t parsed_i64 = 0;
        uint32_t parsed_u32 = 0;
        float parsed_f32 = 0.0f;
        bool parsed_bool = false;

        if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if(value == NULL) {
            fprintf(stderr, "error: missing value for '%s'\n", arg);
            return false;
        }
        if(strcmp(arg, "--backend") == 0) {
            if(!parse_backend_type(value, &options->backend_type)) {
                fprintf(stderr, "error: invalid backend '%s'\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--device") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 < 0) {
                fprintf(stderr, "error: invalid device index '%s'\n", value);
                return false;
            }
            options->device_index = (gsx_index_t)parsed_i64;
            if((int64_t)options->device_index != parsed_i64) {
                fprintf(stderr, "error: device index '%s' out of range for gsx_index_t\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--buffer-type") == 0) {
            if(!parse_buffer_type_class(value, &options->buffer_type_class)) {
                fprintf(stderr, "error: invalid buffer type '%s'\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--width") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 <= 0) {
                fprintf(stderr, "error: invalid width '%s'\n", value);
                return false;
            }
            options->width = (gsx_index_t)parsed_i64;
            if((int64_t)options->width != parsed_i64) {
                fprintf(stderr, "error: width '%s' out of range for gsx_index_t\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--height") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 <= 0) {
                fprintf(stderr, "error: invalid height '%s'\n", value);
                return false;
            }
            options->height = (gsx_index_t)parsed_i64;
            if((int64_t)options->height != parsed_i64) {
                fprintf(stderr, "error: height '%s' out of range for gsx_index_t\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--renderer-output-type") == 0) {
            if(!parse_data_type(value, &options->renderer_output_data_type)) {
                fprintf(stderr, "error: invalid renderer output type '%s'\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--renderer-feature-flags") == 0) {
            if(!parse_u32(value, &parsed_u32)) {
                fprintf(stderr, "error: invalid renderer feature flags '%s'\n", value);
                return false;
            }
            options->renderer_feature_flags = (gsx_renderer_feature_flags)parsed_u32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--renderer-enable-alpha") == 0) {
            if(!parse_bool_value(value, &parsed_bool)) {
                fprintf(stderr, "error: invalid renderer enable alpha value '%s'\n", value);
                return false;
            }
            options->renderer_enable_alpha_output = parsed_bool;
            ++i;
            continue;
        }
        if(strcmp(arg, "--renderer-enable-invdepth") == 0) {
            if(!parse_bool_value(value, &parsed_bool)) {
                fprintf(stderr, "error: invalid renderer enable invdepth value '%s'\n", value);
                return false;
            }
            options->renderer_enable_invdepth_output = parsed_bool;
            ++i;
            continue;
        }
        if(strcmp(arg, "--precision") == 0) {
            if(!parse_render_precision(value, &options->render_precision)) {
                fprintf(stderr, "error: invalid precision '%s'\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--forward-type") == 0) {
            if(!parse_render_forward_type(value, &options->render_forward_type)) {
                fprintf(stderr, "error: invalid forward type '%s'\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--sh-degree") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 < 0 || parsed_i64 > 3) {
                fprintf(stderr, "error: invalid sh degree '%s'\n", value);
                return false;
            }
            options->sh_degree = (gsx_index_t)parsed_i64;
            ++i;
            continue;
        }
        if(strcmp(arg, "--fx") == 0 || strcmp(arg, "--fy") == 0 || strcmp(arg, "--cx") == 0 || strcmp(arg, "--cy") == 0) {
            if(!parse_f32(value, &parsed_f32)) {
                fprintf(stderr, "error: invalid intrinsics value '%s'\n", value);
                return false;
            }
            if(strcmp(arg, "--fx") == 0) {
                options->fx = parsed_f32;
            } else if(strcmp(arg, "--fy") == 0) {
                options->fy = parsed_f32;
            } else if(strcmp(arg, "--cx") == 0) {
                options->cx = parsed_f32;
            } else {
                options->cy = parsed_f32;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--near") == 0 || strcmp(arg, "--far") == 0) {
            if(!parse_f32(value, &parsed_f32)) {
                fprintf(stderr, "error: invalid plane value '%s'\n", value);
                return false;
            }
            if(strcmp(arg, "--near") == 0) {
                options->near_plane = parsed_f32;
            } else {
                options->far_plane = parsed_f32;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--bg") == 0) {
            float parsed_bg[3] = { 0.0f, 0.0f, 0.0f };
            if(!parse_f32_list(value, parsed_bg, 3u)) {
                fprintf(stderr, "error: invalid background color list '%s'\n", value);
                return false;
            }
            options->background_color.x = parsed_bg[0];
            options->background_color.y = parsed_bg[1];
            options->background_color.z = parsed_bg[2];
            ++i;
            continue;
        }
        if(strcmp(arg, "--pose-rot") == 0) {
            float parsed_rot[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            if(!parse_f32_list(value, parsed_rot, 4u)) {
                fprintf(stderr, "error: invalid pose rotation list '%s'\n", value);
                return false;
            }
            options->pose_rotation_xyzw.x = parsed_rot[0];
            options->pose_rotation_xyzw.y = parsed_rot[1];
            options->pose_rotation_xyzw.z = parsed_rot[2];
            options->pose_rotation_xyzw.w = parsed_rot[3];
            ++i;
            continue;
        }
        if(strcmp(arg, "--pose-trans") == 0) {
            float parsed_trans[3] = { 0.0f, 0.0f, 0.0f };
            if(!parse_f32_list(value, parsed_trans, 3u)) {
                fprintf(stderr, "error: invalid pose translation list '%s'\n", value);
                return false;
            }
            options->pose_translation.x = parsed_trans[0];
            options->pose_translation.y = parsed_trans[1];
            options->pose_translation.z = parsed_trans[2];
            ++i;
            continue;
        }
        if(strcmp(arg, "--camera-id") == 0 || strcmp(arg, "--frame-id") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 < 0) {
                fprintf(stderr, "error: invalid id value '%s'\n", value);
                return false;
            }
            if(strcmp(arg, "--camera-id") == 0) {
                options->camera_id = (gsx_index_t)parsed_i64;
                if((int64_t)options->camera_id != parsed_i64) {
                    fprintf(stderr, "error: camera-id '%s' out of range for gsx_index_t\n", value);
                    return false;
                }
            } else {
                options->frame_id = (gsx_index_t)parsed_i64;
                if((int64_t)options->frame_id != parsed_i64) {
                    fprintf(stderr, "error: frame-id '%s' out of range for gsx_index_t\n", value);
                    return false;
                }
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--gaussian-count") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 <= 0) {
                fprintf(stderr, "error: invalid gaussian count '%s'\n", value);
                return false;
            }
            options->gaussian_count = (gsx_index_t)parsed_i64;
            if((int64_t)options->gaussian_count != parsed_i64) {
                fprintf(stderr, "error: gaussian count '%s' out of range for gsx_index_t\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--gaussian-seed") == 0) {
            if(!parse_u32(value, &parsed_u32)) {
                fprintf(stderr, "error: invalid gaussian seed '%s'\n", value);
                return false;
            }
            options->gaussian_seed = parsed_u32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-mean3d") == 0) {
            options->gs_mean3d_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-rotation") == 0) {
            options->gs_rotation_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-logscale") == 0) {
            options->gs_logscale_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-sh0") == 0) {
            options->gs_sh0_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-sh1") == 0) {
            options->gs_sh1_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-sh2") == 0) {
            options->gs_sh2_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-sh3") == 0) {
            options->gs_sh3_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--gs-opacity") == 0) {
            options->gs_opacity_override = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--output") == 0) {
            options->output_path = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--format") == 0) {
            if(!parse_image_format(value, &options->output_format)) {
                fprintf(stderr, "error: invalid output format '%s'\n", value);
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--jpg-quality") == 0) {
            if(!parse_i64(value, &parsed_i64) || parsed_i64 < 1 || parsed_i64 > 100) {
                fprintf(stderr, "error: invalid jpg quality '%s'\n", value);
                return false;
            }
            options->jpg_quality = (gsx_index_t)parsed_i64;
            ++i;
            continue;
        }
        if(strcmp(arg, "--reference-image") == 0) {
            options->reference_image_path = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--compare-with-cpu") == 0) {
            if(!parse_bool_value(value, &parsed_bool)) {
                fprintf(stderr, "error: invalid compare-with-cpu value '%s'\n", value);
                return false;
            }
            options->compare_with_cpu = parsed_bool;
            ++i;
            continue;
        }
        if(strcmp(arg, "--cpu-reference-output") == 0) {
            options->cpu_reference_output_path = value;
            ++i;
            continue;
        }
        if(strcmp(arg, "--compare-max-abs-tol") == 0) {
            if(!parse_f32(value, &parsed_f32) || parsed_f32 < 0.0f) {
                fprintf(stderr, "error: invalid compare max abs tol '%s'\n", value);
                return false;
            }
            options->compare_max_abs_tol = parsed_f32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--compare-rmse-tol") == 0) {
            if(!parse_f32(value, &parsed_f32) || parsed_f32 < 0.0f) {
                fprintf(stderr, "error: invalid compare rmse tol '%s'\n", value);
                return false;
            }
            options->compare_rmse_tol = parsed_f32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--numerical-diff") == 0) {
            if(!parse_bool_value(value, &parsed_bool)) {
                fprintf(stderr, "error: invalid numerical diff switch '%s'\n", value);
                return false;
            }
            options->numerical_diff_enable = parsed_bool;
            ++i;
            continue;
        }
        if(strcmp(arg, "--diff-eps") == 0) {
            if(!parse_f32(value, &parsed_f32) || parsed_f32 <= 0.0f) {
                fprintf(stderr, "error: invalid diff eps '%s'\n", value);
                return false;
            }
            options->numerical_diff_eps = parsed_f32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--diff-tol") == 0) {
            if(!parse_f32(value, &parsed_f32) || parsed_f32 < 0.0f) {
                fprintf(stderr, "error: invalid diff tol '%s'\n", value);
                return false;
            }
            options->numerical_diff_tol = parsed_f32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--diff-rel-tol") == 0) {
            if(!parse_f32(value, &parsed_f32) || parsed_f32 < 0.0f) {
                fprintf(stderr, "error: invalid diff rel tol '%s'\n", value);
                return false;
            }
            options->numerical_diff_rel_tol = parsed_f32;
            ++i;
            continue;
        }
        if(strcmp(arg, "--diff-seed") == 0) {
            if(!parse_u32(value, &parsed_u32)) {
                fprintf(stderr, "error: invalid diff seed '%s'\n", value);
                return false;
            }
            options->numerical_diff_seed = parsed_u32;
            ++i;
            continue;
        }
        fprintf(stderr, "error: unknown argument '%s'\n", arg);
        return false;
    }
    return true;
}

static uint32_t lcg_next(uint32_t *state_ptr)
{
    *state_ptr = (*state_ptr * 1664525u) + 1013904223u;
    return *state_ptr;
}

static float uniform01(uint32_t *state_ptr)
{
    return (float)((lcg_next(state_ptr) >> 8) * (1.0 / 16777216.0));
}

static float randn(uint32_t *state_ptr)
{
    const float pi2 = 6.28318530717958647692f;
    float u1 = uniform01(state_ptr);
    float u2 = uniform01(state_ptr);

    if(u1 < 1.0e-7f) {
        u1 = 1.0e-7f;
    }
    return sqrtf(-2.0f * logf(u1)) * cosf(pi2 * u2);
}

static double dot_f32(const float *lhs, const float *rhs, gsx_size_t count)
{
    double sum = 0.0;
    for(gsx_size_t i = 0; i < count; ++i) {
        sum += (double)lhs[i] * (double)rhs[i];
    }
    return sum;
}

static bool write_render_output(const app_options *options, const app_state *state, const char *path)
{
    gsx_error write_error = { GSX_ERROR_SUCCESS, NULL };

    if(options == NULL || state == NULL || path == NULL || path[0] == '\0') {
        return true;
    }
    if(state->host_rgb == NULL) {
        fprintf(stderr, "error: render output is not available for image write\n");
        return false;
    }

    if(options->output_format == IMAGE_FORMAT_PNG) {
        write_error = gsx_image_write_png(
            path,
            state->host_rgb,
            options->width,
            options->height,
            3,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_CHW);
    } else {
        write_error = gsx_image_write_jpg(
            path,
            state->host_rgb,
            options->width,
            options->height,
            3,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_CHW,
            options->jpg_quality);
    }
    return gsx_check(write_error, "gsx_image_write");
}

static void compute_image_compare_stats(const float *actual, const float *reference, gsx_size_t count, image_compare_stats *out_stats)
{
    double sum_abs = 0.0;
    double sum_sq = 0.0;

    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->count = count;
    for(gsx_size_t i = 0; i < count; ++i) {
        const double diff = (double)actual[i] - (double)reference[i];
        const double abs_diff = fabs(diff);

        sum_abs += abs_diff;
        sum_sq += diff * diff;
        if(abs_diff > out_stats->max_abs) {
            out_stats->max_abs = abs_diff;
            out_stats->max_index = i;
        }
    }
    if(count != 0) {
        out_stats->mean_abs = sum_abs / (double)count;
        out_stats->rmse = sqrt(sum_sq / (double)count);
    }
}

static void image_index_to_chw(gsx_size_t index, gsx_index_t width, gsx_index_t height, gsx_index_t *out_channel, gsx_index_t *out_y, gsx_index_t *out_x)
{
    const gsx_size_t channel_stride = (gsx_size_t)width * (gsx_size_t)height;
    const gsx_size_t channel_index = channel_stride != 0 ? index / channel_stride : 0;
    const gsx_size_t rem = channel_stride != 0 ? index % channel_stride : 0;

    *out_channel = (gsx_index_t)channel_index;
    *out_y = width != 0 ? (gsx_index_t)(rem / (gsx_size_t)width) : 0;
    *out_x = width != 0 ? (gsx_index_t)(rem % (gsx_size_t)width) : 0;
}

static bool compare_host_rgb_buffers(const app_options *options, const float *actual, const float *reference, const char *label)
{
    image_compare_stats stats;
    gsx_index_t channel = 0;
    gsx_index_t y = 0;
    gsx_index_t x = 0;

    if(options == NULL || actual == NULL || reference == NULL || label == NULL) {
        fprintf(stderr, "error: compare_host_rgb_buffers received invalid inputs\n");
        return false;
    }

    compute_image_compare_stats(actual, reference, (gsx_size_t)3u * (gsx_size_t)options->height * (gsx_size_t)options->width, &stats);
    image_index_to_chw(stats.max_index, options->width, options->height, &channel, &y, &x);
    printf(
        "%s: max_abs=%.9e rmse=%.9e mean_abs=%.9e at c=%lld y=%lld x=%lld\n",
        label,
        stats.max_abs,
        stats.rmse,
        stats.mean_abs,
        (long long)channel,
        (long long)y,
        (long long)x);
    if(stats.max_abs > (double)options->compare_max_abs_tol || stats.rmse > (double)options->compare_rmse_tol) {
        fprintf(
            stderr,
            "FAILED: %s exceeds tolerance (max_abs_tol=%.9g rmse_tol=%.9g)\n",
            label,
            (double)options->compare_max_abs_tol,
            (double)options->compare_rmse_tol);
        return false;
    }
    return true;
}

static bool compare_against_reference_image(const app_options *options, const app_state *state)
{
    gsx_image output_image;
    gsx_image reference_image;
    float *output_pixels = NULL;
    float *reference_pixels = NULL;
    gsx_size_t count = 0;
    bool ok = false;

    (void)state;
    memset(&output_image, 0, sizeof(output_image));
    memset(&reference_image, 0, sizeof(reference_image));
    if(options->reference_image_path == NULL) {
        return true;
    }
    if(options->output_path == NULL || options->output_path[0] == '\0') {
        fprintf(stderr, "error: output path must be set when comparing against a reference image\n");
        return false;
    }
    if(!gsx_check(
           gsx_image_load(&output_image, options->output_path, 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_CHW),
           "gsx_image_load(output_image)")) {
        return false;
    }
    if(!gsx_check(
           gsx_image_load(&reference_image, options->reference_image_path, 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_CHW),
           "gsx_image_load(reference_image)")) {
        gsx_check(gsx_image_free(&output_image), "gsx_image_free(output_image)");
        return false;
    }
    if(output_image.width != options->width || output_image.height != options->height || output_image.channels != 3) {
        fprintf(
            stderr,
            "error: output image shape mismatch, expected [3,%lld,%lld], got [3,%lld,%lld]\n",
            (long long)options->height,
            (long long)options->width,
            (long long)output_image.height,
            (long long)output_image.width);
        goto cleanup;
    }
    if(reference_image.width != options->width || reference_image.height != options->height || reference_image.channels != 3) {
        fprintf(
            stderr,
            "error: reference image shape mismatch, expected [3,%lld,%lld], got [3,%lld,%lld]\n",
            (long long)options->height,
            (long long)options->width,
            (long long)reference_image.height,
            (long long)reference_image.width);
        goto cleanup;
    }
    count = (gsx_size_t)3u * (gsx_size_t)options->height * (gsx_size_t)options->width;
    output_pixels = (float *)malloc((size_t)count * sizeof(float));
    reference_pixels = (float *)malloc((size_t)count * sizeof(float));
    if(output_pixels == NULL || reference_pixels == NULL) {
        fprintf(stderr, "error: allocation failed for reference compare buffers\n");
        goto cleanup;
    }
    for(gsx_size_t i = 0; i < count; ++i) {
        output_pixels[i] = (float)((const uint8_t *)output_image.pixels)[i] * (1.0f / 255.0f);
        reference_pixels[i] = (float)((const uint8_t *)reference_image.pixels)[i] * (1.0f / 255.0f);
    }
    ok = compare_host_rgb_buffers(options, output_pixels, reference_pixels, "reference image compare");

cleanup:
    free(output_pixels);
    free(reference_pixels);
    gsx_check(gsx_image_free(&output_image), "gsx_image_free(output_image)");
    gsx_check(gsx_image_free(&reference_image), "gsx_image_free(reference_image)");
    return ok;
}

static bool compare_against_cpu_reference(const app_options *options, const app_state *state)
{
    app_options cpu_options;
    app_state cpu_state;
    bool ok = false;

    if(!options->compare_with_cpu) {
        return true;
    }
    memset(&cpu_state, 0, sizeof(cpu_state));
    cpu_options = *options;
    cpu_options.backend_type = GSX_BACKEND_TYPE_CPU;
    cpu_options.device_index = 0;
    cpu_options.buffer_type_class = GSX_BACKEND_BUFFER_TYPE_HOST;
    cpu_options.reference_image_path = NULL;
    cpu_options.compare_with_cpu = false;
    cpu_options.cpu_reference_output_path = NULL;
    cpu_options.numerical_diff_enable = false;
    cpu_options.output_path = NULL;
    if(!run_render(&cpu_options, &cpu_state)) {
        goto cleanup;
    }
    if(options->cpu_reference_output_path != NULL && !write_render_output(options, &cpu_state, options->cpu_reference_output_path)) {
        goto cleanup;
    }
    if(!compare_host_rgb_buffers(options, (const float *)state->host_rgb, (const float *)cpu_state.host_rgb, "cpu reference compare")) {
        goto cleanup;
    }
    ok = true;

cleanup:
    cleanup_state(&cpu_state);
    return ok;
}

static void configure_numerical_diff_options(const app_options *options, app_options *out_options, bool *out_adjusted)
{
    *out_options = *options;
    out_options->fx = 50.0f * fminf(1.0f, (float)options->width / 64.0f);
    out_options->fy = 66.66667175292969f * fminf(1.0f, (float)options->height / 64.0f);
    out_options->cx = 0.5f * (float)options->width;
    out_options->cy = 0.5f * (float)options->height;
    if(out_adjusted != NULL) {
        *out_adjusted = fabsf(out_options->fx - options->fx) > 1.0e-6f || fabsf(out_options->fy - options->fy) > 1.0e-6f
            || fabsf(out_options->cx - options->cx) > 1.0e-6f || fabsf(out_options->cy - options->cy) > 1.0e-6f;
    }
}

static void configure_numerical_diff_params(const app_options *options, gaussian_params *out_params)
{
    const gsx_size_t n = (gsx_size_t)options->gaussian_count;

    for(gsx_size_t i = 0; i < n; ++i) {
        const gsx_size_t base3 = i * 3u;
        const gsx_size_t base4 = i * 4u;

        out_params->mean3d[base3 + 0u] = ((float)(i % 32u) - 15.5f) * 0.06f;
        out_params->mean3d[base3 + 1u] = ((float)((i / 32u) % 32u) - 15.5f) * 0.04f;
        out_params->mean3d[base3 + 2u] = 3.2f + 0.004f * (float)i;
        out_params->rotation[base4 + 0u] = 0.0f;
        out_params->rotation[base4 + 1u] = 0.0f;
        out_params->rotation[base4 + 2u] = 0.0f;
        out_params->rotation[base4 + 3u] = 1.0f;
        out_params->logscale[base3 + 0u] = -2.0f;
        out_params->logscale[base3 + 1u] = -2.0f;
        out_params->logscale[base3 + 2u] = -2.0f;
        out_params->sh0[base3 + 0u] = 0.0f;
        out_params->sh0[base3 + 1u] = 0.0f;
        out_params->sh0[base3 + 2u] = 0.0f;
        out_params->opacity[i] = -9.0f;
    }
    if(n >= 1u) {
        out_params->mean3d[0] = 0.0f;
        out_params->mean3d[1] = 0.0f;
        out_params->mean3d[2] = 3.6f;
        out_params->rotation[2] = 0.1305262f;
        out_params->rotation[3] = 0.9914449f;
        out_params->logscale[0] = -1.8f;
        out_params->logscale[1] = -2.0f;
        out_params->logscale[2] = -1.9f;
        out_params->sh0[0] = 0.8f;
        out_params->sh0[1] = 0.2f;
        out_params->sh0[2] = 0.1f;
        out_params->opacity[0] = 0.3f;
    }
}

static bool evaluate_objective(
    const app_options *opt,
    app_state *app,
    const gaussian_params *params,
    const float *grad_rgb_host,
    gsx_size_t rgb_count,
    double *out_obj)
{
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    gsx_render_forward_request forward_request;
    const gsx_size_t n = (gsx_size_t)opt->gaussian_count;

    if(!gsx_check(gsx_tensor_upload(app->mean3d, params->mean3d, (n * 3u) * sizeof(float)), "gsx_tensor_upload(mean3d,obj)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_upload(app->rotation, params->rotation, (n * 4u) * sizeof(float)), "gsx_tensor_upload(rotation,obj)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_upload(app->logscale, params->logscale, (n * 3u) * sizeof(float)), "gsx_tensor_upload(logscale,obj)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_upload(app->sh0, params->sh0, (n * 3u) * sizeof(float)), "gsx_tensor_upload(sh0,obj)")) {
        return false;
    }
    if(opt->sh_degree >= 1) {
        if(!gsx_check(gsx_tensor_upload(app->sh1, params->sh1, (n * 9u) * sizeof(float)), "gsx_tensor_upload(sh1,obj)")) {
            return false;
        }
    }
    if(opt->sh_degree >= 2) {
        if(!gsx_check(gsx_tensor_upload(app->sh2, params->sh2, (n * 15u) * sizeof(float)), "gsx_tensor_upload(sh2,obj)")) {
            return false;
        }
    }
    if(opt->sh_degree >= 3) {
        if(!gsx_check(gsx_tensor_upload(app->sh3, params->sh3, (n * 21u) * sizeof(float)), "gsx_tensor_upload(sh3,obj)")) {
            return false;
        }
    }
    if(!gsx_check(gsx_tensor_upload(app->opacity, params->opacity, n * sizeof(float)), "gsx_tensor_upload(opacity,obj)")) {
        return false;
    }

    memset(&intrinsics, 0, sizeof(intrinsics));
    memset(&pose, 0, sizeof(pose));
    memset(&forward_request, 0, sizeof(forward_request));
    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.width = opt->width;
    intrinsics.height = opt->height;
    intrinsics.fx = opt->fx;
    intrinsics.fy = opt->fy;
    intrinsics.cx = opt->cx;
    intrinsics.cy = opt->cy;
    intrinsics.camera_id = opt->camera_id;
    pose.rot = opt->pose_rotation_xyzw;
    pose.transl = opt->pose_translation;
    pose.camera_id = opt->camera_id;
    pose.frame_id = opt->frame_id;
    forward_request.intrinsics = &intrinsics;
    forward_request.pose = &pose;
    forward_request.near_plane = opt->near_plane;
    forward_request.far_plane = opt->far_plane;
    forward_request.background_color = opt->background_color;
    forward_request.precision = opt->render_precision;
    forward_request.sh_degree = opt->sh_degree;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    forward_request.borrow_train_state = false;
    forward_request.gs_mean3d = app->mean3d;
    forward_request.gs_rotation = app->rotation;
    forward_request.gs_logscale = app->logscale;
    forward_request.gs_sh0 = app->sh0;
    forward_request.gs_sh1 = app->sh1;
    forward_request.gs_sh2 = app->sh2;
    forward_request.gs_sh3 = app->sh3;
    forward_request.gs_opacity = app->opacity;
    forward_request.out_rgb = app->out_rgb;
    if(!gsx_check(gsx_renderer_render(app->renderer, app->context, &forward_request), "gsx_renderer_render(obj)")) {
        return false;
    }
    if(!gsx_check(gsx_tensor_download(app->out_rgb, app->host_rgb, app->host_rgb_size_bytes), "gsx_tensor_download(out_rgb,obj)")) {
        return false;
    }
    *out_obj = dot_f32((const float *)app->host_rgb, grad_rgb_host, rgb_count);
    return true;
}

static bool run_numerical_diff_test(const app_options *options, app_state *state)
{
    gsx_tensor_t grad_rgb = NULL;
    gsx_tensor_t grad_mean3d = NULL;
    gsx_tensor_t grad_rotation = NULL;
    gsx_tensor_t grad_logscale = NULL;
    gsx_tensor_t grad_sh0 = NULL;
    gsx_tensor_t grad_sh1 = NULL;
    gsx_tensor_t grad_sh2 = NULL;
    gsx_tensor_t grad_sh3 = NULL;
    gsx_tensor_t grad_opacity = NULL;
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    gsx_render_forward_request forward_request;
    gsx_render_backward_request backward_request;
    gsx_index_t shape_out_rgb[3] = { 3, options->height, options->width };
    gsx_index_t shape_mean3d[2] = { options->gaussian_count, 3 };
    gsx_index_t shape_rotation[2] = { options->gaussian_count, 4 };
    gsx_index_t shape_logscale[2] = { options->gaussian_count, 3 };
    gsx_index_t shape_sh0[2] = { options->gaussian_count, 3 };
    gsx_index_t shape_sh1[3] = { options->gaussian_count, 3, 3 };
    gsx_index_t shape_sh2[3] = { options->gaussian_count, 5, 3 };
    gsx_index_t shape_sh3[3] = { options->gaussian_count, 7, 3 };
    gsx_index_t shape_opacity[1] = { options->gaussian_count };
    gaussian_params params;
    app_options diff_options;
    gaussian_params analytic_grad;
    float *grad_rgb_values = NULL;
    const gsx_size_t n = (gsx_size_t)options->gaussian_count;
    gsx_size_t rgb_count = (gsx_size_t)3u * (gsx_size_t)options->height * (gsx_size_t)options->width;
    uint32_t rng_state = options->numerical_diff_seed;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double max_abs_numeric = 0.0;
    double max_abs_analytic = 0.0;
    double max_rel_numeric = 0.0;
    double max_rel_analytic = 0.0;
    const char *max_param_name = "none";
    gsx_size_t max_param_index = 0;
    const char *max_rel_param_name = "none";
    gsx_size_t max_rel_param_index = 0;
    bool passed = true;
    bool intrinsics_adjusted = false;

    typedef struct param_view {
        const char *name;
        float *values;
        const float *analytic;
        gsx_size_t count;
    } param_view;
    param_view views[8];
    gsx_index_t view_count = 0;

    memset(&params, 0, sizeof(params));
    memset(&analytic_grad, 0, sizeof(analytic_grad));
    if(!options->numerical_diff_enable) {
        return true;
    }
    if(state->host_rgb == NULL) {
        fprintf(stderr, "error: rendered output is not available for numerical diff\n");
        return false;
    }

    if(!gaussian_params_copy_from_options(&params, options)) {
        fprintf(stderr, "error: allocation failed for numerical diff params\n");
        goto failed;
    }
    if(!gaussian_params_alloc(&analytic_grad, options->gaussian_count)) {
        fprintf(stderr, "error: allocation failed for numerical diff grad params\n");
        goto failed;
    }
    memset(analytic_grad.mean3d, 0, (size_t)(n * 3u) * sizeof(float));
    memset(analytic_grad.rotation, 0, (size_t)(n * 4u) * sizeof(float));
    memset(analytic_grad.logscale, 0, (size_t)(n * 3u) * sizeof(float));
    memset(analytic_grad.sh0, 0, (size_t)(n * 3u) * sizeof(float));
    memset(analytic_grad.sh1, 0, (size_t)(n * 9u) * sizeof(float));
    memset(analytic_grad.sh2, 0, (size_t)(n * 15u) * sizeof(float));
    memset(analytic_grad.sh3, 0, (size_t)(n * 21u) * sizeof(float));
    memset(analytic_grad.opacity, 0, (size_t)n * sizeof(float));
    configure_numerical_diff_params(options, &params);
    configure_numerical_diff_options(options, &diff_options, &intrinsics_adjusted);

    grad_rgb_values = (float *)malloc((size_t)rgb_count * sizeof(float));
    if(grad_rgb_values == NULL) {
        fprintf(stderr, "error: allocation failed for numerical diff grad image\n");
        goto failed;
    }
    for(gsx_size_t i = 0; i < rgb_count; ++i) {
        grad_rgb_values[i] = randn(&rng_state);
    }

    if(!init_tensor_f32(&grad_rgb, state->arena, 3, shape_out_rgb, grad_rgb_values, rgb_count)) {
        goto failed;
    }
    if(!init_tensor_f32(&grad_mean3d, state->arena, 2, shape_mean3d, NULL, 0)) {
        goto failed;
    }
    if(!init_tensor_f32(&grad_rotation, state->arena, 2, shape_rotation, NULL, 0)) {
        goto failed;
    }
    if(!init_tensor_f32(&grad_logscale, state->arena, 2, shape_logscale, NULL, 0)) {
        goto failed;
    }
    if(!init_tensor_f32(&grad_sh0, state->arena, 2, shape_sh0, NULL, 0)) {
        goto failed;
    }
    if(options->sh_degree >= 1) {
        if(!init_tensor_f32(&grad_sh1, state->arena, 3, shape_sh1, NULL, 0)) {
            goto failed;
        }
    }
    if(options->sh_degree >= 2) {
        if(!init_tensor_f32(&grad_sh2, state->arena, 3, shape_sh2, NULL, 0)) {
            goto failed;
        }
    }
    if(options->sh_degree >= 3) {
        if(!init_tensor_f32(&grad_sh3, state->arena, 3, shape_sh3, NULL, 0)) {
            goto failed;
        }
    }
    if(!init_tensor_f32(&grad_opacity, state->arena, 1, shape_opacity, NULL, 0)) {
        goto failed;
    }

    if(!gsx_check(gsx_tensor_upload(state->mean3d, params.mean3d, (n * 3u) * sizeof(float)), "gsx_tensor_upload(mean3d,base)")) {
        goto failed;
    }
    if(!gsx_check(gsx_tensor_upload(state->rotation, params.rotation, (n * 4u) * sizeof(float)), "gsx_tensor_upload(rotation,base)")) {
        goto failed;
    }
    if(!gsx_check(gsx_tensor_upload(state->logscale, params.logscale, (n * 3u) * sizeof(float)), "gsx_tensor_upload(logscale,base)")) {
        goto failed;
    }
    if(!gsx_check(gsx_tensor_upload(state->sh0, params.sh0, (n * 3u) * sizeof(float)), "gsx_tensor_upload(sh0,base)")) {
        goto failed;
    }
    if(options->sh_degree >= 1) {
        if(!gsx_check(gsx_tensor_upload(state->sh1, params.sh1, (n * 9u) * sizeof(float)), "gsx_tensor_upload(sh1,base)")) {
            goto failed;
        }
    }
    if(options->sh_degree >= 2) {
        if(!gsx_check(gsx_tensor_upload(state->sh2, params.sh2, (n * 15u) * sizeof(float)), "gsx_tensor_upload(sh2,base)")) {
            goto failed;
        }
    }
    if(options->sh_degree >= 3) {
        if(!gsx_check(gsx_tensor_upload(state->sh3, params.sh3, (n * 21u) * sizeof(float)), "gsx_tensor_upload(sh3,base)")) {
            goto failed;
        }
    }
    if(!gsx_check(gsx_tensor_upload(state->opacity, params.opacity, n * sizeof(float)), "gsx_tensor_upload(opacity,base)")) {
        goto failed;
    }

    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.width = diff_options.width;
    intrinsics.height = diff_options.height;
    intrinsics.fx = diff_options.fx;
    intrinsics.fy = diff_options.fy;
    intrinsics.cx = diff_options.cx;
    intrinsics.cy = diff_options.cy;
    intrinsics.camera_id = diff_options.camera_id;
    pose.rot = diff_options.pose_rotation_xyzw;
    pose.transl = diff_options.pose_translation;
    pose.camera_id = diff_options.camera_id;
    pose.frame_id = diff_options.frame_id;

    memset(&forward_request, 0, sizeof(forward_request));
    forward_request.intrinsics = &intrinsics;
    forward_request.pose = &pose;
    forward_request.near_plane = diff_options.near_plane;
    forward_request.far_plane = diff_options.far_plane;
    forward_request.background_color = diff_options.background_color;
    forward_request.precision = diff_options.render_precision;
    forward_request.sh_degree = diff_options.sh_degree;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    forward_request.borrow_train_state = false;
    forward_request.gs_mean3d = state->mean3d;
    forward_request.gs_rotation = state->rotation;
    forward_request.gs_logscale = state->logscale;
    forward_request.gs_sh0 = state->sh0;
    forward_request.gs_sh1 = state->sh1;
    forward_request.gs_sh2 = state->sh2;
    forward_request.gs_sh3 = state->sh3;
    forward_request.gs_opacity = state->opacity;
    forward_request.out_rgb = state->out_rgb;

    if(!gsx_check(gsx_renderer_render(state->renderer, state->context, &forward_request), "gsx_renderer_render(train)")) {
        goto failed;
    }

    memset(&backward_request, 0, sizeof(backward_request));
    backward_request.grad_rgb = grad_rgb;
    backward_request.grad_gs_mean3d = grad_mean3d;
    backward_request.grad_gs_rotation = grad_rotation;
    backward_request.grad_gs_logscale = grad_logscale;
    backward_request.grad_gs_sh0 = grad_sh0;
    backward_request.grad_gs_sh1 = grad_sh1;
    backward_request.grad_gs_sh2 = grad_sh2;
    backward_request.grad_gs_sh3 = grad_sh3;
    backward_request.grad_gs_opacity = grad_opacity;
    if(!gsx_check(gsx_renderer_backward(state->renderer, state->context, &backward_request), "gsx_renderer_backward")) {
        goto failed;
    }

    if(!gsx_check(gsx_tensor_download(grad_mean3d, analytic_grad.mean3d, (n * 3u) * sizeof(float)), "gsx_tensor_download(grad_mean3d)")) {
        goto failed;
    }
    if(!gsx_check(gsx_tensor_download(grad_rotation, analytic_grad.rotation, (n * 4u) * sizeof(float)), "gsx_tensor_download(grad_rotation)")) {
        goto failed;
    }
    if(!gsx_check(gsx_tensor_download(grad_logscale, analytic_grad.logscale, (n * 3u) * sizeof(float)), "gsx_tensor_download(grad_logscale)")) {
        goto failed;
    }
    if(!gsx_check(gsx_tensor_download(grad_sh0, analytic_grad.sh0, (n * 3u) * sizeof(float)), "gsx_tensor_download(grad_sh0)")) {
        goto failed;
    }
    if(options->sh_degree >= 1) {
        if(!gsx_check(gsx_tensor_download(grad_sh1, analytic_grad.sh1, (n * 9u) * sizeof(float)), "gsx_tensor_download(grad_sh1)")) {
            goto failed;
        }
    }
    if(options->sh_degree >= 2) {
        if(!gsx_check(gsx_tensor_download(grad_sh2, analytic_grad.sh2, (n * 15u) * sizeof(float)), "gsx_tensor_download(grad_sh2)")) {
            goto failed;
        }
    }
    if(options->sh_degree >= 3) {
        if(!gsx_check(gsx_tensor_download(grad_sh3, analytic_grad.sh3, (n * 21u) * sizeof(float)), "gsx_tensor_download(grad_sh3)")) {
            goto failed;
        }
    }
    if(!gsx_check(gsx_tensor_download(grad_opacity, analytic_grad.opacity, n * sizeof(float)), "gsx_tensor_download(grad_opacity)")) {
        goto failed;
    }

    views[view_count++] = (param_view){ "mean3d", params.mean3d, analytic_grad.mean3d, n * 3u };
    views[view_count++] = (param_view){ "rotation", params.rotation, analytic_grad.rotation, n * 4u };
    views[view_count++] = (param_view){ "logscale", params.logscale, analytic_grad.logscale, n * 3u };
    views[view_count++] = (param_view){ "sh0", params.sh0, analytic_grad.sh0, n * 3u };
    if(options->sh_degree >= 1) {
        views[view_count++] = (param_view){ "sh1", params.sh1, analytic_grad.sh1, n * 9u };
    }
    if(options->sh_degree >= 2) {
        views[view_count++] = (param_view){ "sh2", params.sh2, analytic_grad.sh2, n * 15u };
    }
    if(options->sh_degree >= 3) {
        views[view_count++] = (param_view){ "sh3", params.sh3, analytic_grad.sh3, n * 21u };
    }
    views[view_count++] = (param_view){ "opacity", params.opacity, analytic_grad.opacity, n };

    if(intrinsics_adjusted) {
        printf(
            "numerical diff diagnostic scene: sparse active subset with controlled depth spread\n"
            "numerical diff camera adjusted for hard-culling stability: render_fx=%.6f render_fy=%.6f render_cx=%.6f render_cy=%.6f diff_fx=%.6f diff_fy=%.6f diff_cx=%.6f diff_cy=%.6f\n",
            (double)options->fx,
            (double)options->fy,
            (double)options->cx,
            (double)options->cy,
            (double)diff_options.fx,
            (double)diff_options.fy,
            (double)diff_options.cx,
            (double)diff_options.cy
        );
    } else {
        printf(
            "numerical diff diagnostic scene: sparse active subset with controlled depth spread\n"
            "numerical diff camera: fx=%.6f fy=%.6f cx=%.6f cy=%.6f\n",
            (double)diff_options.fx,
            (double)diff_options.fy,
            (double)diff_options.cx,
            (double)diff_options.cy
        );
    }

    for(gsx_index_t v = 0; v < view_count; ++v) {
        for(gsx_size_t i = 0; i < views[v].count; ++i) {
            const float original = views[v].values[i];
            double plus = 0.0;
            double minus = 0.0;
            double numeric = 0.0;
            double abs_diff = 0.0;
            double rel_diff = 0.0;
            double scale = 0.0;
            double allowed = 0.0;

            views[v].values[i] = original + options->numerical_diff_eps;
            if(!evaluate_objective(&diff_options, state, &params, grad_rgb_values, rgb_count, &plus)) {
                goto failed;
            }
            views[v].values[i] = original - options->numerical_diff_eps;
            if(!evaluate_objective(&diff_options, state, &params, grad_rgb_values, rgb_count, &minus)) {
                goto failed;
            }
            views[v].values[i] = original;
            numeric = (plus - minus) / (2.0 * (double)options->numerical_diff_eps);
            abs_diff = fabs(numeric - (double)views[v].analytic[i]);
            scale = fmax(fabs(numeric), fabs((double)views[v].analytic[i]));
            if(scale <= 1.0e-20) {
                rel_diff = 0.0;
            } else {
                rel_diff = abs_diff / scale;
            }
            allowed = (double)options->numerical_diff_tol + (double)options->numerical_diff_rel_tol * scale;
            if(abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                max_abs_numeric = numeric;
                max_abs_analytic = (double)views[v].analytic[i];
                max_param_name = views[v].name;
                max_param_index = i;
            }
            if(rel_diff > max_rel_diff) {
                max_rel_diff = rel_diff;
                max_rel_numeric = numeric;
                max_rel_analytic = (double)views[v].analytic[i];
                max_rel_param_name = views[v].name;
                max_rel_param_index = i;
            }
            if(abs_diff > allowed) {
                passed = false;
            }
        }
    }

    printf(
        "numerical diff backward: eps=%.9g abs_tol=%.9g rel_tol=%.9g seed=%u max_abs=%.9e abs_param=%s abs_index=%llu abs_numeric=%.9e abs_analytic=%.9e max_rel=%.9e rel_param=%s rel_index=%llu rel_numeric=%.9e rel_analytic=%.9e\n",
        (double)options->numerical_diff_eps,
        (double)options->numerical_diff_tol,
        (double)options->numerical_diff_rel_tol,
        options->numerical_diff_seed,
        max_abs_diff,
        max_param_name,
        (unsigned long long)max_param_index,
        max_abs_numeric,
        max_abs_analytic,
        max_rel_diff,
        max_rel_param_name,
        (unsigned long long)max_rel_param_index,
        max_rel_numeric,
        max_rel_analytic
    );
    if(!passed) {
        fprintf(stderr, "FAILED: backward numerical diff exceeds abs/rel tolerance\n");
        goto failed;
    }
    printf("PASSED: backward numerical diff within tolerance\n");

    if(grad_opacity != NULL) {
        gsx_check(gsx_tensor_free(grad_opacity), "gsx_tensor_free(grad_opacity)");
    }
    if(grad_sh3 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh3), "gsx_tensor_free(grad_sh3)");
    }
    if(grad_sh2 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh2), "gsx_tensor_free(grad_sh2)");
    }
    if(grad_sh1 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh1), "gsx_tensor_free(grad_sh1)");
    }
    if(grad_sh0 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh0), "gsx_tensor_free(grad_sh0)");
    }
    if(grad_logscale != NULL) {
        gsx_check(gsx_tensor_free(grad_logscale), "gsx_tensor_free(grad_logscale)");
    }
    if(grad_rotation != NULL) {
        gsx_check(gsx_tensor_free(grad_rotation), "gsx_tensor_free(grad_rotation)");
    }
    if(grad_mean3d != NULL) {
        gsx_check(gsx_tensor_free(grad_mean3d), "gsx_tensor_free(grad_mean3d)");
    }
    if(grad_rgb != NULL) {
        gsx_check(gsx_tensor_free(grad_rgb), "gsx_tensor_free(grad_rgb)");
    }
    free(grad_rgb_values);
    gaussian_params_free(&analytic_grad);
    gaussian_params_free(&params);
    return true;

failed:
    if(grad_opacity != NULL) {
        gsx_check(gsx_tensor_free(grad_opacity), "gsx_tensor_free(grad_opacity)");
    }
    if(grad_sh3 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh3), "gsx_tensor_free(grad_sh3)");
    }
    if(grad_sh2 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh2), "gsx_tensor_free(grad_sh2)");
    }
    if(grad_sh1 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh1), "gsx_tensor_free(grad_sh1)");
    }
    if(grad_sh0 != NULL) {
        gsx_check(gsx_tensor_free(grad_sh0), "gsx_tensor_free(grad_sh0)");
    }
    if(grad_logscale != NULL) {
        gsx_check(gsx_tensor_free(grad_logscale), "gsx_tensor_free(grad_logscale)");
    }
    if(grad_rotation != NULL) {
        gsx_check(gsx_tensor_free(grad_rotation), "gsx_tensor_free(grad_rotation)");
    }
    if(grad_mean3d != NULL) {
        gsx_check(gsx_tensor_free(grad_mean3d), "gsx_tensor_free(grad_mean3d)");
    }
    if(grad_rgb != NULL) {
        gsx_check(gsx_tensor_free(grad_rgb), "gsx_tensor_free(grad_rgb)");
    }
    free(grad_rgb_values);
    gaussian_params_free(&analytic_grad);
    gaussian_params_free(&params);
    return false;
}

static bool init_tensor_f32(gsx_tensor_t *out_tensor, gsx_arena_t arena, gsx_index_t rank, const gsx_index_t *shape, const float *values, gsx_size_t value_count)
{
    gsx_tensor_desc desc;

    memset(&desc, 0, sizeof(desc));
    desc.rank = rank;
    for(gsx_index_t i = 0; i < rank; ++i) {
        desc.shape[i] = shape[i];
    }
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;

    if(!gsx_check(gsx_tensor_init(out_tensor, &desc), "gsx_tensor_init")) {
        return false;
    }
    if(values != NULL && value_count > 0) {
        if(!gsx_check(gsx_tensor_upload(*out_tensor, values, value_count * (gsx_size_t)sizeof(float)), "gsx_tensor_upload")) {
            return false;
        }
    } else {
        if(!gsx_check(gsx_tensor_set_zero(*out_tensor), "gsx_tensor_set_zero")) {
            return false;
        }
    }
    return true;
}

static bool run_render(const app_options *options, app_state *state)
{
    gsx_index_t visible_device_count = 0;
    gsx_backend_device_t device = NULL;
    gsx_backend_desc backend_desc;
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_desc arena_desc;
    gsx_renderer_desc renderer_desc;
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    gsx_render_forward_request request;
    gsx_index_t shape_mean3d[2] = { options->gaussian_count, 3 };
    gsx_index_t shape_rotation[2] = { options->gaussian_count, 4 };
    gsx_index_t shape_logscale[2] = { options->gaussian_count, 3 };
    gsx_index_t shape_sh0[2] = { options->gaussian_count, 3 };
    gsx_index_t shape_sh1[3] = { options->gaussian_count, 3, 3 };
    gsx_index_t shape_sh2[3] = { options->gaussian_count, 5, 3 };
    gsx_index_t shape_sh3[3] = { options->gaussian_count, 7, 3 };
    gsx_index_t shape_opacity[1] = { options->gaussian_count };
    gsx_index_t shape_out_rgb[3] = { 3, 0, 0 };
    gsx_size_t gaussian_count = (gsx_size_t)options->gaussian_count;
    gsx_size_t estimated_rgb_bytes = 0;
    gsx_tensor_info out_rgb_info;
    gsx_renderer_feature_flags renderer_feature_flags = 0u;

    memset(state, 0, sizeof(*state));
    memset(&backend_desc, 0, sizeof(backend_desc));
    memset(&arena_desc, 0, sizeof(arena_desc));
    memset(&renderer_desc, 0, sizeof(renderer_desc));
    memset(&intrinsics, 0, sizeof(intrinsics));
    memset(&pose, 0, sizeof(pose));
    memset(&request, 0, sizeof(request));
    memset(&out_rgb_info, 0, sizeof(out_rgb_info));

    if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
        return false;
    }
    if(!gsx_check(gsx_count_backend_devices_by_type(options->backend_type, &visible_device_count), "gsx_count_backend_devices_by_type")) {
        return false;
    }
    if(visible_device_count <= 0) {
        fprintf(stderr, "error: backend '%s' has no visible devices\n", backend_type_name(options->backend_type));
        return false;
    }
    if(options->device_index < 0 || options->device_index >= visible_device_count) {
        fprintf(
            stderr,
            "error: device index %lld out of range [0, %lld] for backend '%s'\n",
            (long long)options->device_index,
            (long long)(visible_device_count - 1),
            backend_type_name(options->backend_type)
        );
        return false;
    }
    if(!gsx_check(
           gsx_get_backend_device_by_type(options->backend_type, options->device_index, &device),
           "gsx_get_backend_device_by_type")) {
        return false;
    }

    backend_desc.device = device;
    renderer_feature_flags = options->renderer_feature_flags;
    if(options->numerical_diff_enable) {
        renderer_feature_flags |= GSX_RENDERER_FEATURE_DEBUG;
    }
    if(!gsx_check(gsx_backend_init(&state->backend, &backend_desc), "gsx_backend_init")) {
        return false;
    }
    if(!gsx_check(gsx_backend_find_buffer_type(state->backend, options->buffer_type_class, &buffer_type), "gsx_backend_find_buffer_type")) {
        return false;
    }

    estimated_rgb_bytes = (gsx_size_t)options->width * (gsx_size_t)options->height * 3u * (gsx_size_t)sizeof(float);
    arena_desc.initial_capacity_bytes = estimated_rgb_bytes + (1u << 20);
    if(options->numerical_diff_enable) {
        arena_desc.initial_capacity_bytes = (estimated_rgb_bytes * 2u) + (2u << 20);
    }
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    if(!gsx_check(gsx_arena_init(&state->arena, buffer_type, &arena_desc), "gsx_arena_init")) {
        return false;
    }

    renderer_desc.width = options->width;
    renderer_desc.height = options->height;
    renderer_desc.output_data_type = options->renderer_output_data_type;
    renderer_desc.feature_flags = renderer_feature_flags;
    renderer_desc.enable_alpha_output = options->renderer_enable_alpha_output;
    renderer_desc.enable_invdepth_output = options->renderer_enable_invdepth_output;
    if(!gsx_check(gsx_renderer_init(&state->renderer, state->backend, &renderer_desc), "gsx_renderer_init")) {
        return false;
    }
    if(!gsx_check(gsx_render_context_init(&state->context, state->renderer), "gsx_render_context_init")) {
        return false;
    }

    shape_out_rgb[1] = options->height;
    shape_out_rgb[2] = options->width;
    if(!init_tensor_f32(&state->mean3d, state->arena, 2, shape_mean3d, options->gs_mean3d, gaussian_count * 3u)) {
        return false;
    }
    if(!init_tensor_f32(&state->rotation, state->arena, 2, shape_rotation, options->gs_rotation, gaussian_count * 4u)) {
        return false;
    }
    if(!init_tensor_f32(&state->logscale, state->arena, 2, shape_logscale, options->gs_logscale, gaussian_count * 3u)) {
        return false;
    }
    if(!init_tensor_f32(&state->sh0, state->arena, 2, shape_sh0, options->gs_sh0, gaussian_count * 3u)) {
        return false;
    }
    if(options->sh_degree >= 1) {
        if(!init_tensor_f32(&state->sh1, state->arena, 3, shape_sh1, options->gs_sh1, gaussian_count * 9u)) {
            return false;
        }
    }
    if(options->sh_degree >= 2) {
        if(!init_tensor_f32(&state->sh2, state->arena, 3, shape_sh2, options->gs_sh2, gaussian_count * 15u)) {
            return false;
        }
    }
    if(options->sh_degree >= 3) {
        if(!init_tensor_f32(&state->sh3, state->arena, 3, shape_sh3, options->gs_sh3, gaussian_count * 21u)) {
            return false;
        }
    }
    if(!init_tensor_f32(&state->opacity, state->arena, 1, shape_opacity, options->gs_opacity, gaussian_count)) {
        return false;
    }
    if(!init_tensor_f32(&state->out_rgb, state->arena, 3, shape_out_rgb, NULL, 0)) {
        return false;
    }

    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.width = options->width;
    intrinsics.height = options->height;
    intrinsics.fx = options->fx;
    intrinsics.fy = options->fy;
    intrinsics.cx = options->cx;
    intrinsics.cy = options->cy;
    intrinsics.camera_id = options->camera_id;
    pose.rot = options->pose_rotation_xyzw;
    pose.transl = options->pose_translation;
    pose.camera_id = options->camera_id;
    pose.frame_id = options->frame_id;

    request.intrinsics = &intrinsics;
    request.pose = &pose;
    request.near_plane = options->near_plane;
    request.far_plane = options->far_plane;
    request.background_color = options->background_color;
    request.precision = options->render_precision;
    request.sh_degree = options->sh_degree;
    request.forward_type = options->render_forward_type;
    request.borrow_train_state = false;
    request.gs_mean3d = state->mean3d;
    request.gs_rotation = state->rotation;
    request.gs_logscale = state->logscale;
    request.gs_sh0 = state->sh0;
    request.gs_sh1 = state->sh1;
    request.gs_sh2 = state->sh2;
    request.gs_sh3 = state->sh3;
    request.gs_opacity = state->opacity;
    request.out_rgb = state->out_rgb;

    if(!gsx_check(gsx_renderer_render(state->renderer, state->context, &request), "gsx_renderer_render")) {
        return false;
    }

    if(!gsx_check(gsx_tensor_get_info(state->out_rgb, &out_rgb_info), "gsx_tensor_get_info(out_rgb)")) {
        return false;
    }
    state->host_rgb = malloc((size_t)out_rgb_info.size_bytes);
    if(state->host_rgb == NULL) {
        fprintf(stderr, "error: host allocation for output image failed (%llu bytes)\n", (unsigned long long)out_rgb_info.size_bytes);
        return false;
    }
    state->host_rgb_size_bytes = out_rgb_info.size_bytes;
    if(!gsx_check(gsx_tensor_download(state->out_rgb, state->host_rgb, out_rgb_info.size_bytes), "gsx_tensor_download(out_rgb)")) {
        return false;
    }

    return true;
}

static void cleanup_state(app_state *state)
{
    if(state->host_rgb != NULL) {
        free(state->host_rgb);
        state->host_rgb = NULL;
    }
    if(state->out_rgb != NULL) {
        gsx_check(gsx_tensor_free(state->out_rgb), "gsx_tensor_free(out_rgb)");
        state->out_rgb = NULL;
    }
    if(state->opacity != NULL) {
        gsx_check(gsx_tensor_free(state->opacity), "gsx_tensor_free(opacity)");
        state->opacity = NULL;
    }
    if(state->sh3 != NULL) {
        gsx_check(gsx_tensor_free(state->sh3), "gsx_tensor_free(sh3)");
        state->sh3 = NULL;
    }
    if(state->sh2 != NULL) {
        gsx_check(gsx_tensor_free(state->sh2), "gsx_tensor_free(sh2)");
        state->sh2 = NULL;
    }
    if(state->sh1 != NULL) {
        gsx_check(gsx_tensor_free(state->sh1), "gsx_tensor_free(sh1)");
        state->sh1 = NULL;
    }
    if(state->sh0 != NULL) {
        gsx_check(gsx_tensor_free(state->sh0), "gsx_tensor_free(sh0)");
        state->sh0 = NULL;
    }
    if(state->logscale != NULL) {
        gsx_check(gsx_tensor_free(state->logscale), "gsx_tensor_free(logscale)");
        state->logscale = NULL;
    }
    if(state->rotation != NULL) {
        gsx_check(gsx_tensor_free(state->rotation), "gsx_tensor_free(rotation)");
        state->rotation = NULL;
    }
    if(state->mean3d != NULL) {
        gsx_check(gsx_tensor_free(state->mean3d), "gsx_tensor_free(mean3d)");
        state->mean3d = NULL;
    }
    if(state->arena != NULL) {
        gsx_check(gsx_arena_free(state->arena), "gsx_arena_free");
        state->arena = NULL;
    }
    if(state->context != NULL) {
        gsx_check(gsx_render_context_free(state->context), "gsx_render_context_free");
        state->context = NULL;
    }
    if(state->renderer != NULL) {
        gsx_check(gsx_renderer_free(state->renderer), "gsx_renderer_free");
        state->renderer = NULL;
    }
    if(state->backend != NULL) {
        gsx_check(gsx_backend_free(state->backend), "gsx_backend_free");
        state->backend = NULL;
    }
}

int main(int argc, char **argv)
{
    app_options options;
    app_state state;
    int exit_code = EXIT_FAILURE;

    set_default_options(&options);
    memset(&state, 0, sizeof(state));
    if(!parse_args(argc, argv, &options)) {
        print_usage(argv[0]);
        goto cleanup;
    }
    if(options.near_plane <= 0.0f || options.far_plane <= options.near_plane) {
        fprintf(stderr, "error: near/far must satisfy 0 < near < far\n");
        goto cleanup;
    }
    if(options.output_path == NULL || options.output_path[0] == '\0') {
        fprintf(stderr, "error: output path must be non-empty\n");
        goto cleanup;
    }
    if(!initialize_gaussian_params(&options)) {
        goto cleanup;
    }

    if(!run_render(&options, &state)) {
        goto cleanup;
    }
    if(!write_render_output(&options, &state, options.output_path)) {
        goto cleanup;
    }
    if(!compare_against_reference_image(&options, &state)) {
        goto cleanup;
    }
    if(!compare_against_cpu_reference(&options, &state)) {
        goto cleanup;
    }
    if(!run_numerical_diff_test(&options, &state)) {
        goto cleanup;
    }
    printf(
        "rendered %lld gaussians to '%s' (backend=%s device=%lld size=%lldx%lld sh_degree=%lld)\n",
        (long long)options.gaussian_count,
        options.output_path,
        backend_type_name(options.backend_type),
        (long long)options.device_index,
        (long long)options.width,
        (long long)options.height,
        (long long)options.sh_degree
    );
    exit_code = EXIT_SUCCESS;

cleanup:
    cleanup_state(&state);
    free_options(&options);
    return exit_code;
}
