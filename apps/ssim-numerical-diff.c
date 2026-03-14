#include <gsx/gsx.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct cli_options {
    double eps;
    double tol;
    gsx_index_t channels;
    gsx_index_t height;
    gsx_index_t width;
    gsx_storage_format storage_format;
} cli_options;

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

static bool parse_double_value(const char *value, double *out_value)
{
    char *end = NULL;
    double parsed = 0.0;

    errno = 0;
    parsed = strtod(value, &end);
    if(errno != 0 || end == value || *end != '\0' || !isfinite(parsed)) {
        return false;
    }
    *out_value = parsed;
    return true;
}

static bool parse_index_value(const char *value, gsx_index_t *out_value)
{
    char *end = NULL;
    long long parsed = 0;

    errno = 0;
    parsed = strtoll(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0' || parsed <= 0) {
        return false;
    }
    *out_value = (gsx_index_t)parsed;
    if((long long)*out_value != parsed) {
        return false;
    }
    return true;
}

static bool parse_storage_format_value(const char *value, gsx_storage_format *out_format)
{
    if(strcmp(value, "chw") == 0) {
        *out_format = GSX_STORAGE_FORMAT_CHW;
        return true;
    }
    if(strcmp(value, "hwc") == 0) {
        *out_format = GSX_STORAGE_FORMAT_HWC;
        return true;
    }

    return false;
}

static const char *storage_format_name(gsx_storage_format storage_format)
{
    if(storage_format == GSX_STORAGE_FORMAT_HWC) {
        return "HWC";
    }

    return "CHW";
}

static void print_usage(const char *program_name)
{
    fprintf(
        stderr,
        "usage: %s [--eps <value>] [--tol <value>] [--channels <int>] [--height <int>] [--width <int>] [--layout <chw|hwc>]\n",
        program_name
    );
}

static bool parse_cli_options(int argc, char **argv, cli_options *out_options)
{
    cli_options options = { 1e-3, 1e-3, 1, 3, 3, GSX_STORAGE_FORMAT_CHW };

    for(int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if(strcmp(arg, "--eps") == 0) {
            if(i + 1 >= argc || !parse_double_value(argv[i + 1], &options.eps) || options.eps <= 0.0) {
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--tol") == 0) {
            if(i + 1 >= argc || !parse_double_value(argv[i + 1], &options.tol) || options.tol <= 0.0) {
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        if(strcmp(arg, "--channels") == 0) {
            if(i + 1 >= argc || !parse_index_value(argv[i + 1], &options.channels)) {
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--height") == 0) {
            if(i + 1 >= argc || !parse_index_value(argv[i + 1], &options.height)) {
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--width") == 0) {
            if(i + 1 >= argc || !parse_index_value(argv[i + 1], &options.width)) {
                return false;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--layout") == 0) {
            if(i + 1 >= argc || !parse_storage_format_value(argv[i + 1], &options.storage_format)) {
                return false;
            }
            ++i;
            continue;
        }
        return false;
    }
    *out_options = options;
    return true;
}

static bool multiply_size_checked(gsx_size_t lhs, gsx_size_t rhs, gsx_size_t *out_product)
{
    if(lhs != 0 && rhs > ((gsx_size_t)-1) / lhs) {
        return false;
    }
    *out_product = lhs * rhs;
    return true;
}

static gsx_size_t image_index(
    gsx_storage_format storage_format,
    gsx_size_t c,
    gsx_size_t y,
    gsx_size_t x,
    gsx_size_t channels,
    gsx_size_t height,
    gsx_size_t width
)
{
    (void)height;
    if(storage_format == GSX_STORAGE_FORMAT_HWC) {
        return ((y * width + x) * channels + c);
    }

    return ((c * height + y) * width + x);
}

static bool make_input_values(
    float *out_prediction,
    float *out_target,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    gsx_storage_format storage_format)
{
    gsx_size_t c = 0;
    gsx_size_t y = 0;
    gsx_size_t x = 0;
    gsx_size_t index = 0;
    const gsx_size_t c_count = (gsx_size_t)channels;
    const gsx_size_t h_count = (gsx_size_t)height;
    const gsx_size_t w_count = (gsx_size_t)width;

    for(c = 0; c < c_count; ++c) {
        for(y = 0; y < h_count; ++y) {
            for(x = 0; x < w_count; ++x) {
                gsx_size_t value_index = image_index(storage_format, c, y, x, c_count, h_count, w_count);
                const double p_raw = 0.11 * (double)(c + 1) + 0.07 * (double)y + 0.03 * (double)x;
                const double t_raw = p_raw + 0.02 * sin((double)(index + 1));
                const double p_norm = fmod(p_raw, 1.0);
                const double t_norm = fmod(t_raw, 1.0);

                out_prediction[value_index] = (float)(p_norm < 0.0 ? p_norm + 1.0 : p_norm);
                out_target[value_index] = (float)(t_norm < 0.0 ? t_norm + 1.0 : t_norm);
                ++index;
            }
        }
    }
    return true;
}

static double sum_values(const float *values, gsx_size_t count)
{
    double sum = 0.0;
    for(gsx_size_t i = 0; i < count; ++i) {
        sum += (double)values[i];
    }
    return sum;
}

int main(int argc, char **argv)
{
    cli_options options = { 1e-3, 1e-3, 1, 3, 3, GSX_STORAGE_FORMAT_CHW };
    int exit_code = EXIT_FAILURE;
    gsx_backend_device_t device = NULL;
    gsx_backend_t backend = NULL;
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_t arena = NULL;
    gsx_loss_t loss = NULL;
    gsx_tensor_t prediction = NULL;
    gsx_tensor_t target = NULL;
    gsx_tensor_t loss_map = NULL;
    gsx_tensor_t grad = NULL;
    gsx_backend_desc backend_desc = { 0 };
    gsx_arena_desc arena_desc = { 0 };
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_loss_desc loss_desc = { 0 };
    gsx_loss_request request = { 0 };
    gsx_size_t element_count = 0;
    gsx_size_t tensor_bytes = 0;
    float *base_prediction = NULL;
    float *target_values = NULL;
    float *prediction_work = NULL;
    float *grad_values = NULL;
    float *loss_values = NULL;
    float *zero_values = NULL;
    double objective_scale = 0.0;
    double max_abs_diff = 0.0;
    gsx_size_t max_abs_diff_index = 0;

    if(!parse_cli_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if(!multiply_size_checked((gsx_size_t)options.channels, (gsx_size_t)options.height, &element_count)
        || !multiply_size_checked(element_count, (gsx_size_t)options.width, &element_count)
        || !multiply_size_checked(element_count, sizeof(float), &tensor_bytes)
        || element_count == 0
    ) {
        fprintf(stderr, "error: invalid tensor shape\n");
        goto cleanup;
    }
    base_prediction = (float *)calloc(element_count, sizeof(float));
    target_values = (float *)calloc(element_count, sizeof(float));
    prediction_work = (float *)calloc(element_count, sizeof(float));
    grad_values = (float *)calloc(element_count, sizeof(float));
    loss_values = (float *)calloc(element_count, sizeof(float));
    zero_values = (float *)calloc(element_count, sizeof(float));
    if(base_prediction == NULL || target_values == NULL || prediction_work == NULL || grad_values == NULL || loss_values == NULL
        || zero_values == NULL
    ) {
        fprintf(stderr, "error: out of memory\n");
        goto cleanup;
    }
    objective_scale = 1.0 / (double)element_count;
    if(!make_input_values(
           base_prediction,
           target_values,
           options.channels,
           options.height,
           options.width,
           options.storage_format)) {
        fprintf(stderr, "error: failed to build input tensors\n");
        goto cleanup;
    }
    if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &device), "gsx_get_backend_device_by_type")) {
        goto cleanup;
    }
    backend_desc.device = device;
    if(!gsx_check(gsx_backend_init(&backend, &backend_desc), "gsx_backend_init")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type), "gsx_backend_find_buffer_type")) {
        goto cleanup;
    }
    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    if(!gsx_check(gsx_arena_init(&arena, buffer_type, &arena_desc), "gsx_arena_init")) {
        goto cleanup;
    }

    tensor_desc.rank = 3;
    if(options.storage_format == GSX_STORAGE_FORMAT_HWC) {
        tensor_desc.shape[0] = options.height;
        tensor_desc.shape[1] = options.width;
        tensor_desc.shape[2] = options.channels;
    } else {
        tensor_desc.shape[0] = options.channels;
        tensor_desc.shape[1] = options.height;
        tensor_desc.shape[2] = options.width;
    }
    tensor_desc.data_type = GSX_DATA_TYPE_F32;
    tensor_desc.storage_format = options.storage_format;
    tensor_desc.arena = arena;
    if(!gsx_check(gsx_tensor_init(&prediction, &tensor_desc), "gsx_tensor_init(prediction)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_init(&target, &tensor_desc), "gsx_tensor_init(target)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_init(&loss_map, &tensor_desc), "gsx_tensor_init(loss_map)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_init(&grad, &tensor_desc), "gsx_tensor_init(grad)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_upload(prediction, base_prediction, tensor_bytes), "gsx_tensor_upload(prediction)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_upload(target, target_values, tensor_bytes), "gsx_tensor_upload(target)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_upload(loss_map, zero_values, tensor_bytes), "gsx_tensor_upload(loss_map)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_upload(grad, zero_values, tensor_bytes), "gsx_tensor_upload(grad)")) {
        goto cleanup;
    }

    loss_desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    loss_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    if(!gsx_check(gsx_loss_init(&loss, backend, &loss_desc), "gsx_loss_init")) {
        goto cleanup;
    }

    request.prediction = prediction;
    request.target = target;
    request.loss_map_accumulator = loss_map;
    request.grad_prediction_accumulator = grad;
    request.scale = 1.0f;
    if(!gsx_check(gsx_loss_evaluate(loss, &request), "gsx_loss_evaluate(analytic)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_download(grad, grad_values, tensor_bytes), "gsx_tensor_download(grad)")) {
        goto cleanup;
    }

    printf(
        "SSIM numerical diff check (cpu, %s CxHxW=%lldx%lldx%lld)\n",
        storage_format_name(options.storage_format),
        (long long)options.channels,
        (long long)options.height,
        (long long)options.width);
    printf("eps=%.9g tol=%.9g\n", options.eps, options.tol);
    printf("index analytic finite_diff abs_diff\n");

    for(gsx_size_t i = 0; i < element_count; ++i) {
        const double base = (double)base_prediction[i];
        double plus = 0.0;
        double minus = 0.0;
        double finite_diff = 0.0;
        double abs_diff = 0.0;

        memcpy(prediction_work, base_prediction, tensor_bytes);
        prediction_work[i] = (float)(base + options.eps);
        if(!gsx_check(gsx_tensor_upload(prediction, prediction_work, tensor_bytes), "gsx_tensor_upload(prediction + eps)")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_tensor_upload(loss_map, zero_values, tensor_bytes), "gsx_tensor_reset(loss_map, +eps)")) {
            goto cleanup;
        }
        request.grad_prediction_accumulator = NULL;
        if(!gsx_check(gsx_loss_evaluate(loss, &request), "gsx_loss_evaluate(+eps)")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_tensor_download(loss_map, loss_values, tensor_bytes), "gsx_tensor_download(loss_map + eps)")) {
            goto cleanup;
        }
        plus = sum_values(loss_values, element_count) * objective_scale;

        prediction_work[i] = (float)(base - options.eps);
        if(!gsx_check(gsx_tensor_upload(prediction, prediction_work, tensor_bytes), "gsx_tensor_upload(prediction - eps)")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_tensor_upload(loss_map, zero_values, tensor_bytes), "gsx_tensor_reset(loss_map, -eps)")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_loss_evaluate(loss, &request), "gsx_loss_evaluate(-eps)")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_tensor_download(loss_map, loss_values, tensor_bytes), "gsx_tensor_download(loss_map - eps)")) {
            goto cleanup;
        }
        minus = sum_values(loss_values, element_count) * objective_scale;

        finite_diff = (plus - minus) / (2.0 * options.eps);
        abs_diff = fabs(finite_diff - (double)grad_values[i]);
        if(abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_abs_diff_index = i;
        }
        printf("%5llu %+.9e %+.9e %.9e\n", (unsigned long long)i, (double)grad_values[i], finite_diff, abs_diff);
    }

    printf("max_abs_diff=%.9e at index=%llu\n", max_abs_diff, (unsigned long long)max_abs_diff_index);
    if(max_abs_diff > options.tol) {
        fprintf(stderr, "FAILED: max_abs_diff exceeds tolerance\n");
        goto cleanup;
    }
    printf("PASSED\n");
    exit_code = EXIT_SUCCESS;

cleanup:
    if(loss != NULL) {
        gsx_check(gsx_loss_free(loss), "gsx_loss_free");
    }
    if(grad != NULL) {
        gsx_check(gsx_tensor_free(grad), "gsx_tensor_free(grad)");
    }
    if(loss_map != NULL) {
        gsx_check(gsx_tensor_free(loss_map), "gsx_tensor_free(loss_map)");
    }
    if(target != NULL) {
        gsx_check(gsx_tensor_free(target), "gsx_tensor_free(target)");
    }
    if(prediction != NULL) {
        gsx_check(gsx_tensor_free(prediction), "gsx_tensor_free(prediction)");
    }
    if(arena != NULL) {
        gsx_check(gsx_arena_free(arena), "gsx_arena_free");
    }
    if(backend != NULL) {
        gsx_check(gsx_backend_free(backend), "gsx_backend_free");
    }
    free(zero_values);
    free(loss_values);
    free(grad_values);
    free(prediction_work);
    free(target_values);
    free(base_prediction);
    return exit_code;
}
