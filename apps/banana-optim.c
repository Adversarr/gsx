#include <gsx/gsx.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if GSX_HAS_CUDA
#include <cuda_runtime.h>
#endif

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

static const char *backend_type_name(enum gsx_backend_type backend_type)
{
    switch(backend_type) {
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

static bool parse_backend_type(const char *value, enum gsx_backend_type *out_backend_type)
{
    if(strcmp(value, "cpu") == 0) {
        *out_backend_type = GSX_BACKEND_TYPE_CPU;
        return true;
    }
    if(strcmp(value, "cuda") == 0) {
        *out_backend_type = GSX_BACKEND_TYPE_CUDA;
        return true;
    }
    return false;
}

static bool parse_device_index(const char *value, gsx_index_t *out_device_index)
{
    char *end = NULL;
    long parsed = 0;
    errno = 0;
    parsed = strtol(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0' || parsed < 0) {
        return false;
    }
    *out_device_index = (gsx_index_t)parsed;
    if((long)*out_device_index != parsed) {
        return false;
    }
    return true;
}

static void print_usage(const char *program_name)
{
    fprintf(stderr, "usage: %s [--backend cpu|cuda] [--device <index>]\n", program_name);
}

static float banana_loss(float x, float y)
{
    const float dx = 1.0f - x;
    const float t = y - x * x;
    return dx * dx + 100.0f * t * t;
}

static void banana_grad(float x, float y, float *out_gx, float *out_gy)
{
    const float t = y - x * x;
    *out_gx = 2.0f * (x - 1.0f) - 400.0f * x * t;
    *out_gy = 200.0f * t;
}

static bool sync_backend_if_cuda(gsx_backend_t backend)
{
#if GSX_HAS_CUDA
    gsx_backend_info backend_info = {0};
    void *stream = NULL;

    if(!gsx_check(gsx_backend_get_info(backend, &backend_info), "gsx_backend_get_info")) {
        return false;
    }
    if(backend_info.backend_type != GSX_BACKEND_TYPE_CUDA) {
        return true;
    }
    if(!gsx_check(gsx_backend_get_major_stream(backend, &stream), "gsx_backend_get_major_stream")) {
        return false;
    }
    if(cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess) {
        fprintf(stderr, "error: cudaStreamSynchronize failed\n");
        return false;
    }
#else
    (void)backend;
#endif
    return true;
}

int main(int argc, char **argv)
{
    const float init_params[2] = { -0.8f, 0.8f };
    const int max_steps = 3000;
    const int log_interval = 200;
    enum gsx_backend_type selected_backend_type = GSX_BACKEND_TYPE_CPU;
    gsx_index_t selected_device_index = 0;
    gsx_index_t selected_backend_device_count = 0;
    int exit_code = EXIT_FAILURE;
    gsx_backend_device_t device = NULL;
    gsx_backend_t backend = NULL;
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_t arena = NULL;
    gsx_tensor_t params = NULL;
    gsx_tensor_t grads = NULL;
    gsx_optim_t optim = NULL;
    gsx_backend_desc backend_desc = {0};
    gsx_arena_desc arena_desc = {0};
    gsx_tensor_desc tensor_desc = {0};
    gsx_optim_param_group_desc group_desc = {0};
    gsx_optim_desc optim_desc = {0};
    gsx_optim_step_request step_request = {0};
    float host_params[2] = {0.0f, 0.0f};
    float host_grads[2] = {0.0f, 0.0f};

    for(int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if(strcmp(arg, "--backend") == 0) {
            if(i + 1 >= argc || !parse_backend_type(argv[i + 1], &selected_backend_type)) {
                fprintf(stderr, "error: invalid backend '%s'\n", (i + 1 < argc) ? argv[i + 1] : "");
                print_usage(argv[0]);
                goto cleanup;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--device") == 0) {
            if(i + 1 >= argc || !parse_device_index(argv[i + 1], &selected_device_index)) {
                fprintf(stderr, "error: invalid device index '%s'\n", (i + 1 < argc) ? argv[i + 1] : "");
                print_usage(argv[0]);
                goto cleanup;
            }
            ++i;
            continue;
        }
        if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            exit_code = EXIT_SUCCESS;
            goto cleanup;
        }
        fprintf(stderr, "error: unknown argument '%s'\n", arg);
        print_usage(argv[0]);
        goto cleanup;
    }

    if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_count_backend_devices_by_type(selected_backend_type, &selected_backend_device_count), "gsx_count_backend_devices_by_type")) {
        goto cleanup;
    }
    if(selected_backend_device_count <= 0) {
        fprintf(stderr, "error: backend '%s' has no available devices\n", backend_type_name(selected_backend_type));
        goto cleanup;
    }
    if(selected_device_index >= selected_backend_device_count) {
        fprintf(
            stderr,
            "error: backend '%s' device index %lld out of range [0, %lld]\n",
            backend_type_name(selected_backend_type),
            (long long)selected_device_index,
            (long long)(selected_backend_device_count - 1)
        );
        goto cleanup;
    }
    if(!gsx_check(gsx_get_backend_device_by_type(selected_backend_type, selected_device_index, &device), "gsx_get_backend_device_by_type")) {
        goto cleanup;
    }
    printf("using backend=%s device=%lld\n", backend_type_name(selected_backend_type), (long long)selected_device_index);
    backend_desc.device = device;
    if(!gsx_check(gsx_backend_init(&backend, &backend_desc), "gsx_backend_init")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type), "gsx_backend_find_buffer_type")) {
        goto cleanup;
    }

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    if(!gsx_check(gsx_arena_init(&arena, buffer_type, &arena_desc), "gsx_arena_init")) {
        goto cleanup;
    }

    tensor_desc.rank = 1;
    tensor_desc.shape[0] = 2;
    tensor_desc.data_type = GSX_DATA_TYPE_F32;
    tensor_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    tensor_desc.arena = arena;
    if(!gsx_check(gsx_tensor_init(&params, &tensor_desc), "gsx_tensor_init(params)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_init(&grads, &tensor_desc), "gsx_tensor_init(grads)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_upload(params, init_params, sizeof(init_params)), "gsx_tensor_upload(params)")) {
        goto cleanup;
    }
    if(!gsx_check(gsx_tensor_set_zero(grads), "gsx_tensor_set_zero(grads)")) {
        goto cleanup;
    }

    group_desc.role = GSX_OPTIM_PARAM_ROLE_CUSTOM;
    group_desc.label = "banana-xy";
    group_desc.parameter = params;
    group_desc.gradient = grads;
    group_desc.learning_rate = 0.02f;
    group_desc.beta1 = 0.9f;
    group_desc.beta2 = 0.999f;
    group_desc.weight_decay = 0.0f;
    group_desc.epsilon = 1e-8f;
    group_desc.max_grad = 0.0f;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group_desc;
    optim_desc.param_group_count = 1;
    if(!gsx_check(gsx_optim_init(&optim, backend, &optim_desc), "gsx_optim_init")) {
        goto cleanup;
    }

    step_request.force_all = true;

    for(int step = 1; step <= max_steps; ++step) {
        if(!gsx_check(gsx_tensor_download(params, host_params, sizeof(host_params)), "gsx_tensor_download(params)")) {
            goto cleanup;
        }
        if(!sync_backend_if_cuda(backend)) {
            goto cleanup;
        }
        banana_grad(host_params[0], host_params[1], &host_grads[0], &host_grads[1]);
        if(!gsx_check(gsx_tensor_upload(grads, host_grads, sizeof(host_grads)), "gsx_tensor_upload(grads)")) {
            goto cleanup;
        }
        if(!gsx_check(gsx_optim_step(optim, &step_request), "gsx_optim_step")) {
            goto cleanup;
        }
        if(step == 1 || step % log_interval == 0 || step == max_steps) {
            const float loss = banana_loss(host_params[0], host_params[1]);
            printf(
                "step=%d x=%.6f y=%.6f f=%.8f grad=(%.6f, %.6f)\n",
                step,
                host_params[0],
                host_params[1],
                loss,
                host_grads[0],
                host_grads[1]
            );
        }
    }

    if(!gsx_check(gsx_tensor_download(params, host_params, sizeof(host_params)), "gsx_tensor_download(final params)")) {
        goto cleanup;
    }
    if(!sync_backend_if_cuda(backend)) {
        goto cleanup;
    }
    {
        const float loss = banana_loss(host_params[0], host_params[1]);
        const float dx = host_params[0] - 1.0f;
        const float dy = host_params[1] - 1.0f;
        const float distance = sqrtf(dx * dx + dy * dy);
        printf("\nresult: x=%.6f y=%.6f f=%.10f distance_to_gt=%.8f\n", host_params[0], host_params[1], loss, distance);
    }

    exit_code = EXIT_SUCCESS;

cleanup:
    if(optim != NULL) {
        gsx_check(gsx_optim_free(optim), "gsx_optim_free");
    }
    if(grads != NULL) {
        gsx_check(gsx_tensor_free(grads), "gsx_tensor_free(grads)");
    }
    if(params != NULL) {
        gsx_check(gsx_tensor_free(params), "gsx_tensor_free(params)");
    }
    if(arena != NULL) {
        gsx_check(gsx_arena_free(arena), "gsx_arena_free");
    }
    if(backend != NULL) {
        gsx_check(gsx_backend_free(backend), "gsx_backend_free");
    }
    return exit_code;
}
