#include "gsx/gsx.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#if GSX_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

struct gsx_error_benchmark_fixture {
    gsx_error success = { GSX_ERROR_SUCCESS, nullptr };
    gsx_error failure = { GSX_ERROR_INVALID_ARGUMENT, "invalid argument" };
};

static void BM_Base_ErrorIsSuccess(benchmark::State &state)
{
    gsx_error_benchmark_fixture fixture;

    for(auto _ : state) {
        benchmark::DoNotOptimize(gsx_error_is_success(fixture.success));
        benchmark::DoNotOptimize(gsx_error_is_success(fixture.failure));
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * 2);
}

BENCHMARK(BM_Base_ErrorIsSuccess);

struct SsimBenchmarkContext {
    gsx_backend_t backend = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_loss_t loss = nullptr;
    gsx_loss_context_t loss_context = nullptr;
    gsx_tensor_t prediction = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t loss_map = nullptr;
    gsx_tensor_t grad_prediction = nullptr;
    gsx_loss_request request{};
    gsx_size_t element_count = 0;
};

static bool gsx_error_ok(gsx_error error)
{
    return gsx_error_is_success(error);
}

static bool gsx_sync_backend_if_cuda(gsx_backend_t backend, std::string *out_error)
{
#if GSX_HAS_CUDA
    gsx_backend_info backend_info{};
    void *stream = nullptr;

    if(out_error == nullptr) {
        return false;
    }
    if(!gsx_error_ok(gsx_backend_get_info(backend, &backend_info))) {
        *out_error = "gsx_backend_get_info failed";
        return false;
    }
    if(backend_info.backend_type != GSX_BACKEND_TYPE_CUDA) {
        return true;
    }
    if(!gsx_error_ok(gsx_backend_get_major_stream(backend, &stream))) {
        *out_error = "gsx_backend_get_major_stream failed";
        return false;
    }
    if(cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess) {
        *out_error = "cudaStreamSynchronize failed";
        return false;
    }
#else
    (void)backend;
    (void)out_error;
#endif
    return true;
}

static void gsx_maybe_free_tensor(gsx_tensor_t *tensor)
{
    if(tensor != nullptr && *tensor != nullptr) {
        (void)gsx_tensor_free(*tensor);
        *tensor = nullptr;
    }
}

static void gsx_cleanup_ssim_context(SsimBenchmarkContext *ctx)
{
    if(ctx == nullptr) {
        return;
    }
    if(ctx->loss != nullptr) {
        if(ctx->loss_context != nullptr) {
            (void)gsx_loss_context_free(ctx->loss_context);
            ctx->loss_context = nullptr;
        }
        (void)gsx_loss_free(ctx->loss);
        ctx->loss = nullptr;
    }
    gsx_maybe_free_tensor(&ctx->prediction);
    gsx_maybe_free_tensor(&ctx->target);
    gsx_maybe_free_tensor(&ctx->loss_map);
    gsx_maybe_free_tensor(&ctx->grad_prediction);
    if(ctx->arena != nullptr) {
        (void)gsx_arena_free(ctx->arena);
        ctx->arena = nullptr;
    }
    if(ctx->backend != nullptr) {
        (void)gsx_backend_free(ctx->backend);
        ctx->backend = nullptr;
    }
}

static gsx_tensor_t gsx_make_f32_tensor(gsx_arena_t arena, gsx_index_t channels, gsx_index_t height, gsx_index_t width)
{
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc desc{};

    desc.rank = 3;
    desc.shape[0] = channels;
    desc.shape[1] = height;
    desc.shape[2] = width;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    if(!gsx_error_ok(gsx_tensor_init(&tensor, &desc))) {
        return nullptr;
    }
    return tensor;
}

static bool gsx_init_ssim_context(
    SsimBenchmarkContext *out_ctx,
    gsx_backend_type backend_type,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width,
    bool with_grad,
    std::string *out_error)
{
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_desc backend_desc{};
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_loss_desc loss_desc{};
    std::vector<float> prediction_values;
    std::vector<float> target_values;
    gsx_size_t element_count = 0;
    gsx_size_t tensor_bytes = 0;
    gsx_size_t tensor_count = with_grad ? 4 : 3;
    gsx_size_t total_tensor_bytes = 0;
    gsx_size_t capacity_bytes = 0;
    const gsx_size_t max_size = std::numeric_limits<gsx_size_t>::max();
    SsimBenchmarkContext ctx = {};

    if(out_ctx == nullptr || out_error == nullptr) {
        return false;
    }
    if(!gsx_error_ok(gsx_backend_registry_init())) {
        *out_error = "gsx_backend_registry_init failed";
        return false;
    }
    if(!gsx_error_ok(gsx_get_backend_device_by_type(backend_type, 0, &backend_device)) || backend_device == nullptr) {
        *out_error = backend_type == GSX_BACKEND_TYPE_CUDA ? "CUDA backend device unavailable" : "CPU backend device unavailable";
        return false;
    }
    if((gsx_size_t)channels != 0 && (gsx_size_t)height > max_size / (gsx_size_t)channels) {
        *out_error = "benchmark tensor size overflow";
        return false;
    }
    element_count = (gsx_size_t)channels * (gsx_size_t)height;
    if((gsx_size_t)width != 0 && element_count > max_size / (gsx_size_t)width) {
        *out_error = "benchmark tensor size overflow";
        return false;
    }
    element_count *= (gsx_size_t)width;
    if(sizeof(float) != 0 && element_count > max_size / sizeof(float)) {
        *out_error = "benchmark tensor size overflow";
        return false;
    }
    tensor_bytes = element_count * sizeof(float);
    if(tensor_count != 0 && tensor_bytes > max_size / tensor_count) {
        *out_error = "benchmark tensor size overflow";
        return false;
    }
    total_tensor_bytes = tensor_bytes * tensor_count;
    if(total_tensor_bytes > max_size - (gsx_size_t)65536) {
        *out_error = "benchmark tensor size overflow";
        return false;
    }
    capacity_bytes = total_tensor_bytes + (gsx_size_t)65536;

    backend_desc.device = backend_device;
    if(!gsx_error_ok(gsx_backend_init(&ctx.backend, &backend_desc))) {
        *out_error = "gsx_backend_init failed";
        return false;
    }
    if(!gsx_error_ok(gsx_backend_find_buffer_type(ctx.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type))) {
        *out_error = "gsx_backend_find_buffer_type failed";
        gsx_cleanup_ssim_context(&ctx);
        return false;
    }

    arena_desc.initial_capacity_bytes = capacity_bytes;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    if(!gsx_error_ok(gsx_arena_init(&ctx.arena, device_buffer_type, &arena_desc))) {
        *out_error = "gsx_arena_init failed";
        gsx_cleanup_ssim_context(&ctx);
        return false;
    }

    ctx.prediction = gsx_make_f32_tensor(ctx.arena, channels, height, width);
    ctx.target = gsx_make_f32_tensor(ctx.arena, channels, height, width);
    ctx.loss_map = gsx_make_f32_tensor(ctx.arena, channels, height, width);
    ctx.grad_prediction = with_grad ? gsx_make_f32_tensor(ctx.arena, channels, height, width) : nullptr;
    if(ctx.prediction == nullptr || ctx.target == nullptr || ctx.loss_map == nullptr || (with_grad && ctx.grad_prediction == nullptr)) {
        *out_error = "tensor allocation failed";
        gsx_cleanup_ssim_context(&ctx);
        return false;
    }

    prediction_values.resize((std::size_t)element_count);
    target_values.resize((std::size_t)element_count);
    for(std::size_t i = 0; i < prediction_values.size(); ++i) {
        prediction_values[i] = (float)((i % 31) * 0.03125);
        target_values[i] = (float)(((i + 7) % 29) * 0.0333333333);
    }
    if(!gsx_error_ok(gsx_tensor_upload(ctx.prediction, prediction_values.data(), element_count * sizeof(float)))
        || !gsx_error_ok(gsx_tensor_upload(ctx.target, target_values.data(), element_count * sizeof(float)))
        || !gsx_error_ok(gsx_tensor_set_zero(ctx.loss_map))
        || (with_grad && !gsx_error_ok(gsx_tensor_set_zero(ctx.grad_prediction)))) {
        *out_error = "tensor initialization failed";
        gsx_cleanup_ssim_context(&ctx);
        return false;
    }

    loss_desc.algorithm = GSX_LOSS_ALGORITHM_SSIM;
    loss_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    if(!gsx_error_ok(gsx_loss_init(&ctx.loss, ctx.backend, &loss_desc))) {
        *out_error = "gsx_loss_init failed";
        gsx_cleanup_ssim_context(&ctx);
        return false;
    }
    if(!gsx_error_ok(gsx_loss_context_init(&ctx.loss_context, ctx.loss))) {
        *out_error = "gsx_loss_context_init failed";
        gsx_cleanup_ssim_context(&ctx);
        return false;
    }

    ctx.request.prediction = ctx.prediction;
    ctx.request.target = ctx.target;
    ctx.request.loss_map_accumulator = ctx.loss_map;
    ctx.request.grad_prediction_accumulator = ctx.grad_prediction;
    ctx.request.scale = 1.0f;
    ctx.element_count = element_count;
    *out_ctx = ctx;
    return true;
}

static void BM_Loss_SsimCpu(benchmark::State &state)
{
    SsimBenchmarkContext ctx = {};
    std::string error;
    const gsx_index_t channels = 3;
    const gsx_index_t height = (gsx_index_t)state.range(0);
    const gsx_index_t width = (gsx_index_t)state.range(1);

    if(!gsx_init_ssim_context(&ctx, GSX_BACKEND_TYPE_CPU, channels, height, width, true, &error)) {
        state.SkipWithError(error.c_str());
        return;
    }

    for(auto _ : state) {
        gsx_loss_forward_request forward_request{};
        gsx_loss_backward_request backward_request{};

        forward_request.prediction = ctx.request.prediction;
        forward_request.target = ctx.request.target;
        forward_request.loss_map_accumulator = ctx.request.loss_map_accumulator;
        forward_request.train = true;
        forward_request.scale = ctx.request.scale;
        backward_request.grad_prediction_accumulator = ctx.request.grad_prediction_accumulator;
        backward_request.scale = ctx.request.scale;
        if(!gsx_error_ok(gsx_tensor_set_zero(ctx.loss_map))
            || !gsx_error_ok(gsx_tensor_set_zero(ctx.grad_prediction))
            || !gsx_error_ok(gsx_loss_forward(ctx.loss, ctx.loss_context, &forward_request))
            || !gsx_error_ok(gsx_loss_backward(ctx.loss, ctx.loss_context, &backward_request))) {
            state.SkipWithError("CPU SSIM loss evaluation failed");
            break;
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
    gsx_cleanup_ssim_context(&ctx);
}

static void BM_Loss_SsimCuda(benchmark::State &state)
{
    SsimBenchmarkContext ctx = {};
    std::string error;
    const gsx_index_t channels = 3;
    const gsx_index_t height = (gsx_index_t)state.range(0);
    const gsx_index_t width = (gsx_index_t)state.range(1);

    if(!gsx_init_ssim_context(&ctx, GSX_BACKEND_TYPE_CUDA, channels, height, width, true, &error)) {
        state.SkipWithError(error.c_str());
        return;
    }

    for(auto _ : state) {
        gsx_loss_forward_request forward_request{};
        gsx_loss_backward_request backward_request{};

        forward_request.prediction = ctx.request.prediction;
        forward_request.target = ctx.request.target;
        forward_request.loss_map_accumulator = ctx.request.loss_map_accumulator;
        forward_request.train = true;
        forward_request.scale = ctx.request.scale;
        backward_request.grad_prediction_accumulator = ctx.request.grad_prediction_accumulator;
        backward_request.scale = ctx.request.scale;
        if(!gsx_error_ok(gsx_tensor_set_zero(ctx.loss_map))
            || !gsx_error_ok(gsx_tensor_set_zero(ctx.grad_prediction))
            || !gsx_error_ok(gsx_loss_forward(ctx.loss, ctx.loss_context, &forward_request))
            || !gsx_error_ok(gsx_loss_backward(ctx.loss, ctx.loss_context, &backward_request))) {
            state.SkipWithError("CUDA SSIM loss evaluation failed");
            break;
        }
        if(!gsx_sync_backend_if_cuda(ctx.backend, &error)) {
            state.SkipWithError(error.c_str());
            break;
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
    gsx_cleanup_ssim_context(&ctx);
}

BENCHMARK(BM_Loss_SsimCpu)->Args({ 128, 128 });
BENCHMARK(BM_Loss_SsimCuda)->Args({ 512, 512 });

}  // namespace

BENCHMARK_MAIN();
