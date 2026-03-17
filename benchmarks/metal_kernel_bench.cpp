#include "gsx/gsx.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

struct ReduceUnaryBenchmarkContext {
	gsx_backend_t backend = nullptr;
	gsx_arena_t arena = nullptr;
	gsx_tensor_t input = nullptr;
	gsx_tensor_t output = nullptr;
	gsx_size_t element_count = 0;
	gsx_index_t start_axis = 1;
};

struct ReduceBinaryBenchmarkContext {
	gsx_backend_t backend = nullptr;
	gsx_arena_t arena = nullptr;
	gsx_tensor_t lhs = nullptr;
	gsx_tensor_t rhs = nullptr;
	gsx_tensor_t output = nullptr;
	gsx_size_t element_count = 0;
	gsx_index_t start_axis = 1;
};

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

enum class BenchmarkUnaryReduceOp {
	Sum,
	Mean,
	Max
};

enum class BenchmarkBinaryReduceOp {
	Mse,
	Mae
};

static bool gsx_error_ok(gsx_error error)
{
	return gsx_error_is_success(error);
}

static gsx_tensor_t gsx_make_f32_tensor_rank2(gsx_arena_t arena, gsx_index_t dim0, gsx_index_t dim1)
{
	gsx_tensor_t tensor = nullptr;
	gsx_tensor_desc desc{};

	desc.rank = 2;
	desc.shape[0] = dim0;
	desc.shape[1] = dim1;
	desc.data_type = GSX_DATA_TYPE_F32;
	desc.storage_format = GSX_STORAGE_FORMAT_CHW;
	desc.arena = arena;
	if(!gsx_error_ok(gsx_tensor_init(&tensor, &desc))) {
		return nullptr;
	}
	return tensor;
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

static void gsx_maybe_free_tensor(gsx_tensor_t *tensor)
{
	if(tensor != nullptr && *tensor != nullptr) {
		(void)gsx_tensor_free(*tensor);
		*tensor = nullptr;
	}
}

static bool gsx_sync_backend(gsx_backend_t backend, std::string *out_error)
{
	if(out_error == nullptr) {
		return false;
	}
	if(!gsx_error_ok(gsx_backend_major_stream_sync(backend))) {
		*out_error = "gsx_backend_major_stream_sync failed";
		return false;
	}
	return true;
}

static void gsx_cleanup_reduce_unary_context(ReduceUnaryBenchmarkContext *ctx)
{
	if(ctx == nullptr) {
		return;
	}
	gsx_maybe_free_tensor(&ctx->input);
	gsx_maybe_free_tensor(&ctx->output);
	if(ctx->arena != nullptr) {
		(void)gsx_arena_free(ctx->arena);
		ctx->arena = nullptr;
	}
	if(ctx->backend != nullptr) {
		(void)gsx_backend_free(ctx->backend);
		ctx->backend = nullptr;
	}
}

static void gsx_cleanup_reduce_binary_context(ReduceBinaryBenchmarkContext *ctx)
{
	if(ctx == nullptr) {
		return;
	}
	gsx_maybe_free_tensor(&ctx->lhs);
	gsx_maybe_free_tensor(&ctx->rhs);
	gsx_maybe_free_tensor(&ctx->output);
	if(ctx->arena != nullptr) {
		(void)gsx_arena_free(ctx->arena);
		ctx->arena = nullptr;
	}
	if(ctx->backend != nullptr) {
		(void)gsx_backend_free(ctx->backend);
		ctx->backend = nullptr;
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
		*out_error = backend_type == GSX_BACKEND_TYPE_METAL ? "Metal backend device unavailable" : "CPU backend device unavailable";
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

static bool gsx_plan_unary_reduce_arena_capacity(
	gsx_backend_buffer_type_t buffer_type,
	gsx_index_t outer_count,
	gsx_index_t reduce_count,
	gsx_size_t *out_capacity_bytes,
	std::string *out_error)
{
	gsx_arena_t dry_arena = nullptr;
	gsx_arena_desc dry_arena_desc{};
	gsx_tensor_t dry_input = nullptr;
	gsx_tensor_t dry_output = nullptr;

	if(buffer_type == nullptr || out_capacity_bytes == nullptr || out_error == nullptr) {
		return false;
	}

	dry_arena_desc.initial_capacity_bytes = 0;
	dry_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
	dry_arena_desc.dry_run = true;
	if(!gsx_error_ok(gsx_arena_init(&dry_arena, buffer_type, &dry_arena_desc))) {
		*out_error = "dry-run arena init failed";
		return false;
	}

	dry_input = gsx_make_f32_tensor_rank2(dry_arena, outer_count, reduce_count);
	dry_output = gsx_make_f32_tensor_rank2(dry_arena, outer_count, 1);
	if(dry_input == nullptr || dry_output == nullptr) {
		*out_error = "dry-run tensor allocation failed";
		gsx_maybe_free_tensor(&dry_output);
		gsx_maybe_free_tensor(&dry_input);
		(void)gsx_arena_free(dry_arena);
		return false;
	}

	if(!gsx_error_ok(gsx_tensor_sum(dry_arena, dry_input, dry_output, 1))
		|| !gsx_error_ok(gsx_tensor_mean(dry_arena, dry_input, dry_output, 1))
		|| !gsx_error_ok(gsx_tensor_max(dry_arena, dry_input, dry_output, 1))
		|| !gsx_error_ok(gsx_arena_get_required_bytes(dry_arena, out_capacity_bytes))) {
		*out_error = "dry-run unary reduce planning failed";
		gsx_maybe_free_tensor(&dry_output);
		gsx_maybe_free_tensor(&dry_input);
		(void)gsx_arena_free(dry_arena);
		return false;
	}

	gsx_maybe_free_tensor(&dry_output);
	gsx_maybe_free_tensor(&dry_input);
	if(!gsx_error_ok(gsx_arena_free(dry_arena))) {
		*out_error = "dry-run arena free failed";
		return false;
	}
	return true;
}

static bool gsx_plan_binary_reduce_arena_capacity(
	gsx_backend_buffer_type_t buffer_type,
	gsx_index_t outer_count,
	gsx_index_t reduce_count,
	gsx_size_t *out_capacity_bytes,
	std::string *out_error)
{
	gsx_arena_t dry_arena = nullptr;
	gsx_arena_desc dry_arena_desc{};
	gsx_tensor_t dry_lhs = nullptr;
	gsx_tensor_t dry_rhs = nullptr;
	gsx_tensor_t dry_output = nullptr;

	if(buffer_type == nullptr || out_capacity_bytes == nullptr || out_error == nullptr) {
		return false;
	}

	dry_arena_desc.initial_capacity_bytes = 0;
	dry_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
	dry_arena_desc.dry_run = true;
	if(!gsx_error_ok(gsx_arena_init(&dry_arena, buffer_type, &dry_arena_desc))) {
		*out_error = "dry-run arena init failed";
		return false;
	}

	dry_lhs = gsx_make_f32_tensor_rank2(dry_arena, outer_count, reduce_count);
	dry_rhs = gsx_make_f32_tensor_rank2(dry_arena, outer_count, reduce_count);
	dry_output = gsx_make_f32_tensor_rank2(dry_arena, outer_count, 1);
	if(dry_lhs == nullptr || dry_rhs == nullptr || dry_output == nullptr) {
		*out_error = "dry-run tensor allocation failed";
		gsx_maybe_free_tensor(&dry_output);
		gsx_maybe_free_tensor(&dry_rhs);
		gsx_maybe_free_tensor(&dry_lhs);
		(void)gsx_arena_free(dry_arena);
		return false;
	}

	if(!gsx_error_ok(gsx_tensor_mse(dry_arena, dry_lhs, dry_rhs, dry_output, 1))
		|| !gsx_error_ok(gsx_tensor_mae(dry_arena, dry_lhs, dry_rhs, dry_output, 1))
		|| !gsx_error_ok(gsx_arena_get_required_bytes(dry_arena, out_capacity_bytes))) {
		*out_error = "dry-run binary reduce planning failed";
		gsx_maybe_free_tensor(&dry_output);
		gsx_maybe_free_tensor(&dry_rhs);
		gsx_maybe_free_tensor(&dry_lhs);
		(void)gsx_arena_free(dry_arena);
		return false;
	}

	gsx_maybe_free_tensor(&dry_output);
	gsx_maybe_free_tensor(&dry_rhs);
	gsx_maybe_free_tensor(&dry_lhs);
	if(!gsx_error_ok(gsx_arena_free(dry_arena))) {
		*out_error = "dry-run arena free failed";
		return false;
	}
	return true;
}

static bool gsx_init_metal_reduce_unary_context(
	ReduceUnaryBenchmarkContext *out_ctx,
	gsx_index_t outer_count,
	gsx_index_t reduce_count,
	std::string *out_error)
{
	gsx_backend_device_t backend_device = nullptr;
	gsx_backend_desc backend_desc{};
	gsx_backend_buffer_type_t device_buffer_type = nullptr;
	gsx_arena_desc arena_desc{};
	std::vector<float> input_values;
	gsx_size_t element_count = 0;
	gsx_size_t input_bytes = 0;
	gsx_size_t arena_capacity_bytes = 0;
	const gsx_size_t max_size = std::numeric_limits<gsx_size_t>::max();
	ReduceUnaryBenchmarkContext ctx = {};

	if(out_ctx == nullptr || out_error == nullptr) {
		return false;
	}
	if(outer_count <= 0 || reduce_count <= 0) {
		*out_error = "reduce dimensions must be positive";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_registry_init())) {
		*out_error = "gsx_backend_registry_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device)) || backend_device == nullptr) {
		*out_error = "Metal backend device unavailable";
		return false;
	}
	if((gsx_size_t)outer_count != 0 && (gsx_size_t)reduce_count > max_size / (gsx_size_t)outer_count) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	element_count = (gsx_size_t)outer_count * (gsx_size_t)reduce_count;
	if(sizeof(float) != 0 && element_count > max_size / sizeof(float)) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	input_bytes = element_count * sizeof(float);

	backend_desc.device = backend_device;
	if(!gsx_error_ok(gsx_backend_init(&ctx.backend, &backend_desc))) {
		*out_error = "gsx_backend_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_find_buffer_type(ctx.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type))) {
		*out_error = "gsx_backend_find_buffer_type failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}
	if(!gsx_plan_unary_reduce_arena_capacity(device_buffer_type, outer_count, reduce_count, &arena_capacity_bytes, out_error)) {
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}
	arena_desc.initial_capacity_bytes = arena_capacity_bytes;
	arena_desc.growth_mode = arena_capacity_bytes == 0 ? GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND : GSX_ARENA_GROWTH_MODE_FIXED;
	if(!gsx_error_ok(gsx_arena_init(&ctx.arena, device_buffer_type, &arena_desc))) {
		*out_error = "gsx_arena_init failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}

	ctx.input = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, reduce_count);
	ctx.output = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, 1);
	if(ctx.input == nullptr || ctx.output == nullptr) {
		*out_error = "tensor allocation failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}

	input_values.resize((std::size_t)element_count);
	for(std::size_t i = 0; i < input_values.size(); ++i) {
		input_values[i] = (float)((i % 257) * 0.0078125);
	}
	if(!gsx_error_ok(gsx_tensor_upload(ctx.input, input_values.data(), input_bytes)) || !gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
		*out_error = "tensor initialization failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}

	ctx.element_count = element_count;
	*out_ctx = ctx;
	return true;
}

static bool gsx_init_metal_reduce_binary_context(
	ReduceBinaryBenchmarkContext *out_ctx,
	gsx_index_t outer_count,
	gsx_index_t reduce_count,
	std::string *out_error)
{
	gsx_backend_device_t backend_device = nullptr;
	gsx_backend_desc backend_desc{};
	gsx_backend_buffer_type_t device_buffer_type = nullptr;
	gsx_arena_desc arena_desc{};
	std::vector<float> lhs_values;
	std::vector<float> rhs_values;
	gsx_size_t element_count = 0;
	gsx_size_t input_bytes = 0;
	gsx_size_t arena_capacity_bytes = 0;
	const gsx_size_t max_size = std::numeric_limits<gsx_size_t>::max();
	ReduceBinaryBenchmarkContext ctx = {};

	if(out_ctx == nullptr || out_error == nullptr) {
		return false;
	}
	if(outer_count <= 0 || reduce_count <= 0) {
		*out_error = "reduce dimensions must be positive";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_registry_init())) {
		*out_error = "gsx_backend_registry_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device)) || backend_device == nullptr) {
		*out_error = "Metal backend device unavailable";
		return false;
	}
	if((gsx_size_t)outer_count != 0 && (gsx_size_t)reduce_count > max_size / (gsx_size_t)outer_count) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	element_count = (gsx_size_t)outer_count * (gsx_size_t)reduce_count;
	if(sizeof(float) != 0 && element_count > max_size / sizeof(float)) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	input_bytes = element_count * sizeof(float);

	backend_desc.device = backend_device;
	if(!gsx_error_ok(gsx_backend_init(&ctx.backend, &backend_desc))) {
		*out_error = "gsx_backend_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_find_buffer_type(ctx.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type))) {
		*out_error = "gsx_backend_find_buffer_type failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}
	if(!gsx_plan_binary_reduce_arena_capacity(device_buffer_type, outer_count, reduce_count, &arena_capacity_bytes, out_error)) {
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}
	arena_desc.initial_capacity_bytes = arena_capacity_bytes;
	arena_desc.growth_mode = arena_capacity_bytes == 0 ? GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND : GSX_ARENA_GROWTH_MODE_FIXED;
	if(!gsx_error_ok(gsx_arena_init(&ctx.arena, device_buffer_type, &arena_desc))) {
		*out_error = "gsx_arena_init failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}

	ctx.lhs = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, reduce_count);
	ctx.rhs = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, reduce_count);
	ctx.output = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, 1);
	if(ctx.lhs == nullptr || ctx.rhs == nullptr || ctx.output == nullptr) {
		*out_error = "tensor allocation failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}

	lhs_values.resize((std::size_t)element_count);
	rhs_values.resize((std::size_t)element_count);
	for(std::size_t i = 0; i < lhs_values.size(); ++i) {
		lhs_values[i] = (float)((i % 127) * 0.015625);
		rhs_values[i] = (float)(((i + 19) % 131) * 0.0125);
	}
	if(!gsx_error_ok(gsx_tensor_upload(ctx.lhs, lhs_values.data(), input_bytes))
		|| !gsx_error_ok(gsx_tensor_upload(ctx.rhs, rhs_values.data(), input_bytes))
		|| !gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
		*out_error = "tensor initialization failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}

	ctx.element_count = element_count;
	*out_ctx = ctx;
	return true;
}

static bool gsx_init_cpu_reduce_unary_context(
	ReduceUnaryBenchmarkContext *out_ctx,
	gsx_index_t outer_count,
	gsx_index_t reduce_count,
	std::string *out_error)
{
	gsx_backend_device_t backend_device = nullptr;
	gsx_backend_desc backend_desc{};
	gsx_backend_buffer_type_t device_buffer_type = nullptr;
	gsx_arena_desc arena_desc{};
	std::vector<float> input_values;
	gsx_size_t element_count = 0;
	gsx_size_t input_bytes = 0;
	gsx_size_t arena_capacity_bytes = 0;
	const gsx_size_t max_size = std::numeric_limits<gsx_size_t>::max();
	ReduceUnaryBenchmarkContext ctx = {};

	if(out_ctx == nullptr || out_error == nullptr) {
		return false;
	}
	if(outer_count <= 0 || reduce_count <= 0) {
		*out_error = "reduce dimensions must be positive";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_registry_init())) {
		*out_error = "gsx_backend_registry_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device)) || backend_device == nullptr) {
		*out_error = "CPU backend device unavailable";
		return false;
	}
	if((gsx_size_t)outer_count != 0 && (gsx_size_t)reduce_count > max_size / (gsx_size_t)outer_count) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	element_count = (gsx_size_t)outer_count * (gsx_size_t)reduce_count;
	if(sizeof(float) != 0 && element_count > max_size / sizeof(float)) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	input_bytes = element_count * sizeof(float);

	backend_desc.device = backend_device;
	if(!gsx_error_ok(gsx_backend_init(&ctx.backend, &backend_desc))) {
		*out_error = "gsx_backend_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_find_buffer_type(ctx.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type))) {
		*out_error = "gsx_backend_find_buffer_type failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}
	if(!gsx_plan_unary_reduce_arena_capacity(device_buffer_type, outer_count, reduce_count, &arena_capacity_bytes, out_error)) {
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}
	arena_desc.initial_capacity_bytes = arena_capacity_bytes;
	arena_desc.growth_mode = arena_capacity_bytes == 0 ? GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND : GSX_ARENA_GROWTH_MODE_FIXED;
	if(!gsx_error_ok(gsx_arena_init(&ctx.arena, device_buffer_type, &arena_desc))) {
		*out_error = "gsx_arena_init failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}

	ctx.input = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, reduce_count);
	ctx.output = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, 1);
	if(ctx.input == nullptr || ctx.output == nullptr) {
		*out_error = "tensor allocation failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}

	input_values.resize((std::size_t)element_count);
	for(std::size_t i = 0; i < input_values.size(); ++i) {
		input_values[i] = (float)((i % 257) * 0.0078125);
	}
	if(!gsx_error_ok(gsx_tensor_upload(ctx.input, input_values.data(), input_bytes)) || !gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
		*out_error = "tensor initialization failed";
		gsx_cleanup_reduce_unary_context(&ctx);
		return false;
	}

	ctx.element_count = element_count;
	*out_ctx = ctx;
	return true;
}

static bool gsx_init_cpu_reduce_binary_context(
	ReduceBinaryBenchmarkContext *out_ctx,
	gsx_index_t outer_count,
	gsx_index_t reduce_count,
	std::string *out_error)
{
	gsx_backend_device_t backend_device = nullptr;
	gsx_backend_desc backend_desc{};
	gsx_backend_buffer_type_t device_buffer_type = nullptr;
	gsx_arena_desc arena_desc{};
	std::vector<float> lhs_values;
	std::vector<float> rhs_values;
	gsx_size_t element_count = 0;
	gsx_size_t input_bytes = 0;
	gsx_size_t arena_capacity_bytes = 0;
	const gsx_size_t max_size = std::numeric_limits<gsx_size_t>::max();
	ReduceBinaryBenchmarkContext ctx = {};

	if(out_ctx == nullptr || out_error == nullptr) {
		return false;
	}
	if(outer_count <= 0 || reduce_count <= 0) {
		*out_error = "reduce dimensions must be positive";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_registry_init())) {
		*out_error = "gsx_backend_registry_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device)) || backend_device == nullptr) {
		*out_error = "CPU backend device unavailable";
		return false;
	}
	if((gsx_size_t)outer_count != 0 && (gsx_size_t)reduce_count > max_size / (gsx_size_t)outer_count) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	element_count = (gsx_size_t)outer_count * (gsx_size_t)reduce_count;
	if(sizeof(float) != 0 && element_count > max_size / sizeof(float)) {
		*out_error = "benchmark tensor size overflow";
		return false;
	}
	input_bytes = element_count * sizeof(float);

	backend_desc.device = backend_device;
	if(!gsx_error_ok(gsx_backend_init(&ctx.backend, &backend_desc))) {
		*out_error = "gsx_backend_init failed";
		return false;
	}
	if(!gsx_error_ok(gsx_backend_find_buffer_type(ctx.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type))) {
		*out_error = "gsx_backend_find_buffer_type failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}
	if(!gsx_plan_binary_reduce_arena_capacity(device_buffer_type, outer_count, reduce_count, &arena_capacity_bytes, out_error)) {
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}
	arena_desc.initial_capacity_bytes = arena_capacity_bytes;
	arena_desc.growth_mode = arena_capacity_bytes == 0 ? GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND : GSX_ARENA_GROWTH_MODE_FIXED;
	if(!gsx_error_ok(gsx_arena_init(&ctx.arena, device_buffer_type, &arena_desc))) {
		*out_error = "gsx_arena_init failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}

	ctx.lhs = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, reduce_count);
	ctx.rhs = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, reduce_count);
	ctx.output = gsx_make_f32_tensor_rank2(ctx.arena, outer_count, 1);
	if(ctx.lhs == nullptr || ctx.rhs == nullptr || ctx.output == nullptr) {
		*out_error = "tensor allocation failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}

	lhs_values.resize((std::size_t)element_count);
	rhs_values.resize((std::size_t)element_count);
	for(std::size_t i = 0; i < lhs_values.size(); ++i) {
		lhs_values[i] = (float)((i % 127) * 0.015625);
		rhs_values[i] = (float)(((i + 19) % 131) * 0.0125);
	}
	if(!gsx_error_ok(gsx_tensor_upload(ctx.lhs, lhs_values.data(), input_bytes))
		|| !gsx_error_ok(gsx_tensor_upload(ctx.rhs, rhs_values.data(), input_bytes))
		|| !gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
		*out_error = "tensor initialization failed";
		gsx_cleanup_reduce_binary_context(&ctx);
		return false;
	}

	ctx.element_count = element_count;
	*out_ctx = ctx;
	return true;
}

static void bm_reduce_unary_metal(benchmark::State &state, BenchmarkUnaryReduceOp op)
{
	ReduceUnaryBenchmarkContext ctx = {};
	std::string error;
	const gsx_index_t outer_count = (gsx_index_t)state.range(0);
	const gsx_index_t reduce_count = (gsx_index_t)state.range(1);

	if(!gsx_init_metal_reduce_unary_context(&ctx, outer_count, reduce_count, &error)) {
		state.SkipWithError(error.c_str());
		return;
	}

	for(auto _ : state) {
		gsx_error op_error = { GSX_ERROR_SUCCESS, nullptr };

		if(!gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
			state.SkipWithError("Metal unary reduce output reset failed");
			break;
		}
		switch(op) {
		case BenchmarkUnaryReduceOp::Sum:
			op_error = gsx_tensor_sum(ctx.arena, ctx.input, ctx.output, ctx.start_axis);
			break;
		case BenchmarkUnaryReduceOp::Mean:
			op_error = gsx_tensor_mean(ctx.arena, ctx.input, ctx.output, ctx.start_axis);
			break;
		case BenchmarkUnaryReduceOp::Max:
			op_error = gsx_tensor_max(ctx.arena, ctx.input, ctx.output, ctx.start_axis);
			break;
		default:
			state.SkipWithError("unsupported unary reduce op");
			gsx_cleanup_reduce_unary_context(&ctx);
			return;
		}

		if(!gsx_error_ok(op_error)) {
			state.SkipWithError("Metal unary reduce failed");
			break;
		}
		if(!gsx_sync_backend(ctx.backend, &error)) {
			state.SkipWithError(error.c_str());
			break;
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
	gsx_cleanup_reduce_unary_context(&ctx);
}

static void bm_reduce_binary_metal(benchmark::State &state, BenchmarkBinaryReduceOp op)
{
	ReduceBinaryBenchmarkContext ctx = {};
	std::string error;
	const gsx_index_t outer_count = (gsx_index_t)state.range(0);
	const gsx_index_t reduce_count = (gsx_index_t)state.range(1);

	if(!gsx_init_metal_reduce_binary_context(&ctx, outer_count, reduce_count, &error)) {
		state.SkipWithError(error.c_str());
		return;
	}

	for(auto _ : state) {
		gsx_error op_error = { GSX_ERROR_SUCCESS, nullptr };

		if(!gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
			state.SkipWithError("Metal binary reduce output reset failed");
			break;
		}
		switch(op) {
		case BenchmarkBinaryReduceOp::Mse:
			op_error = gsx_tensor_mse(ctx.arena, ctx.lhs, ctx.rhs, ctx.output, ctx.start_axis);
			break;
		case BenchmarkBinaryReduceOp::Mae:
			op_error = gsx_tensor_mae(ctx.arena, ctx.lhs, ctx.rhs, ctx.output, ctx.start_axis);
			break;
		default:
			state.SkipWithError("unsupported binary reduce op");
			gsx_cleanup_reduce_binary_context(&ctx);
			return;
		}

		if(!gsx_error_ok(op_error)) {
			state.SkipWithError("Metal binary reduce failed");
			break;
		}
		if(!gsx_sync_backend(ctx.backend, &error)) {
			state.SkipWithError(error.c_str());
			break;
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
	gsx_cleanup_reduce_binary_context(&ctx);
}

static void bm_reduce_unary_cpu(benchmark::State &state, BenchmarkUnaryReduceOp op)
{
	ReduceUnaryBenchmarkContext ctx = {};
	std::string error;
	const gsx_index_t outer_count = (gsx_index_t)state.range(0);
	const gsx_index_t reduce_count = (gsx_index_t)state.range(1);

	if(!gsx_init_cpu_reduce_unary_context(&ctx, outer_count, reduce_count, &error)) {
		state.SkipWithError(error.c_str());
		return;
	}

	for(auto _ : state) {
		gsx_error op_error = { GSX_ERROR_SUCCESS, nullptr };

		if(!gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
			state.SkipWithError("CPU unary reduce output reset failed");
			break;
		}
		switch(op) {
		case BenchmarkUnaryReduceOp::Sum:
			op_error = gsx_tensor_sum(ctx.arena, ctx.input, ctx.output, ctx.start_axis);
			break;
		case BenchmarkUnaryReduceOp::Mean:
			op_error = gsx_tensor_mean(ctx.arena, ctx.input, ctx.output, ctx.start_axis);
			break;
		case BenchmarkUnaryReduceOp::Max:
			op_error = gsx_tensor_max(ctx.arena, ctx.input, ctx.output, ctx.start_axis);
			break;
		default:
			state.SkipWithError("unsupported unary reduce op");
			gsx_cleanup_reduce_unary_context(&ctx);
			return;
		}

		if(!gsx_error_ok(op_error)) {
			state.SkipWithError("CPU unary reduce failed");
			break;
		}
		if(!gsx_sync_backend(ctx.backend, &error)) {
			state.SkipWithError(error.c_str());
			break;
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
	gsx_cleanup_reduce_unary_context(&ctx);
}

static void bm_reduce_binary_cpu(benchmark::State &state, BenchmarkBinaryReduceOp op)
{
	ReduceBinaryBenchmarkContext ctx = {};
	std::string error;
	const gsx_index_t outer_count = (gsx_index_t)state.range(0);
	const gsx_index_t reduce_count = (gsx_index_t)state.range(1);

	if(!gsx_init_cpu_reduce_binary_context(&ctx, outer_count, reduce_count, &error)) {
		state.SkipWithError(error.c_str());
		return;
	}

	for(auto _ : state) {
		gsx_error op_error = { GSX_ERROR_SUCCESS, nullptr };

		if(!gsx_error_ok(gsx_tensor_set_zero(ctx.output))) {
			state.SkipWithError("CPU binary reduce output reset failed");
			break;
		}
		switch(op) {
		case BenchmarkBinaryReduceOp::Mse:
			op_error = gsx_tensor_mse(ctx.arena, ctx.lhs, ctx.rhs, ctx.output, ctx.start_axis);
			break;
		case BenchmarkBinaryReduceOp::Mae:
			op_error = gsx_tensor_mae(ctx.arena, ctx.lhs, ctx.rhs, ctx.output, ctx.start_axis);
			break;
		default:
			state.SkipWithError("unsupported binary reduce op");
			gsx_cleanup_reduce_binary_context(&ctx);
			return;
		}

		if(!gsx_error_ok(op_error)) {
			state.SkipWithError("CPU binary reduce failed");
			break;
		}
		if(!gsx_sync_backend(ctx.backend, &error)) {
			state.SkipWithError(error.c_str());
			break;
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
	gsx_cleanup_reduce_binary_context(&ctx);
}

static void bm_loss_ssim(benchmark::State &state, gsx_backend_type backend_type, const char *label)
{
	SsimBenchmarkContext ctx = {};
	std::string error;
	const gsx_index_t channels = 3;
	const gsx_index_t height = (gsx_index_t)state.range(0);
	const gsx_index_t width = (gsx_index_t)state.range(1);

	if(!gsx_init_ssim_context(&ctx, backend_type, channels, height, width, true, &error)) {
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
			state.SkipWithError(label);
			break;
		}
		if(!gsx_sync_backend(ctx.backend, &error)) {
			state.SkipWithError(error.c_str());
			break;
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)ctx.element_count);
	gsx_cleanup_ssim_context(&ctx);
}

static void BM_Core_ReduceMetalSum(benchmark::State &state)
{
	bm_reduce_unary_metal(state, BenchmarkUnaryReduceOp::Sum);
}

static void BM_Core_ReduceMetalMean(benchmark::State &state)
{
	bm_reduce_unary_metal(state, BenchmarkUnaryReduceOp::Mean);
}

static void BM_Core_ReduceMetalMax(benchmark::State &state)
{
	bm_reduce_unary_metal(state, BenchmarkUnaryReduceOp::Max);
}

static void BM_Core_ReduceMetalMse(benchmark::State &state)
{
	bm_reduce_binary_metal(state, BenchmarkBinaryReduceOp::Mse);
}

static void BM_Core_ReduceMetalMae(benchmark::State &state)
{
	bm_reduce_binary_metal(state, BenchmarkBinaryReduceOp::Mae);
}

static void BM_Core_ReduceCpuSum(benchmark::State &state)
{
	bm_reduce_unary_cpu(state, BenchmarkUnaryReduceOp::Sum);
}

static void BM_Core_ReduceCpuMean(benchmark::State &state)
{
	bm_reduce_unary_cpu(state, BenchmarkUnaryReduceOp::Mean);
}

static void BM_Core_ReduceCpuMax(benchmark::State &state)
{
	bm_reduce_unary_cpu(state, BenchmarkUnaryReduceOp::Max);
}

static void BM_Core_ReduceCpuMse(benchmark::State &state)
{
	bm_reduce_binary_cpu(state, BenchmarkBinaryReduceOp::Mse);
}

static void BM_Core_ReduceCpuMae(benchmark::State &state)
{
	bm_reduce_binary_cpu(state, BenchmarkBinaryReduceOp::Mae);
}

static void BM_Loss_SsimCpu(benchmark::State &state)
{
	bm_loss_ssim(state, GSX_BACKEND_TYPE_CPU, "CPU SSIM loss evaluation failed");
}

static void BM_Loss_SsimMetal(benchmark::State &state)
{
	bm_loss_ssim(state, GSX_BACKEND_TYPE_METAL, "Metal SSIM loss evaluation failed");
}

BENCHMARK(BM_Loss_SsimCpu)->Args({ 256, 256 })->Args({ 512, 512 });
BENCHMARK(BM_Loss_SsimMetal)->Args({ 256, 256 })->Args({ 512, 512 });
BENCHMARK(BM_Core_ReduceCpuSum)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceCpuMean)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceCpuMax)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceCpuMse)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceCpuMae)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceMetalSum)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceMetalMean)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceMetalMax)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceMetalMse)->Args({ 4096, 256 })->Args({ 1024, 4096 });
BENCHMARK(BM_Core_ReduceMetalMae)->Args({ 4096, 256 })->Args({ 1024, 4096 });

}  // namespace

BENCHMARK_MAIN();
