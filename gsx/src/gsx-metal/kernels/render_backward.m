#include "../objc-helpers.h"

extern const char gsx_metal_render_metallib_start[];
extern const char gsx_metal_render_metallib_end[];

static gsx_error gsx_metal_backend_ensure_render_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
	return gsx_metal_backend_ensure_embedded_library(
		metal_backend,
		&metal_backend->render_library,
		gsx_metal_render_metallib_start,
		gsx_metal_render_metallib_end,
		"embedded Metal render metallib is empty",
		"failed to create dispatch data for embedded Metal render metallib",
		"failed to load embedded Metal render metallib",
		out_library);
}

static gsx_error gsx_metal_backend_ensure_render_blend_backward_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
	return gsx_metal_backend_ensure_compute_pipeline(
		metal_backend,
		&metal_backend->render_blend_backward_pipeline,
		gsx_metal_backend_ensure_render_library,
		"gsx_metal_render_blend_backward_kernel",
		"failed to look up Metal render blend backward kernel function",
		"failed to create Metal render blend backward pipeline state",
		out_pipeline);
}

static gsx_error gsx_metal_backend_ensure_render_preprocess_backward_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
	return gsx_metal_backend_ensure_compute_pipeline(
		metal_backend,
		&metal_backend->render_preprocess_backward_pipeline,
		gsx_metal_backend_ensure_render_library,
		"gsx_metal_render_preprocess_backward_kernel",
		"failed to look up Metal render preprocess backward kernel function",
		"failed to create Metal render preprocess backward pipeline state",
		out_pipeline);
}

gsx_error gsx_metal_backend_dispatch_render_blend_backward(
	gsx_backend_t backend,
	const gsx_backend_tensor_view *tile_ranges_view,
	const gsx_backend_tensor_view *tile_bucket_offsets_view,
	const gsx_backend_tensor_view *bucket_tile_index_view,
	const gsx_backend_tensor_view *instance_primitive_ids_view,
	const gsx_backend_tensor_view *mean2d_view,
	const gsx_backend_tensor_view *conic_opacity_view,
	const gsx_backend_tensor_view *color_view,
	const gsx_backend_tensor_view *image_view,
	const gsx_backend_tensor_view *alpha_view,
	const gsx_backend_tensor_view *tile_max_n_contributions_view,
	const gsx_backend_tensor_view *tile_n_contributions_view,
	const gsx_backend_tensor_view *bucket_color_transmittance_view,
	const gsx_backend_tensor_view *grad_rgb_view,
	const gsx_backend_tensor_view *grad_mean2d_view,
	const gsx_backend_tensor_view *absgrad_mean2d_view,
	const gsx_backend_tensor_view *grad_conic_view,
	const gsx_backend_tensor_view *grad_raw_opacity_view,
	const gsx_backend_tensor_view *grad_color_view,
	const gsx_metal_render_blend_backward_params *params)
{
	gsx_metal_backend *metal_backend = NULL;
	id<MTLComputePipelineState> pipeline = nil;
	id<MTLCommandBuffer> command_buffer = nil;
	id<MTLComputeCommandEncoder> encoder = nil;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(backend == NULL || tile_ranges_view == NULL || tile_bucket_offsets_view == NULL || bucket_tile_index_view == NULL
		|| instance_primitive_ids_view == NULL || mean2d_view == NULL
		|| conic_opacity_view == NULL || color_view == NULL || image_view == NULL || alpha_view == NULL
		|| tile_max_n_contributions_view == NULL || tile_n_contributions_view == NULL || bucket_color_transmittance_view == NULL
		|| grad_rgb_view == NULL || grad_mean2d_view == NULL || absgrad_mean2d_view == NULL
		|| grad_conic_view == NULL || grad_raw_opacity_view == NULL || grad_color_view == NULL || params == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render blend backward dispatch arguments must be non-null");
	}
	if(params->gaussian_count == 0) {
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}

	metal_backend = gsx_metal_backend_from_base(backend);
	error = gsx_metal_backend_ensure_render_blend_backward_pipeline(metal_backend, &pipeline);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_ranges_view->buffer)->mtl_buffer offset:(NSUInteger)tile_ranges_view->offset_bytes atIndex:0];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_bucket_offsets_view->buffer)->mtl_buffer offset:(NSUInteger)tile_bucket_offsets_view->offset_bytes atIndex:1];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bucket_tile_index_view->buffer)->mtl_buffer offset:(NSUInteger)bucket_tile_index_view->offset_bytes atIndex:2];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)instance_primitive_ids_view->offset_bytes atIndex:3];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:4];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:5];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(color_view->buffer)->mtl_buffer offset:(NSUInteger)color_view->offset_bytes atIndex:6];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(image_view->buffer)->mtl_buffer offset:(NSUInteger)image_view->offset_bytes atIndex:7];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(alpha_view->buffer)->mtl_buffer offset:(NSUInteger)alpha_view->offset_bytes atIndex:8];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_max_n_contributions_view->buffer)->mtl_buffer offset:(NSUInteger)tile_max_n_contributions_view->offset_bytes atIndex:9];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(tile_n_contributions_view->buffer)->mtl_buffer offset:(NSUInteger)tile_n_contributions_view->offset_bytes atIndex:10];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(bucket_color_transmittance_view->buffer)->mtl_buffer offset:(NSUInteger)bucket_color_transmittance_view->offset_bytes atIndex:11];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_rgb_view->buffer)->mtl_buffer offset:(NSUInteger)grad_rgb_view->offset_bytes atIndex:12];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)grad_mean2d_view->offset_bytes atIndex:13];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(absgrad_mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)absgrad_mean2d_view->offset_bytes atIndex:14];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_conic_view->buffer)->mtl_buffer offset:(NSUInteger)grad_conic_view->offset_bytes atIndex:15];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_raw_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)grad_raw_opacity_view->offset_bytes atIndex:16];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_color_view->buffer)->mtl_buffer offset:(NSUInteger)grad_color_view->offset_bytes atIndex:17];
	[encoder setBytes:params length:sizeof(*params) atIndex:18];

	if((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup < 16u * 16u) {
		[encoder endEncoding];
		return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "render blend backward kernel requires 256-thread Metal threadgroups");
	}
	if(params->total_bucket_count == 0u) {
		[encoder endEncoding];
		[command_buffer commit];
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}

	[encoder
		dispatchThreadgroups:MTLSizeMake(((NSUInteger)params->total_bucket_count + 7u) / 8u, 1, 1)
		threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[encoder endEncoding];
	[command_buffer commit];
	return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_render_preprocess_backward(
	gsx_backend_t backend,
	const gsx_backend_tensor_view *mean3d_view,
	const gsx_backend_tensor_view *rotation_view,
	const gsx_backend_tensor_view *logscale_view,
	const gsx_backend_tensor_view *sh0_view,
	const gsx_backend_tensor_view *sh1_view,
	const gsx_backend_tensor_view *sh2_view,
	const gsx_backend_tensor_view *sh3_view,
	const gsx_backend_tensor_view *opacity_view,
	const gsx_backend_tensor_view *mean2d_view,
	const gsx_backend_tensor_view *conic_opacity_view,
	const gsx_backend_tensor_view *grad_mean2d_view,
	const gsx_backend_tensor_view *absgrad_mean2d_view,
	const gsx_backend_tensor_view *grad_conic_view,
	const gsx_backend_tensor_view *grad_raw_opacity_partial_view,
	const gsx_backend_tensor_view *grad_color_view,
	const gsx_backend_tensor_view *grad_mean3d_view,
	const gsx_backend_tensor_view *grad_rotation_view,
	const gsx_backend_tensor_view *grad_logscale_view,
	const gsx_backend_tensor_view *grad_sh0_view,
	const gsx_backend_tensor_view *grad_sh1_view,
	const gsx_backend_tensor_view *grad_sh2_view,
	const gsx_backend_tensor_view *grad_sh3_view,
	const gsx_backend_tensor_view *grad_opacity_view,
	const gsx_backend_tensor_view *visible_counter_view,
	const gsx_backend_tensor_view *grad_acc_view,
	const gsx_backend_tensor_view *absgrad_acc_view,
	const gsx_metal_render_preprocess_backward_params *params)
{
	gsx_metal_backend *metal_backend = NULL;
	id<MTLComputePipelineState> pipeline = nil;
	id<MTLCommandBuffer> command_buffer = nil;
	id<MTLComputeCommandEncoder> encoder = nil;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(backend == NULL || mean3d_view == NULL || rotation_view == NULL || logscale_view == NULL || sh0_view == NULL
		|| opacity_view == NULL || mean2d_view == NULL || conic_opacity_view == NULL || grad_mean2d_view == NULL
		|| absgrad_mean2d_view == NULL || grad_conic_view == NULL || grad_raw_opacity_partial_view == NULL || grad_color_view == NULL
		|| grad_mean3d_view == NULL || grad_rotation_view == NULL || grad_logscale_view == NULL
		|| grad_sh0_view == NULL || grad_opacity_view == NULL || params == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render preprocess backward dispatch arguments must be non-null");
	}
	if(params->sh_degree > 3u) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render preprocess backward sh_degree must be in [0,3]");
	}
	if(sh1_view == NULL || sh2_view == NULL || sh3_view == NULL || grad_sh1_view == NULL || grad_sh2_view == NULL
		|| grad_sh3_view == NULL || visible_counter_view == NULL || grad_acc_view == NULL || absgrad_acc_view == NULL) {
		return gsx_make_error(
			GSX_ERROR_INVALID_ARGUMENT,
			"render preprocess backward requires sh1, sh2, sh3, grad_sh1, grad_sh2, grad_sh3, visible_counter, grad_acc, and absgrad_acc views; bind a dummy tensor when unused");
	}
	if(params->gaussian_count == 0) {
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}

	metal_backend = gsx_metal_backend_from_base(backend);
	error = gsx_metal_backend_ensure_render_preprocess_backward_pipeline(metal_backend, &pipeline);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean3d_view->buffer)->mtl_buffer offset:(NSUInteger)mean3d_view->offset_bytes atIndex:0];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(rotation_view->buffer)->mtl_buffer offset:(NSUInteger)rotation_view->offset_bytes atIndex:1];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(logscale_view->buffer)->mtl_buffer offset:(NSUInteger)logscale_view->offset_bytes atIndex:2];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh0_view->buffer)->mtl_buffer offset:(NSUInteger)sh0_view->offset_bytes atIndex:3];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh1_view->buffer)->mtl_buffer
		offset:(NSUInteger)sh1_view->offset_bytes
		atIndex:4];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh2_view->buffer)->mtl_buffer
		offset:(NSUInteger)sh2_view->offset_bytes
		atIndex:5];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(sh3_view->buffer)->mtl_buffer
		offset:(NSUInteger)sh3_view->offset_bytes
		atIndex:6];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(opacity_view->buffer)->mtl_buffer offset:(NSUInteger)opacity_view->offset_bytes atIndex:7];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:8];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:9];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)grad_mean2d_view->offset_bytes atIndex:10];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(absgrad_mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)absgrad_mean2d_view->offset_bytes atIndex:11];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_conic_view->buffer)->mtl_buffer offset:(NSUInteger)grad_conic_view->offset_bytes atIndex:12];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_raw_opacity_partial_view->buffer)->mtl_buffer offset:(NSUInteger)grad_raw_opacity_partial_view->offset_bytes atIndex:13];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_color_view->buffer)->mtl_buffer offset:(NSUInteger)grad_color_view->offset_bytes atIndex:14];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_mean3d_view->buffer)->mtl_buffer offset:(NSUInteger)grad_mean3d_view->offset_bytes atIndex:15];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_rotation_view->buffer)->mtl_buffer offset:(NSUInteger)grad_rotation_view->offset_bytes atIndex:16];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_logscale_view->buffer)->mtl_buffer offset:(NSUInteger)grad_logscale_view->offset_bytes atIndex:17];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_sh0_view->buffer)->mtl_buffer offset:(NSUInteger)grad_sh0_view->offset_bytes atIndex:18];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_sh1_view->buffer)->mtl_buffer
		offset:(NSUInteger)grad_sh1_view->offset_bytes
		atIndex:19];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_sh2_view->buffer)->mtl_buffer
		offset:(NSUInteger)grad_sh2_view->offset_bytes
		atIndex:20];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_sh3_view->buffer)->mtl_buffer
		offset:(NSUInteger)grad_sh3_view->offset_bytes
		atIndex:21];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)grad_opacity_view->offset_bytes atIndex:22];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(visible_counter_view->buffer)->mtl_buffer
		offset:(NSUInteger)visible_counter_view->offset_bytes
		atIndex:23];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_acc_view->buffer)->mtl_buffer
		offset:(NSUInteger)grad_acc_view->offset_bytes
		atIndex:24];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(absgrad_acc_view->buffer)->mtl_buffer
		offset:(NSUInteger)absgrad_acc_view->offset_bytes
		atIndex:25];
	[encoder setBytes:params length:sizeof(*params) atIndex:26];

	gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
	[encoder endEncoding];
	[command_buffer commit];
	return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
