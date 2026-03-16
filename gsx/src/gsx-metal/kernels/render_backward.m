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
	const gsx_backend_tensor_view *instance_primitive_ids_view,
	const gsx_backend_tensor_view *mean2d_view,
	const gsx_backend_tensor_view *conic_opacity_view,
	const gsx_backend_tensor_view *color_view,
	const gsx_backend_tensor_view *grad_rgb_view,
	const gsx_backend_tensor_view *grad_mean2d_view,
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

	if(backend == NULL || tile_ranges_view == NULL || instance_primitive_ids_view == NULL || mean2d_view == NULL
		|| conic_opacity_view == NULL || color_view == NULL || grad_rgb_view == NULL || grad_mean2d_view == NULL
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
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(instance_primitive_ids_view->buffer)->mtl_buffer offset:(NSUInteger)instance_primitive_ids_view->offset_bytes atIndex:1];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:2];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:3];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(color_view->buffer)->mtl_buffer offset:(NSUInteger)color_view->offset_bytes atIndex:4];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_rgb_view->buffer)->mtl_buffer offset:(NSUInteger)grad_rgb_view->offset_bytes atIndex:5];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)grad_mean2d_view->offset_bytes atIndex:6];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_conic_view->buffer)->mtl_buffer offset:(NSUInteger)grad_conic_view->offset_bytes atIndex:7];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_raw_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)grad_raw_opacity_view->offset_bytes atIndex:8];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_color_view->buffer)->mtl_buffer offset:(NSUInteger)grad_color_view->offset_bytes atIndex:9];
	[encoder setBytes:params length:sizeof(*params) atIndex:10];

	gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
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
	const gsx_backend_tensor_view *opacity_view,
	const gsx_backend_tensor_view *mean2d_view,
	const gsx_backend_tensor_view *conic_opacity_view,
	const gsx_backend_tensor_view *grad_mean2d_view,
	const gsx_backend_tensor_view *grad_conic_view,
	const gsx_backend_tensor_view *grad_raw_opacity_partial_view,
	const gsx_backend_tensor_view *grad_color_view,
	const gsx_backend_tensor_view *grad_mean3d_view,
	const gsx_backend_tensor_view *grad_rotation_view,
	const gsx_backend_tensor_view *grad_logscale_view,
	const gsx_backend_tensor_view *grad_sh0_view,
	const gsx_backend_tensor_view *grad_opacity_view,
	const gsx_metal_render_preprocess_backward_params *params)
{
	gsx_metal_backend *metal_backend = NULL;
	id<MTLComputePipelineState> pipeline = nil;
	id<MTLCommandBuffer> command_buffer = nil;
	id<MTLComputeCommandEncoder> encoder = nil;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(backend == NULL || mean3d_view == NULL || rotation_view == NULL || logscale_view == NULL || sh0_view == NULL
		|| opacity_view == NULL || mean2d_view == NULL || conic_opacity_view == NULL || grad_mean2d_view == NULL
		|| grad_conic_view == NULL || grad_raw_opacity_partial_view == NULL || grad_color_view == NULL
		|| grad_mean3d_view == NULL || grad_rotation_view == NULL || grad_logscale_view == NULL
		|| grad_sh0_view == NULL || grad_opacity_view == NULL || params == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "render preprocess backward dispatch arguments must be non-null");
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
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(opacity_view->buffer)->mtl_buffer offset:(NSUInteger)opacity_view->offset_bytes atIndex:4];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)mean2d_view->offset_bytes atIndex:5];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(conic_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)conic_opacity_view->offset_bytes atIndex:6];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_mean2d_view->buffer)->mtl_buffer offset:(NSUInteger)grad_mean2d_view->offset_bytes atIndex:7];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_conic_view->buffer)->mtl_buffer offset:(NSUInteger)grad_conic_view->offset_bytes atIndex:8];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_raw_opacity_partial_view->buffer)->mtl_buffer offset:(NSUInteger)grad_raw_opacity_partial_view->offset_bytes atIndex:9];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_color_view->buffer)->mtl_buffer offset:(NSUInteger)grad_color_view->offset_bytes atIndex:10];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_mean3d_view->buffer)->mtl_buffer offset:(NSUInteger)grad_mean3d_view->offset_bytes atIndex:11];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_rotation_view->buffer)->mtl_buffer offset:(NSUInteger)grad_rotation_view->offset_bytes atIndex:12];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_logscale_view->buffer)->mtl_buffer offset:(NSUInteger)grad_logscale_view->offset_bytes atIndex:13];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_sh0_view->buffer)->mtl_buffer offset:(NSUInteger)grad_sh0_view->offset_bytes atIndex:14];
	[encoder setBuffer:(id<MTLBuffer>)gsx_metal_backend_buffer_from_base(grad_opacity_view->buffer)->mtl_buffer offset:(NSUInteger)grad_opacity_view->offset_bytes atIndex:15];
	[encoder setBytes:params length:sizeof(*params) atIndex:16];

	gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->gaussian_count);
	[encoder endEncoding];
	[command_buffer commit];
	return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
