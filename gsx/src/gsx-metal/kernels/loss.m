#include "../internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Pointwise loss kernels are compiled lazily and cached per backend. */
static NSString *const gsx_metal_loss_kernel_source = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
\n\
struct gsx_metal_loss_pointwise_params {\n\
    uint element_count;\n\
    float scale;\n\
};\n\
\n\
kernel void gsx_metal_loss_mse_f32_kernel(\n\
    device const float *prediction [[buffer(0)]],\n\
    device const float *target [[buffer(1)]],\n\
    device float *loss_map [[buffer(2)]],\n\
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    if(gid >= params.element_count) {\n\
        return;\n\
    }\n\
\n\
    float diff = prediction[gid] - target[gid];\n\
    loss_map[gid] += params.scale * diff * diff;\n\
}\n\
\n\
kernel void gsx_metal_loss_l1_f32_kernel(\n\
    device const float *prediction [[buffer(0)]],\n\
    device const float *target [[buffer(1)]],\n\
    device float *loss_map [[buffer(2)]],\n\
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    if(gid >= params.element_count) {\n\
        return;\n\
    }\n\
\n\
    float diff = prediction[gid] - target[gid];\n\
    loss_map[gid] += params.scale * fabs(diff);\n\
}\n\
\n\
kernel void gsx_metal_loss_mse_backward_f32_kernel(\n\
    device const float *prediction [[buffer(0)]],\n\
    device const float *target [[buffer(1)]],\n\
    device float *grad_prediction [[buffer(2)]],\n\
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    if(gid >= params.element_count) {\n\
        return;\n\
    }\n\
\n\
    float diff = prediction[gid] - target[gid];\n\
    grad_prediction[gid] += 2.0f * diff * params.scale;\n\
}\n\
\n\
kernel void gsx_metal_loss_l1_backward_f32_kernel(\n\
    device const float *prediction [[buffer(0)]],\n\
    device const float *target [[buffer(1)]],\n\
    device float *grad_prediction [[buffer(2)]],\n\
    constant gsx_metal_loss_pointwise_params &params [[buffer(3)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    float sign = 0.0f;\n\
    if(gid >= params.element_count) {\n\
        return;\n\
    }\n\
\n\
    if(prediction[gid] > target[gid]) {\n\
        sign = 1.0f;\n\
    } else if(prediction[gid] < target[gid]) {\n\
        sign = -1.0f;\n\
    }\n\
    grad_prediction[gid] += sign * params.scale;\n\
}\n";

static gsx_error gsx_metal_backend_make_loss_pipeline(
    gsx_metal_backend *metal_backend,
    const char *function_name,
    id<MTLComputePipelineState> *out_pipeline)
{
    NSError *ns_error = nil;
    MTLCompileOptions *compile_options = nil;
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || function_name == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend, function_name, and out_pipeline must be non-null");
    }

    compile_options = [[MTLCompileOptions alloc] init];
    if(compile_options == nil) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal compile options");
    }
    if(@available(macOS 10.13, *)) {
        compile_options.languageVersion = MTLLanguageVersion2_0;
    }

    library = [(id<MTLDevice>)metal_backend->mtl_device
        newLibraryWithSource:gsx_metal_loss_kernel_source
        options:compile_options
        error:&ns_error];
    [compile_options release];
    if(library == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to compile Metal loss kernel library");
    }

    function = [library newFunctionWithName:[NSString stringWithUTF8String:function_name]];
    [library release];
    if(function == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to look up Metal loss kernel function");
    }

    pipeline = [(id<MTLDevice>)metal_backend->mtl_device newComputePipelineStateWithFunction:function error:&ns_error];
    [function release];
    if(pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to create Metal loss pipeline state");
    }

    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_loss_mse_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->loss_mse_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->loss_mse_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_backend_make_loss_pipeline(metal_backend, "gsx_metal_loss_mse_f32_kernel", &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend->loss_mse_pipeline = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_loss_l1_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->loss_l1_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->loss_l1_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_backend_make_loss_pipeline(metal_backend, "gsx_metal_loss_l1_f32_kernel", &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend->loss_l1_pipeline = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_loss_mse_backward_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->loss_mse_backward_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->loss_mse_backward_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_backend_make_loss_pipeline(metal_backend, "gsx_metal_loss_mse_backward_f32_kernel", &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend->loss_mse_backward_pipeline = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_loss_l1_backward_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->loss_l1_backward_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->loss_l1_backward_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_backend_make_loss_pipeline(metal_backend, "gsx_metal_loss_l1_backward_f32_kernel", &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend->loss_l1_backward_pipeline = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_dispatch_loss_pointwise_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params,
    gsx_error (*ensure_pipeline)(gsx_metal_backend *, id<MTLComputePipelineState> *))
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *prediction_buffer = NULL;
    gsx_metal_backend_buffer *target_buffer = NULL;
    gsx_metal_backend_buffer *accumulator_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    NSUInteger threadgroup_width = 0;

    if(backend == NULL || prediction_view == NULL || target_view == NULL || accumulator_view == NULL || params == NULL
        || ensure_pipeline == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, params, and ensure_pipeline must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    prediction_buffer = gsx_metal_backend_buffer_from_base(prediction_view->buffer);
    target_buffer = gsx_metal_backend_buffer_from_base(target_view->buffer);
    accumulator_buffer = gsx_metal_backend_buffer_from_base(accumulator_view->buffer);

    error = ensure_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
    }

    encoder = [command_buffer computeCommandEncoder];
    if(encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal compute encoder");
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)prediction_buffer->mtl_buffer offset:(NSUInteger)prediction_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)target_buffer->mtl_buffer offset:(NSUInteger)target_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)accumulator_buffer->mtl_buffer offset:(NSUInteger)accumulator_view->offset_bytes atIndex:2];
    [encoder setBytes:params length:sizeof(*params) atIndex:3];

    threadgroup_width = (NSUInteger)pipeline.threadExecutionWidth;
    if(threadgroup_width == 0) {
        threadgroup_width = 64;
    }
    if(threadgroup_width > (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup) {
        threadgroup_width = (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup;
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)params->element_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];

    [encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_loss_mse_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        accumulator_view,
        params,
        gsx_metal_backend_ensure_loss_mse_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_l1_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *accumulator_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        accumulator_view,
        params,
        gsx_metal_backend_ensure_loss_l1_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_mse_backward_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        grad_view,
        params,
        gsx_metal_backend_ensure_loss_mse_backward_pipeline);
}

gsx_error gsx_metal_backend_dispatch_loss_l1_backward_f32(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *prediction_view,
    const gsx_backend_tensor_view *target_view,
    const gsx_backend_tensor_view *grad_view,
    const gsx_metal_loss_pointwise_params *params)
{
    return gsx_metal_backend_dispatch_loss_pointwise_f32(
        backend,
        prediction_view,
        target_view,
        grad_view,
        params,
        gsx_metal_backend_ensure_loss_l1_backward_pipeline);
}
