#include "internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Adam optimizer kernel is compiled on first use and cached per backend. */
static NSString *const gsx_metal_optim_kernel_source = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
\n\
struct gsx_metal_adam_step_params {\n\
    float learning_rate;\n\
    float beta1;\n\
    float beta2;\n\
    float epsilon;\n\
    float weight_decay;\n\
    float max_grad;\n\
    float inv_beta1_correction;\n\
    float inv_beta2_correction;\n\
    uint element_count;\n\
};\n\
\n\
kernel void gsx_metal_adam_step_f32_kernel(\n\
    device float *parameter [[buffer(0)]],\n\
    device const float *gradient [[buffer(1)]],\n\
    device float *first_moment [[buffer(2)]],\n\
    device float *second_moment [[buffer(3)]],\n\
    constant gsx_metal_adam_step_params &params [[buffer(4)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    if(gid >= params.element_count) {\n\
        return;\n\
    }\n\
\n\
    float grad = gradient[gid];\n\
    if(params.max_grad > 0.0f) {\n\
        grad = clamp(grad, -params.max_grad, params.max_grad);\n\
    }\n\
\n\
    float m = params.beta1 * first_moment[gid] + (1.0f - params.beta1) * grad;\n\
    float v = params.beta2 * second_moment[gid] + (1.0f - params.beta2) * grad * grad;\n\
    float param = parameter[gid];\n\
\n\
    first_moment[gid] = m;\n\
    second_moment[gid] = v;\n\
\n\
    if(params.weight_decay > 0.0f) {\n\
        param -= params.learning_rate * params.weight_decay * param;\n\
    }\n\
\n\
    float m_hat = m * params.inv_beta1_correction;\n\
    float v_hat = v * params.inv_beta2_correction;\n\
    param -= params.learning_rate * (m_hat / (sqrt(v_hat) + params.epsilon));\n\
    parameter[gid] = param;\n\
}\n";

/* Compile the optimizer kernel library and create Adam pipeline state once.
 * No-op when already cached; always leaves the backend in a valid state on failure. */
static gsx_error gsx_metal_backend_ensure_adam_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    NSError *ns_error = nil;
    MTLCompileOptions *compile_options = nil;
    id<MTLLibrary> library = nil;
    id<MTLFunction> adam_fn = nil;
    id<MTLComputePipelineState> adam_pso = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->optim_adam_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->optim_adam_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    compile_options = [[MTLCompileOptions alloc] init];
    if(compile_options == nil) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal compile options");
    }
    if(@available(macOS 10.13, *)) {
        compile_options.languageVersion = MTLLanguageVersion2_0;
    }

    library = [(id<MTLDevice>)metal_backend->mtl_device
        newLibraryWithSource:gsx_metal_optim_kernel_source
        options:compile_options
        error:&ns_error];
    [compile_options release];
    if(library == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to compile Metal optimizer kernel library");
    }

    adam_fn = [library newFunctionWithName:@"gsx_metal_adam_step_f32_kernel"];
    [library release];
    if(adam_fn == nil) {
        [adam_fn release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to look up Metal optimizer kernel functions");
    }

    adam_pso = [(id<MTLDevice>)metal_backend->mtl_device
        newComputePipelineStateWithFunction:adam_fn
        error:&ns_error];
    [adam_fn release];
    if(adam_pso == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to create Metal Adam step pipeline state");
    }

    metal_backend->optim_adam_pipeline = adam_pso;
    *out_pipeline = adam_pso;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_adam_step(
    gsx_backend_t backend,
    gsx_tensor_t parameter,
    gsx_tensor_t gradient,
    gsx_tensor_t first_moment,
    gsx_tensor_t second_moment,
    const gsx_metal_adam_step_params *params
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *parameter_buffer = NULL;
    gsx_metal_backend_buffer *gradient_buffer = NULL;
    gsx_metal_backend_buffer *first_moment_buffer = NULL;
    gsx_metal_backend_buffer *second_moment_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    NSUInteger threadgroup_width = 0;

    if(backend == NULL || parameter == NULL || gradient == NULL || first_moment == NULL || second_moment == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensors, and params must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    parameter_buffer = gsx_metal_backend_buffer_from_base(parameter->backing_buffer);
    gradient_buffer = gsx_metal_backend_buffer_from_base(gradient->backing_buffer);
    first_moment_buffer = gsx_metal_backend_buffer_from_base(first_moment->backing_buffer);
    second_moment_buffer = gsx_metal_backend_buffer_from_base(second_moment->backing_buffer);

    error = gsx_metal_backend_ensure_adam_pipeline(metal_backend, &pipeline);
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
    [encoder setBuffer:(id<MTLBuffer>)parameter_buffer->mtl_buffer offset:(NSUInteger)parameter->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gradient_buffer->mtl_buffer offset:(NSUInteger)gradient->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)first_moment_buffer->mtl_buffer offset:(NSUInteger)first_moment->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)second_moment_buffer->mtl_buffer offset:(NSUInteger)second_moment->offset_bytes atIndex:3];
    [encoder setBytes:params length:sizeof(*params) atIndex:4];

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

