#include "internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/*
 * Both optimizer kernels are compiled from one library on first use so the
 * Metal compiler is invoked only once per backend lifetime.
 */
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
}\n\
\n\
struct gsx_metal_row_gather_params {\n\
    uint row_floats;\n\
    uint row_count;\n\
};\n\
\n\
/* dst[dst_row][col] = src[indices[dst_row]][col] for all dst_row in [0, row_count) and col in [0, row_floats). */\n\
kernel void gsx_metal_row_gather_f32_kernel(\n\
    device float *dst [[buffer(0)]],\n\
    device const float *src [[buffer(1)]],\n\
    device const int *indices [[buffer(2)]],\n\
    constant gsx_metal_row_gather_params &params [[buffer(3)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    if(gid >= params.row_floats * params.row_count) {\n\
        return;\n\
    }\n\
    uint dst_row = gid / params.row_floats;\n\
    uint col     = gid % params.row_floats;\n\
    uint src_row = (uint)indices[dst_row];\n\
    dst[gid] = src[src_row * params.row_floats + col];\n\
}\n";

/* Compile the optimizer kernel library and create both pipeline states once.
 * No-op when both are already cached; always leaves the backend in a valid state on failure. */
static gsx_error gsx_metal_backend_ensure_optim_pipelines(gsx_metal_backend *metal_backend)
{
    NSError *ns_error = nil;
    MTLCompileOptions *compile_options = nil;
    id<MTLLibrary> library = nil;
    id<MTLFunction> adam_fn = nil;
    id<MTLFunction> gather_fn = nil;
    id<MTLComputePipelineState> adam_pso = nil;
    id<MTLComputePipelineState> gather_pso = nil;

    if(metal_backend == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend must be non-null");
    }
    if(metal_backend->optim_adam_pipeline != NULL && metal_backend->optim_row_gather_pipeline != NULL) {
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

    adam_fn   = [library newFunctionWithName:@"gsx_metal_adam_step_f32_kernel"];
    gather_fn = [library newFunctionWithName:@"gsx_metal_row_gather_f32_kernel"];
    [library release];
    if(adam_fn == nil || gather_fn == nil) {
        [adam_fn release];
        [gather_fn release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to look up Metal optimizer kernel functions");
    }

    adam_pso = [(id<MTLDevice>)metal_backend->mtl_device
        newComputePipelineStateWithFunction:adam_fn
        error:&ns_error];
    [adam_fn release];
    if(adam_pso == nil) {
        [gather_fn release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to create Metal Adam step pipeline state");
    }

    gather_pso = [(id<MTLDevice>)metal_backend->mtl_device
        newComputePipelineStateWithFunction:gather_fn
        error:&ns_error];
    [gather_fn release];
    if(gather_pso == nil) {
        [adam_pso release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to create Metal row gather pipeline state");
    }

    metal_backend->optim_adam_pipeline = adam_pso;
    metal_backend->optim_row_gather_pipeline = gather_pso;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_adam_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    error = gsx_metal_backend_ensure_optim_pipelines(metal_backend);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    *out_pipeline = (id<MTLComputePipelineState>)metal_backend->optim_adam_pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_row_gather_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    error = gsx_metal_backend_ensure_optim_pipelines(metal_backend);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    *out_pipeline = (id<MTLComputePipelineState>)metal_backend->optim_row_gather_pipeline;
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

gsx_error gsx_metal_backend_dispatch_row_gather(
    gsx_backend_t backend,
    gsx_tensor_t dst,
    gsx_tensor_t src,
    gsx_backend_buffer_t indices_buffer,
    gsx_size_t indices_offset_bytes,
    uint32_t row_floats,
    uint32_t row_count
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *dst_metal = NULL;
    gsx_metal_backend_buffer *src_metal = NULL;
    gsx_metal_backend_buffer *idx_metal = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_metal_row_gather_params kernel_params;
    NSUInteger total_threads = 0;
    NSUInteger threadgroup_width = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || dst == NULL || src == NULL || indices_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensors, and indices_buffer must be non-null");
    }
    if(row_count == 0 || row_floats == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    dst_metal = gsx_metal_backend_buffer_from_base(dst->backing_buffer);
    src_metal = gsx_metal_backend_buffer_from_base(src->backing_buffer);
    idx_metal = gsx_metal_backend_buffer_from_base(indices_buffer);

    error = gsx_metal_backend_ensure_row_gather_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer for row gather");
    }

    encoder = [command_buffer computeCommandEncoder];
    if(encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal compute encoder for row gather");
    }

    kernel_params.row_floats = row_floats;
    kernel_params.row_count  = row_count;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)dst_metal->mtl_buffer offset:(NSUInteger)dst->offset_bytes        atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)src_metal->mtl_buffer offset:(NSUInteger)src->offset_bytes        atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)idx_metal->mtl_buffer offset:(NSUInteger)indices_offset_bytes     atIndex:2];
    [encoder setBytes:&kernel_params length:sizeof(kernel_params) atIndex:3];

    total_threads   = (NSUInteger)row_floats * (NSUInteger)row_count;
    threadgroup_width = (NSUInteger)pipeline.threadExecutionWidth;
    if(threadgroup_width == 0) {
        threadgroup_width = 64;
    }
    if(threadgroup_width > (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup) {
        threadgroup_width = (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup;
    }

    [encoder
        dispatchThreads:MTLSizeMake(total_threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];

    [encoder endEncoding];
    [command_buffer commit];

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_grow_blit(
    gsx_backend_t backend,
    gsx_tensor_t dst,
    gsx_tensor_t src,
    gsx_size_t copy_bytes,
    gsx_size_t total_dst_bytes
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *dst_metal = NULL;
    gsx_metal_backend_buffer *src_metal = NULL;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit_encoder = nil;

    if(backend == NULL || dst == NULL || src == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend and tensors must be non-null");
    }
    if(total_dst_bytes < copy_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "total_dst_bytes must be >= copy_bytes");
    }
    if(total_dst_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    dst_metal = gsx_metal_backend_buffer_from_base(dst->backing_buffer);
    src_metal = gsx_metal_backend_buffer_from_base(src->backing_buffer);

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer for grow blit");
    }

    blit_encoder = [command_buffer blitCommandEncoder];
    if(blit_encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder for grow blit");
    }

    if(copy_bytes > 0) {
        [blit_encoder
            copyFromBuffer:(id<MTLBuffer>)src_metal->mtl_buffer
            sourceOffset:(NSUInteger)src->offset_bytes
            toBuffer:(id<MTLBuffer>)dst_metal->mtl_buffer
            destinationOffset:(NSUInteger)dst->offset_bytes
            size:(NSUInteger)copy_bytes];
    }
    if(total_dst_bytes > copy_bytes) {
        [blit_encoder
            fillBuffer:(id<MTLBuffer>)dst_metal->mtl_buffer
            range:NSMakeRange((NSUInteger)(dst->offset_bytes + copy_bytes), (NSUInteger)(total_dst_bytes - copy_bytes))
            value:0];
    }

    [blit_encoder endEncoding];
    [command_buffer commit];

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

