#include "../internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// TODO: Tensor kernel source is compiled twice (once per pipeline ensure path)
// TODO: vectorized kernels for wider data types and better performance

/* Tensor utility kernels are compiled lazily and cached on the backend object. */
static NSString *const gsx_metal_tensor_kernel_source = @"\
#include <metal_stdlib>\n\
#include <metal_atomic>\n\
using namespace metal;\n\
\n\
struct gsx_metal_tensor_gather_params {\n\
    uint x_row_count;\n\
    uint out_row_count;\n\
    uint row_bytes;\n\
};\n\
\n\
struct gsx_metal_tensor_exp_params {\n\
    uint element_count;\n\
};\n\
\n\
kernel void gsx_metal_tensor_gather_kernel(\n\
    device const uchar *x_bytes [[buffer(0)]],\n\
    device const int *index_data [[buffer(1)]],\n\
    device uchar *out_bytes [[buffer(2)]],\n\
    constant gsx_metal_tensor_gather_params &params [[buffer(3)]],\n\
    device atomic_uint *out_status [[buffer(4)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    uint total_bytes = params.out_row_count * params.row_bytes;\n\
    if(gid >= total_bytes) {\n\
        return;\n\
    }\n\
\n\
    uint row = gid / params.row_bytes;\n\
    uint col = gid - row * params.row_bytes;\n\
    int src_row = index_data[row];\n\
    if(src_row < 0 || (uint)src_row >= params.x_row_count) {\n\
        atomic_fetch_or_explicit(out_status, 1u, memory_order_relaxed);\n\
        return;\n\
    }\n\
\n\
    out_bytes[gid] = x_bytes[(uint)src_row * params.row_bytes + col];\n\
}\n\
\n\
kernel void gsx_metal_tensor_exp_f32_kernel(\n\
    device const float *x_values [[buffer(0)]],\n\
    device float *out_values [[buffer(1)]],\n\
    constant gsx_metal_tensor_exp_params &params [[buffer(2)]],\n\
    uint gid [[thread_position_in_grid]])\n\
{\n\
    if(gid >= params.element_count) {\n\
        return;\n\
    }\n\
    out_values[gid] = exp(x_values[gid]);\n\
}\n";

static gsx_error gsx_metal_backend_make_tensor_pipeline(
    gsx_metal_backend *metal_backend,
    const char *function_name,
    id<MTLComputePipelineState> *out_pipeline
)
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
        newLibraryWithSource:gsx_metal_tensor_kernel_source
        options:compile_options
        error:&ns_error];
    [compile_options release];
    if(library == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to compile Metal tensor kernel library");
    }

    function = [library newFunctionWithName:[NSString stringWithUTF8String:function_name]];
    [library release];
    if(function == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to look up Metal tensor kernel function");
    }

    pipeline = [(id<MTLDevice>)metal_backend->mtl_device newComputePipelineStateWithFunction:function error:&ns_error];
    [function release];
    if(pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to create Metal tensor pipeline state");
    }

    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_tensor_gather_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->tensor_gather_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->tensor_gather_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_backend_make_tensor_pipeline(metal_backend, "gsx_metal_tensor_gather_kernel", &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend->tensor_gather_pipeline = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_ensure_tensor_exp_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend and out_pipeline must be non-null");
    }
    if(metal_backend->tensor_exp_pipeline != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)metal_backend->tensor_exp_pipeline;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_metal_backend_make_tensor_pipeline(metal_backend, "gsx_metal_tensor_exp_f32_kernel", &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    metal_backend->tensor_exp_pipeline = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_tensor_gather(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_gather_params *params
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *index_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLBuffer> status_buffer = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    NSUInteger threadgroup_width = 0;
    NSUInteger total_bytes = 0;
    uint32_t *status_ptr = NULL;

    if(backend == NULL || x_view == NULL || index_view == NULL || out_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, and params must be non-null");
    }
    if(params->out_row_count == 0 || params->row_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    index_buffer = gsx_metal_backend_buffer_from_base(index_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_view->buffer);

    error = gsx_metal_backend_ensure_tensor_gather_pipeline(metal_backend, &pipeline);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    status_buffer = [(id<MTLDevice>)metal_backend->mtl_device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
    if(status_buffer == nil) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal gather status buffer");
    }
    status_ptr = (uint32_t *)[status_buffer contents];
    if(status_ptr == NULL) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to access Metal gather status buffer contents");
    }
    *status_ptr = 0;

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
    }

    encoder = [command_buffer computeCommandEncoder];
    if(encoder == nil) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal compute encoder");
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)x_buffer->mtl_buffer offset:(NSUInteger)x_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)index_buffer->mtl_buffer offset:(NSUInteger)index_view->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_view->offset_bytes atIndex:2];
    [encoder setBytes:params length:sizeof(*params) atIndex:3];
    [encoder setBuffer:status_buffer offset:0 atIndex:4];

    threadgroup_width = (NSUInteger)pipeline.threadExecutionWidth;
    if(threadgroup_width == 0) {
        threadgroup_width = 64;
    }
    if(threadgroup_width > (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup) {
        threadgroup_width = (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup;
    }

    total_bytes = (NSUInteger)params->out_row_count * (NSUInteger)params->row_bytes;
    [encoder
        dispatchThreads:MTLSizeMake(total_bytes, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if(*status_ptr != 0) {
        [status_buffer release];
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather index is out of range");
    }

    [status_buffer release];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_dispatch_tensor_exp(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_metal_tensor_exp_params *params
)
{
    gsx_metal_backend *metal_backend = NULL;
    gsx_metal_backend_buffer *x_buffer = NULL;
    gsx_metal_backend_buffer *out_buffer = NULL;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    NSUInteger threadgroup_width = 0;

    if(backend == NULL || x_view == NULL || out_view == NULL || params == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "backend, tensor views, and params must be non-null");
    }
    if(params->element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metal_backend = gsx_metal_backend_from_base(backend);
    x_buffer = gsx_metal_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_metal_backend_buffer_from_base(out_view->buffer);

    error = gsx_metal_backend_ensure_tensor_exp_pipeline(metal_backend, &pipeline);
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
    [encoder setBuffer:(id<MTLBuffer>)x_buffer->mtl_buffer offset:(NSUInteger)x_view->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)out_buffer->mtl_buffer offset:(NSUInteger)out_view->offset_bytes atIndex:1];
    [encoder setBytes:params length:sizeof(*params) atIndex:2];

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
