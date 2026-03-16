#ifndef GSX_METAL_OBJC_HELPERS_H
#define GSX_METAL_OBJC_HELPERS_H

#include "internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

typedef gsx_error (*gsx_metal_library_ensure_fn)(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library);

static inline gsx_error gsx_metal_backend_ensure_embedded_library(
    gsx_metal_backend *metal_backend,
    void **library_slot,
    const char *metallib_start,
    const char *metallib_end,
    const char *empty_message,
    const char *create_data_message,
    const char *load_message,
    id<MTLLibrary> *out_library
)
{
    id<MTLLibrary> library = nil;
    dispatch_data_t metallib_data = NULL;
    size_t metallib_size = 0;

    if(metal_backend == NULL || library_slot == NULL || metallib_start == NULL || metallib_end == NULL || out_library == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend, embedded metallib range, and out_library must be non-null");
    }
    if(*library_slot != NULL) {
        *out_library = (id<MTLLibrary>)*library_slot;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    metallib_size = (size_t)(metallib_end - metallib_start);
    if(metallib_size == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, empty_message);
    }

    metallib_data = dispatch_data_create(metallib_start, metallib_size, dispatch_get_main_queue(), ^{});
    if(metallib_data == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, create_data_message);
    }

    library = [(id<MTLDevice>)metal_backend->mtl_device newLibraryWithData:metallib_data error:NULL];
#if !OS_OBJECT_USE_OBJC
    dispatch_release(metallib_data);
#endif
    if(library == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, load_message);
    }

    *library_slot = library;
    *out_library = library;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_metal_backend_ensure_compute_pipeline(
    gsx_metal_backend *metal_backend,
    void **pipeline_slot,
    gsx_metal_library_ensure_fn ensure_library,
    const char *function_name,
    const char *lookup_message,
    const char *create_message,
    id<MTLComputePipelineState> *out_pipeline
)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    id<MTLLibrary> library = nil;
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> pipeline = nil;

    if(metal_backend == NULL || pipeline_slot == NULL || ensure_library == NULL || function_name == NULL || out_pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend, pipeline cache, function_name, and out_pipeline must be non-null");
    }
    if(*pipeline_slot != NULL) {
        *out_pipeline = (id<MTLComputePipelineState>)*pipeline_slot;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = ensure_library(metal_backend, &library);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    function = [library newFunctionWithName:[NSString stringWithUTF8String:function_name]];
    if(function == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, lookup_message);
    }

    pipeline = [(id<MTLDevice>)metal_backend->mtl_device newComputePipelineStateWithFunction:function error:NULL];
    [function release];
    if(pipeline == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, create_message);
    }

    *pipeline_slot = pipeline;
    *out_pipeline = pipeline;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline gsx_error gsx_metal_backend_begin_compute_command(
    gsx_metal_backend *metal_backend,
    id<MTLComputePipelineState> pipeline,
    id<MTLCommandBuffer> *out_command_buffer,
    id<MTLComputeCommandEncoder> *out_encoder
)
{
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;

    if(metal_backend == NULL || pipeline == nil || out_command_buffer == nil || out_encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal backend, pipeline, and command outputs must be non-null");
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
    *out_command_buffer = command_buffer;
    *out_encoder = encoder;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static inline NSUInteger gsx_metal_backend_compute_threadgroup_width(id<MTLComputePipelineState> pipeline)
{
    NSUInteger threadgroup_width = 0;

    threadgroup_width = (NSUInteger)pipeline.threadExecutionWidth;
    if(threadgroup_width == 0) {
        threadgroup_width = 64;
    }
    if(threadgroup_width > (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup) {
        threadgroup_width = (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup;
    }

    return threadgroup_width;
}

static inline void gsx_metal_backend_dispatch_threads_1d(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger thread_count
)
{
    NSUInteger threadgroup_width = gsx_metal_backend_compute_threadgroup_width(pipeline);

    [encoder
        dispatchThreads:MTLSizeMake(thread_count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];
}

#endif /* GSX_METAL_OBJC_HELPERS_H */
