#include "../objc-helpers.h"

extern const char gsx_metal_optim_metallib_start[];
extern const char gsx_metal_optim_metallib_end[];

static gsx_error gsx_metal_backend_ensure_optim_library(gsx_metal_backend *metal_backend, id<MTLLibrary> *out_library)
{
    return gsx_metal_backend_ensure_embedded_library(
        metal_backend,
        &metal_backend->optim_library,
        gsx_metal_optim_metallib_start,
        gsx_metal_optim_metallib_end,
        "embedded Metal optimizer metallib is empty",
        "failed to create dispatch data for embedded Metal optimizer metallib",
        "failed to load embedded Metal optimizer metallib",
        out_library);
}

/* Compile the optimizer kernel library and create Adam pipeline state once.
 * No-op when already cached; always leaves the backend in a valid state on failure. */
static gsx_error gsx_metal_backend_ensure_adam_pipeline(gsx_metal_backend *metal_backend, id<MTLComputePipelineState> *out_pipeline)
{
    return gsx_metal_backend_ensure_compute_pipeline(
        metal_backend,
        &metal_backend->optim_adam_pipeline,
        gsx_metal_backend_ensure_optim_library,
        "gsx_metal_adam_step_f32_kernel",
        "failed to look up Metal optimizer kernel functions",
        "failed to create Metal Adam step pipeline state",
        out_pipeline);
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

    error = gsx_metal_backend_begin_compute_command(metal_backend, pipeline, &command_buffer, &encoder);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    [encoder setBuffer:(id<MTLBuffer>)parameter_buffer->mtl_buffer offset:(NSUInteger)parameter->offset_bytes atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)gradient_buffer->mtl_buffer offset:(NSUInteger)gradient->offset_bytes atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)first_moment_buffer->mtl_buffer offset:(NSUInteger)first_moment->offset_bytes atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)second_moment_buffer->mtl_buffer offset:(NSUInteger)second_moment->offset_bytes atIndex:3];
    [encoder setBytes:params length:sizeof(*params) atIndex:4];

    gsx_metal_backend_dispatch_threads_1d(encoder, pipeline, (NSUInteger)params->element_count);

    [encoder endEncoding];
    [command_buffer commit];

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

