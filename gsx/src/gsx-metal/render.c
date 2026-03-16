#include "internal.h"

#include <stdlib.h>

static gsx_error gsx_metal_renderer_destroy(gsx_renderer_t renderer);
static gsx_error gsx_metal_renderer_create_context(gsx_renderer_t renderer, gsx_render_context_t *out_context);
static gsx_error gsx_metal_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request);
static gsx_error gsx_metal_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request);
static gsx_error gsx_metal_render_context_destroy(gsx_render_context_t context);

static const gsx_renderer_i gsx_metal_renderer_iface = {
    gsx_metal_renderer_destroy,
    gsx_metal_renderer_create_context,
    gsx_metal_renderer_render,
    gsx_metal_renderer_backward
};

static const gsx_render_context_i gsx_metal_render_context_iface = {
    gsx_metal_render_context_destroy
};

gsx_error gsx_metal_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer)
{
    gsx_metal_renderer *metal_renderer = NULL;
    gsx_renderer_capabilities capabilities = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_renderer == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_renderer and desc must be non-null");
    }
    *out_renderer = NULL;

    if(desc->output_data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer currently supports only float32 output_data_type");
    }
    if((desc->feature_flags & GSX_RENDERER_FEATURE_ANTIALIASING) != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer antialiasing feature flag is not implemented");
    }
    if(desc->enable_alpha_output || desc->enable_invdepth_output) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer currently supports only rgb output");
    }

    metal_renderer = (gsx_metal_renderer *)calloc(1, sizeof(*metal_renderer));
    if(metal_renderer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal renderer");
    }

    capabilities.supported_precisions = GSX_RENDER_PRECISION_FLAG_FLOAT32;
    capabilities.supports_alpha_output = false;
    capabilities.supports_invdepth_output = false;
    capabilities.supports_cov3d_input = false;
    error = gsx_renderer_base_init(&metal_renderer->base, &gsx_metal_renderer_iface, backend, desc, &capabilities);
    if(!gsx_error_is_success(error)) {
        free(metal_renderer);
        return error;
    }

    error = gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &metal_renderer->device_buffer_type);
    if(!gsx_error_is_success(error)) {
        gsx_renderer_base_deinit(&metal_renderer->base);
        free(metal_renderer);
        return error;
    }

    *out_renderer = &metal_renderer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_renderer_destroy(gsx_renderer_t renderer)
{
    gsx_metal_renderer *metal_renderer = (gsx_metal_renderer *)renderer;

    if(renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer must be non-null");
    }

    gsx_renderer_base_deinit(&metal_renderer->base);
    free(metal_renderer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_renderer_create_context(gsx_renderer_t renderer, gsx_render_context_t *out_context)
{
    gsx_metal_renderer *metal_renderer = (gsx_metal_renderer *)renderer;
    gsx_metal_render_context *metal_context = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(renderer == NULL || out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer and out_context must be non-null");
    }
    *out_context = NULL;

    metal_context = (gsx_metal_render_context *)calloc(1, sizeof(*metal_context));
    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate metal render context");
    }

    error = gsx_render_context_base_init(&metal_context->base, &gsx_metal_render_context_iface, renderer);
    if(!gsx_error_is_success(error)) {
        free(metal_context);
        return error;
    }

    error = gsx_metal_render_context_init(
        metal_context,
        metal_renderer->device_buffer_type,
        renderer->info.width,
        renderer->info.height);
    if(!gsx_error_is_success(error)) {
        gsx_render_context_base_deinit(&metal_context->base);
        free(metal_context);
        return error;
    }

    *out_context = &metal_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_context_destroy(gsx_render_context_t context)
{
    gsx_metal_render_context *metal_context = (gsx_metal_render_context *)context;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    error = gsx_metal_render_context_dispose(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    gsx_render_context_base_deinit(&metal_context->base);
    free(metal_context);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request)
{
    return gsx_metal_renderer_forward(renderer, context, request);
}

static gsx_error gsx_metal_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request)
{
    (void)renderer;
    (void)context;
    (void)request;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal renderer backward is not implemented");
}
