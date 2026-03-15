#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GSX_CUDA_RENDER_IMAGE_TILE = 8,
    GSX_CUDA_RENDER_IMAGE_TILE_LOG2 = 3,
    GSX_CUDA_RENDER_IMAGE_TILE_MASK = GSX_CUDA_RENDER_IMAGE_TILE - 1
};

typedef struct gsx_cuda_resize_blob {
    gsx_arena_t arena;
    gsx_tensor_t blob;
} gsx_cuda_resize_blob;

typedef struct gsx_cuda_pose_block {
    float4 w2c[4];
    float4 cam_position;
    float4 w2c_grad[4];
} gsx_cuda_pose_block;

typedef struct gsx_cuda_renderer {
    struct gsx_renderer base;
    gsx_backend_buffer_type_t device_buffer_type;
} gsx_cuda_renderer;

typedef struct gsx_cuda_render_context {
    struct gsx_render_context base;
    gsx_cuda_resize_blob per_primitive_blob;
    gsx_cuda_resize_blob per_tile_blob;
    gsx_cuda_resize_blob per_instance_blob;
    gsx_cuda_resize_blob per_bucket_blob;
    gsx_arena_t helper_arena;
    gsx_arena_t retain_arena;
    gsx_tensor_t helper_image_tiled;
    gsx_tensor_t helper_alpha_tiled;
    gsx_tensor_t helper_grad_mean2d;
    gsx_tensor_t helper_grad_conic;
    gsx_tensor_t helper_grad_color;
    gsx_tensor_t helper_absgrad_mean2d;
    gsx_tensor_t helper_grad_w2c;
    gsx_tensor_t saved_mean3d;
    gsx_tensor_t saved_rotation;
    gsx_tensor_t saved_logscale;
    gsx_tensor_t saved_sh0;
    gsx_tensor_t saved_sh1;
    gsx_tensor_t saved_sh2;
    gsx_tensor_t saved_sh3;
    gsx_tensor_t saved_opacity;
    gsx_tensor_t saved_image_tiled;
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    gsx_vec3 background_color;
    gsx_float_t near_plane;
    gsx_float_t far_plane;
    gsx_index_t sh_degree;
    bool has_train_state;
    bool train_state_borrowed;
    int n_visible_primitives;
    int n_instances;
    int n_buckets;
    int primitive_selector;
    int instance_selector;
    cudaStream_t helper_stream;
    cudaEvent_t memset_per_tile_done;
    cudaEvent_t copy_n_instances_done;
    cudaEvent_t preprocess_done;
    char *zero_copy;
    gsx_cuda_pose_block *host_pose_block;
    gsx_cuda_pose_block *device_pose_block;
} gsx_cuda_render_context;

static gsx_error gsx_cuda_renderer_destroy(gsx_renderer_t renderer);
static gsx_error gsx_cuda_renderer_create_context(gsx_renderer_t renderer, gsx_render_context_t *out_context);
static gsx_error gsx_cuda_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request);
static gsx_error gsx_cuda_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request);
static gsx_error gsx_cuda_render_context_destroy(gsx_render_context_t context);
static void gsx_cuda_render_free_tensor_handle(gsx_tensor_t *tensor);

static const gsx_renderer_i gsx_cuda_renderer_iface = {
    gsx_cuda_renderer_destroy,
    gsx_cuda_renderer_create_context,
    gsx_cuda_renderer_render,
    gsx_cuda_renderer_backward
};

static const gsx_render_context_i gsx_cuda_render_context_iface = {
    gsx_cuda_render_context_destroy
};

static void gsx_cuda_render_log_cleanup_failure(const char *operation, gsx_error error)
{
    fprintf(
        stderr,
        "gsx cuda renderer cleanup warning: %s failed with code %d%s%s\n",
        operation,
        (int)error.code,
        error.message != NULL ? " - " : "",
        error.message != NULL ? error.message : "");
}

static bool gsx_cuda_render_tensor_is_device(gsx_tensor_t tensor)
{
    return tensor != NULL
        && tensor->backing_buffer != NULL
        && gsx_cuda_backend_buffer_get_type_class(tensor->backing_buffer) == GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static unsigned char *gsx_cuda_render_tensor_device_bytes(gsx_tensor_t tensor)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(tensor->backing_buffer);

    return (unsigned char *)cuda_buffer->ptr + (size_t)tensor->offset_bytes;
}

static float *gsx_cuda_render_tensor_device_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cuda_render_tensor_device_bytes(tensor);
}

static gsx_error gsx_cuda_render_init_arena(gsx_backend_buffer_type_t buffer_type, bool dry_run, gsx_arena_t *out_arena)
{
    gsx_arena_desc arena_desc = { 0 };

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.requested_alignment_bytes = 128;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = dry_run;
    return gsx_arena_init(out_arena, buffer_type, &arena_desc);
}

static gsx_error gsx_cuda_render_make_float_tensor(gsx_arena_t arena, gsx_size_t element_count, gsx_tensor_t *out_tensor)
{
    gsx_tensor_desc desc = { 0 };

    if(out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor must be non-null");
    }
    *out_tensor = NULL;
    if(element_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(element_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "internal float tensor exceeds rank-1 limits");
    }

    desc.rank = 1;
    desc.shape[0] = (gsx_index_t)element_count;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.requested_alignment_bytes = 128;
    desc.arena = arena;
    return gsx_tensor_init(out_tensor, &desc);
}

static gsx_error gsx_cuda_render_propagate_error(gsx_error error, const char *message)
{
    if(gsx_error_is_success(error)) {
        return error;
    }
    return gsx_make_error(error.code, message);
}

static gsx_error gsx_cuda_render_make_byte_tensor(gsx_arena_t arena, gsx_size_t size_bytes, gsx_tensor_t *out_tensor)
{
    gsx_tensor_desc desc = { 0 };
    gsx_size_t padded_element_count = 0;

    if(out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor must be non-null");
    }
    *out_tensor = NULL;
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_add_overflows(size_bytes, sizeof(float) - 1u, &padded_element_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "internal blob tensor size overflows");
    }
    padded_element_count /= sizeof(float);
    if(padded_element_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "internal blob tensor exceeds rank-1 limits");
    }

    desc.rank = 1;
    desc.shape[0] = (gsx_index_t)padded_element_count;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.requested_alignment_bytes = 128;
    desc.arena = arena;
    return gsx_tensor_init(out_tensor, &desc);
}

static gsx_error gsx_cuda_render_plan_forward_helper_required_bytes(
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t image_element_count,
    gsx_size_t alpha_element_count,
    gsx_size_t *out_required_bytes)
{
    gsx_arena_t dry_run_arena = NULL;
    gsx_tensor_t image_tiled = NULL;
    gsx_tensor_t alpha_tiled = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_required_bytes must be non-null");
    }

    *out_required_bytes = 0;
    error = gsx_cuda_render_init_arena(buffer_type, true, &dry_run_arena);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_arena_reset(dry_run_arena);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_cuda_render_make_float_tensor(dry_run_arena, image_element_count, &image_tiled);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_cuda_render_make_float_tensor(dry_run_arena, alpha_element_count, &alpha_tiled);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);

cleanup:
    gsx_cuda_render_free_tensor_handle(&alpha_tiled);
    gsx_cuda_render_free_tensor_handle(&image_tiled);
    if(dry_run_arena != NULL) {
        gsx_error cleanup_error = gsx_arena_free(dry_run_arena);

        if(gsx_error_is_success(error) && !gsx_error_is_success(cleanup_error)) {
            error = cleanup_error;
        }
    }
    return error;
}

static gsx_error gsx_cuda_render_plan_backward_helper_required_bytes(
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t grad_mean2d_element_count,
    gsx_size_t grad_conic_element_count,
    gsx_size_t grad_color_element_count,
    gsx_size_t absgrad_mean2d_element_count,
    gsx_size_t grad_w2c_element_count,
    gsx_size_t *out_required_bytes)
{
    gsx_arena_t dry_run_arena = NULL;
    gsx_tensor_t grad_mean2d = NULL;
    gsx_tensor_t grad_conic = NULL;
    gsx_tensor_t grad_color = NULL;
    gsx_tensor_t absgrad_mean2d = NULL;
    gsx_tensor_t grad_w2c = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(buffer_type == NULL || out_required_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer_type and out_required_bytes must be non-null");
    }

    *out_required_bytes = 0;
    error = gsx_cuda_render_init_arena(buffer_type, true, &dry_run_arena);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_arena_reset(dry_run_arena);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }

    error = gsx_cuda_render_make_float_tensor(dry_run_arena, grad_mean2d_element_count, &grad_mean2d);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_cuda_render_make_float_tensor(dry_run_arena, grad_conic_element_count, &grad_conic);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_cuda_render_make_float_tensor(dry_run_arena, grad_color_element_count, &grad_color);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_cuda_render_make_float_tensor(dry_run_arena, absgrad_mean2d_element_count, &absgrad_mean2d);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_cuda_render_make_float_tensor(dry_run_arena, grad_w2c_element_count, &grad_w2c);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_arena_get_required_bytes(dry_run_arena, out_required_bytes);

cleanup:
    gsx_cuda_render_free_tensor_handle(&grad_w2c);
    gsx_cuda_render_free_tensor_handle(&absgrad_mean2d);
    gsx_cuda_render_free_tensor_handle(&grad_color);
    gsx_cuda_render_free_tensor_handle(&grad_conic);
    gsx_cuda_render_free_tensor_handle(&grad_mean2d);
    if(dry_run_arena != NULL) {
        gsx_error cleanup_error = gsx_arena_free(dry_run_arena);

        if(gsx_error_is_success(error) && !gsx_error_is_success(cleanup_error)) {
            error = cleanup_error;
        }
    }
    return error;
}

static void gsx_cuda_render_free_tensor_handle(gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        gsx_error error = gsx_tensor_free(*tensor);

        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_log_cleanup_failure("gsx_tensor_free", error);
        }
        *tensor = NULL;
    }
}

static void gsx_cuda_render_dispose_resize_blob(gsx_cuda_resize_blob *blob)
{
    if(blob == NULL) {
        return;
    }
    gsx_cuda_render_free_tensor_handle(&blob->blob);
    if(blob->arena != NULL) {
        gsx_error error = gsx_arena_free(blob->arena);

        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_log_cleanup_failure("gsx_arena_free", error);
        }
        blob->arena = NULL;
    }
}

static gsx_error gsx_cuda_render_reset_resize_blob(gsx_cuda_resize_blob *blob, gsx_size_t size_bytes)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(blob == NULL || blob->arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "resize blob arena is not initialized");
    }
    gsx_cuda_render_free_tensor_handle(&blob->blob);
    error = gsx_arena_reset(blob->arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_cuda_render_make_byte_tensor(blob->arena, size_bytes, &blob->blob);
}

static char *gsx_cuda_render_resize_blob_callback(void *user_data, gsx_size_t size_bytes)
{
    gsx_cuda_resize_blob *blob = (gsx_cuda_resize_blob *)user_data;
    gsx_error error = gsx_cuda_render_reset_resize_blob(blob, size_bytes);

    if(!gsx_error_is_success(error) || blob->blob == NULL) {
        fprintf(
            stderr,
            "gsx cuda resize callback failed: size=%llu code=%d%s%s\n",
            (unsigned long long)size_bytes,
            (int)error.code,
            error.message != NULL ? " - " : "",
            error.message != NULL ? error.message : "");
        return NULL;
    }
    return (char *)gsx_cuda_render_tensor_device_bytes(blob->blob);
}

static gsx_error gsx_cuda_render_clone_tensor(gsx_tensor_t src, gsx_arena_t arena, gsx_tensor_t *out_clone)
{
    gsx_tensor_desc desc = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(src == NULL || arena == NULL || out_clone == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src, arena, and out_clone must be non-null");
    }

    error = gsx_tensor_get_desc(src, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    desc.arena = arena;
    error = gsx_tensor_init(out_clone, &desc);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_tensor_copy(src, *out_clone);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_render_free_tensor_handle(out_clone);
        return error;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cuda_render_clear_helper(gsx_cuda_render_context *cuda_context)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_context == NULL) {
        return;
    }
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_image_tiled);
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_alpha_tiled);
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_grad_mean2d);
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_grad_conic);
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_grad_color);
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_absgrad_mean2d);
    gsx_cuda_render_free_tensor_handle(&cuda_context->helper_grad_w2c);
    if(cuda_context->helper_arena != NULL) {
        error = gsx_arena_reset(cuda_context->helper_arena);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_log_cleanup_failure("gsx_arena_reset", error);
        }
    }
}

static void gsx_cuda_render_clear_snapshot(gsx_cuda_render_context *cuda_context)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(cuda_context == NULL) {
        return;
    }
    if(!cuda_context->train_state_borrowed) {
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_mean3d);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_rotation);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_logscale);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_sh0);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_sh1);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_sh2);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_sh3);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_opacity);
        gsx_cuda_render_free_tensor_handle(&cuda_context->saved_image_tiled);
    }
    cuda_context->saved_mean3d = NULL;
    cuda_context->saved_rotation = NULL;
    cuda_context->saved_logscale = NULL;
    cuda_context->saved_sh0 = NULL;
    cuda_context->saved_sh1 = NULL;
    cuda_context->saved_sh2 = NULL;
    cuda_context->saved_sh3 = NULL;
    cuda_context->saved_opacity = NULL;
    cuda_context->saved_image_tiled = NULL;
    if(cuda_context->retain_arena != NULL) {
        error = gsx_arena_reset(cuda_context->retain_arena);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_log_cleanup_failure("gsx_arena_reset", error);
        }
    }
    cuda_context->has_train_state = false;
    cuda_context->train_state_borrowed = false;
    memset(&cuda_context->intrinsics, 0, sizeof(cuda_context->intrinsics));
    memset(&cuda_context->pose, 0, sizeof(cuda_context->pose));
    cuda_context->background_color.x = 0.0f;
    cuda_context->background_color.y = 0.0f;
    cuda_context->background_color.z = 0.0f;
    cuda_context->near_plane = 0.0f;
    cuda_context->far_plane = 0.0f;
    cuda_context->sh_degree = 0;
    cuda_context->n_visible_primitives = 0;
    cuda_context->n_instances = 0;
    cuda_context->n_buckets = 0;
    cuda_context->primitive_selector = 0;
    cuda_context->instance_selector = 0;
}

static gsx_error gsx_cuda_render_reserve_snapshot_arena(gsx_cuda_render_context *cuda_context, const gsx_render_forward_request *request)
{
    gsx_tensor_t tensors[9];
    gsx_size_t required_bytes = 0;
    gsx_size_t size_bytes = 0;
    gsx_size_t alignment_bytes = 0;
    gsx_index_t i = 0;
    gsx_tensor_desc desc = { 0 };
    gsx_arena_info arena_info;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    tensors[0] = request->gs_mean3d;
    tensors[1] = request->gs_rotation;
    tensors[2] = request->gs_logscale;
    tensors[3] = request->gs_sh0;
    tensors[4] = request->gs_sh1;
    tensors[5] = request->gs_sh2;
    tensors[6] = request->gs_sh3;
    tensors[7] = request->gs_opacity;
    tensors[8] = cuda_context->helper_image_tiled;

    memset(&arena_info, 0, sizeof(arena_info));
    error = gsx_arena_get_info(cuda_context->retain_arena, &arena_info);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(i = 0; i < (gsx_index_t)(sizeof(tensors) / sizeof(tensors[0])); ++i) {
        if(tensors[i] == NULL) {
            continue;
        }
        error = gsx_tensor_get_desc(tensors[i], &desc);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_tensor_get_size_bytes(tensors[i], &size_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        alignment_bytes = arena_info.effective_alignment_bytes;
        if(desc.requested_alignment_bytes > alignment_bytes) {
            alignment_bytes = desc.requested_alignment_bytes;
        }
        if(gsx_round_up_overflows(required_bytes, alignment_bytes, &required_bytes)
            || gsx_size_add_overflows(required_bytes, size_bytes, &required_bytes)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer retained snapshot size overflows");
        }
    }

    return gsx_arena_reserve(cuda_context->retain_arena, required_bytes);
}

static gsx_error gsx_cuda_render_snapshot_request(gsx_cuda_render_context *cuda_context, const gsx_render_forward_request *request)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    void *stream = NULL;
    cudaError_t cuda_err = cudaSuccess;
    int n_visible_primitives = 0;
    int n_instances = 0;
    int n_buckets = 0;
    int primitive_selector = 0;
    int instance_selector = 0;

    n_visible_primitives = cuda_context->n_visible_primitives;
    n_instances = cuda_context->n_instances;
    n_buckets = cuda_context->n_buckets;
    primitive_selector = cuda_context->primitive_selector;
    instance_selector = cuda_context->instance_selector;

    gsx_cuda_render_clear_snapshot(cuda_context);
    if(request->borrow_train_state) {
        cuda_context->saved_mean3d = request->gs_mean3d;
        cuda_context->saved_rotation = request->gs_rotation;
        cuda_context->saved_logscale = request->gs_logscale;
        cuda_context->saved_sh0 = request->gs_sh0;
        cuda_context->saved_sh1 = request->gs_sh1;
        cuda_context->saved_sh2 = request->gs_sh2;
        cuda_context->saved_sh3 = request->gs_sh3;
        cuda_context->saved_opacity = request->gs_opacity;
        cuda_context->saved_image_tiled = cuda_context->helper_image_tiled;
        cuda_context->train_state_borrowed = true;
    } else {
        error = gsx_cuda_render_reserve_snapshot_arena(cuda_context, request);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to reserve retained TRAIN snapshot arena");
        }
        error = gsx_cuda_render_clone_tensor(request->gs_mean3d, cuda_context->retain_arena, &cuda_context->saved_mean3d);
        if(!gsx_error_is_success(error)) {
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_mean3d");
        }
        error = gsx_cuda_render_clone_tensor(request->gs_rotation, cuda_context->retain_arena, &cuda_context->saved_rotation);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_clear_snapshot(cuda_context);
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_rotation");
        }
        error = gsx_cuda_render_clone_tensor(request->gs_logscale, cuda_context->retain_arena, &cuda_context->saved_logscale);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_clear_snapshot(cuda_context);
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_logscale");
        }
        error = gsx_cuda_render_clone_tensor(request->gs_sh0, cuda_context->retain_arena, &cuda_context->saved_sh0);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_clear_snapshot(cuda_context);
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_sh0");
        }
        if(request->gs_sh1 != NULL) {
            error = gsx_cuda_render_clone_tensor(request->gs_sh1, cuda_context->retain_arena, &cuda_context->saved_sh1);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_render_clear_snapshot(cuda_context);
                return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_sh1");
            }
        }
        if(request->gs_sh2 != NULL) {
            error = gsx_cuda_render_clone_tensor(request->gs_sh2, cuda_context->retain_arena, &cuda_context->saved_sh2);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_render_clear_snapshot(cuda_context);
                return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_sh2");
            }
        }
        if(request->gs_sh3 != NULL) {
            error = gsx_cuda_render_clone_tensor(request->gs_sh3, cuda_context->retain_arena, &cuda_context->saved_sh3);
            if(!gsx_error_is_success(error)) {
                gsx_cuda_render_clear_snapshot(cuda_context);
                return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_sh3");
            }
        }
        error = gsx_cuda_render_clone_tensor(request->gs_opacity, cuda_context->retain_arena, &cuda_context->saved_opacity);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_clear_snapshot(cuda_context);
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained gs_opacity");
        }
        error = gsx_cuda_render_clone_tensor(cuda_context->helper_image_tiled, cuda_context->retain_arena, &cuda_context->saved_image_tiled);
        if(!gsx_error_is_success(error)) {
            gsx_cuda_render_clear_snapshot(cuda_context);
            return gsx_cuda_render_propagate_error(error, "cuda renderer failed to clone retained RGB image");
        }
        cuda_context->train_state_borrowed = false;
    }
    error = gsx_backend_get_major_stream(cuda_context->base.renderer->backend, &stream);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_render_clear_snapshot(cuda_context);
        return error;
    }
    cuda_err = gsx_cuda_render_compose_background_tiled_f32_kernel_launch(
        gsx_cuda_render_tensor_device_f32(cuda_context->saved_image_tiled),
        gsx_cuda_render_tensor_device_f32(cuda_context->helper_alpha_tiled),
        request->intrinsics->width,
        request->intrinsics->height,
        request->background_color,
        (cudaStream_t)stream
    );
    if(cuda_err != cudaSuccess) {
        gsx_cuda_render_clear_snapshot(cuda_context);
        return gsx_cuda_make_error(cuda_err, "cuda renderer failed to compose retained RGB image with background");
    }

    cuda_context->intrinsics = *request->intrinsics;
    cuda_context->pose = *request->pose;
    cuda_context->background_color = request->background_color;
    cuda_context->near_plane = request->near_plane;
    cuda_context->far_plane = request->far_plane;
    cuda_context->sh_degree = request->sh_degree;
    cuda_context->n_visible_primitives = n_visible_primitives;
    cuda_context->n_instances = n_instances;
    cuda_context->n_buckets = n_buckets;
    cuda_context->primitive_selector = primitive_selector;
    cuda_context->instance_selector = instance_selector;
    cuda_context->has_train_state = true;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_require_device_tensor(gsx_tensor_t tensor, const char *message)
{
    if(!gsx_cuda_render_tensor_is_device(tensor)) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, message);
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_validate_forward_request(const gsx_render_forward_request *request)
{
    if(request->precision != GSX_RENDER_PRECISION_FLOAT32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer currently supports float32 precision only");
    }
    if(request->forward_type == GSX_RENDER_FORWARD_TYPE_METRIC) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer metric mode is not implemented");
    }
    if(request->gs_cov3d != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer does not implement gs_cov3d input yet");
    }
    if(request->out_invdepth != NULL || request->out_alpha != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer does not implement inverse-depth or alpha outputs yet");
    }
    if(request->metric_map != NULL || request->gs_metric_accumulator != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer does not implement metric mode yet");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_validate_backward_request(gsx_renderer_t renderer, const gsx_cuda_render_context *cuda_context, const gsx_render_backward_request *request)
{
    gsx_index_t rgb_shape[3];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(request->grad_invdepth != NULL || request->grad_alpha != NULL || request->grad_gs_cov3d != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer implements RGB backward only");
    }
    if(request->grad_rgb == NULL
        || request->grad_gs_mean3d == NULL
        || request->grad_gs_rotation == NULL
        || request->grad_gs_logscale == NULL
        || request->grad_gs_sh0 == NULL
        || request->grad_gs_opacity == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cuda renderer backward requires RGB and core Gaussian gradient sinks");
    }

    rgb_shape[0] = 3;
    rgb_shape[1] = renderer->info.height;
    rgb_shape[2] = renderer->info.width;
    if(request->grad_rgb->data_type != GSX_DATA_TYPE_F32
        || request->grad_rgb->storage_format != GSX_STORAGE_FORMAT_CHW
        || request->grad_rgb->rank != 3
        || request->grad_rgb->shape[0] != rgb_shape[0]
        || request->grad_rgb->shape[1] != rgb_shape[1]
        || request->grad_rgb->shape[2] != rgb_shape[2]) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_rgb must be float32 CHW with shape [3,H,W]");
    }

    error = gsx_cuda_render_require_device_tensor(request->grad_rgb, "cuda renderer requires device-backed grad_rgb");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->grad_gs_mean3d, "cuda renderer requires device-backed grad_gs_mean3d");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->grad_gs_rotation, "cuda renderer requires device-backed grad_gs_rotation");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->grad_gs_logscale, "cuda renderer requires device-backed grad_gs_logscale");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->grad_gs_sh0, "cuda renderer requires device-backed grad_gs_sh0");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->grad_gs_opacity, "cuda renderer requires device-backed grad_gs_opacity");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(cuda_context->sh_degree >= 1) {
        if(request->grad_gs_sh1 == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh1 must be non-null when sh_degree is at least 1");
        }
        error = gsx_cuda_render_require_device_tensor(request->grad_gs_sh1, "cuda renderer requires device-backed grad_gs_sh1");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->grad_gs_sh1 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh1 must be null when sh_degree is 0");
    }
    if(cuda_context->sh_degree >= 2) {
        if(request->grad_gs_sh2 == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh2 must be non-null when sh_degree is at least 2");
        }
        error = gsx_cuda_render_require_device_tensor(request->grad_gs_sh2, "cuda renderer requires device-backed grad_gs_sh2");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->grad_gs_sh2 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh2 must be null when sh_degree is less than 2");
    }
    if(cuda_context->sh_degree >= 3) {
        if(request->grad_gs_sh3 == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh3 must be non-null when sh_degree is 3");
        }
        error = gsx_cuda_render_require_device_tensor(request->grad_gs_sh3, "cuda renderer requires device-backed grad_gs_sh3");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->grad_gs_sh3 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh3 must be null when sh_degree is less than 3");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_sync_major_stream(gsx_backend_t backend)
{
    void *stream = NULL;
    gsx_error error = gsx_backend_get_major_stream(backend, &stream);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_cuda_make_error(cudaStreamSynchronize((cudaStream_t)stream), "cudaStreamSynchronize failed");
}

static void gsx_cuda_render_quaternion_to_matrix(const gsx_quat *quat, float out_matrix[9])
{
    float qx = quat->x;
    float qy = quat->y;
    float qz = quat->z;
    float qw = quat->w;
    float norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    float inv_norm = 1.0f;

    if(norm_sq > 1.0e-20f) {
        inv_norm = 1.0f / sqrtf(norm_sq);
        qx *= inv_norm;
        qy *= inv_norm;
        qz *= inv_norm;
        qw *= inv_norm;
    } else {
        qx = 0.0f;
        qy = 0.0f;
        qz = 0.0f;
        qw = 1.0f;
    }

    out_matrix[0] = 1.0f - 2.0f * (qy * qy + qz * qz);
    out_matrix[1] = 2.0f * (qx * qy - qw * qz);
    out_matrix[2] = 2.0f * (qx * qz + qw * qy);
    out_matrix[3] = 2.0f * (qx * qy + qw * qz);
    out_matrix[4] = 1.0f - 2.0f * (qx * qx + qz * qz);
    out_matrix[5] = 2.0f * (qy * qz - qw * qx);
    out_matrix[6] = 2.0f * (qx * qz - qw * qy);
    out_matrix[7] = 2.0f * (qy * qz + qw * qx);
    out_matrix[8] = 1.0f - 2.0f * (qx * qx + qy * qy);
}

static void gsx_cuda_render_fill_pose_block(gsx_cuda_pose_block *pose_block, const gsx_camera_pose *pose)
{
    float rotation[9];
    float camera_position_x = 0.0f;
    float camera_position_y = 0.0f;
    float camera_position_z = 0.0f;

    gsx_cuda_render_quaternion_to_matrix(&pose->rot, rotation);
    pose_block->w2c[0] = make_float4(rotation[0], rotation[1], rotation[2], pose->transl.x);
    pose_block->w2c[1] = make_float4(rotation[3], rotation[4], rotation[5], pose->transl.y);
    pose_block->w2c[2] = make_float4(rotation[6], rotation[7], rotation[8], pose->transl.z);
    pose_block->w2c[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    camera_position_x = -(rotation[0] * pose->transl.x + rotation[3] * pose->transl.y + rotation[6] * pose->transl.z);
    camera_position_y = -(rotation[1] * pose->transl.x + rotation[4] * pose->transl.y + rotation[7] * pose->transl.z);
    camera_position_z = -(rotation[2] * pose->transl.x + rotation[5] * pose->transl.y + rotation[8] * pose->transl.z);
    pose_block->cam_position = make_float4(camera_position_x, camera_position_y, camera_position_z, 0.0f);
    memset(pose_block->w2c_grad, 0, sizeof(pose_block->w2c_grad));
}

static gsx_error gsx_cuda_render_prepare_pose_block(gsx_cuda_render_context *cuda_context, gsx_backend_t backend, const gsx_camera_pose *pose)
{
    void *stream = NULL;
    gsx_error error = gsx_backend_get_major_stream(backend, &stream);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    gsx_cuda_render_fill_pose_block(cuda_context->host_pose_block, pose);
    return gsx_cuda_make_error(
        cudaMemcpyAsync(
            cuda_context->device_pose_block,
            cuda_context->host_pose_block,
            sizeof(*cuda_context->device_pose_block),
            cudaMemcpyHostToDevice,
            (cudaStream_t)stream),
        "cudaMemcpyAsync pose upload failed");
}

static gsx_error gsx_cuda_render_compute_tiled_layout(
    const gsx_renderer *renderer,
    gsx_size_t *out_width_in_tile,
    gsx_size_t *out_height_in_tile,
    gsx_size_t *out_tile_count,
    gsx_size_t *out_channel_stride)
{
    gsx_size_t width_in_tile = 0;
    gsx_size_t height_in_tile = 0;
    gsx_size_t tile_count = 0;
    gsx_size_t channel_stride = 0;

    if(renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer must be non-null");
    }

    width_in_tile = (gsx_size_t)(((int)renderer->info.width + GSX_CUDA_RENDER_IMAGE_TILE_MASK) >> GSX_CUDA_RENDER_IMAGE_TILE_LOG2);
    height_in_tile = (gsx_size_t)(((int)renderer->info.height + GSX_CUDA_RENDER_IMAGE_TILE_MASK) >> GSX_CUDA_RENDER_IMAGE_TILE_LOG2);
    if(width_in_tile > (gsx_size_t)INT32_MAX || height_in_tile > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer tile grid exceeds kernel integer range");
    }
    if(gsx_size_mul_overflows(width_in_tile, height_in_tile, &tile_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer tile grid size overflows");
    }
    if(tile_count > (gsx_size_t)UINT16_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer tile count exceeds 16-bit instance key range");
    }
    if(gsx_size_mul_overflows(tile_count, (gsx_size_t)(GSX_CUDA_RENDER_IMAGE_TILE * GSX_CUDA_RENDER_IMAGE_TILE), &channel_stride)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer tiled image stride overflows");
    }
    if(channel_stride > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer tiled image stride exceeds kernel integer range");
    }

    if(out_width_in_tile != NULL) {
        *out_width_in_tile = width_in_tile;
    }
    if(out_height_in_tile != NULL) {
        *out_height_in_tile = height_in_tile;
    }
    if(out_tile_count != NULL) {
        *out_tile_count = tile_count;
    }
    if(out_channel_stride != NULL) {
        *out_channel_stride = channel_stride;
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_alloc_forward_scratch(gsx_cuda_render_context *cuda_context, gsx_renderer_t renderer)
{
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_size_t channel_stride = 0;
    gsx_size_t image_element_count = 0;
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(gsx_size_mul_overflows((gsx_size_t)renderer->info.width, (gsx_size_t)renderer->info.height, &channel_stride)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer image stride overflows");
    }
    if(gsx_size_mul_overflows(3u, channel_stride, &image_element_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer RGB scratch size overflows");
    }
    gsx_cuda_render_clear_helper(cuda_context);
    error = gsx_arena_get_buffer_type(cuda_context->helper_arena, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_plan_forward_helper_required_bytes(buffer_type, image_element_count, channel_stride, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to plan forward helper arena");
    }
    error = gsx_arena_reserve(cuda_context->helper_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to reserve forward helper arena");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, image_element_count, &cuda_context->helper_image_tiled);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate RGB scratch");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, channel_stride, &cuda_context->helper_alpha_tiled);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate alpha scratch");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_alloc_backward_scratch(gsx_cuda_render_context *cuda_context, gsx_size_t gaussian_count)
{
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_size_t grad_mean2d_element_count = 0;
    gsx_size_t grad_conic_element_count = 0;
    gsx_size_t grad_color_element_count = 0;
    gsx_size_t absgrad_mean2d_element_count = 0;
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(gsx_size_mul_overflows(gaussian_count, 2u, &grad_mean2d_element_count)
        || gsx_size_mul_overflows(gaussian_count, 3u, &grad_conic_element_count)
        || gsx_size_mul_overflows(gaussian_count, 3u, &grad_color_element_count)
        || gsx_size_mul_overflows(gaussian_count, 2u, &absgrad_mean2d_element_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer backward scratch element count overflows");
    }
    gsx_cuda_render_clear_helper(cuda_context);
    error = gsx_arena_get_buffer_type(cuda_context->helper_arena, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_plan_backward_helper_required_bytes(
        buffer_type,
        grad_mean2d_element_count,
        grad_conic_element_count,
        grad_color_element_count,
        absgrad_mean2d_element_count,
        16u,
        &required_bytes);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to plan backward helper arena");
    }
    error = gsx_arena_reserve(cuda_context->helper_arena, required_bytes);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to reserve backward helper arena");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, grad_mean2d_element_count, &cuda_context->helper_grad_mean2d);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate mean2d gradient scratch");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, grad_conic_element_count, &cuda_context->helper_grad_conic);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate conic gradient scratch");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, grad_color_element_count, &cuda_context->helper_grad_color);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate color gradient scratch");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, absgrad_mean2d_element_count, &cuda_context->helper_absgrad_mean2d);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate abs-mean2d scratch");
    }
    error = gsx_cuda_render_make_float_tensor(cuda_context->helper_arena, 16u, &cuda_context->helper_grad_w2c);
    if(!gsx_error_is_success(error)) {
        return gsx_cuda_render_propagate_error(error, "cuda renderer failed to allocate w2c gradient scratch");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_renderer_destroy(gsx_renderer_t renderer)
{
    gsx_cuda_renderer *cuda_renderer = (gsx_cuda_renderer *)renderer;

    if(renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer must be non-null");
    }
    gsx_renderer_base_deinit(&cuda_renderer->base);
    free(cuda_renderer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_renderer_create_context(gsx_renderer_t renderer, gsx_render_context_t *out_context)
{
    gsx_cuda_renderer *cuda_renderer = (gsx_cuda_renderer *)renderer;
    gsx_cuda_render_context *cuda_context = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_context must be non-null");
    }
    *out_context = NULL;

    cuda_context = (gsx_cuda_render_context *)calloc(1, sizeof(*cuda_context));
    if(cuda_context == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda render context");
    }

    error = gsx_render_context_base_init(&cuda_context->base, &gsx_cuda_render_context_iface, renderer);
    if(!gsx_error_is_success(error)) {
        free(cuda_context);
        return error;
    }
    error = gsx_cuda_render_init_arena(cuda_renderer->device_buffer_type, false, &cuda_context->per_primitive_blob.arena);
    if(!gsx_error_is_success(error)) {
        gsx_render_context_base_deinit(&cuda_context->base);
        free(cuda_context);
        return error;
    }
    error = gsx_cuda_render_init_arena(cuda_renderer->device_buffer_type, false, &cuda_context->per_tile_blob.arena);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_primitive_blob);
        gsx_render_context_base_deinit(&cuda_context->base);
        free(cuda_context);
        return error;
    }
    error = gsx_cuda_render_init_arena(cuda_renderer->device_buffer_type, false, &cuda_context->per_instance_blob.arena);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_primitive_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_tile_blob);
        gsx_render_context_base_deinit(&cuda_context->base);
        free(cuda_context);
        return error;
    }
    error = gsx_cuda_render_init_arena(cuda_renderer->device_buffer_type, false, &cuda_context->per_bucket_blob.arena);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_primitive_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_tile_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_instance_blob);
        gsx_render_context_base_deinit(&cuda_context->base);
        free(cuda_context);
        return error;
    }
    error = gsx_cuda_render_init_arena(cuda_renderer->device_buffer_type, false, &cuda_context->helper_arena);
    if(!gsx_error_is_success(error)) {
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_primitive_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_tile_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_instance_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_bucket_blob);
        gsx_render_context_base_deinit(&cuda_context->base);
        free(cuda_context);
        return error;
    }
    error = gsx_cuda_render_init_arena(cuda_renderer->device_buffer_type, false, &cuda_context->retain_arena);
    if(!gsx_error_is_success(error)) {
        (void)gsx_arena_free(cuda_context->helper_arena);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_primitive_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_tile_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_instance_blob);
        gsx_cuda_render_dispose_resize_blob(&cuda_context->per_bucket_blob);
        gsx_render_context_base_deinit(&cuda_context->base);
        free(cuda_context);
        return error;
    }

    if(cudaStreamCreateWithFlags(&cuda_context->helper_stream, cudaStreamNonBlocking) != cudaSuccess
        || cudaEventCreateWithFlags(&cuda_context->memset_per_tile_done, cudaEventDisableTiming) != cudaSuccess
        || cudaEventCreateWithFlags(&cuda_context->copy_n_instances_done, cudaEventDisableTiming) != cudaSuccess
        || cudaEventCreateWithFlags(&cuda_context->preprocess_done, cudaEventDisableTiming) != cudaSuccess
        || cudaHostAlloc((void **)&cuda_context->zero_copy, 1024, cudaHostAllocMapped) != cudaSuccess
        || cudaHostAlloc((void **)&cuda_context->host_pose_block, sizeof(*cuda_context->host_pose_block), cudaHostAllocDefault) != cudaSuccess
        || cudaMalloc((void **)&cuda_context->device_pose_block, sizeof(*cuda_context->device_pose_block)) != cudaSuccess) {
        (void)gsx_cuda_render_context_destroy(&cuda_context->base);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to initialize CUDA render context resources");
    }

    *out_context = &cuda_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_context_destroy(gsx_render_context_t context)
{
    gsx_cuda_render_context *cuda_context = (gsx_cuda_render_context *)context;
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    gsx_cuda_render_clear_helper(cuda_context);
    gsx_cuda_render_clear_snapshot(cuda_context);
    if(cuda_context->helper_arena != NULL) {
        error = gsx_arena_free(cuda_context->helper_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        cuda_context->helper_arena = NULL;
    }
    if(cuda_context->retain_arena != NULL) {
        error = gsx_arena_free(cuda_context->retain_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        cuda_context->retain_arena = NULL;
    }
    gsx_cuda_render_dispose_resize_blob(&cuda_context->per_primitive_blob);
    gsx_cuda_render_dispose_resize_blob(&cuda_context->per_tile_blob);
    gsx_cuda_render_dispose_resize_blob(&cuda_context->per_instance_blob);
    gsx_cuda_render_dispose_resize_blob(&cuda_context->per_bucket_blob);
    if(cuda_context->device_pose_block != NULL) {
        (void)cudaFree(cuda_context->device_pose_block);
    }
    if(cuda_context->host_pose_block != NULL) {
        (void)cudaFreeHost(cuda_context->host_pose_block);
    }
    if(cuda_context->zero_copy != NULL) {
        (void)cudaFreeHost(cuda_context->zero_copy);
    }
    if(cuda_context->preprocess_done != NULL) {
        (void)cudaEventDestroy(cuda_context->preprocess_done);
    }
    if(cuda_context->copy_n_instances_done != NULL) {
        (void)cudaEventDestroy(cuda_context->copy_n_instances_done);
    }
    if(cuda_context->memset_per_tile_done != NULL) {
        (void)cudaEventDestroy(cuda_context->memset_per_tile_done);
    }
    if(cuda_context->helper_stream != NULL) {
        (void)cudaStreamDestroy(cuda_context->helper_stream);
    }
    gsx_render_context_base_deinit(&cuda_context->base);
    free(cuda_context);
    return first_error;
}

static gsx_error gsx_cuda_render_require_forward_device_tensors(const gsx_render_forward_request *request)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_render_require_device_tensor(request->gs_mean3d, "cuda renderer requires device-backed gs_mean3d");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->gs_rotation, "cuda renderer requires device-backed gs_rotation");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->gs_logscale, "cuda renderer requires device-backed gs_logscale");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->gs_sh0, "cuda renderer requires device-backed gs_sh0");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->gs_opacity, "cuda renderer requires device-backed gs_opacity");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_device_tensor(request->out_rgb, "cuda renderer requires device-backed out_rgb");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->gs_sh1 != NULL) {
        error = gsx_cuda_render_require_device_tensor(request->gs_sh1, "cuda renderer requires device-backed gs_sh1");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->gs_sh2 != NULL) {
        error = gsx_cuda_render_require_device_tensor(request->gs_sh2, "cuda renderer requires device-backed gs_sh2");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->gs_sh3 != NULL) {
        error = gsx_cuda_render_require_device_tensor(request->gs_sh3, "cuda renderer requires device-backed gs_sh3");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_render_finalize_forward(
    gsx_renderer_t renderer,
    gsx_cuda_render_context *cuda_context,
    const gsx_render_forward_request *request,
    cudaStream_t stream)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;

    cuda_err = gsx_cuda_render_tiled_to_chw_f32_kernel_launch(
        gsx_cuda_render_tensor_device_f32(cuda_context->helper_image_tiled),
        gsx_cuda_render_tensor_device_f32(cuda_context->helper_alpha_tiled),
        gsx_cuda_render_tensor_device_f32(request->out_rgb),
        renderer->info.width,
        renderer->info.height,
        request->background_color,
        stream
    );
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "render CHW output compose failed");
    }
    error = gsx_cuda_render_sync_major_stream(renderer->backend);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN) {
        return gsx_cuda_render_snapshot_request(cuda_context, request);
    }
    gsx_cuda_render_clear_snapshot(cuda_context);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cuda_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request)
{
    gsx_cuda_render_context *cuda_context = (gsx_cuda_render_context *)context;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error result = { GSX_ERROR_SUCCESS, NULL };
    void *stream = NULL;
    gsx_size_t gaussian_count = 0;
    cudaError_t cuda_err = cudaSuccess;

    error = gsx_cuda_render_validate_forward_request(request);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_require_forward_device_tensors(request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gaussian_count = (gsx_size_t)request->gs_mean3d->shape[0];
    if(gaussian_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "cuda renderer gaussian count exceeds kernel integer range");
    }
    error = gsx_cuda_render_compute_tiled_layout(renderer, NULL, NULL, NULL, NULL);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_backend_get_major_stream(renderer->backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_prepare_pose_block(cuda_context, renderer->backend, request->pose);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_alloc_forward_scratch(cuda_context, renderer);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_image_tiled);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_alpha_tiled);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(gaussian_count == 0) {
        cuda_context->n_visible_primitives = 0;
        cuda_context->n_instances = 0;
        cuda_context->n_buckets = 0;
        cuda_context->primitive_selector = 0;
        cuda_context->instance_selector = 0;
        return gsx_cuda_render_finalize_forward(renderer, cuda_context, request, (cudaStream_t)stream);
    }
    cuda_err = gsx_cuda_fastgs_forward_launch(
        gsx_cuda_render_resize_blob_callback,
        &cuda_context->per_primitive_blob,
        gsx_cuda_render_resize_blob_callback,
        &cuda_context->per_tile_blob,
        gsx_cuda_render_resize_blob_callback,
        &cuda_context->per_instance_blob,
        gsx_cuda_render_resize_blob_callback,
        &cuda_context->per_bucket_blob,
        (const float3 *)gsx_cuda_render_tensor_device_bytes(request->gs_mean3d),
        (const float3 *)gsx_cuda_render_tensor_device_bytes(request->gs_logscale),
        (const float4 *)gsx_cuda_render_tensor_device_bytes(request->gs_rotation),
        (const float *)gsx_cuda_render_tensor_device_bytes(request->gs_opacity),
        (const float *)gsx_cuda_render_tensor_device_bytes(request->gs_sh0),
        request->gs_sh1 != NULL ? (const float *)gsx_cuda_render_tensor_device_bytes(request->gs_sh1) : NULL,
        request->gs_sh2 != NULL ? (const float *)gsx_cuda_render_tensor_device_bytes(request->gs_sh2) : NULL,
        request->gs_sh3 != NULL ? (const float *)gsx_cuda_render_tensor_device_bytes(request->gs_sh3) : NULL,
        (const float4 *)cuda_context->device_pose_block,
        (const float3 *)((unsigned char *)cuda_context->device_pose_block + offsetof(gsx_cuda_pose_block, cam_position)),
        gsx_cuda_render_tensor_device_f32(cuda_context->helper_image_tiled),
        gsx_cuda_render_tensor_device_f32(cuda_context->helper_alpha_tiled),
        (int)gaussian_count,
        (int)((request->sh_degree + 1) * (request->sh_degree + 1)),
        renderer->info.width,
        renderer->info.height,
        request->intrinsics->fx,
        request->intrinsics->fy,
        request->intrinsics->cx,
        request->intrinsics->cy,
        request->near_plane,
        request->far_plane,
        (cudaStream_t)stream,
        cuda_context->helper_stream,
        cuda_context->zero_copy,
        cuda_context->memset_per_tile_done,
        cuda_context->copy_n_instances_done,
        cuda_context->preprocess_done,
        &cuda_context->n_visible_primitives,
        &cuda_context->n_instances,
        &cuda_context->n_buckets,
        &cuda_context->primitive_selector,
        &cuda_context->instance_selector
    );
    if(cuda_err != cudaSuccess) {
        result = gsx_cuda_make_error(cuda_err, "fastgs forward launch failed");
        goto cleanup;
    }
    error = gsx_cuda_render_finalize_forward(renderer, cuda_context, request, (cudaStream_t)stream);
    if(!gsx_error_is_success(error)) {
        result = error;
        goto cleanup;
    }
    result = gsx_make_error(GSX_ERROR_SUCCESS, NULL);
cleanup:
    return result;
}

static gsx_error gsx_cuda_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request)
{
    gsx_cuda_render_context *cuda_context = (gsx_cuda_render_context *)context;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error result = { GSX_ERROR_SUCCESS, NULL };
    void *stream = NULL;
    gsx_size_t gaussian_count = 0;
    cudaError_t cuda_err = cudaSuccess;

    if(!cuda_context->has_train_state) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backward requires a retained TRAIN forward on the same context");
    }
    error = gsx_cuda_render_validate_backward_request(renderer, cuda_context, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gaussian_count = (gsx_size_t)cuda_context->saved_mean3d->shape[0];
    error = gsx_backend_get_major_stream(renderer->backend, &stream);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_prepare_pose_block(cuda_context, renderer->backend, &cuda_context->pose);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_render_alloc_backward_scratch(cuda_context, gaussian_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_mean3d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_rotation);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_logscale);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(request->grad_gs_sh0);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(request->grad_gs_sh1 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh1);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->grad_gs_sh2 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh2);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    if(request->grad_gs_sh3 != NULL) {
        error = gsx_tensor_set_zero(request->grad_gs_sh3);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }
    error = gsx_tensor_set_zero(request->grad_gs_opacity);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_grad_mean2d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_grad_conic);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_grad_color);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_absgrad_mean2d);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(cuda_context->helper_grad_w2c);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(cuda_context->n_visible_primitives == 0 || cuda_context->n_instances == 0 || cuda_context->n_buckets == 0) {
        return gsx_cuda_render_sync_major_stream(renderer->backend);
    }

    cuda_err = gsx_cuda_fastgs_backward_launch(
        gsx_cuda_render_tensor_device_f32(request->grad_rgb),
        gsx_cuda_render_tensor_device_f32(cuda_context->saved_image_tiled),
        (const float3 *)gsx_cuda_render_tensor_device_bytes(cuda_context->saved_mean3d),
        (const float3 *)gsx_cuda_render_tensor_device_bytes(cuda_context->saved_logscale),
        (const float4 *)gsx_cuda_render_tensor_device_bytes(cuda_context->saved_rotation),
        cuda_context->saved_sh1 != NULL ? (const float *)gsx_cuda_render_tensor_device_bytes(cuda_context->saved_sh1) : NULL,
        cuda_context->saved_sh2 != NULL ? (const float *)gsx_cuda_render_tensor_device_bytes(cuda_context->saved_sh2) : NULL,
        cuda_context->saved_sh3 != NULL ? (const float *)gsx_cuda_render_tensor_device_bytes(cuda_context->saved_sh3) : NULL,
        (const float4 *)cuda_context->device_pose_block,
        (const float3 *)((unsigned char *)cuda_context->device_pose_block + offsetof(gsx_cuda_pose_block, cam_position)),
        cuda_context->per_primitive_blob.blob != NULL ? (char *)gsx_cuda_render_tensor_device_bytes(cuda_context->per_primitive_blob.blob) : NULL,
        cuda_context->per_tile_blob.blob != NULL ? (char *)gsx_cuda_render_tensor_device_bytes(cuda_context->per_tile_blob.blob) : NULL,
        cuda_context->per_instance_blob.blob != NULL ? (char *)gsx_cuda_render_tensor_device_bytes(cuda_context->per_instance_blob.blob) : NULL,
        cuda_context->per_bucket_blob.blob != NULL ? (char *)gsx_cuda_render_tensor_device_bytes(cuda_context->per_bucket_blob.blob) : NULL,
        (float3 *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_mean3d),
        (float3 *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_logscale),
        (float4 *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_rotation),
        (float *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_opacity),
        (float *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_sh0),
        request->grad_gs_sh1 != NULL ? (float *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_sh1) : NULL,
        request->grad_gs_sh2 != NULL ? (float *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_sh2) : NULL,
        request->grad_gs_sh3 != NULL ? (float *)gsx_cuda_render_tensor_device_bytes(request->grad_gs_sh3) : NULL,
        (float2 *)gsx_cuda_render_tensor_device_bytes(cuda_context->helper_grad_mean2d),
        (float *)gsx_cuda_render_tensor_device_bytes(cuda_context->helper_grad_conic),
        (float3 *)gsx_cuda_render_tensor_device_bytes(cuda_context->helper_grad_color),
        (float4 *)gsx_cuda_render_tensor_device_bytes(cuda_context->helper_grad_w2c),
        (float2 *)gsx_cuda_render_tensor_device_bytes(cuda_context->helper_absgrad_mean2d),
        (int)gaussian_count,
        cuda_context->n_visible_primitives,
        cuda_context->n_instances,
        cuda_context->n_buckets,
        cuda_context->primitive_selector,
        cuda_context->instance_selector,
        (int)((cuda_context->sh_degree + 1) * (cuda_context->sh_degree + 1)),
        renderer->info.width,
        renderer->info.height,
        cuda_context->intrinsics.fx,
        cuda_context->intrinsics.fy,
        cuda_context->intrinsics.cx,
        cuda_context->intrinsics.cy,
        (cudaStream_t)stream
    );
    if(cuda_err != cudaSuccess) {
        result = gsx_cuda_make_error(cuda_err, "fastgs backward launch failed");
        goto cleanup;
    }
    result = gsx_cuda_render_sync_major_stream(renderer->backend);
cleanup:
    return result;
}

gsx_error gsx_cuda_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer)
{
    gsx_cuda_renderer *cuda_renderer = NULL;
    gsx_renderer_capabilities capabilities = { 0 };
    gsx_backend_buffer_type_t device_buffer_type = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_renderer must be non-null");
    }
    *out_renderer = NULL;
    if(desc->output_data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer currently supports float32 outputs only");
    }
    if(desc->enable_alpha_output || desc->enable_invdepth_output) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer does not implement alpha or inverse-depth outputs yet");
    }
    if((desc->feature_flags & GSX_RENDERER_FEATURE_ANTIALIASING) != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda renderer does not implement anti-aliasing yet");
    }

    error = gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cuda_renderer = (gsx_cuda_renderer *)calloc(1, sizeof(*cuda_renderer));
    if(cuda_renderer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cuda renderer");
    }

    capabilities.supported_precisions = GSX_RENDER_PRECISION_FLAG_FLOAT32;
    capabilities.supports_invdepth_output = false;
    capabilities.supports_alpha_output = false;
    capabilities.supports_cov3d_input = false;
    error = gsx_renderer_base_init(&cuda_renderer->base, &gsx_cuda_renderer_iface, backend, desc, &capabilities);
    if(!gsx_error_is_success(error)) {
        free(cuda_renderer);
        return error;
    }
    cuda_renderer->device_buffer_type = device_buffer_type;
    *out_renderer = &cuda_renderer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
