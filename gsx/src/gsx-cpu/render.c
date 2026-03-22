#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_cpu_vec3 {
    float x;
    float y;
    float z;
} gsx_cpu_vec3;

typedef struct gsx_cpu_mat3 {
    float m[3][3];
} gsx_cpu_mat3;

typedef struct gsx_cpu_render_visible {
    gsx_size_t gaussian_index;
    int32_t x_min;
    int32_t x_max;
    int32_t y_min;
    int32_t y_max;
    float mean2d_x;
    float mean2d_y;
    float cov2d_a;
    float cov2d_b;
    float cov2d_c;
    float inv_det;
    float depth;
    float x_cam;
    float y_cam;
    float z_cam;
    float color[3];
    float opacity;
    float raw_opacity;
    float dir[3];
} gsx_cpu_render_visible;

typedef struct gsx_cpu_render_pixel_contrib {
    int32_t pixel_index;
    float alpha;
    float t_before;
} gsx_cpu_render_pixel_contrib;

typedef struct gsx_cpu_renderer {
    struct gsx_renderer base;
} gsx_cpu_renderer;

typedef struct gsx_cpu_render_context {
    struct gsx_render_context base;
    gsx_arena_t scratch_arena;
    gsx_arena_t retain_arena;
    gsx_tensor_t saved_mean3d;
    gsx_tensor_t saved_rotation;
    gsx_tensor_t saved_logscale;
    gsx_tensor_t saved_sh0;
    gsx_tensor_t saved_sh1;
    gsx_tensor_t saved_sh2;
    gsx_tensor_t saved_sh3;
    gsx_tensor_t saved_opacity;
    gsx_camera_intrinsics intrinsics;
    gsx_camera_pose pose;
    gsx_vec3 background_color;
    gsx_float_t near_plane;
    gsx_float_t far_plane;
    gsx_index_t sh_degree;
    bool has_train_state;
    bool train_state_borrowed;
} gsx_cpu_render_context;

typedef struct gsx_cpu_render_projected {
    float mean2d_x;
    float mean2d_y;
    float cov2d_a;
    float cov2d_b;
    float cov2d_c;
    float depth;
    float x_cam;
    float y_cam;
    float z_cam;
} gsx_cpu_render_projected;

static const float gsx_cpu_render_sh_c0 = 0.28209479177387814f;
static const float gsx_cpu_render_sh_c1 = 0.4886025119029199f;
static const float gsx_cpu_render_sh_c2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
static const float gsx_cpu_render_sh_c3[7] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};
static const float gsx_cpu_render_min_abs_z = 1.0e-8f;
static const float gsx_cpu_render_min_quaternion_norm_sq = 1.0e-20f;
static const float gsx_cpu_render_cov2d_dilation = 0.3f;
static const float gsx_cpu_render_power_threshold = -4.5f;
static const float gsx_cpu_render_min_alpha_threshold_rcp = 255.0f;
static const float gsx_cpu_render_min_alpha_threshold = 1.0f / gsx_cpu_render_min_alpha_threshold_rcp;
static const float gsx_cpu_render_min_alpha_threshold_deactivated = -5.537334267018537f;
static const float gsx_cpu_render_max_fragment_alpha = 0.999f;
static const float gsx_cpu_render_transmittance_threshold = 1.0e-4f;

static const gsx_renderer_i gsx_cpu_renderer_iface;
static const gsx_render_context_i gsx_cpu_render_context_iface;

static void gsx_cpu_render_log_cleanup_failure(const char *operation, gsx_error error)
{
    fprintf(
        stderr,
        "gsx cpu renderer cleanup warning: %s failed with code %d%s%s\n",
        operation,
        (int)error.code,
        error.message != NULL ? " - " : "",
        error.message != NULL ? error.message : "");
}

static unsigned char *gsx_cpu_render_tensor_data_bytes(gsx_tensor_t tensor)
{
    gsx_cpu_backend_buffer *cpu_buffer = (gsx_cpu_backend_buffer *)tensor->backing_buffer;

    return (unsigned char *)cpu_buffer->data + (size_t)tensor->offset_bytes;
}

static float *gsx_cpu_render_tensor_data_f32(gsx_tensor_t tensor)
{
    return (float *)gsx_cpu_render_tensor_data_bytes(tensor);
}

static const float *gsx_cpu_render_tensor_data_f32_const(gsx_tensor_t tensor)
{
    return (const float *)gsx_cpu_render_tensor_data_bytes(tensor);
}

static gsx_error gsx_cpu_render_validate_tensor_shape(
    gsx_tensor_t tensor,
    gsx_data_type data_type,
    gsx_storage_format storage_format,
    gsx_index_t rank,
    const gsx_index_t *shape,
    const char *message)
{
    gsx_index_t dim = 0;

    if(tensor->data_type != data_type || tensor->storage_format != storage_format || tensor->rank != rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
    }
    for(dim = 0; dim < rank; ++dim) {
        if(tensor->shape[dim] != shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, message);
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_render_init_arena(gsx_backend_buffer_type_t buffer_type, gsx_arena_t *out_arena)
{
    gsx_arena_desc arena_desc = { 0 };

    if(out_arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_arena must be non-null");
    }

    arena_desc.initial_capacity_bytes = 4096;
    return gsx_arena_init(out_arena, buffer_type, &arena_desc);
}

static gsx_error gsx_cpu_render_alloc_byte_tensor(gsx_arena_t arena, gsx_size_t size_bytes, gsx_tensor_t *out_tensor)
{
    gsx_tensor_desc desc = { 0 };

    if(out_tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_tensor must be non-null");
    }
    *out_tensor = NULL;
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(size_bytes > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "scratch allocation exceeds tensor rank-1 limits");
    }

    desc.rank = 1;
    desc.shape[0] = (gsx_index_t)size_bytes;
    desc.data_type = GSX_DATA_TYPE_U8;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    return gsx_tensor_init(out_tensor, &desc);
}

static gsx_error gsx_cpu_render_clone_tensor(gsx_tensor_t src, gsx_arena_t arena, gsx_tensor_t *out_clone)
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
        (void)gsx_tensor_free(*out_clone);
        *out_clone = NULL;
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_cpu_render_free_tensor_handle(gsx_tensor_t *tensor)
{
    if(tensor != NULL && *tensor != NULL) {
        gsx_error error = gsx_tensor_free(*tensor);

        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_log_cleanup_failure("gsx_tensor_free", error);
        }
        *tensor = NULL;
    }
}

static void gsx_cpu_render_clear_snapshot(gsx_cpu_render_context *cpu_context)
{
    if(cpu_context == NULL) {
        return;
    }

    if(!cpu_context->train_state_borrowed) {
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_mean3d);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_rotation);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_logscale);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_sh0);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_sh1);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_sh2);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_sh3);
        gsx_cpu_render_free_tensor_handle(&cpu_context->saved_opacity);
    }
    cpu_context->saved_mean3d = NULL;
    cpu_context->saved_rotation = NULL;
    cpu_context->saved_logscale = NULL;
    cpu_context->saved_sh0 = NULL;
    cpu_context->saved_sh1 = NULL;
    cpu_context->saved_sh2 = NULL;
    cpu_context->saved_sh3 = NULL;
    cpu_context->saved_opacity = NULL;
    if(cpu_context->retain_arena != NULL) {
        gsx_error error = gsx_arena_reset(cpu_context->retain_arena);

        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_log_cleanup_failure("gsx_arena_reset", error);
        }
    }
    cpu_context->has_train_state = false;
    cpu_context->train_state_borrowed = false;
    memset(&cpu_context->intrinsics, 0, sizeof(cpu_context->intrinsics));
    memset(&cpu_context->pose, 0, sizeof(cpu_context->pose));
    cpu_context->near_plane = 0.0f;
    cpu_context->far_plane = 0.0f;
    cpu_context->background_color.x = 0.0f;
    cpu_context->background_color.y = 0.0f;
    cpu_context->background_color.z = 0.0f;
    cpu_context->sh_degree = 0;
}

static gsx_error gsx_cpu_render_snapshot_request(gsx_cpu_render_context *cpu_context, const gsx_render_forward_request *request)
{
    /* Tensors to clone, in the same order used below. NULL-checked before sizing. */
    const gsx_tensor_t snapshot_sources[] = {
        request->gs_mean3d, request->gs_rotation, request->gs_logscale, request->gs_sh0,
        request->gs_sh1,    request->gs_sh2,       request->gs_sh3,      request->gs_opacity
    };
    gsx_arena_info arena_info = { 0 };
    gsx_tensor_info tensor_info = { 0 };
    gsx_size_t reserve_bytes = 0;
    gsx_index_t i = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    gsx_cpu_render_clear_snapshot(cpu_context);

    /*
     * Pre-reserve the retain arena to fit all cloned tensors before any allocation.
     * After clear_snapshot the arena cursor is at 0 and active_tensor_count == 0,
     * so gsx_arena_reserve is allowed.  Without this pre-reservation, the arena
     * cannot grow once even one tensor is live inside it, causing failures when
     * the Gaussian count is large enough that the combined snapshot exceeds the
     * initial 4096-byte capacity.
     */
    if(!request->borrow_train_state) {
        error = gsx_arena_get_info(cpu_context->retain_arena, &arena_info);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        for(i = 0; i < (gsx_index_t)(sizeof(snapshot_sources) / sizeof(snapshot_sources[0])); ++i) {
            if(snapshot_sources[i] == NULL) {
                continue;
            }
            error = gsx_tensor_get_info(snapshot_sources[i], &tensor_info);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            /* Per-allocation alignment overhead is at most (alignment - 1) bytes. */
            if(gsx_size_add_overflows(reserve_bytes, tensor_info.size_bytes, &reserve_bytes)
                || gsx_size_add_overflows(reserve_bytes, arena_info.effective_alignment_bytes - 1u, &reserve_bytes)) {
                return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "retain arena reserve size overflow");
            }
        }
        if(reserve_bytes > arena_info.capacity_bytes) {
            error = gsx_arena_reserve(cpu_context->retain_arena, reserve_bytes);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
    }

    if(request->borrow_train_state) {
        cpu_context->saved_mean3d = request->gs_mean3d;
        cpu_context->saved_rotation = request->gs_rotation;
        cpu_context->saved_logscale = request->gs_logscale;
        cpu_context->saved_sh0 = request->gs_sh0;
        cpu_context->saved_sh1 = request->gs_sh1;
        cpu_context->saved_sh2 = request->gs_sh2;
        cpu_context->saved_sh3 = request->gs_sh3;
        cpu_context->saved_opacity = request->gs_opacity;
        cpu_context->train_state_borrowed = true;
    } else {
        error = gsx_cpu_render_clone_tensor(request->gs_mean3d, cpu_context->retain_arena, &cpu_context->saved_mean3d);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_cpu_render_clone_tensor(request->gs_rotation, cpu_context->retain_arena, &cpu_context->saved_rotation);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_clear_snapshot(cpu_context);
            return error;
        }
        error = gsx_cpu_render_clone_tensor(request->gs_logscale, cpu_context->retain_arena, &cpu_context->saved_logscale);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_clear_snapshot(cpu_context);
            return error;
        }
        error = gsx_cpu_render_clone_tensor(request->gs_sh0, cpu_context->retain_arena, &cpu_context->saved_sh0);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_clear_snapshot(cpu_context);
            return error;
        }
        if(request->gs_sh1 != NULL) {
            error = gsx_cpu_render_clone_tensor(request->gs_sh1, cpu_context->retain_arena, &cpu_context->saved_sh1);
            if(!gsx_error_is_success(error)) {
                gsx_cpu_render_clear_snapshot(cpu_context);
                return error;
            }
        }
        if(request->gs_sh2 != NULL) {
            error = gsx_cpu_render_clone_tensor(request->gs_sh2, cpu_context->retain_arena, &cpu_context->saved_sh2);
            if(!gsx_error_is_success(error)) {
                gsx_cpu_render_clear_snapshot(cpu_context);
                return error;
            }
        }
        if(request->gs_sh3 != NULL) {
            error = gsx_cpu_render_clone_tensor(request->gs_sh3, cpu_context->retain_arena, &cpu_context->saved_sh3);
            if(!gsx_error_is_success(error)) {
                gsx_cpu_render_clear_snapshot(cpu_context);
                return error;
            }
        }
        error = gsx_cpu_render_clone_tensor(request->gs_opacity, cpu_context->retain_arena, &cpu_context->saved_opacity);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_clear_snapshot(cpu_context);
            return error;
        }
        cpu_context->train_state_borrowed = false;
    }

    cpu_context->intrinsics = *request->intrinsics;
    cpu_context->pose = *request->pose;
    cpu_context->near_plane = request->near_plane;
    cpu_context->far_plane = request->far_plane;
    cpu_context->background_color = request->background_color;
    cpu_context->sh_degree = request->sh_degree;
    cpu_context->has_train_state = true;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_render_debug_check_tensor(gsx_tensor_t tensor, const char *label)
{
    bool is_finite = true;
    gsx_error error = gsx_tensor_check_finite(tensor, &is_finite);

    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!is_finite) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, label);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static float gsx_cpu_render_sigmoid(float x)
{
    if(x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static float gsx_cpu_render_sigmoid_grad_from_raw(float x)
{
    float y = gsx_cpu_render_sigmoid(x);

    return y * (1.0f - y);
}

static float gsx_cpu_render_exp(float x)
{
    return expf(x);
}

static bool gsx_cpu_render_normalize_quaternion_xyzw(
    float qx_raw,
    float qy_raw,
    float qz_raw,
    float qw_raw,
    float *out_qx,
    float *out_qy,
    float *out_qz,
    float *out_qw)
{
    float norm_sq = qx_raw * qx_raw + qy_raw * qy_raw + qz_raw * qz_raw + qw_raw * qw_raw;
    float inv_norm = 0.0f;

    if(out_qx == NULL || out_qy == NULL || out_qz == NULL || out_qw == NULL) {
        return false;
    }
    if(norm_sq <= gsx_cpu_render_min_quaternion_norm_sq) {
        *out_qx = 0.0f;
        *out_qy = 0.0f;
        *out_qz = 0.0f;
        *out_qw = 1.0f;
        return false;
    }

    inv_norm = 1.0f / sqrtf(norm_sq);
    *out_qx = qx_raw * inv_norm;
    *out_qy = qy_raw * inv_norm;
    *out_qz = qz_raw * inv_norm;
    *out_qw = qw_raw * inv_norm;
    return true;
}

static gsx_cpu_vec3 gsx_cpu_render_vec3(float x, float y, float z)
{
    gsx_cpu_vec3 value;

    value.x = x;
    value.y = y;
    value.z = z;
    return value;
}

static gsx_cpu_vec3 gsx_cpu_render_vec3_add(gsx_cpu_vec3 lhs, gsx_cpu_vec3 rhs)
{
    return gsx_cpu_render_vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

static gsx_cpu_vec3 gsx_cpu_render_vec3_sub(gsx_cpu_vec3 lhs, gsx_cpu_vec3 rhs)
{
    return gsx_cpu_render_vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

static gsx_cpu_vec3 gsx_cpu_render_vec3_scale(gsx_cpu_vec3 value, float scale)
{
    return gsx_cpu_render_vec3(value.x * scale, value.y * scale, value.z * scale);
}

static float gsx_cpu_render_vec3_dot(gsx_cpu_vec3 lhs, gsx_cpu_vec3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

static float gsx_cpu_render_vec3_length(gsx_cpu_vec3 value)
{
    return sqrtf(gsx_cpu_render_vec3_dot(value, value));
}

static gsx_cpu_vec3 gsx_cpu_render_vec3_normalize(gsx_cpu_vec3 value)
{
    float length = gsx_cpu_render_vec3_length(value);

    if(length <= 0.0f) {
        return gsx_cpu_render_vec3(0.0f, 0.0f, 0.0f);
    }
    return gsx_cpu_render_vec3_scale(value, 1.0f / length);
}

static gsx_cpu_mat3 gsx_cpu_render_mat3_zero(void)
{
    gsx_cpu_mat3 result;

    memset(&result, 0, sizeof(result));
    return result;
}

static gsx_cpu_mat3 gsx_cpu_render_mat3_transpose(gsx_cpu_mat3 value)
{
    gsx_cpu_mat3 result;
    int row = 0;
    int col = 0;

    for(row = 0; row < 3; ++row) {
        for(col = 0; col < 3; ++col) {
            result.m[row][col] = value.m[col][row];
        }
    }
    return result;
}

static gsx_cpu_mat3 gsx_cpu_render_mat3_add(gsx_cpu_mat3 lhs, gsx_cpu_mat3 rhs)
{
    gsx_cpu_mat3 result;
    int row = 0;
    int col = 0;

    for(row = 0; row < 3; ++row) {
        for(col = 0; col < 3; ++col) {
            result.m[row][col] = lhs.m[row][col] + rhs.m[row][col];
        }
    }
    return result;
}

static gsx_cpu_mat3 gsx_cpu_render_mat3_mul(gsx_cpu_mat3 lhs, gsx_cpu_mat3 rhs)
{
    gsx_cpu_mat3 result = gsx_cpu_render_mat3_zero();
    int row = 0;
    int col = 0;
    int k = 0;

    for(row = 0; row < 3; ++row) {
        for(col = 0; col < 3; ++col) {
            for(k = 0; k < 3; ++k) {
                result.m[row][col] += lhs.m[row][k] * rhs.m[k][col];
            }
        }
    }

    return result;
}

static gsx_cpu_vec3 gsx_cpu_render_mat3_mul_vec3(gsx_cpu_mat3 matrix, gsx_cpu_vec3 value)
{
    return gsx_cpu_render_vec3(
        matrix.m[0][0] * value.x + matrix.m[0][1] * value.y + matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x + matrix.m[1][1] * value.y + matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x + matrix.m[2][1] * value.y + matrix.m[2][2] * value.z
    );
}

static gsx_cpu_mat3 gsx_cpu_render_rotation_matrix_from_xyzw(float qx_raw, float qy_raw, float qz_raw, float qw_raw)
{
    gsx_cpu_mat3 matrix;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 1.0f;

    (void)gsx_cpu_render_normalize_quaternion_xyzw(qx_raw, qy_raw, qz_raw, qw_raw, &qx, &qy, &qz, &qw);

    matrix.m[0][0] = 1.0f - 2.0f * (qy * qy + qz * qz);
    matrix.m[0][1] = 2.0f * (qx * qy - qw * qz);
    matrix.m[0][2] = 2.0f * (qx * qz + qw * qy);
    matrix.m[1][0] = 2.0f * (qx * qy + qw * qz);
    matrix.m[1][1] = 1.0f - 2.0f * (qx * qx + qz * qz);
    matrix.m[1][2] = 2.0f * (qy * qz - qw * qx);
    matrix.m[2][0] = 2.0f * (qx * qz - qw * qy);
    matrix.m[2][1] = 2.0f * (qy * qz + qw * qx);
    matrix.m[2][2] = 1.0f - 2.0f * (qx * qx + qy * qy);
    return matrix;
}

static gsx_cpu_mat3 gsx_cpu_render_cov3d_from_inputs(float qx, float qy, float qz, float qw, float sx_raw, float sy_raw, float sz_raw)
{
    gsx_cpu_mat3 rotation = gsx_cpu_render_rotation_matrix_from_xyzw(qx, qy, qz, qw);
    gsx_cpu_mat3 scaled_rotation = rotation;
    gsx_cpu_mat3 cov3d;
    float sx = gsx_cpu_render_exp(sx_raw);
    float sy = gsx_cpu_render_exp(sy_raw);
    float sz = gsx_cpu_render_exp(sz_raw);
    int row = 0;

    for(row = 0; row < 3; ++row) {
        scaled_rotation.m[row][0] *= sx;
        scaled_rotation.m[row][1] *= sy;
        scaled_rotation.m[row][2] *= sz;
    }

    cov3d = gsx_cpu_render_mat3_mul(scaled_rotation, gsx_cpu_render_mat3_transpose(scaled_rotation));
    return cov3d;
}

static gsx_cpu_vec3 gsx_cpu_render_camera_position(const gsx_camera_pose *pose, gsx_cpu_mat3 world_to_camera)
{
    gsx_cpu_vec3 translation = gsx_cpu_render_vec3(pose->transl.x, pose->transl.y, pose->transl.z);
    gsx_cpu_vec3 camera_in_rot = gsx_cpu_render_mat3_mul_vec3(gsx_cpu_render_mat3_transpose(world_to_camera), translation);

    return gsx_cpu_render_vec3_scale(camera_in_rot, -1.0f);
}

static bool gsx_cpu_render_project_gaussian(
    gsx_cpu_vec3 mean3d,
    gsx_cpu_mat3 cov3d,
    const gsx_camera_intrinsics *intrinsics,
    const gsx_camera_pose *pose,
    gsx_cpu_mat3 world_to_camera,
    gsx_cpu_render_projected *out_projected)
{
    gsx_cpu_render_projected projected;
    gsx_cpu_vec3 mean_cam = gsx_cpu_render_vec3_add(
        gsx_cpu_render_mat3_mul_vec3(world_to_camera, mean3d),
        gsx_cpu_render_vec3(pose->transl.x, pose->transl.y, pose->transl.z));
    float inv_z = 0.0f;
    float inv_z2 = 0.0f;
    float j00 = 0.0f;
    float j02 = 0.0f;
    float j11 = 0.0f;
    float j12 = 0.0f;
    float jc00 = 0.0f;
    float jc01 = 0.0f;
    float jc02 = 0.0f;
    float jc11 = 0.0f;
    float jc12 = 0.0f;
    gsx_cpu_mat3 cov_cam = gsx_cpu_render_mat3_mul(
        gsx_cpu_render_mat3_mul(world_to_camera, cov3d),
        gsx_cpu_render_mat3_transpose(world_to_camera));

    if(out_projected == NULL) {
        return false;
    }
    if(fabsf(mean_cam.z) < gsx_cpu_render_min_abs_z) {
        memset(out_projected, 0, sizeof(*out_projected));
        return false;
    }

    inv_z = 1.0f / mean_cam.z;
    inv_z2 = inv_z * inv_z;
    j00 = intrinsics->fx * inv_z;
    j02 = -intrinsics->fx * mean_cam.x * inv_z2;
    j11 = intrinsics->fy * inv_z;
    j12 = -intrinsics->fy * mean_cam.y * inv_z2;
    jc00 = j00 * cov_cam.m[0][0] + j02 * cov_cam.m[2][0];
    jc01 = j00 * cov_cam.m[0][1] + j02 * cov_cam.m[2][1];
    jc02 = j00 * cov_cam.m[0][2] + j02 * cov_cam.m[2][2];
    jc11 = j11 * cov_cam.m[1][1] + j12 * cov_cam.m[2][1];
    jc12 = j11 * cov_cam.m[1][2] + j12 * cov_cam.m[2][2];
    projected.mean2d_x = intrinsics->fx * mean_cam.x * inv_z + intrinsics->cx;
    projected.mean2d_y = intrinsics->fy * mean_cam.y * inv_z + intrinsics->cy;
    projected.cov2d_a = jc00 * j00 + jc02 * j02 + gsx_cpu_render_cov2d_dilation;
    projected.cov2d_b = jc01 * j11 + jc02 * j12;
    projected.cov2d_c = jc11 * j11 + jc12 * j12 + gsx_cpu_render_cov2d_dilation;
    projected.depth = mean_cam.z;
    projected.x_cam = mean_cam.x;
    projected.y_cam = mean_cam.y;
    projected.z_cam = mean_cam.z;
    *out_projected = projected;
    return true;
}

static void gsx_cpu_render_compute_bbox(
    float cov_a,
    float cov_b,
    float cov_c,
    float mean_x,
    float mean_y,
    gsx_index_t width,
    gsx_index_t height,
    int32_t *out_x_min,
    int32_t *out_x_max,
    int32_t *out_y_min,
    int32_t *out_y_max)
{
    float trace = cov_a + cov_c;
    float det = cov_a * cov_c - cov_b * cov_b;
    float disc = sqrtf(fmaxf(0.0f, trace * trace * 0.25f - det));
    float lambda1 = trace * 0.5f + disc;
    float lambda2 = trace * 0.5f - disc;
    float radius = 3.0f * sqrtf(fmaxf(lambda1, lambda2));

    *out_x_min = (int32_t)fmaxf(0.0f, floorf(mean_x - radius));
    *out_x_max = (int32_t)fminf((float)(width - 1), ceilf(mean_x + radius));
    *out_y_min = (int32_t)fmaxf(0.0f, floorf(mean_y - radius));
    *out_y_max = (int32_t)fminf((float)(height - 1), ceilf(mean_y + radius));
}

static float gsx_cpu_render_read_sh(const float *values, gsx_size_t gaussian_index, gsx_index_t coeff_count, gsx_index_t coeff_index, gsx_index_t channel)
{
    return values[((gaussian_index * (gsx_size_t)coeff_count + (gsx_size_t)coeff_index) * 3u) + (gsx_size_t)channel];
}

static void gsx_cpu_render_accum_sh_grad(
    float *values,
    gsx_size_t gaussian_index,
    gsx_index_t coeff_count,
    gsx_index_t coeff_index,
    gsx_cpu_vec3 grad)
{
    gsx_size_t base = ((gaussian_index * (gsx_size_t)coeff_count + (gsx_size_t)coeff_index) * 3u);

    values[base + 0] += grad.x;
    values[base + 1] += grad.y;
    values[base + 2] += grad.z;
}

static void gsx_cpu_render_accum_direction_grad(
    gsx_cpu_vec3 *out_grad_direction,
    gsx_cpu_vec3 grad_color,
    gsx_cpu_vec3 coeff,
    float basis_dx,
    float basis_dy,
    float basis_dz)
{
    float coeff_scale = gsx_cpu_render_vec3_dot(grad_color, coeff);

    out_grad_direction->x += coeff_scale * basis_dx;
    out_grad_direction->y += coeff_scale * basis_dy;
    out_grad_direction->z += coeff_scale * basis_dz;
}

static gsx_cpu_vec3 gsx_cpu_render_evaluate_sh(
    gsx_size_t gaussian_index,
    gsx_index_t sh_degree,
    gsx_cpu_vec3 direction,
    const float *sh0,
    const float *sh1,
    const float *sh2,
    const float *sh3)
{
    gsx_cpu_vec3 result = gsx_cpu_render_vec3(
        gsx_cpu_render_sh_c0 * sh0[gaussian_index * 3u + 0],
        gsx_cpu_render_sh_c0 * sh0[gaussian_index * 3u + 1],
        gsx_cpu_render_sh_c0 * sh0[gaussian_index * 3u + 2]);

    if(sh_degree >= 1) {
        float x = direction.x;
        float y = direction.y;
        float z = direction.z;
        gsx_cpu_vec3 c0 = gsx_cpu_render_vec3(
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 0, 0),
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 0, 1),
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 0, 2));
        gsx_cpu_vec3 c1 = gsx_cpu_render_vec3(
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 1, 0),
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 1, 1),
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 1, 2));
        gsx_cpu_vec3 c2 = gsx_cpu_render_vec3(
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 2, 0),
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 2, 1),
            gsx_cpu_render_read_sh(sh1, gaussian_index, 3, 2, 2));

        result = gsx_cpu_render_vec3_add(result, gsx_cpu_render_vec3_scale(c0, -gsx_cpu_render_sh_c1 * y));
        result = gsx_cpu_render_vec3_add(result, gsx_cpu_render_vec3_scale(c1, gsx_cpu_render_sh_c1 * z));
        result = gsx_cpu_render_vec3_add(result, gsx_cpu_render_vec3_scale(c2, -gsx_cpu_render_sh_c1 * x));

        if(sh_degree >= 2) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            int coeff = 0;

            for(coeff = 0; coeff < 5; ++coeff) {
                gsx_cpu_vec3 value = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(sh2, gaussian_index, 5, coeff, 0),
                    gsx_cpu_render_read_sh(sh2, gaussian_index, 5, coeff, 1),
                    gsx_cpu_render_read_sh(sh2, gaussian_index, 5, coeff, 2));
                float basis = 0.0f;

                switch(coeff) {
                case 0:
                    basis = gsx_cpu_render_sh_c2[0] * xy;
                    break;
                case 1:
                    basis = gsx_cpu_render_sh_c2[1] * yz;
                    break;
                case 2:
                    basis = gsx_cpu_render_sh_c2[2] * (2.0f * zz - xx - yy);
                    break;
                case 3:
                    basis = gsx_cpu_render_sh_c2[3] * xz;
                    break;
                default:
                    basis = gsx_cpu_render_sh_c2[4] * (xx - yy);
                    break;
                }
                result = gsx_cpu_render_vec3_add(result, gsx_cpu_render_vec3_scale(value, basis));
            }

            if(sh_degree >= 3) {
                float basis_values[7];
                int coeff = 0;

                basis_values[0] = gsx_cpu_render_sh_c3[0] * y * (3.0f * xx - yy);
                basis_values[1] = gsx_cpu_render_sh_c3[1] * xy * z;
                basis_values[2] = gsx_cpu_render_sh_c3[2] * y * (4.0f * zz - xx - yy);
                basis_values[3] = gsx_cpu_render_sh_c3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
                basis_values[4] = gsx_cpu_render_sh_c3[4] * x * (4.0f * zz - xx - yy);
                basis_values[5] = gsx_cpu_render_sh_c3[5] * z * (xx - yy);
                basis_values[6] = gsx_cpu_render_sh_c3[6] * x * (xx - 3.0f * yy);
                for(coeff = 0; coeff < 7; ++coeff) {
                    gsx_cpu_vec3 value = gsx_cpu_render_vec3(
                        gsx_cpu_render_read_sh(sh3, gaussian_index, 7, coeff, 0),
                        gsx_cpu_render_read_sh(sh3, gaussian_index, 7, coeff, 1),
                        gsx_cpu_render_read_sh(sh3, gaussian_index, 7, coeff, 2));

                    result = gsx_cpu_render_vec3_add(result, gsx_cpu_render_vec3_scale(value, basis_values[coeff]));
                }
            }
        }
    }

    result.x = fmaxf(result.x + 0.5f, 0.0f);
    result.y = fmaxf(result.y + 0.5f, 0.0f);
    result.z = fmaxf(result.z + 0.5f, 0.0f);
    return result;
}

static int gsx_cpu_render_compare_visible_depth(const void *lhs, const void *rhs)
{
    const gsx_cpu_render_visible *left = (const gsx_cpu_render_visible *)lhs;
    const gsx_cpu_render_visible *right = (const gsx_cpu_render_visible *)rhs;

    if(left->depth < right->depth) {
        return -1;
    }
    if(left->depth > right->depth) {
        return 1;
    }
    if(left->gaussian_index < right->gaussian_index) {
        return -1;
    }
    if(left->gaussian_index > right->gaussian_index) {
        return 1;
    }
    return 0;
}

static gsx_size_t gsx_cpu_render_count_visible(
    const gsx_render_forward_request *request,
    gsx_cpu_mat3 world_to_camera,
    gsx_size_t *out_contrib_capacity)
{
    const float *means = gsx_cpu_render_tensor_data_f32_const(request->gs_mean3d);
    const float *rotations = gsx_cpu_render_tensor_data_f32_const(request->gs_rotation);
    const float *logscales = gsx_cpu_render_tensor_data_f32_const(request->gs_logscale);
    const float *opacities = gsx_cpu_render_tensor_data_f32_const(request->gs_opacity);
    gsx_size_t gaussian_count = (gsx_size_t)request->gs_mean3d->shape[0];
    gsx_size_t visible_count = 0;
    gsx_size_t contrib_capacity = 0;
    gsx_size_t i = 0;

    for(i = 0; i < gaussian_count; ++i) {
        gsx_cpu_vec3 mean = gsx_cpu_render_vec3(means[i * 3u + 0], means[i * 3u + 1], means[i * 3u + 2]);
        gsx_cpu_mat3 cov3d = gsx_cpu_render_cov3d_from_inputs(
            rotations[i * 4u + 0],
            rotations[i * 4u + 1],
            rotations[i * 4u + 2],
            rotations[i * 4u + 3],
            logscales[i * 3u + 0],
            logscales[i * 3u + 1],
            logscales[i * 3u + 2]);
        gsx_cpu_render_projected projected;
        float det = 0.0f;
        int32_t x_min = 0;
        int32_t x_max = 0;
        int32_t y_min = 0;
        int32_t y_max = 0;

        if(opacities[i] <= gsx_cpu_render_min_alpha_threshold_deactivated) {
            continue;
        }
        if(!gsx_cpu_render_project_gaussian(mean, cov3d, request->intrinsics, request->pose, world_to_camera, &projected)) {
            continue;
        }
        det = projected.cov2d_a * projected.cov2d_c - projected.cov2d_b * projected.cov2d_b;
        if(projected.depth < request->near_plane || projected.depth > request->far_plane || det <= 0.0f) {
            continue;
        }
        gsx_cpu_render_compute_bbox(
            projected.cov2d_a,
            projected.cov2d_b,
            projected.cov2d_c,
            projected.mean2d_x,
            projected.mean2d_y,
            request->intrinsics->width,
            request->intrinsics->height,
            &x_min,
            &x_max,
            &y_min,
            &y_max);
        if(x_min > x_max || y_min > y_max) {
            continue;
        }

        visible_count += 1;
        contrib_capacity += (gsx_size_t)(x_max - x_min + 1) * (gsx_size_t)(y_max - y_min + 1);
    }

    if(out_contrib_capacity != NULL) {
        *out_contrib_capacity = contrib_capacity;
    }
    return visible_count;
}

static void gsx_cpu_render_fill_visible(
    const gsx_render_forward_request *request,
    gsx_cpu_mat3 world_to_camera,
    gsx_cpu_vec3 camera_position,
    gsx_cpu_render_visible *visible)
{
    const float *means = gsx_cpu_render_tensor_data_f32_const(request->gs_mean3d);
    const float *rotations = gsx_cpu_render_tensor_data_f32_const(request->gs_rotation);
    const float *logscales = gsx_cpu_render_tensor_data_f32_const(request->gs_logscale);
    const float *opacities = gsx_cpu_render_tensor_data_f32_const(request->gs_opacity);
    const float *sh0 = gsx_cpu_render_tensor_data_f32_const(request->gs_sh0);
    const float *sh1 = request->gs_sh1 != NULL ? gsx_cpu_render_tensor_data_f32_const(request->gs_sh1) : NULL;
    const float *sh2 = request->gs_sh2 != NULL ? gsx_cpu_render_tensor_data_f32_const(request->gs_sh2) : NULL;
    const float *sh3 = request->gs_sh3 != NULL ? gsx_cpu_render_tensor_data_f32_const(request->gs_sh3) : NULL;
    gsx_size_t gaussian_count = (gsx_size_t)request->gs_mean3d->shape[0];
    gsx_size_t visible_index = 0;
    gsx_size_t i = 0;

    for(i = 0; i < gaussian_count; ++i) {
        gsx_cpu_vec3 mean = gsx_cpu_render_vec3(means[i * 3u + 0], means[i * 3u + 1], means[i * 3u + 2]);
        gsx_cpu_mat3 cov3d = gsx_cpu_render_cov3d_from_inputs(
            rotations[i * 4u + 0],
            rotations[i * 4u + 1],
            rotations[i * 4u + 2],
            rotations[i * 4u + 3],
            logscales[i * 3u + 0],
            logscales[i * 3u + 1],
            logscales[i * 3u + 2]);
        gsx_cpu_render_projected projected;
        float det = 0.0f;
        int32_t x_min = 0;
        int32_t x_max = 0;
        int32_t y_min = 0;
        int32_t y_max = 0;

        if(opacities[i] <= gsx_cpu_render_min_alpha_threshold_deactivated) {
            continue;
        }
        if(!gsx_cpu_render_project_gaussian(mean, cov3d, request->intrinsics, request->pose, world_to_camera, &projected)) {
            continue;
        }
        det = projected.cov2d_a * projected.cov2d_c - projected.cov2d_b * projected.cov2d_b;
        if(projected.depth < request->near_plane || projected.depth > request->far_plane || det <= 0.0f) {
            continue;
        }
        gsx_cpu_render_compute_bbox(
            projected.cov2d_a,
            projected.cov2d_b,
            projected.cov2d_c,
            projected.mean2d_x,
            projected.mean2d_y,
            request->intrinsics->width,
            request->intrinsics->height,
            &x_min,
            &x_max,
            &y_min,
            &y_max);
        if(x_min > x_max || y_min > y_max) {
            continue;
        }

        visible[visible_index].gaussian_index = i;
        visible[visible_index].x_min = x_min;
        visible[visible_index].x_max = x_max;
        visible[visible_index].y_min = y_min;
        visible[visible_index].y_max = y_max;
        visible[visible_index].mean2d_x = projected.mean2d_x;
        visible[visible_index].mean2d_y = projected.mean2d_y;
        visible[visible_index].cov2d_a = projected.cov2d_a;
        visible[visible_index].cov2d_b = projected.cov2d_b;
        visible[visible_index].cov2d_c = projected.cov2d_c;
        visible[visible_index].inv_det = 1.0f / det;
        visible[visible_index].depth = projected.depth;
        visible[visible_index].x_cam = projected.x_cam;
        visible[visible_index].y_cam = projected.y_cam;
        visible[visible_index].z_cam = projected.z_cam;
        visible[visible_index].raw_opacity = opacities[i];
        visible[visible_index].opacity = gsx_cpu_render_sigmoid(opacities[i]);

        {
            gsx_cpu_vec3 direction = gsx_cpu_render_vec3_normalize(gsx_cpu_render_vec3_sub(mean, camera_position));
            gsx_cpu_vec3 color = gsx_cpu_render_evaluate_sh(i, request->sh_degree, direction, sh0, sh1, sh2, sh3);

            visible[visible_index].dir[0] = direction.x;
            visible[visible_index].dir[1] = direction.y;
            visible[visible_index].dir[2] = direction.z;
            visible[visible_index].color[0] = color.x;
            visible[visible_index].color[1] = color.y;
            visible[visible_index].color[2] = color.z;
        }

        visible_index += 1;
    }
}

static gsx_error gsx_cpu_render_forward_impl(
    const gsx_render_forward_request *request,
    gsx_cpu_render_visible *visible,
    gsx_size_t visible_count,
    float *transmittance)
{
    float *output = gsx_cpu_render_tensor_data_f32(request->out_rgb);
    gsx_size_t pixel_count = (gsx_size_t)request->intrinsics->width * (gsx_size_t)request->intrinsics->height;
    gsx_size_t index = 0;
    gsx_size_t visible_index = 0;

    memset(output, 0, (size_t)request->out_rgb->size_bytes);
    for(index = 0; index < pixel_count; ++index) {
        transmittance[index] = 1.0f;
    }

    qsort(visible, (size_t)visible_count, sizeof(*visible), gsx_cpu_render_compare_visible_depth);
    for(visible_index = 0; visible_index < visible_count; ++visible_index) {
        const gsx_cpu_render_visible *entry = &visible[visible_index];
        float inv_a = entry->cov2d_c * entry->inv_det;
        float inv_b = -entry->cov2d_b * entry->inv_det;
        float inv_c = entry->cov2d_a * entry->inv_det;
        int32_t py = 0;

        for(py = entry->y_min; py <= entry->y_max; ++py) {
            int32_t px = 0;

            for(px = entry->x_min; px <= entry->x_max; ++px) {
                float dx = ((float)px + 0.5f) - entry->mean2d_x;
                float dy = ((float)py + 0.5f) - entry->mean2d_y;
                float power = -0.5f * (inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_c * dy * dy);
                float gaussian_value = 0.0f;
                float alpha = 0.0f;
                gsx_size_t pixel_index = (gsx_size_t)py * (gsx_size_t)request->intrinsics->width + (gsx_size_t)px;
                float t_before = 0.0f;
                float weight = 0.0f;

                if(power > 0.0f) {
                    power = 0.0f;
                }
                if(power < gsx_cpu_render_power_threshold) {
                    continue;
                }

                gaussian_value = expf(power);
                alpha = fminf(gsx_cpu_render_max_fragment_alpha, entry->opacity * gaussian_value);
                if(alpha < gsx_cpu_render_min_alpha_threshold) {
                    continue;
                }

                t_before = transmittance[pixel_index];
                if(t_before < gsx_cpu_render_transmittance_threshold) {
                    continue;
                }
                weight = alpha * t_before;
                output[0 * pixel_count + pixel_index] += weight * entry->color[0];
                output[1 * pixel_count + pixel_index] += weight * entry->color[1];
                output[2 * pixel_count + pixel_index] += weight * entry->color[2];
                transmittance[pixel_index] = t_before * (1.0f - alpha);
            }
        }
    }

    for(index = 0; index < pixel_count; ++index) {
        output[0 * pixel_count + index] += transmittance[index] * request->background_color.x;
        output[1 * pixel_count + index] += transmittance[index] * request->background_color.y;
        output[2 * pixel_count + index] += transmittance[index] * request->background_color.z;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_render_validate_backward_sinks(
    const gsx_cpu_render_context *cpu_context,
    const gsx_render_backward_request *request)
{
    gsx_index_t rgb_shape[3];
    gsx_index_t pair_shape[2];
    gsx_index_t opacity_shape[1];
    gsx_index_t sh0_shape[2];
    gsx_index_t sh_shape[3];
    gsx_size_t gaussian_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    gaussian_count = (gsx_size_t)cpu_context->saved_mean3d->shape[0];
    rgb_shape[0] = 3;
    rgb_shape[1] = cpu_context->intrinsics.height;
    rgb_shape[2] = cpu_context->intrinsics.width;
    error = gsx_cpu_render_validate_tensor_shape(
        request->grad_rgb, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, rgb_shape, "grad_rgb must be float32 CHW with shape [3,H,W]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    pair_shape[0] = (gsx_index_t)gaussian_count;
    pair_shape[1] = 3;
    error = gsx_cpu_render_validate_tensor_shape(
        request->grad_gs_mean3d, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, pair_shape, "grad_gs_mean3d must match [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cpu_render_validate_tensor_shape(
        request->grad_gs_logscale, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, pair_shape, "grad_gs_logscale must match [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    pair_shape[1] = 4;
    error = gsx_cpu_render_validate_tensor_shape(
        request->grad_gs_rotation, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, pair_shape, "grad_gs_rotation must match [N,4]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    opacity_shape[0] = (gsx_index_t)gaussian_count;
    error = gsx_cpu_render_validate_tensor_shape(
        request->grad_gs_opacity, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 1, opacity_shape, "grad_gs_opacity must match [N]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    sh0_shape[0] = (gsx_index_t)gaussian_count;
    sh0_shape[1] = 3;
    error = gsx_cpu_render_validate_tensor_shape(
        request->grad_gs_sh0, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 2, sh0_shape, "grad_gs_sh0 must match [N,3]");
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(cpu_context->sh_degree >= 1) {
        if(request->grad_gs_sh1 == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh1 must be non-null when sh_degree is at least 1");
        }
        sh_shape[0] = (gsx_index_t)gaussian_count;
        sh_shape[1] = 3;
        sh_shape[2] = 3;
        error = gsx_cpu_render_validate_tensor_shape(
            request->grad_gs_sh1, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "grad_gs_sh1 must match [N,3,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->grad_gs_sh1 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh1 must be null when sh_degree is 0");
    }
    if(cpu_context->sh_degree >= 2) {
        if(request->grad_gs_sh2 == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh2 must be non-null when sh_degree is at least 2");
        }
        sh_shape[1] = 5;
        error = gsx_cpu_render_validate_tensor_shape(
            request->grad_gs_sh2, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "grad_gs_sh2 must match [N,5,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->grad_gs_sh2 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh2 must be null when sh_degree is less than 2");
    }
    if(cpu_context->sh_degree >= 3) {
        if(request->grad_gs_sh3 == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh3 must be non-null when sh_degree is 3");
        }
        sh_shape[1] = 7;
        error = gsx_cpu_render_validate_tensor_shape(
            request->grad_gs_sh3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, 3, sh_shape, "grad_gs_sh3 must match [N,7,3]");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(request->grad_gs_sh3 != NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "grad_gs_sh3 must be null when sh_degree is less than 3");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_render_validate_forward_request(const gsx_render_forward_request *request)
{
    if(request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "request must be non-null");
    }
    if(request->precision != GSX_RENDER_PRECISION_FLOAT32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "requested render precision is not supported");
    }
    if(request->forward_type == GSX_RENDER_FORWARD_TYPE_METRIC) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metric render mode is not implemented");
    }
    if(request->out_rgb == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_rgb must be non-null for inference and train forwards");
    }
    if(request->out_alpha != NULL || request->out_invdepth != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "alpha and inverse-depth outputs are not implemented");
    }
    if(request->gs_cov3d != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "gs_cov3d input is not implemented");
    }
    if(request->metric_map != NULL || request->gs_metric_accumulator != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metric render inputs are not implemented");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_render_validate_backward_core_request(gsx_renderer_t renderer, const gsx_render_backward_request *request)
{
    gsx_index_t rgb_shape[3];

    if(renderer == NULL || request == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer and request must be non-null");
    }
    if(request->grad_alpha != NULL || request->grad_invdepth != NULL || request->grad_gs_cov3d != NULL) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "only RGB backward is implemented");
    }
    if(request->grad_rgb == NULL
        || request->grad_gs_mean3d == NULL
        || request->grad_gs_rotation == NULL
        || request->grad_gs_logscale == NULL
        || request->grad_gs_sh0 == NULL
        || request->grad_gs_opacity == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "cpu renderer backward requires RGB and core Gaussian gradient sinks");
    }

    rgb_shape[0] = 3;
    rgb_shape[1] = renderer->info.height;
    rgb_shape[2] = renderer->info.width;
    return gsx_cpu_render_validate_tensor_shape(
        request->grad_rgb,
        GSX_DATA_TYPE_F32,
        GSX_STORAGE_FORMAT_CHW,
        3,
        rgb_shape,
        "grad_rgb must be float32 CHW with shape [3,H,W]");
}

static gsx_error gsx_cpu_renderer_destroy(gsx_renderer_t renderer)
{
    gsx_cpu_renderer *cpu_renderer = (gsx_cpu_renderer *)renderer;

    if(renderer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "renderer must be non-null");
    }

    gsx_renderer_base_deinit(&cpu_renderer->base);
    free(cpu_renderer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_renderer_create_context(gsx_renderer_t renderer, gsx_render_context_t *out_context)
{
    gsx_cpu_render_context *cpu_context = NULL;
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_context must be non-null");
    }
    *out_context = NULL;

    cpu_context = (gsx_cpu_render_context *)calloc(1, sizeof(*cpu_context));
    if(cpu_context == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu render context");
    }

    error = gsx_backend_find_buffer_type(renderer->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type);
    if(!gsx_error_is_success(error)) {
        free(cpu_context);
        return error;
    }

    error = gsx_render_context_base_init(&cpu_context->base, &gsx_cpu_render_context_iface, renderer);
    if(!gsx_error_is_success(error)) {
        free(cpu_context);
        return error;
    }
    error = gsx_cpu_render_init_arena(buffer_type, &cpu_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        gsx_render_context_base_deinit(&cpu_context->base);
        free(cpu_context);
        return error;
    }
    error = gsx_cpu_render_init_arena(buffer_type, &cpu_context->retain_arena);
    if(!gsx_error_is_success(error)) {
        gsx_error cleanup_error = gsx_arena_free(cpu_context->scratch_arena);

        if(!gsx_error_is_success(cleanup_error)) {
            gsx_cpu_render_log_cleanup_failure("gsx_arena_free", cleanup_error);
        }
        gsx_render_context_base_deinit(&cpu_context->base);
        free(cpu_context);
        return error;
    }

    *out_context = &cpu_context->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_render_context_destroy(gsx_render_context_t context)
{
    gsx_cpu_render_context *cpu_context = (gsx_cpu_render_context *)context;
    gsx_error first_error = { GSX_ERROR_SUCCESS, NULL };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "context must be non-null");
    }

    gsx_cpu_render_clear_snapshot(cpu_context);
    if(cpu_context->scratch_arena != NULL) {
        error = gsx_arena_free(cpu_context->scratch_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_log_cleanup_failure("gsx_arena_free", error);
        }
        cpu_context->scratch_arena = NULL;
    }
    if(cpu_context->retain_arena != NULL) {
        error = gsx_arena_free(cpu_context->retain_arena);
        if(!gsx_error_is_success(error) && gsx_error_is_success(first_error)) {
            first_error = error;
        }
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_log_cleanup_failure("gsx_arena_free", error);
        }
        cpu_context->retain_arena = NULL;
    }
    gsx_render_context_base_deinit(&cpu_context->base);
    free(cpu_context);
    return first_error;
}

static gsx_error gsx_cpu_renderer_render(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request)
{
    gsx_cpu_render_context *cpu_context = (gsx_cpu_render_context *)context;
    gsx_cpu_mat3 world_to_camera = gsx_cpu_render_rotation_matrix_from_xyzw(
        request->pose->rot.x, request->pose->rot.y, request->pose->rot.z, request->pose->rot.w);
    gsx_cpu_vec3 camera_position = gsx_cpu_render_camera_position(request->pose, world_to_camera);
    gsx_size_t contrib_capacity = 0;
    gsx_size_t visible_count = 0;
    gsx_size_t scratch_bytes = 0;
    gsx_tensor_t scratch_tensor = NULL;
    unsigned char *scratch_base = NULL;
    gsx_cpu_render_visible *visible = NULL;
    float *transmittance = NULL;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t pixel_count = (gsx_size_t)request->intrinsics->width * (gsx_size_t)request->intrinsics->height;
    bool debug_enabled = (renderer->info.feature_flags & GSX_RENDERER_FEATURE_DEBUG) != 0;

    (void)contrib_capacity;
    error = gsx_cpu_render_validate_forward_request(request);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(cpu_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(debug_enabled) {
        error = gsx_cpu_render_debug_check_tensor(request->gs_mean3d, "render debug validation failed: gs_mean3d is non-finite");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_cpu_render_debug_check_tensor(request->gs_rotation, "render debug validation failed: gs_rotation is non-finite");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_cpu_render_debug_check_tensor(request->gs_logscale, "render debug validation failed: gs_logscale is non-finite");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_cpu_render_debug_check_tensor(request->gs_opacity, "render debug validation failed: gs_opacity is non-finite");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_cpu_render_debug_check_tensor(request->gs_sh0, "render debug validation failed: gs_sh0 is non-finite");
        if(!gsx_error_is_success(error)) {
            return error;
        }
        if(request->sh_degree >= 1) {
            error = gsx_cpu_render_debug_check_tensor(request->gs_sh1, "render debug validation failed: gs_sh1 is non-finite");
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        if(request->sh_degree >= 2) {
            error = gsx_cpu_render_debug_check_tensor(request->gs_sh2, "render debug validation failed: gs_sh2 is non-finite");
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        if(request->sh_degree >= 3) {
            error = gsx_cpu_render_debug_check_tensor(request->gs_sh3, "render debug validation failed: gs_sh3 is non-finite");
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
    }

    visible_count = gsx_cpu_render_count_visible(request, world_to_camera, NULL);
    if(visible_count != 0) {
        if(gsx_size_mul_overflows((gsx_size_t)visible_count, (gsx_size_t)sizeof(*visible), &scratch_bytes)
            || gsx_size_add_overflows(scratch_bytes, pixel_count * sizeof(*transmittance), &scratch_bytes)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render scratch sizing overflow");
        }
        error = gsx_cpu_render_alloc_byte_tensor(cpu_context->scratch_arena, scratch_bytes, &scratch_tensor);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        scratch_base = gsx_cpu_render_tensor_data_bytes(scratch_tensor);
        visible = (gsx_cpu_render_visible *)scratch_base;
        transmittance = (float *)(scratch_base + (size_t)visible_count * sizeof(*visible));
        gsx_cpu_render_fill_visible(request, world_to_camera, camera_position, visible);
    } else {
        if(pixel_count != 0) {
            if(gsx_size_mul_overflows(pixel_count, sizeof(*transmittance), &scratch_bytes)) {
                return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render scratch sizing overflow");
            }
            error = gsx_cpu_render_alloc_byte_tensor(cpu_context->scratch_arena, scratch_bytes, &scratch_tensor);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            transmittance = (float *)gsx_cpu_render_tensor_data_bytes(scratch_tensor);
        }
    }

    error = gsx_cpu_render_forward_impl(request, visible, visible_count, transmittance);
    if(!gsx_error_is_success(error)) {
        gsx_cpu_render_free_tensor_handle(&scratch_tensor);
        return error;
    }
    if(request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN) {
        error = gsx_cpu_render_snapshot_request(cpu_context, request);
        if(!gsx_error_is_success(error)) {
            gsx_cpu_render_free_tensor_handle(&scratch_tensor);
            return error;
        }
    } else {
        gsx_cpu_render_clear_snapshot(cpu_context);
    }

    gsx_cpu_render_free_tensor_handle(&scratch_tensor);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_cpu_renderer_backward(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_backward_request *request)
{
    gsx_cpu_render_context *cpu_context = (gsx_cpu_render_context *)context;
    gsx_cpu_mat3 world_to_camera;
    gsx_cpu_vec3 camera_position;
    gsx_render_forward_request forward_request = { 0 };
    gsx_size_t visible_count = 0;
    gsx_size_t contrib_capacity = 0;
    gsx_size_t pixel_count = 0;
    gsx_size_t scratch_bytes = 0;
    gsx_tensor_t scratch_tensor = NULL;
    unsigned char *scratch_base = NULL;
    gsx_cpu_render_visible *visible = NULL;
    gsx_size_t *contrib_offsets = NULL;
    gsx_cpu_render_pixel_contrib *contribs = NULL;
    float *transmittance = NULL;
    gsx_cpu_vec3 *suffix_rgb = NULL;
    const float *saved_means = NULL;
    const float *saved_sh1 = NULL;
    const float *saved_sh2 = NULL;
    const float *saved_sh3 = NULL;
    float *grad_rgb = NULL;
    float *grad_mean3d = NULL;
    float *grad_rotation = NULL;
    float *grad_logscale = NULL;
    float *grad_sh0 = NULL;
    float *grad_sh1 = NULL;
    float *grad_sh2 = NULL;
    float *grad_sh3 = NULL;
    float *grad_opacity = NULL;
    bool debug_enabled = (renderer->info.feature_flags & GSX_RENDERER_FEATURE_DEBUG) != 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t visible_index = 0;
    gsx_size_t total_contrib_count = 0;

    error = gsx_cpu_render_validate_backward_core_request(renderer, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!cpu_context->has_train_state) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backward requires a retained TRAIN forward on the same context");
    }
    error = gsx_cpu_render_validate_backward_sinks(cpu_context, request);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(debug_enabled) {
        error = gsx_cpu_render_debug_check_tensor(request->grad_rgb, "render debug validation failed: grad_rgb is non-finite");
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    forward_request.intrinsics = &cpu_context->intrinsics;
    forward_request.pose = &cpu_context->pose;
    forward_request.near_plane = cpu_context->near_plane;
    forward_request.far_plane = cpu_context->far_plane;
    forward_request.background_color = cpu_context->background_color;
    forward_request.precision = GSX_RENDER_PRECISION_FLOAT32;
    forward_request.sh_degree = cpu_context->sh_degree;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_TRAIN;
    forward_request.borrow_train_state = false;
    forward_request.gs_mean3d = cpu_context->saved_mean3d;
    forward_request.gs_rotation = cpu_context->saved_rotation;
    forward_request.gs_logscale = cpu_context->saved_logscale;
    forward_request.gs_sh0 = cpu_context->saved_sh0;
    forward_request.gs_sh1 = cpu_context->saved_sh1;
    forward_request.gs_sh2 = cpu_context->saved_sh2;
    forward_request.gs_sh3 = cpu_context->saved_sh3;
    forward_request.gs_opacity = cpu_context->saved_opacity;
    world_to_camera = gsx_cpu_render_rotation_matrix_from_xyzw(
        cpu_context->pose.rot.x, cpu_context->pose.rot.y, cpu_context->pose.rot.z, cpu_context->pose.rot.w);
    camera_position = gsx_cpu_render_camera_position(&cpu_context->pose, world_to_camera);
    visible_count = gsx_cpu_render_count_visible(&forward_request, world_to_camera, &contrib_capacity);
    pixel_count = (gsx_size_t)cpu_context->intrinsics.width * (gsx_size_t)cpu_context->intrinsics.height;

    error = gsx_arena_reset(cpu_context->scratch_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gsx_size_mul_overflows((gsx_size_t)visible_count, (gsx_size_t)sizeof(*visible), &scratch_bytes)
        || gsx_size_add_overflows(scratch_bytes, ((gsx_size_t)visible_count + 1u) * sizeof(*contrib_offsets), &scratch_bytes)
        || gsx_size_add_overflows(scratch_bytes, contrib_capacity * sizeof(*contribs), &scratch_bytes)
        || gsx_size_add_overflows(scratch_bytes, pixel_count * sizeof(*transmittance), &scratch_bytes)
        || gsx_size_add_overflows(scratch_bytes, pixel_count * sizeof(*suffix_rgb), &scratch_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "render backward scratch sizing overflow");
    }
    error = gsx_cpu_render_alloc_byte_tensor(cpu_context->scratch_arena, scratch_bytes, &scratch_tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    scratch_base = gsx_cpu_render_tensor_data_bytes(scratch_tensor);
    visible = (gsx_cpu_render_visible *)scratch_base;
    scratch_base += (size_t)visible_count * sizeof(*visible);
    contrib_offsets = (gsx_size_t *)scratch_base;
    scratch_base += ((size_t)visible_count + 1u) * sizeof(*contrib_offsets);
    contribs = (gsx_cpu_render_pixel_contrib *)scratch_base;
    scratch_base += (size_t)contrib_capacity * sizeof(*contribs);
    transmittance = (float *)scratch_base;
    scratch_base += (size_t)pixel_count * sizeof(*transmittance);
    suffix_rgb = (gsx_cpu_vec3 *)scratch_base;
    gsx_cpu_render_fill_visible(&forward_request, world_to_camera, camera_position, visible);
    qsort(visible, (size_t)visible_count, sizeof(*visible), gsx_cpu_render_compare_visible_depth);
    for(visible_index = 0; visible_index < pixel_count; ++visible_index) {
        transmittance[visible_index] = 1.0f;
    }

    for(visible_index = 0; visible_index < visible_count; ++visible_index) {
        const gsx_cpu_render_visible *entry = &visible[visible_index];
        float inv_a = entry->cov2d_c * entry->inv_det;
        float inv_b = -entry->cov2d_b * entry->inv_det;
        float inv_c = entry->cov2d_a * entry->inv_det;
        int32_t py = 0;

        contrib_offsets[visible_index] = total_contrib_count;
        for(py = entry->y_min; py <= entry->y_max; ++py) {
            int32_t px = 0;

            for(px = entry->x_min; px <= entry->x_max; ++px) {
                float dx = ((float)px + 0.5f) - entry->mean2d_x;
                float dy = ((float)py + 0.5f) - entry->mean2d_y;
                float power = -0.5f * (inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_c * dy * dy);
                float gaussian_value = 0.0f;
                float alpha = 0.0f;
                gsx_size_t pixel_index = (gsx_size_t)py * (gsx_size_t)cpu_context->intrinsics.width + (gsx_size_t)px;
                float t_before = transmittance[pixel_index];

                if(power > 0.0f) {
                    power = 0.0f;
                }
                if(power < gsx_cpu_render_power_threshold) {
                    continue;
                }
                gaussian_value = expf(power);
                alpha = fminf(gsx_cpu_render_max_fragment_alpha, entry->opacity * gaussian_value);
                if(alpha < gsx_cpu_render_min_alpha_threshold || t_before < gsx_cpu_render_transmittance_threshold) {
                    continue;
                }
                contribs[total_contrib_count].pixel_index = (int32_t)pixel_index;
                contribs[total_contrib_count].alpha = alpha;
                contribs[total_contrib_count].t_before = t_before;
                total_contrib_count += 1;
                transmittance[pixel_index] = t_before * (1.0f - alpha);
            }
        }
    }
    contrib_offsets[visible_count] = total_contrib_count;

    grad_rgb = gsx_cpu_render_tensor_data_f32(request->grad_rgb);
    saved_means = gsx_cpu_render_tensor_data_f32_const(cpu_context->saved_mean3d);
    grad_mean3d = gsx_cpu_render_tensor_data_f32(request->grad_gs_mean3d);
    grad_rotation = gsx_cpu_render_tensor_data_f32(request->grad_gs_rotation);
    grad_logscale = gsx_cpu_render_tensor_data_f32(request->grad_gs_logscale);
    grad_sh0 = gsx_cpu_render_tensor_data_f32(request->grad_gs_sh0);
    grad_opacity = gsx_cpu_render_tensor_data_f32(request->grad_gs_opacity);
    if(request->grad_gs_sh1 != NULL) {
        grad_sh1 = gsx_cpu_render_tensor_data_f32(request->grad_gs_sh1);
        saved_sh1 = gsx_cpu_render_tensor_data_f32_const(cpu_context->saved_sh1);
    }
    if(request->grad_gs_sh2 != NULL) {
        grad_sh2 = gsx_cpu_render_tensor_data_f32(request->grad_gs_sh2);
        saved_sh2 = gsx_cpu_render_tensor_data_f32_const(cpu_context->saved_sh2);
    }
    if(request->grad_gs_sh3 != NULL) {
        grad_sh3 = gsx_cpu_render_tensor_data_f32(request->grad_gs_sh3);
        saved_sh3 = gsx_cpu_render_tensor_data_f32_const(cpu_context->saved_sh3);
    }

    memset(grad_mean3d, 0, (size_t)request->grad_gs_mean3d->size_bytes);
    memset(grad_rotation, 0, (size_t)request->grad_gs_rotation->size_bytes);
    memset(grad_logscale, 0, (size_t)request->grad_gs_logscale->size_bytes);
    memset(grad_sh0, 0, (size_t)request->grad_gs_sh0->size_bytes);
    memset(grad_opacity, 0, (size_t)request->grad_gs_opacity->size_bytes);
    if(grad_sh1 != NULL) {
        memset(grad_sh1, 0, (size_t)request->grad_gs_sh1->size_bytes);
    }
    if(grad_sh2 != NULL) {
        memset(grad_sh2, 0, (size_t)request->grad_gs_sh2->size_bytes);
    }
    if(grad_sh3 != NULL) {
        memset(grad_sh3, 0, (size_t)request->grad_gs_sh3->size_bytes);
    }

    for(visible_index = 0; visible_index < pixel_count; ++visible_index) {
        suffix_rgb[visible_index] = gsx_cpu_render_vec3(
            transmittance[visible_index] * cpu_context->background_color.x,
            transmittance[visible_index] * cpu_context->background_color.y,
            transmittance[visible_index] * cpu_context->background_color.z);
    }

    for(visible_index = visible_count; visible_index > 0; --visible_index) {
        gsx_size_t entry_index = visible_index - 1u;
        const gsx_cpu_render_visible *entry = &visible[entry_index];
        const float *rotations = gsx_cpu_render_tensor_data_f32_const(cpu_context->saved_rotation);
        const float *logscales = gsx_cpu_render_tensor_data_f32_const(cpu_context->saved_logscale);
        gsx_size_t gaussian_index = entry->gaussian_index;
        gsx_size_t contrib_index = 0;
        float dcolor_x = 0.0f;
        float dcolor_y = 0.0f;
        float dcolor_z = 0.0f;
        float dmean2d_x = 0.0f;
        float dmean2d_y = 0.0f;
        float dcov_a = 0.0f;
        float dcov_b = 0.0f;
        float dcov_c = 0.0f;
        float dopacity = 0.0f;
        float inv_a = entry->cov2d_c * entry->inv_det;
        float inv_b = -entry->cov2d_b * entry->inv_det;
        float inv_c = entry->cov2d_a * entry->inv_det;
        float det = entry->cov2d_a * entry->cov2d_c - entry->cov2d_b * entry->cov2d_b;
        float inv_det2 = 1.0f / (det * det);
        gsx_cpu_mat3 cov3d = gsx_cpu_render_cov3d_from_inputs(
            rotations[gaussian_index * 4u + 0],
            rotations[gaussian_index * 4u + 1],
            rotations[gaussian_index * 4u + 2],
            rotations[gaussian_index * 4u + 3],
            logscales[gaussian_index * 3u + 0],
            logscales[gaussian_index * 3u + 1],
            logscales[gaussian_index * 3u + 2]);
        gsx_cpu_mat3 rotation = gsx_cpu_render_rotation_matrix_from_xyzw(
            rotations[gaussian_index * 4u + 0],
            rotations[gaussian_index * 4u + 1],
            rotations[gaussian_index * 4u + 2],
            rotations[gaussian_index * 4u + 3]);
        gsx_cpu_mat3 scaled_rotation = rotation;
        gsx_cpu_mat3 d_cov3d = gsx_cpu_render_mat3_zero();
        gsx_cpu_mat3 d_rs = gsx_cpu_render_mat3_zero();
        gsx_cpu_mat3 d_r = gsx_cpu_render_mat3_zero();
        float scale_x = gsx_cpu_render_exp(logscales[gaussian_index * 3u + 0]);
        float scale_y = gsx_cpu_render_exp(logscales[gaussian_index * 3u + 1]);
        float scale_z = gsx_cpu_render_exp(logscales[gaussian_index * 3u + 2]);
        float j00 = cpu_context->intrinsics.fx / entry->z_cam;
        float j02 = -cpu_context->intrinsics.fx * entry->x_cam / (entry->z_cam * entry->z_cam);
        float j11 = cpu_context->intrinsics.fy / entry->z_cam;
        float j12 = -cpu_context->intrinsics.fy * entry->y_cam / (entry->z_cam * entry->z_cam);
        float m_matrix[2][3];
        int row = 0;
        int col = 0;

        for(contrib_index = contrib_offsets[entry_index]; contrib_index < contrib_offsets[entry_index + 1u]; ++contrib_index) {
            const gsx_cpu_render_pixel_contrib *contrib = &contribs[contrib_index];
            gsx_size_t pixel_index = (gsx_size_t)contrib->pixel_index;
            int32_t py = contrib->pixel_index / cpu_context->intrinsics.width;
            int32_t px = contrib->pixel_index % cpu_context->intrinsics.width;
            float dout_x = grad_rgb[0 * pixel_count + pixel_index];
            float dout_y = grad_rgb[1 * pixel_count + pixel_index];
            float dout_z = grad_rgb[2 * pixel_count + pixel_index];
            float dx = ((float)px + 0.5f) - entry->mean2d_x;
            float dy = ((float)py + 0.5f) - entry->mean2d_y;
            float power = -0.5f * (inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_c * dy * dy);
            float gaussian_value = expf(power > 0.0f ? 0.0f : power);
            float alpha = contrib->alpha;
            float t_before = contrib->t_before;
            float weight = alpha * t_before;
            gsx_cpu_vec3 suffix = suffix_rgb[pixel_index];
            float dalpha = t_before * (entry->color[0] * dout_x + entry->color[1] * dout_y + entry->color[2] * dout_z);
            float dgauss = 0.0f;
            float dpower = 0.0f;
            float dinv_a = 0.0f;
            float dinv_b = 0.0f;
            float dinv_c = 0.0f;

            dcolor_x += weight * dout_x;
            dcolor_y += weight * dout_y;
            dcolor_z += weight * dout_z;
            if(fabsf(1.0f - alpha) > 1.0e-6f) {
                dalpha -= (suffix.x * dout_x + suffix.y * dout_y + suffix.z * dout_z) / (1.0f - alpha);
            }
            if(entry->opacity * gaussian_value < gsx_cpu_render_max_fragment_alpha) {
                dgauss = dalpha * entry->opacity;
                dopacity += dalpha * gaussian_value;
            }
            dpower = dgauss * gaussian_value;
            dmean2d_x += dpower * (inv_a * dx + inv_b * dy);
            dmean2d_y += dpower * (inv_b * dx + inv_c * dy);
            dinv_a = dpower * (-0.5f * dx * dx);
            dinv_b = dpower * (-dx * dy);
            dinv_c = dpower * (-0.5f * dy * dy);
            dcov_a += dinv_a * (-entry->cov2d_c * entry->cov2d_c * inv_det2)
                + dinv_b * (entry->cov2d_b * entry->cov2d_c * inv_det2)
                + dinv_c * (-entry->cov2d_b * entry->cov2d_b * inv_det2);
            dcov_b += dinv_a * (2.0f * entry->cov2d_b * entry->cov2d_c * inv_det2)
                + dinv_b * (-(entry->cov2d_a * entry->cov2d_c + entry->cov2d_b * entry->cov2d_b) * inv_det2)
                + dinv_c * (2.0f * entry->cov2d_a * entry->cov2d_b * inv_det2);
            dcov_c += dinv_a * (-entry->cov2d_b * entry->cov2d_b * inv_det2)
                + dinv_b * (entry->cov2d_a * entry->cov2d_b * inv_det2)
                + dinv_c * (-entry->cov2d_a * entry->cov2d_a * inv_det2);
        }

        for(contrib_index = contrib_offsets[entry_index]; contrib_index < contrib_offsets[entry_index + 1u]; ++contrib_index) {
            gsx_size_t pixel_index = (gsx_size_t)contribs[contrib_index].pixel_index;
            float weight = contribs[contrib_index].alpha * contribs[contrib_index].t_before;

            suffix_rgb[pixel_index].x += weight * entry->color[0];
            suffix_rgb[pixel_index].y += weight * entry->color[1];
            suffix_rgb[pixel_index].z += weight * entry->color[2];
        }

        {
            float mask_x = entry->color[0] > 0.0f ? 1.0f : 0.0f;
            float mask_y = entry->color[1] > 0.0f ? 1.0f : 0.0f;
            float mask_z = entry->color[2] > 0.0f ? 1.0f : 0.0f;
            gsx_cpu_vec3 dcolor_clamped = gsx_cpu_render_vec3(dcolor_x * mask_x, dcolor_y * mask_y, dcolor_z * mask_z);
            gsx_cpu_vec3 ddir = gsx_cpu_render_vec3(0.0f, 0.0f, 0.0f);
            float x = entry->dir[0];
            float y = entry->dir[1];
            float z = entry->dir[2];
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;

            grad_sh0[gaussian_index * 3u + 0] += gsx_cpu_render_sh_c0 * dcolor_clamped.x;
            grad_sh0[gaussian_index * 3u + 1] += gsx_cpu_render_sh_c0 * dcolor_clamped.y;
            grad_sh0[gaussian_index * 3u + 2] += gsx_cpu_render_sh_c0 * dcolor_clamped.z;

            if(cpu_context->sh_degree >= 1 && grad_sh1 != NULL) {
                gsx_cpu_vec3 c0 = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 0, 0),
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 0, 1),
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 0, 2));
                gsx_cpu_vec3 c1 = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 1, 0),
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 1, 1),
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 1, 2));
                gsx_cpu_vec3 c2 = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 2, 0),
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 2, 1),
                    gsx_cpu_render_read_sh(saved_sh1, gaussian_index, 3, 2, 2));

                gsx_cpu_render_accum_sh_grad(grad_sh1, gaussian_index, 3, 0, gsx_cpu_render_vec3_scale(dcolor_clamped, -gsx_cpu_render_sh_c1 * y));
                gsx_cpu_render_accum_sh_grad(grad_sh1, gaussian_index, 3, 1, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c1 * z));
                gsx_cpu_render_accum_sh_grad(grad_sh1, gaussian_index, 3, 2, gsx_cpu_render_vec3_scale(dcolor_clamped, -gsx_cpu_render_sh_c1 * x));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, c0, 0.0f, -gsx_cpu_render_sh_c1, 0.0f);
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, c1, 0.0f, 0.0f, gsx_cpu_render_sh_c1);
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, c2, -gsx_cpu_render_sh_c1, 0.0f, 0.0f);
            }
            if(cpu_context->sh_degree >= 2 && grad_sh2 != NULL) {
                gsx_cpu_vec3 coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 0, 0),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 0, 1),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 0, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh2, gaussian_index, 5, 0, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c2[0] * xy));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, gsx_cpu_render_sh_c2[0] * y, gsx_cpu_render_sh_c2[0] * x, 0.0f);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 1, 0),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 1, 1),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 1, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh2, gaussian_index, 5, 1, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c2[1] * yz));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, 0.0f, gsx_cpu_render_sh_c2[1] * z, gsx_cpu_render_sh_c2[1] * y);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 2, 0),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 2, 1),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 2, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh2, gaussian_index, 5, 2, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c2[2] * (2.0f * zz - xx - yy)));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, -2.0f * gsx_cpu_render_sh_c2[2] * x, -2.0f * gsx_cpu_render_sh_c2[2] * y, 4.0f * gsx_cpu_render_sh_c2[2] * z);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 3, 0),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 3, 1),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 3, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh2, gaussian_index, 5, 3, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c2[3] * xz));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, gsx_cpu_render_sh_c2[3] * z, 0.0f, gsx_cpu_render_sh_c2[3] * x);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 4, 0),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 4, 1),
                    gsx_cpu_render_read_sh(saved_sh2, gaussian_index, 5, 4, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh2, gaussian_index, 5, 4, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c2[4] * (xx - yy)));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, 2.0f * gsx_cpu_render_sh_c2[4] * x, -2.0f * gsx_cpu_render_sh_c2[4] * y, 0.0f);
            }
            if(cpu_context->sh_degree >= 3 && grad_sh3 != NULL) {
                gsx_cpu_vec3 coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 0, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 0, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 0, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 0, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[0] * y * (3.0f * xx - yy)));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, 6.0f * gsx_cpu_render_sh_c3[0] * x * y, 3.0f * gsx_cpu_render_sh_c3[0] * (xx - yy), 0.0f);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 1, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 1, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 1, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 1, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[1] * xy * z));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, gsx_cpu_render_sh_c3[1] * y * z, gsx_cpu_render_sh_c3[1] * x * z, gsx_cpu_render_sh_c3[1] * xy);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 2, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 2, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 2, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 2, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[2] * y * (4.0f * zz - xx - yy)));
                gsx_cpu_render_accum_direction_grad(
                    &ddir,
                    dcolor_clamped,
                    coeff,
                    -2.0f * gsx_cpu_render_sh_c3[2] * x * y,
                    gsx_cpu_render_sh_c3[2] * (4.0f * zz - xx - 3.0f * yy),
                    8.0f * gsx_cpu_render_sh_c3[2] * y * z);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 3, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 3, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 3, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 3, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy)));
                gsx_cpu_render_accum_direction_grad(
                    &ddir,
                    dcolor_clamped,
                    coeff,
                    -6.0f * gsx_cpu_render_sh_c3[3] * x * z,
                    -6.0f * gsx_cpu_render_sh_c3[3] * y * z,
                    gsx_cpu_render_sh_c3[3] * (6.0f * zz - 3.0f * xx - 3.0f * yy));
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 4, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 4, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 4, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 4, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[4] * x * (4.0f * zz - xx - yy)));
                gsx_cpu_render_accum_direction_grad(
                    &ddir,
                    dcolor_clamped,
                    coeff,
                    gsx_cpu_render_sh_c3[4] * (4.0f * zz - 3.0f * xx - yy),
                    -2.0f * gsx_cpu_render_sh_c3[4] * x * y,
                    8.0f * gsx_cpu_render_sh_c3[4] * x * z);
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 5, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 5, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 5, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 5, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[5] * z * (xx - yy)));
                gsx_cpu_render_accum_direction_grad(
                    &ddir,
                    dcolor_clamped,
                    coeff,
                    2.0f * gsx_cpu_render_sh_c3[5] * x * z,
                    -2.0f * gsx_cpu_render_sh_c3[5] * y * z,
                    gsx_cpu_render_sh_c3[5] * (xx - yy));
                coeff = gsx_cpu_render_vec3(
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 6, 0),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 6, 1),
                    gsx_cpu_render_read_sh(saved_sh3, gaussian_index, 7, 6, 2));
                gsx_cpu_render_accum_sh_grad(grad_sh3, gaussian_index, 7, 6, gsx_cpu_render_vec3_scale(dcolor_clamped, gsx_cpu_render_sh_c3[6] * x * (xx - 3.0f * yy)));
                gsx_cpu_render_accum_direction_grad(&ddir, dcolor_clamped, coeff, 3.0f * gsx_cpu_render_sh_c3[6] * (xx - yy), -6.0f * gsx_cpu_render_sh_c3[6] * x * y, 0.0f);
            }

            if(cpu_context->sh_degree >= 1) {
                gsx_cpu_vec3 mean = gsx_cpu_render_vec3(
                    saved_means[gaussian_index * 3u + 0],
                    saved_means[gaussian_index * 3u + 1],
                    saved_means[gaussian_index * 3u + 2]);
                gsx_cpu_vec3 view = gsx_cpu_render_vec3_sub(mean, camera_position);
                float view_norm = gsx_cpu_render_vec3_length(view);

                if(view_norm > 0.0f) {
                    float dir_dot = gsx_cpu_render_vec3_dot(ddir, gsx_cpu_render_vec3(entry->dir[0], entry->dir[1], entry->dir[2]));
                    gsx_cpu_vec3 dview = gsx_cpu_render_vec3_scale(
                        gsx_cpu_render_vec3_sub(ddir, gsx_cpu_render_vec3_scale(gsx_cpu_render_vec3(entry->dir[0], entry->dir[1], entry->dir[2]), dir_dot)),
                        1.0f / view_norm);

                    grad_mean3d[gaussian_index * 3u + 0] += dview.x;
                    grad_mean3d[gaussian_index * 3u + 1] += dview.y;
                    grad_mean3d[gaussian_index * 3u + 2] += dview.z;
                }
            }
        }

        grad_opacity[gaussian_index] += dopacity * gsx_cpu_render_sigmoid_grad_from_raw(entry->raw_opacity);

        {
            float inv_z = 1.0f / entry->z_cam;
            float inv_z2 = inv_z * inv_z;
            gsx_cpu_vec3 dmean_cam = gsx_cpu_render_vec3(
                dmean2d_x * cpu_context->intrinsics.fx * inv_z,
                dmean2d_y * cpu_context->intrinsics.fy * inv_z,
                dmean2d_x * (-cpu_context->intrinsics.fx * entry->x_cam * inv_z2)
                    + dmean2d_y * (-cpu_context->intrinsics.fy * entry->y_cam * inv_z2));
            gsx_cpu_vec3 dmean_world = gsx_cpu_render_mat3_mul_vec3(gsx_cpu_render_mat3_transpose(world_to_camera), dmean_cam);

            grad_mean3d[gaussian_index * 3u + 0] += dmean_world.x;
            grad_mean3d[gaussian_index * 3u + 1] += dmean_world.y;
            grad_mean3d[gaussian_index * 3u + 2] += dmean_world.z;
        }

        for(col = 0; col < 3; ++col) {
            m_matrix[0][col] = j00 * world_to_camera.m[0][col] + j02 * world_to_camera.m[2][col];
            m_matrix[1][col] = j11 * world_to_camera.m[1][col] + j12 * world_to_camera.m[2][col];
        }
        for(row = 0; row < 3; ++row) {
            for(col = 0; col < 3; ++col) {
                d_cov3d.m[row][col] =
                    m_matrix[0][row] * (dcov_a * m_matrix[0][col] + 0.5f * dcov_b * m_matrix[1][col])
                    + m_matrix[1][row] * (0.5f * dcov_b * m_matrix[0][col] + dcov_c * m_matrix[1][col]);
            }
        }
        for(row = 0; row < 3; ++row) {
            scaled_rotation.m[row][0] *= scale_x;
            scaled_rotation.m[row][1] *= scale_y;
            scaled_rotation.m[row][2] *= scale_z;
        }
        d_rs = gsx_cpu_render_mat3_mul(gsx_cpu_render_mat3_add(d_cov3d, gsx_cpu_render_mat3_transpose(d_cov3d)), scaled_rotation);
        grad_logscale[gaussian_index * 3u + 0] += (
            d_rs.m[0][0] * rotation.m[0][0] + d_rs.m[1][0] * rotation.m[1][0] + d_rs.m[2][0] * rotation.m[2][0]) * scale_x;
        grad_logscale[gaussian_index * 3u + 1] += (
            d_rs.m[0][1] * rotation.m[0][1] + d_rs.m[1][1] * rotation.m[1][1] + d_rs.m[2][1] * rotation.m[2][1]) * scale_y;
        grad_logscale[gaussian_index * 3u + 2] += (
            d_rs.m[0][2] * rotation.m[0][2] + d_rs.m[1][2] * rotation.m[1][2] + d_rs.m[2][2] * rotation.m[2][2]) * scale_z;

        for(row = 0; row < 3; ++row) {
            d_r.m[row][0] = d_rs.m[row][0] * scale_x;
            d_r.m[row][1] = d_rs.m[row][1] * scale_y;
            d_r.m[row][2] = d_rs.m[row][2] * scale_z;
        }

        {
            float qx = rotations[gaussian_index * 4u + 0];
            float qy = rotations[gaussian_index * 4u + 1];
            float qz = rotations[gaussian_index * 4u + 2];
            float qw = rotations[gaussian_index * 4u + 3];
            bool has_normalized_quaternion = false;
            float inv_norm = 0.0f;
            float nx = 0.0f;
            float ny = 0.0f;
            float nz = 0.0f;
            float nw = 1.0f;
            float dqw = 0.0f;
            float dqx = 0.0f;
            float dqy = 0.0f;
            float dqz = 0.0f;
            float dot_q = 0.0f;

            has_normalized_quaternion = gsx_cpu_render_normalize_quaternion_xyzw(qx, qy, qz, qw, &nx, &ny, &nz, &nw);
            if(has_normalized_quaternion) {
                inv_norm = 1.0f / sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
                dqw += d_r.m[0][1] * (-2.0f * nz);
                dqw += d_r.m[0][2] * (2.0f * ny);
                dqw += d_r.m[1][0] * (2.0f * nz);
                dqw += d_r.m[1][2] * (-2.0f * nx);
                dqw += d_r.m[2][0] * (-2.0f * ny);
                dqw += d_r.m[2][1] * (2.0f * nx);

                dqx += d_r.m[0][1] * (2.0f * ny);
                dqx += d_r.m[0][2] * (2.0f * nz);
                dqx += d_r.m[1][0] * (2.0f * ny);
                dqx += d_r.m[1][1] * (-4.0f * nx);
                dqx += d_r.m[1][2] * (-2.0f * nw);
                dqx += d_r.m[2][0] * (2.0f * nz);
                dqx += d_r.m[2][1] * (2.0f * nw);
                dqx += d_r.m[2][2] * (-4.0f * nx);

                dqy += d_r.m[0][0] * (-4.0f * ny);
                dqy += d_r.m[0][1] * (2.0f * nx);
                dqy += d_r.m[0][2] * (2.0f * nw);
                dqy += d_r.m[1][0] * (2.0f * nx);
                dqy += d_r.m[1][2] * (2.0f * nz);
                dqy += d_r.m[2][0] * (-2.0f * nw);
                dqy += d_r.m[2][1] * (2.0f * nz);
                dqy += d_r.m[2][2] * (-4.0f * ny);

                dqz += d_r.m[0][0] * (-4.0f * nz);
                dqz += d_r.m[0][1] * (-2.0f * nw);
                dqz += d_r.m[0][2] * (2.0f * nx);
                dqz += d_r.m[1][0] * (2.0f * nw);
                dqz += d_r.m[1][1] * (-4.0f * nz);
                dqz += d_r.m[1][2] * (2.0f * ny);
                dqz += d_r.m[2][0] * (2.0f * nx);
                dqz += d_r.m[2][1] * (2.0f * ny);

                dot_q = nx * dqx + ny * dqy + nz * dqz + nw * dqw;
                grad_rotation[gaussian_index * 4u + 0] += (dqx - nx * dot_q) * inv_norm;
                grad_rotation[gaussian_index * 4u + 1] += (dqy - ny * dot_q) * inv_norm;
                grad_rotation[gaussian_index * 4u + 2] += (dqz - nz * dot_q) * inv_norm;
                grad_rotation[gaussian_index * 4u + 3] += (dqw - nw * dot_q) * inv_norm;
            }
        }

        {
            gsx_cpu_mat3 cov_cam = gsx_cpu_render_mat3_mul(
                gsx_cpu_render_mat3_mul(world_to_camera, cov3d),
                gsx_cpu_render_mat3_transpose(world_to_camera));
            float d_j[2][3];
            float inv_z = 1.0f / entry->z_cam;
            float inv_z2 = inv_z * inv_z;
            float inv_z3 = inv_z2 * inv_z;
            gsx_cpu_vec3 dmean_cam_j;
            gsx_cpu_vec3 dmean_world_j;

            for(col = 0; col < 3; ++col) {
                d_j[0][col] =
                    2.0f * dcov_a * (j00 * cov_cam.m[0][col] + j02 * cov_cam.m[2][col])
                    + dcov_b * (j11 * cov_cam.m[1][col] + j12 * cov_cam.m[2][col]);
                d_j[1][col] =
                    dcov_b * (j00 * cov_cam.m[0][col] + j02 * cov_cam.m[2][col])
                    + 2.0f * dcov_c * (j11 * cov_cam.m[1][col] + j12 * cov_cam.m[2][col]);
            }

            dmean_cam_j.x = d_j[0][2] * (-cpu_context->intrinsics.fx * inv_z2);
            dmean_cam_j.y = d_j[1][2] * (-cpu_context->intrinsics.fy * inv_z2);
            dmean_cam_j.z = d_j[0][0] * (-cpu_context->intrinsics.fx * inv_z2)
                + d_j[0][2] * (2.0f * cpu_context->intrinsics.fx * entry->x_cam * inv_z3)
                + d_j[1][1] * (-cpu_context->intrinsics.fy * inv_z2)
                + d_j[1][2] * (2.0f * cpu_context->intrinsics.fy * entry->y_cam * inv_z3);
            dmean_world_j = gsx_cpu_render_mat3_mul_vec3(gsx_cpu_render_mat3_transpose(world_to_camera), dmean_cam_j);
            grad_mean3d[gaussian_index * 3u + 0] += dmean_world_j.x;
            grad_mean3d[gaussian_index * 3u + 1] += dmean_world_j.y;
            grad_mean3d[gaussian_index * 3u + 2] += dmean_world_j.z;
        }
    }

    gsx_cpu_render_free_tensor_handle(&scratch_tensor);
    gsx_cpu_render_clear_snapshot(cpu_context);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static const gsx_renderer_i gsx_cpu_renderer_iface = {
    gsx_cpu_renderer_destroy,
    gsx_cpu_renderer_create_context,
    gsx_cpu_renderer_render,
    gsx_cpu_renderer_backward
};

static const gsx_render_context_i gsx_cpu_render_context_iface = {
    gsx_cpu_render_context_destroy
};

gsx_error gsx_cpu_backend_create_renderer(gsx_backend_t backend, const gsx_renderer_desc *desc, gsx_renderer_t *out_renderer)
{
    gsx_cpu_renderer *cpu_renderer = NULL;
    gsx_renderer_capabilities capabilities = { 0 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_renderer == NULL || desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_renderer and desc must be non-null");
    }
    *out_renderer = NULL;
    if(desc->output_data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu renderer only supports float32 output_data_type");
    }
    if((desc->feature_flags & GSX_RENDERER_FEATURE_ANTIALIASING) != 0) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu renderer antialiasing feature flag is not implemented");
    }
    if(desc->enable_alpha_output || desc->enable_invdepth_output) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cpu renderer only supports RGB outputs");
    }

    cpu_renderer = (gsx_cpu_renderer *)calloc(1, sizeof(*cpu_renderer));
    if(cpu_renderer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate cpu renderer");
    }

    capabilities.supported_precisions = GSX_RENDER_PRECISION_FLAG_FLOAT32;
    capabilities.supports_alpha_output = false;
    capabilities.supports_invdepth_output = false;
    capabilities.supports_cov3d_input = false;
    error = gsx_renderer_base_init(&cpu_renderer->base, &gsx_cpu_renderer_iface, backend, desc, &capabilities);
    if(!gsx_error_is_success(error)) {
        free(cpu_renderer);
        return error;
    }

    *out_renderer = &cpu_renderer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
