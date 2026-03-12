#include "gsx/gsx.h"

#include <stdalign.h>
#include <stddef.h>

#define GSX_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#define GSX_TYPE_MATCHES(expr, type) _Generic((expr), type: 1, default: 0)

static gsx_error test_dataset_get_length(void *object, gsx_size_t *out_length)
{
    (void)object;
    if(out_length != NULL) {
        *out_length = 1;
    }
    return (gsx_error){ GSX_ERROR_SUCCESS, NULL };
}

static gsx_error test_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    (void)object;
    (void)sample_index;
    if(out_sample != NULL) {
        out_sample->rgb.channel_count = 3;
    }
    return (gsx_error){ GSX_ERROR_SUCCESS, NULL };
}

static void test_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    (void)object;
    (void)sample;
}

GSX_STATIC_ASSERT(GSX_VERSION_MAJOR == 0, "GSX must report major version 0.");
GSX_STATIC_ASSERT(sizeof(((gsx_tensor_desc *)0)->shape) / sizeof(gsx_index_t) == GSX_TENSOR_MAX_DIM, "Tensor shape array must match GSX_TENSOR_MAX_DIM.");
GSX_STATIC_ASSERT(sizeof(gsx_tensor_t) == sizeof(void *), "Opaque handles must be pointer-sized.");
GSX_STATIC_ASSERT(sizeof(gsx_backend_buffer_type_t) == sizeof(void *), "Buffer type handles must be pointer-sized.");
GSX_STATIC_ASSERT(sizeof(gsx_optim_param_role_flags) == sizeof(gsx_flags32_t), "Optimizer role flags must stay a 32-bit public bitmask.");
GSX_STATIC_ASSERT(alignof(gsx_vec4) == 16, "gsx_vec4 must keep 16-byte alignment.");
GSX_STATIC_ASSERT(sizeof(gsx_vec4) == 16, "gsx_vec4 must stay 16 bytes.");
GSX_STATIC_ASSERT(sizeof(gsx_camera_intrinsics) >= sizeof(gsx_float_t) * 9, "Camera intrinsics must remain a plain value type.");
GSX_STATIC_ASSERT(sizeof(gsx_camera_pose) >= sizeof(gsx_quat) + sizeof(gsx_vec3), "Camera pose must remain a plain value type.");
GSX_STATIC_ASSERT(offsetof(gsx_camera_pose, rot) == 0, "Camera pose rotation field must stay first.");
GSX_STATIC_ASSERT(offsetof(gsx_camera_pose, transl) >= sizeof(gsx_quat), "Camera pose translation must follow rotation.");
GSX_STATIC_ASSERT(GSX_OPTIM_PARAM_ROLE_MEAN3D == 0, "Built-in optimizer roles must keep stable ordering.");
GSX_STATIC_ASSERT(GSX_OPTIM_PARAM_ROLE_SH3 == 7, "Built-in optimizer roles must keep stable ordering.");
GSX_STATIC_ASSERT(GSX_OPTIM_PARAM_ROLE_CUSTOM > GSX_OPTIM_PARAM_ROLE_SH3, "Custom optimizer role must stay outside the built-in range.");
GSX_STATIC_ASSERT(sizeof(gsx_dataset_info) == sizeof(gsx_size_t), "Dataset info should expose a cached sample length only.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_cpu_image_view *)0)->data, const void *), "CPU image view data pointer must stay a borrowed const pointer.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataset_cpu_sample *)0)->release_token, void *), "Dataset samples must carry an opaque release token.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataloader_result *)0)->rgb_image, gsx_tensor_t), "Dataloader results must expose the RGB tensor directly.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataloader_result *)0)->alpha_image, gsx_tensor_t), "Dataloader results must expose the alpha tensor directly.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataloader_result *)0)->invdepth_image, gsx_tensor_t), "Dataloader results must expose the inverse-depth tensor directly.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_get_major_stream, gsx_error (*)(gsx_backend_t, void **)), "Major-stream getter signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_dataset_init, gsx_error (*)(gsx_dataset_t *, const gsx_dataset_desc *)), "Dataset init signature must match the callback-backed dataset contract.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&test_dataset_get_length, gsx_dataset_get_length_fn), "Dataset length callback signature must stay stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&test_dataset_get_sample, gsx_dataset_get_sample_fn), "Dataset sample callback signature must stay stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&test_dataset_release_sample, gsx_dataset_release_sample_fn), "Dataset release callback signature must stay stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_render_context_init, gsx_error (*)(gsx_render_context_t *, gsx_renderer_t)), "Render-context init signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_get_param_group_desc_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_optim_param_group_desc *)), "Role-based optimizer descriptor lookup signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_get_learning_rate_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t *)), "Role-based optimizer LR query signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_set_learning_rate_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t)), "Role-based optimizer LR update signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_reset_param_group_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role)), "Role-based optimizer reset signature must remain stable.");

int main(void)
{
    gsx_error ok = { GSX_ERROR_SUCCESS, NULL };
    gsx_arena_desc arena_desc = { 0 };
    gsx_dataset_desc dataset_desc = { 0 };
    gsx_dataset_info dataset_info = { 0 };
    gsx_cpu_image_view image_view = { 0 };
    gsx_dataset_cpu_sample dataset_sample = { 0 };
    gsx_dataloader_desc dataloader_desc = { 0 };
    gsx_dataloader_result dataloader_result = { 0 };
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_renderer_desc renderer_desc = { 0 };
    gsx_dataloader_state dataloader_state = { 0 };
    gsx_optim_param_group_desc param_group = { 0 };
    gsx_optim_step_request step_request = { 0 };
    gsx_checkpoint_info checkpoint = { 0 };
    gsx_backend_capabilities backend_capabilities = { 0 };
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_camera_intrinsics intrinsics = { 0 };
    gsx_camera_pose pose = { 0 };
    gsx_backend_buffer_type_class buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_optim_param_role role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    gsx_optim_param_role_flags role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D | GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY;
    gsx_index_t custom_group_indices[] = { 0 };
    void *major_stream = NULL;

    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    pose.rot.w = 1.0f;
    pose.transl.z = 1.0f;
    pose.camera_id = 1;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataset_sample.rgb.channel_count = 3;
    dataloader_result.alpha_image = NULL;
    dataloader_result.invdepth_image = NULL;
    param_group.role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    param_group.label = "mean3d";
    step_request.role_flags = role_flags;
    step_request.param_group_indices = custom_group_indices;
    step_request.param_group_index_count = 1;
    (void)buffer_type_class;
    (void)arena_desc;
    (void)dataset_desc;
    (void)dataset_info;
    (void)image_view;
    (void)dataset_sample;
    (void)dataloader_desc;
    (void)dataloader_result;
    (void)tensor_desc;
    (void)renderer_desc;
    (void)dataloader_state;
    (void)param_group;
    (void)checkpoint;
    (void)backend_capabilities;
    (void)buffer_type_info;
    (void)role;
    (void)major_stream;

    return gsx_error_is_success(ok) ? 0 : 1;
}
