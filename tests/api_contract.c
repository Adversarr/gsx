#include "gsx/gsx.h"
#include "gsx/extra/gsx-stbi.h"

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
        out_sample->rgb_data = NULL;
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
GSX_STATIC_ASSERT(GSX_TENSOR_MAX_DIM == 4, "Tensor rank limit must stay at 4.");
GSX_STATIC_ASSERT(sizeof(gsx_backend_device_t) == sizeof(void *), "Backend-device handles must be pointer-sized.");
GSX_STATIC_ASSERT(sizeof(gsx_tensor_t) == sizeof(void *), "Opaque handles must be pointer-sized.");
GSX_STATIC_ASSERT(sizeof(gsx_backend_buffer_type_t) == sizeof(void *), "Buffer type handles must be pointer-sized.");
GSX_STATIC_ASSERT(sizeof(gsx_backend_buffer_t) == sizeof(void *), "Backend-buffer handles must be pointer-sized.");
GSX_STATIC_ASSERT(sizeof(gsx_arena_mark) == sizeof(gsx_size_t) + sizeof(gsx_id_t), "Arena marks must stay compact POD values.");
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
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataset_desc *)0)->image_data_type, gsx_data_type), "Dataset descriptors must keep a fixed source image dtype.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataset_desc *)0)->width, gsx_index_t), "Dataset descriptors must keep a fixed width field.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataset_cpu_sample *)0)->rgb_data, const void *), "Dataset samples must expose borrowed RGB payload pointers.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataset_cpu_sample *)0)->release_token, void *), "Dataset samples must carry an opaque release token.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataloader_result *)0)->rgb_image, gsx_tensor_t), "Dataloader results must expose the RGB tensor directly.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataloader_result *)0)->alpha_image, gsx_tensor_t), "Dataloader results must expose the alpha tensor directly.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_dataloader_result *)0)->invdepth_image, gsx_tensor_t), "Dataloader results must expose the inverse-depth tensor directly.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_registry_init, gsx_error (*)(void)), "Backend-registry init signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_count_backend_devices, gsx_error (*)(gsx_index_t *)), "Backend-device count signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_get_backend_device, gsx_error (*)(gsx_index_t, gsx_backend_device_t *)), "Backend-device lookup signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_count_backend_devices_by_type, gsx_error (*)(gsx_backend_type, gsx_index_t *)), "Backend-device filtered count signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_get_backend_device_by_type, gsx_error (*)(gsx_backend_type, gsx_index_t, gsx_backend_device_t *)), "Backend-device filtered lookup signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_device_get_info, gsx_error (*)(gsx_backend_device_t, gsx_backend_device_info *)), "Backend-device info signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_get_major_stream, gsx_error (*)(gsx_backend_t, void **)), "Major-stream getter signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_type_get_info, gsx_error (*)(gsx_backend_buffer_type_t, gsx_backend_buffer_type_info *)), "Buffer-type info signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_init, gsx_error (*)(gsx_backend_buffer_t *, const gsx_backend_buffer_desc *)), "Backend-buffer init signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_get_info, gsx_error (*)(gsx_backend_buffer_t, gsx_backend_buffer_info *)), "Backend-buffer info signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_get_native_handle, gsx_error (*)(gsx_backend_buffer_t, void **)), "Backend-buffer native-handle signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_upload, gsx_error (*)(gsx_backend_buffer_t, gsx_size_t, const void *, gsx_size_t)), "Backend-buffer upload signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_download, gsx_error (*)(gsx_backend_buffer_t, gsx_size_t, void *, gsx_size_t)), "Backend-buffer download signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_backend_buffer_set_zero, gsx_error (*)(gsx_backend_buffer_t)), "Backend-buffer zero signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_arena_reserve, gsx_error (*)(gsx_arena_t, gsx_size_t)), "Arena reserve signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_arena_reset, gsx_error (*)(gsx_arena_t)), "Arena reset signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_arena_get_mark, gsx_error (*)(gsx_arena_t, gsx_arena_mark *)), "Arena mark signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_arena_rewind, gsx_error (*)(gsx_arena_t, gsx_arena_mark)), "Arena rewind signature must remain stable.");
GSX_STATIC_ASSERT(
    GSX_TYPE_MATCHES(
        &gsx_arena_plan_required_bytes,
        gsx_error (*)(gsx_backend_buffer_type_t, const gsx_arena_desc *, gsx_arena_plan_callback, void *, gsx_size_t *)),
    "Arena required-bytes planner signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_tensor_get_native_handle, gsx_error (*)(gsx_tensor_t, void **, gsx_size_t *)), "Tensor native-handle signature must remain stable.");
GSX_STATIC_ASSERT(
    GSX_TYPE_MATCHES(
        &gsx_tensor_init_many,
        gsx_error (*)(gsx_tensor_t *, gsx_arena_t, const gsx_tensor_desc *, gsx_index_t)),
    "Tensor batch init signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_tensor_free_many, gsx_error (*)(gsx_tensor_t *, gsx_index_t)), "Tensor batch free signature must remain stable.");
GSX_STATIC_ASSERT(
    GSX_TYPE_MATCHES(
        &gsx_tensor_plan_required_bytes,
        gsx_error (*)(gsx_backend_buffer_type_t, const gsx_arena_desc *, const gsx_tensor_desc *, gsx_index_t, gsx_size_t *)),
    "Tensor required-bytes planner signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_tensor_copy, gsx_error (*)(gsx_tensor_t, gsx_tensor_t)), "Tensor copy signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_tensor_fill, gsx_error (*)(gsx_tensor_t, const void *, gsx_size_t)), "Tensor fill signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_gs_get_field, gsx_error (*)(gsx_gs_t, gsx_gs_field, gsx_tensor_t *)), "GS field getter signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_gs_set_field, gsx_error (*)(gsx_gs_t, gsx_gs_field, gsx_tensor_t)), "GS field setter signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_gs_gather, gsx_error (*)(gsx_gs_t, gsx_tensor_t)), "GS gather signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_gs_resize, gsx_error (*)(gsx_gs_t, gsx_size_t)), "GS resize signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_dataset_init, gsx_error (*)(gsx_dataset_t *, const gsx_dataset_desc *)), "Dataset init signature must match the callback-backed dataset contract.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&test_dataset_get_length, gsx_dataset_get_length_fn), "Dataset length callback signature must stay stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&test_dataset_get_sample, gsx_dataset_get_sample_fn), "Dataset sample callback signature must stay stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&test_dataset_release_sample, gsx_dataset_release_sample_fn), "Dataset release callback signature must stay stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_render_context_init, gsx_error (*)(gsx_render_context_t *, gsx_renderer_t)), "Render-context init signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_get_param_group_desc_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_optim_param_group_desc *)), "Role-based optimizer descriptor lookup signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_get_learning_rate_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t *)), "Role-based optimizer LR query signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_set_learning_rate_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t)), "Role-based optimizer LR update signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_reset_param_group_by_role, gsx_error (*)(gsx_optim_t, gsx_optim_param_role)), "Role-based optimizer reset signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_optim_rebind_param_groups_from_gs, gsx_error (*)(gsx_optim_t, gsx_gs_t)), "Optimizer GS rebinding signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_image *)0)->pixels, void *), "Image pixels must remain an owned mutable pointer.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_image *)0)->data_type, gsx_data_type), "Image data type metadata must remain explicit.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(((gsx_image *)0)->storage_format, gsx_storage_format), "Image storage format metadata must remain explicit.");
GSX_STATIC_ASSERT(
    GSX_TYPE_MATCHES(
        &gsx_image_load,
        gsx_error (*)(gsx_image *, const char *, gsx_index_t, gsx_data_type, gsx_storage_format)),
    "Image load signature must remain stable.");
GSX_STATIC_ASSERT(GSX_TYPE_MATCHES(&gsx_image_free, gsx_error (*)(gsx_image *)), "Image free signature must remain stable.");
GSX_STATIC_ASSERT(
    GSX_TYPE_MATCHES(
        &gsx_image_write_png,
        gsx_error (*)(const char *, const void *, gsx_index_t, gsx_index_t, gsx_index_t, gsx_data_type, gsx_storage_format)),
    "PNG write signature must remain stable.");
GSX_STATIC_ASSERT(
    GSX_TYPE_MATCHES(
        &gsx_image_write_jpg,
        gsx_error (*)(const char *, const void *, gsx_index_t, gsx_index_t, gsx_index_t, gsx_data_type, gsx_storage_format, gsx_index_t)),
    "JPG write signature must remain stable.");

int main(void)
{
    gsx_error ok = { GSX_ERROR_SUCCESS, NULL };
    gsx_arena_desc arena_desc = { 0 };
    gsx_arena_mark arena_mark = { 0 };
    gsx_arena_plan_callback arena_plan_callback = NULL;
    gsx_dataset_desc dataset_desc = { 0 };
    gsx_dataset_info dataset_info = { 0 };
    gsx_dataset_cpu_sample dataset_sample = { 0 };
    gsx_dataloader_desc dataloader_desc = { 0 };
    gsx_dataloader_result dataloader_result = { 0 };
    gsx_tensor_desc tensor_desc = { 0 };
    gsx_renderer_desc renderer_desc = { 0 };
    gsx_optim_desc optim_desc = { 0 };
    gsx_optim_param_group_desc param_group = { 0 };
    gsx_optim_step_request step_request = { 0 };
    gsx_checkpoint_info checkpoint = { 0 };
    gsx_backend_device_info backend_device_info = { 0 };
    gsx_backend_capabilities backend_capabilities = { 0 };
    gsx_backend_buffer_desc backend_buffer_desc = { 0 };
    gsx_backend_buffer_info backend_buffer_info = { 0 };
    gsx_backend_buffer_type_info buffer_type_info = { 0 };
    gsx_camera_intrinsics intrinsics = { 0 };
    gsx_camera_pose pose = { 0 };
    gsx_backend_buffer_type_class buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_optim_param_role role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    gsx_optim_param_role_flags role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D | GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY;
    gsx_image image = { 0 };
    gsx_index_t custom_group_indices[] = { 0 };
    void *major_stream = NULL;

    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    pose.rot.w = 1.0f;
    pose.transl.z = 1.0f;
    pose.camera_id = 1;
    arena_desc.requested_alignment_bytes = 64;
    arena_mark.reset_epoch = 1;
    tensor_desc.requested_alignment_bytes = 64;
    buffer_type_info.alignment_bytes = 64;
    backend_buffer_info.alignment_bytes = 64;
    dataset_desc.image_data_type = GSX_DATA_TYPE_U8;
    dataset_desc.width = 1;
    dataset_desc.height = 1;
    dataset_desc.has_rgb = true;
    dataset_desc.has_alpha = false;
    dataset_desc.has_invdepth = false;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataset_sample.rgb_data = NULL;
    dataloader_result.alpha_image = NULL;
    dataloader_result.invdepth_image = NULL;
    image.data_type = GSX_DATA_TYPE_U8;
    image.storage_format = GSX_STORAGE_FORMAT_HWC;
    param_group.role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    param_group.label = "mean3d";
    optim_desc.state_buffer_type = NULL;
    step_request.role_flags = role_flags;
    step_request.param_group_indices = custom_group_indices;
    step_request.param_group_index_count = 1;
    (void)buffer_type_class;
    (void)arena_desc;
    (void)arena_mark;
    (void)arena_plan_callback;
    (void)dataset_desc;
    (void)dataset_info;
    (void)dataset_sample;
    (void)dataloader_desc;
    (void)dataloader_result;
    (void)tensor_desc;
    (void)renderer_desc;
    (void)optim_desc;
    (void)param_group;
    (void)checkpoint;
    (void)backend_device_info;
    (void)backend_capabilities;
    (void)backend_buffer_desc;
    (void)backend_buffer_info;
    (void)buffer_type_info;
    (void)role;
    (void)image;
    (void)major_stream;

    return gsx_error_is_success(ok) ? 0 : 1;
}
