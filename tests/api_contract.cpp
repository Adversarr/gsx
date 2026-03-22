#include "gsx/gsx.h"
#include "gsx/extra/gsx-stbi.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

namespace {

static gsx_error test_dataset_get_length(void *object, gsx_size_t *out_length)
{
    (void)object;
    if(out_length != nullptr) {
        *out_length = 1;
    }
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error test_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    (void)object;
    (void)sample_index;
    if(out_sample != nullptr) {
        out_sample->rgb.channel_count = 3;
    }
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static void test_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    (void)object;
    (void)sample;
}

using gsx_dataloader_rgb_image_field_t = decltype(((gsx_dataloader_result *)nullptr)->rgb_image);
using gsx_dataloader_alpha_image_field_t = decltype(((gsx_dataloader_result *)nullptr)->alpha_image);
using gsx_dataloader_invdepth_image_field_t = decltype(((gsx_dataloader_result *)nullptr)->invdepth_image);

static_assert(GSX_VERSION_MAJOR == 0, "GSX must report major version 0.");
static_assert(std::is_pointer<gsx_backend_device_t>::value, "Backend-device handles must stay pointer types.");
static_assert(std::is_pointer<gsx_tensor_t>::value, "Opaque handles must stay pointer types.");
static_assert(std::is_pointer<gsx_backend_buffer_type_t>::value, "Buffer type handles must stay pointer types.");
static_assert(std::is_pointer<gsx_backend_buffer_t>::value, "Backend-buffer handles must stay pointer types.");
static_assert(std::is_pointer<gsx_session_t>::value, "Runtime handles must stay pointer types.");
static_assert(GSX_TENSOR_MAX_DIM == 4, "Tensor rank limit must stay at 4.");
static_assert(sizeof(gsx_arena_mark) == sizeof(gsx_size_t) + sizeof(gsx_id_t), "Arena marks must stay compact POD values.");
static_assert(std::is_trivially_copyable<gsx_camera_intrinsics>::value, "Camera intrinsics must stay trivially copyable.");
static_assert(std::is_trivially_copyable<gsx_camera_pose>::value, "Camera pose must stay trivially copyable.");
static_assert(std::is_trivially_copyable<gsx_cpu_image_view>::value, "CPU image views must stay trivially copyable.");
static_assert(std::is_trivially_copyable<gsx_dataset_cpu_sample>::value, "Dataset CPU samples must stay trivially copyable.");
static_assert(std::is_standard_layout<gsx_camera_intrinsics>::value, "Camera intrinsics must stay standard-layout.");
static_assert(std::is_standard_layout<gsx_camera_pose>::value, "Camera pose must stay standard-layout.");
static_assert(alignof(gsx_vec4) == 16, "gsx_vec4 must keep 16-byte alignment.");
static_assert(sizeof(gsx_vec4) == 16, "gsx_vec4 must stay 16 bytes.");
static_assert(offsetof(gsx_camera_pose, rot) == 0, "Camera pose rotation field must stay first.");
static_assert(sizeof(gsx_optim_param_role_flags) == sizeof(gsx_flags32_t), "Optimizer role flags must stay a 32-bit public bitmask.");
static_assert(GSX_OPTIM_PARAM_ROLE_MEAN3D == 0, "Built-in optimizer roles must keep stable ordering.");
static_assert(GSX_OPTIM_PARAM_ROLE_SH3 == 7, "Built-in optimizer roles must keep stable ordering.");
static_assert(GSX_OPTIM_PARAM_ROLE_CUSTOM > GSX_OPTIM_PARAM_ROLE_SH3, "Custom optimizer role must stay outside the built-in range.");
static_assert(std::is_same<decltype(&gsx_backend_registry_init), gsx_error (*)()>::value, "Backend-registry init signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_count_backend_devices), gsx_error (*)(gsx_index_t *)>::value, "Backend-device count signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_get_backend_device), gsx_error (*)(gsx_index_t, gsx_backend_device_t *)>::value, "Backend-device lookup signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_count_backend_devices_by_type), gsx_error (*)(gsx_backend_type, gsx_index_t *)>::value, "Backend-device filtered count signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_get_backend_device_by_type), gsx_error (*)(gsx_backend_type, gsx_index_t, gsx_backend_device_t *)>::value, "Backend-device filtered lookup signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_device_get_info), gsx_error (*)(gsx_backend_device_t, gsx_backend_device_info *)>::value, "Backend-device info signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_get_major_stream), gsx_error (*)(gsx_backend_t, void **)>::value, "Major-stream getter signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_type_get_info), gsx_error (*)(gsx_backend_buffer_type_t, gsx_backend_buffer_type_info *)>::value, "Buffer-type info signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_init), gsx_error (*)(gsx_backend_buffer_t *, const gsx_backend_buffer_desc *)>::value, "Backend-buffer init signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_get_info), gsx_error (*)(gsx_backend_buffer_t, gsx_backend_buffer_info *)>::value, "Backend-buffer info signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_get_native_handle), gsx_error (*)(gsx_backend_buffer_t, void **)>::value, "Backend-buffer native-handle signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_upload), gsx_error (*)(gsx_backend_buffer_t, gsx_size_t, const void *, gsx_size_t)>::value, "Backend-buffer upload signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_download), gsx_error (*)(gsx_backend_buffer_t, gsx_size_t, void *, gsx_size_t)>::value, "Backend-buffer download signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_backend_buffer_set_zero), gsx_error (*)(gsx_backend_buffer_t)>::value, "Backend-buffer zero signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_arena_reserve), gsx_error (*)(gsx_arena_t, gsx_size_t)>::value, "Arena reserve signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_arena_reset), gsx_error (*)(gsx_arena_t)>::value, "Arena reset signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_arena_get_mark), gsx_error (*)(gsx_arena_t, gsx_arena_mark *)>::value, "Arena mark signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_arena_rewind), gsx_error (*)(gsx_arena_t, gsx_arena_mark)>::value, "Arena rewind signature must remain stable.");
static_assert(
    std::is_same<
        decltype(&gsx_arena_plan_required_bytes),
        gsx_error (*)(gsx_backend_buffer_type_t, const gsx_arena_desc *, gsx_arena_plan_callback, void *, gsx_size_t *)>::value,
    "Arena required-bytes planner signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_tensor_get_native_handle), gsx_error (*)(gsx_tensor_t, void **, gsx_size_t *)>::value, "Tensor native-handle signature must remain stable.");
static_assert(
    std::is_same<
        decltype(&gsx_tensor_init_many),
        gsx_error (*)(gsx_tensor_t *, gsx_arena_t, const gsx_tensor_desc *, gsx_index_t)>::value,
    "Tensor batch init signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_tensor_free_many), gsx_error (*)(gsx_tensor_t *, gsx_index_t)>::value, "Tensor batch free signature must remain stable.");
static_assert(
    std::is_same<
        decltype(&gsx_tensor_plan_required_bytes),
        gsx_error (*)(gsx_backend_buffer_type_t, const gsx_arena_desc *, const gsx_tensor_desc *, gsx_index_t, gsx_size_t *)>::value,
    "Tensor required-bytes planner signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_gs_get_field), gsx_error (*)(gsx_gs_t, gsx_gs_field, gsx_tensor_t *)>::value, "GS field getter signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_gs_set_field), gsx_error (*)(gsx_gs_t, gsx_gs_field, gsx_tensor_t)>::value, "GS field setter signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_gs_gather), gsx_error (*)(gsx_gs_t, gsx_tensor_t)>::value, "GS gather signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_gs_resize), gsx_error (*)(gsx_gs_t, gsx_size_t)>::value, "GS resize signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_dataset_init), gsx_error (*)(gsx_dataset_t *, const gsx_dataset_desc *)>::value, "Dataset init signature must match the callback-backed dataset contract.");
static_assert(std::is_same<decltype(&test_dataset_get_length), gsx_dataset_get_length_fn>::value, "Dataset length callback signature must stay stable.");
static_assert(std::is_same<decltype(&test_dataset_get_sample), gsx_dataset_get_sample_fn>::value, "Dataset sample callback signature must stay stable.");
static_assert(std::is_same<decltype(&test_dataset_release_sample), gsx_dataset_release_sample_fn>::value, "Dataset release callback signature must stay stable.");
static_assert(std::is_same<decltype(&gsx_render_context_init), gsx_error (*)(gsx_render_context_t *, gsx_renderer_t)>::value, "Render-context init signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_optim_get_param_group_desc_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_optim_param_group_desc *)>::value, "Role-based optimizer descriptor lookup signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_optim_get_learning_rate_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t *)>::value, "Role-based optimizer LR query signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_optim_set_learning_rate_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t)>::value, "Role-based optimizer LR update signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_optim_reset_param_group_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role)>::value, "Role-based optimizer reset signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_session_get_desc), gsx_error (*)(gsx_session_t, gsx_session_desc *)>::value, "Session descriptor getter signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_session_reset), gsx_error (*)(gsx_session_t)>::value, "Session reset signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_session_get_last_step_report), gsx_error (*)(gsx_session_t, gsx_session_step_report *)>::value, "Session report getter signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_session_get_last_outputs), gsx_error (*)(gsx_session_t, gsx_session_outputs *)>::value, "Session output getter signature must remain stable.");
static_assert(std::is_same<decltype(((gsx_image *)nullptr)->pixels), void *>::value, "Image pixels must remain an owned mutable pointer.");
static_assert(std::is_same<decltype(((gsx_image *)nullptr)->data_type), gsx_data_type>::value, "Image data type metadata must remain explicit.");
static_assert(std::is_same<decltype(((gsx_image *)nullptr)->storage_format), gsx_storage_format>::value, "Image storage format metadata must remain explicit.");
static_assert(std::is_same<decltype(&gsx_image_load), gsx_error (*)(gsx_image *, const char *, gsx_index_t, gsx_data_type, gsx_storage_format)>::value, "Image load signature must remain stable.");
static_assert(std::is_same<decltype(&gsx_image_free), gsx_error (*)(gsx_image *)>::value, "Image free signature must remain stable.");
static_assert(
    std::is_same<
        decltype(&gsx_image_write_png),
        gsx_error (*)(const char *, const void *, gsx_index_t, gsx_index_t, gsx_index_t, gsx_data_type, gsx_storage_format)>::value,
    "PNG write signature must remain stable.");
static_assert(
    std::is_same<
        decltype(&gsx_image_write_jpg),
        gsx_error (*)(const char *, const void *, gsx_index_t, gsx_index_t, gsx_index_t, gsx_data_type, gsx_storage_format, gsx_index_t)>::value,
    "JPG write signature must remain stable.");
static_assert(std::is_same<gsx_dataloader_rgb_image_field_t, gsx_tensor_t>::value, "Dataloader results must expose the RGB tensor directly.");
static_assert(std::is_same<gsx_dataloader_alpha_image_field_t, gsx_tensor_t>::value, "Dataloader results must expose the alpha tensor directly.");
static_assert(std::is_same<gsx_dataloader_invdepth_image_field_t, gsx_tensor_t>::value, "Dataloader results must expose the inverse-depth tensor directly.");

TEST(VersionAndHandleContract, VersionMarkersRemainStable)
{
    EXPECT_EQ(GSX_VERSION_MAJOR, 0);
    EXPECT_EQ(GSX_VERSION_MINOR, 0);
    EXPECT_EQ(GSX_VERSION_PATCH, 1);
    EXPECT_EQ(GSX_VERSION, GSX_MAKE_VERSION(GSX_VERSION_MAJOR, GSX_VERSION_MINOR, GSX_VERSION_PATCH));
    EXPECT_TRUE((std::is_pointer<gsx_backend_device_t>::value));
    EXPECT_TRUE((std::is_pointer<gsx_tensor_t>::value));
    EXPECT_TRUE((std::is_pointer<gsx_backend_buffer_type_t>::value));
    EXPECT_TRUE((std::is_pointer<gsx_backend_buffer_t>::value));
    EXPECT_TRUE((std::is_pointer<gsx_session_t>::value));
    EXPECT_EQ(GSX_TENSOR_MAX_DIM, 4);
    EXPECT_EQ(sizeof(gsx_arena_mark), sizeof(gsx_size_t) + sizeof(gsx_id_t));
}

TEST(ValueTypeContract, PublicStructsRemainPlainCopyableValues)
{
    EXPECT_TRUE((std::is_trivially_copyable<gsx_camera_intrinsics>::value));
    EXPECT_TRUE((std::is_trivially_copyable<gsx_camera_pose>::value));
    EXPECT_TRUE((std::is_trivially_copyable<gsx_cpu_image_view>::value));
    EXPECT_TRUE((std::is_trivially_copyable<gsx_dataset_cpu_sample>::value));
    EXPECT_TRUE((std::is_standard_layout<gsx_camera_intrinsics>::value));
    EXPECT_TRUE((std::is_standard_layout<gsx_camera_pose>::value));
}

TEST(LayoutContract, AlignmentAndFieldOrderRemainStable)
{
    EXPECT_EQ(alignof(gsx_vec4), 16U);
    EXPECT_EQ(sizeof(gsx_vec4), 16U);
    EXPECT_EQ(offsetof(gsx_camera_pose, rot), 0U);
    EXPECT_GE(offsetof(gsx_camera_pose, transl), sizeof(gsx_quat));
    EXPECT_EQ(sizeof(gsx_optim_param_role_flags), sizeof(gsx_flags32_t));
    EXPECT_EQ(GSX_OPTIM_PARAM_ROLE_MEAN3D, 0);
    EXPECT_EQ(GSX_OPTIM_PARAM_ROLE_SH3, 7);
    EXPECT_GT(GSX_OPTIM_PARAM_ROLE_CUSTOM, GSX_OPTIM_PARAM_ROLE_SH3);
}

TEST(SignatureContract, CallbackAndPublicFunctionSignaturesRemainStable)
{
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_registry_init), gsx_error (*)()>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_count_backend_devices), gsx_error (*)(gsx_index_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_get_backend_device), gsx_error (*)(gsx_index_t, gsx_backend_device_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_count_backend_devices_by_type), gsx_error (*)(gsx_backend_type, gsx_index_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_get_backend_device_by_type), gsx_error (*)(gsx_backend_type, gsx_index_t, gsx_backend_device_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_device_get_info), gsx_error (*)(gsx_backend_device_t, gsx_backend_device_info *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_get_major_stream), gsx_error (*)(gsx_backend_t, void **)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_type_get_info), gsx_error (*)(gsx_backend_buffer_type_t, gsx_backend_buffer_type_info *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_init), gsx_error (*)(gsx_backend_buffer_t *, const gsx_backend_buffer_desc *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_get_info), gsx_error (*)(gsx_backend_buffer_t, gsx_backend_buffer_info *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_get_native_handle), gsx_error (*)(gsx_backend_buffer_t, void **)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_upload), gsx_error (*)(gsx_backend_buffer_t, gsx_size_t, const void *, gsx_size_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_download), gsx_error (*)(gsx_backend_buffer_t, gsx_size_t, void *, gsx_size_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_backend_buffer_set_zero), gsx_error (*)(gsx_backend_buffer_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_arena_reserve), gsx_error (*)(gsx_arena_t, gsx_size_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_arena_reset), gsx_error (*)(gsx_arena_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_arena_get_mark), gsx_error (*)(gsx_arena_t, gsx_arena_mark *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_arena_rewind), gsx_error (*)(gsx_arena_t, gsx_arena_mark)>::value));
    EXPECT_TRUE((
        std::is_same<
            decltype(&gsx_arena_plan_required_bytes),
            gsx_error (*)(gsx_backend_buffer_type_t, const gsx_arena_desc *, gsx_arena_plan_callback, void *, gsx_size_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_tensor_get_native_handle), gsx_error (*)(gsx_tensor_t, void **, gsx_size_t *)>::value));
    EXPECT_TRUE((
        std::is_same<
            decltype(&gsx_tensor_init_many),
            gsx_error (*)(gsx_tensor_t *, gsx_arena_t, const gsx_tensor_desc *, gsx_index_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_tensor_free_many), gsx_error (*)(gsx_tensor_t *, gsx_index_t)>::value));
    EXPECT_TRUE((
        std::is_same<
            decltype(&gsx_tensor_plan_required_bytes),
            gsx_error (*)(gsx_backend_buffer_type_t, const gsx_arena_desc *, const gsx_tensor_desc *, gsx_index_t, gsx_size_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_gs_get_field), gsx_error (*)(gsx_gs_t, gsx_gs_field, gsx_tensor_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_gs_set_field), gsx_error (*)(gsx_gs_t, gsx_gs_field, gsx_tensor_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_gs_gather), gsx_error (*)(gsx_gs_t, gsx_tensor_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_gs_resize), gsx_error (*)(gsx_gs_t, gsx_size_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_dataset_init), gsx_error (*)(gsx_dataset_t *, const gsx_dataset_desc *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&test_dataset_get_length), gsx_dataset_get_length_fn>::value));
    EXPECT_TRUE((std::is_same<decltype(&test_dataset_get_sample), gsx_dataset_get_sample_fn>::value));
    EXPECT_TRUE((std::is_same<decltype(&test_dataset_release_sample), gsx_dataset_release_sample_fn>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_render_context_init), gsx_error (*)(gsx_render_context_t *, gsx_renderer_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_optim_get_param_group_desc_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_optim_param_group_desc *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_optim_get_learning_rate_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_optim_set_learning_rate_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role, gsx_float_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_optim_reset_param_group_by_role), gsx_error (*)(gsx_optim_t, gsx_optim_param_role)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_session_get_desc), gsx_error (*)(gsx_session_t, gsx_session_desc *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_session_reset), gsx_error (*)(gsx_session_t)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_session_get_last_step_report), gsx_error (*)(gsx_session_t, gsx_session_step_report *)>::value));
    EXPECT_TRUE((std::is_same<decltype(&gsx_session_get_last_outputs), gsx_error (*)(gsx_session_t, gsx_session_outputs *)>::value));
}

TEST(DescriptorAndResultContract, RepresentativePublicTypesRemainUsable)
{
    gsx_dataset_desc dataset_desc{};
    gsx_dataset_info dataset_info{};
    gsx_cpu_image_view image_view{};
    gsx_dataset_cpu_sample dataset_sample{};
    gsx_dataloader_desc dataloader_desc{};
    gsx_dataloader_result dataloader_result{};
    gsx_loss_desc loss_desc{};
    gsx_metric_desc metric_desc{};
    gsx_render_forward_request forward_request{};
    gsx_render_backward_request backward_request{};
    gsx_scheduler_desc scheduler_desc{};
    gsx_session_desc session_desc{};
    gsx_session_render_desc session_render_desc{};
    gsx_session_optim_step_desc session_optim_step_desc{};
    gsx_session_adc_step_desc session_adc_step_desc{};
    gsx_session_workspace_desc session_workspace_desc{};
    gsx_session_report_desc session_report_desc{};
    gsx_session_outputs session_outputs{};
    gsx_session_step_report session_step_report{};
    gsx_session_step_timing session_step_timing{};
    gsx_arena_mark arena_mark{};
    gsx_arena_plan_callback arena_plan_callback = nullptr;
    gsx_optim_desc optim_desc{};
    gsx_optim_param_group_desc param_group{};
    gsx_optim_step_request step_request{};
    gsx_backend_device_info backend_device_info{};
    gsx_backend_capabilities backend_capabilities{};
    gsx_backend_buffer_desc backend_buffer_desc{};
    gsx_backend_buffer_info backend_buffer_info{};
    gsx_backend_buffer_type_info buffer_type_info{};
    gsx_image image{};
    gsx_camera_pose pose{};
    gsx_optim_param_role role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    gsx_optim_param_role_flags role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D | GSX_OPTIM_PARAM_ROLE_FLAG_OPACITY;
    gsx_index_t custom_group_indices[] = { 0 };
    void *major_stream = nullptr;

    pose.rot.w = 1.0f;
    pose.transl.x = 0.0f;
    arena_mark.reset_epoch = 1;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataset_sample.rgb.channel_count = 3;
    dataloader_result.alpha_image = nullptr;
    dataloader_result.invdepth_image = nullptr;
    loss_desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
    optim_desc.state_buffer_type = nullptr;
    param_group.role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    param_group.label = "mean3d";
    step_request.role_flags = role_flags;
    step_request.param_group_indices = custom_group_indices;
    step_request.param_group_index_count = 1;
    image.data_type = GSX_DATA_TYPE_U8;
    image.storage_format = GSX_STORAGE_FORMAT_HWC;

    EXPECT_EQ(dataset_desc.get_length, &test_dataset_get_length);
    EXPECT_EQ(dataset_desc.get_sample, &test_dataset_get_sample);
    EXPECT_EQ(dataset_desc.release_sample, &test_dataset_release_sample);
    EXPECT_EQ(dataloader_desc.image_data_type, GSX_DATA_TYPE_F32);
    EXPECT_EQ(dataloader_desc.storage_format, GSX_STORAGE_FORMAT_CHW);
    EXPECT_EQ(dataset_sample.rgb.channel_count, 3);
    EXPECT_EQ(dataloader_result.alpha_image, nullptr);
    EXPECT_EQ(dataloader_result.invdepth_image, nullptr);
    EXPECT_EQ(loss_desc.grad_normalization, GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN);
    EXPECT_EQ(param_group.role, GSX_OPTIM_PARAM_ROLE_MEAN3D);
    EXPECT_STREQ(param_group.label, "mean3d");
    EXPECT_EQ(step_request.role_flags, role_flags);
    EXPECT_EQ(step_request.param_group_indices, custom_group_indices);
    EXPECT_EQ(step_request.param_group_index_count, 1U);
    EXPECT_TRUE((std::is_same<gsx_dataloader_rgb_image_field_t, gsx_tensor_t>::value));
    EXPECT_TRUE((std::is_same<gsx_dataloader_alpha_image_field_t, gsx_tensor_t>::value));
    EXPECT_TRUE((std::is_same<gsx_dataloader_invdepth_image_field_t, gsx_tensor_t>::value));

    (void)dataset_info;
    (void)image_view;
    (void)loss_desc;
    (void)metric_desc;
    (void)forward_request;
    (void)backward_request;
    (void)scheduler_desc;
    (void)session_desc;
    (void)session_render_desc;
    (void)session_optim_step_desc;
    (void)session_adc_step_desc;
    (void)session_workspace_desc;
    (void)session_report_desc;
    (void)session_outputs;
    (void)session_step_report;
    (void)session_step_timing;
    (void)arena_mark;
    (void)arena_plan_callback;
    (void)optim_desc;
    (void)backend_device_info;
    (void)backend_capabilities;
    (void)backend_buffer_desc;
    (void)backend_buffer_info;
    (void)buffer_type_info;
    (void)image;
    (void)role;
    (void)major_stream;
}

TEST(ErrorHelperContract, SuccessPredicateTracksStatusCode)
{
    const gsx_error success = { GSX_ERROR_SUCCESS, nullptr };
    const gsx_error failure = { GSX_ERROR_INVALID_ARGUMENT, "invalid" };

    EXPECT_TRUE(gsx_error_is_success(success));
    EXPECT_FALSE(gsx_error_is_success(failure));
}

}  // namespace
