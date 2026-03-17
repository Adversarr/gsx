#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

#define ASSERT_GSX_SUCCESS(expr)                                                                                     \
    do {                                                                                                             \
        const gsx_error gsx_assert_success_error__ = (expr);                                                         \
        ASSERT_EQ(gsx_assert_success_error__.code, GSX_ERROR_SUCCESS)                                                \
            << (gsx_assert_success_error__.message != nullptr ? gsx_assert_success_error__.message : "");           \
    } while(false)

#define EXPECT_GSX_CODE(expr, expected_code)                                                                         \
    do {                                                                                                             \
        const gsx_error gsx_expect_code_error__ = (expr);                                                            \
        EXPECT_EQ(gsx_expect_code_error__.code, (expected_code))                                                     \
            << (gsx_expect_code_error__.message != nullptr ? gsx_expect_code_error__.message : "");                \
    } while(false)

static gsx_backend_device_t get_cpu_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device), GSX_ERROR_SUCCESS);
    return backend_device;
}

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};

    backend_desc.device = get_cpu_backend_device();
    EXPECT_NE(backend_desc.device, nullptr);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, type, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static gsx_tensor_desc make_f32_tensor_desc(gsx_arena_t arena, gsx_index_t length, gsx_size_t requested_alignment_bytes = 0)
{
    gsx_tensor_desc desc{};

    desc.rank = 1;
    desc.shape[0] = length;
    desc.requested_alignment_bytes = requested_alignment_bytes;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    return desc;
}

static gsx_tensor_desc make_f32_tensor_desc_with_shape(
    gsx_arena_t arena,
    const std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> &shape,
    gsx_index_t rank,
    gsx_size_t requested_alignment_bytes = 0,
    gsx_storage_format storage_format = GSX_STORAGE_FORMAT_CHW
)
{
    gsx_tensor_desc desc{};

    desc.rank = rank;
    for(std::size_t i = 0; i < shape.size(); ++i) {
        desc.shape[i] = shape[i];
    }
    desc.requested_alignment_bytes = requested_alignment_bytes;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = storage_format;
    desc.arena = arena;
    return desc;
}

static gsx_tensor_desc make_rank1_tensor_desc(gsx_arena_t arena, gsx_index_t length, gsx_data_type data_type)
{
    gsx_tensor_desc desc{};

    desc.rank = 1;
    desc.shape[0] = length;
    desc.data_type = data_type;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    return desc;
}

TEST(CoreRuntime, ArenaInitReportsRoundedCapacityAndAlignment)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_arena_info info{};

    desc.initial_capacity_bytes = 65;
    desc.requested_alignment_bytes = 256;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;

    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_EQ(info.capacity_bytes, 128U);
    EXPECT_EQ(info.effective_alignment_bytes, 256U);
    EXPECT_EQ(info.used_bytes, 0U);
    EXPECT_EQ(info.peak_bytes, 0U);
    EXPECT_EQ(info.active_tensor_count, 0U);
    EXPECT_EQ(info.growth_mode, GSX_ARENA_GROWTH_MODE_FIXED);
    EXPECT_EQ(info.buffer_type, buffer_type);
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, DryRunMirrorsRealAllocationLayoutAndRejectsDataAccess)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t real_arena = nullptr;
    gsx_arena_t dry_arena = nullptr;
    gsx_arena_desc real_desc{};
    gsx_arena_desc dry_desc{};
    gsx_tensor_t real_tensor_a = nullptr;
    gsx_tensor_t real_tensor_b = nullptr;
    gsx_tensor_t dry_tensor_a = nullptr;
    gsx_tensor_t dry_tensor_b = nullptr;
    gsx_tensor_desc desc_a{};
    gsx_tensor_desc desc_b{};
    gsx_size_t real_required_bytes = 0;
    gsx_size_t dry_required_bytes = 0;
    gsx_tensor_info real_info_b{};
    gsx_tensor_info dry_info_b{};
    std::array<float, 8> values{};

    real_desc.initial_capacity_bytes = 1024;
    real_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    dry_desc = real_desc;
    dry_desc.dry_run = true;

    desc_a = make_f32_tensor_desc(real_arena, 8, 0);
    desc_b = make_f32_tensor_desc(real_arena, 4, 256);

    ASSERT_GSX_SUCCESS(gsx_arena_init(&real_arena, buffer_type, &real_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&dry_arena, buffer_type, &dry_desc));

    desc_a.arena = real_arena;
    desc_b.arena = real_arena;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&real_tensor_a, &desc_a));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&real_tensor_b, &desc_b));

    desc_a.arena = dry_arena;
    desc_b.arena = dry_arena;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&dry_tensor_a, &desc_a));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&dry_tensor_b, &desc_b));

    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(real_arena, &real_required_bytes));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(dry_arena, &dry_required_bytes));
    EXPECT_EQ(real_required_bytes, dry_required_bytes);

    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(real_tensor_b, &real_info_b));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(dry_tensor_b, &dry_info_b));
    EXPECT_EQ(real_info_b.size_bytes, dry_info_b.size_bytes);
    EXPECT_EQ(real_info_b.effective_alignment_bytes, dry_info_b.effective_alignment_bytes);
    EXPECT_GSX_CODE(gsx_tensor_upload(dry_tensor_a, values.data(), sizeof(float)), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_tensor_download(dry_tensor_a, values.data(), sizeof(float)), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_tensor_set_zero(dry_tensor_a), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(real_tensor_a));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(real_tensor_b));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(dry_tensor_a));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(dry_tensor_b));
    ASSERT_GSX_SUCCESS(gsx_arena_free(real_arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(dry_arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, FixedArenaOverflowFailsFastAndReportsRequiredBytes)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    gsx_size_t required_bytes = 0;

    desc.initial_capacity_bytes = 64;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;

    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));
    tensor_desc = make_f32_tensor_desc(arena, 32);
    EXPECT_GSX_CODE(gsx_tensor_init(&tensor, &tensor_desc), GSX_ERROR_OUT_OF_RANGE);
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(arena, &required_bytes));
    EXPECT_EQ(required_bytes, 128U);
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GrowOnDemandArenaGrowsOnlyWithoutLiveTensors)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t tensor_a = nullptr;
    gsx_tensor_desc desc_a{};
    gsx_tensor_t tensor_b = nullptr;
    gsx_tensor_desc desc_b{};
    gsx_arena_info info{};
    gsx_size_t required_bytes = 0;

    desc.initial_capacity_bytes = 64;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;

    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));

    desc_a = make_f32_tensor_desc(arena, 32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor_a, &desc_a));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_GE(info.capacity_bytes, 128U);
    EXPECT_EQ(info.active_tensor_count, 1U);

    desc_b = make_f32_tensor_desc(arena, 16);
    EXPECT_GSX_CODE(gsx_tensor_init(&tensor_b, &desc_b), GSX_ERROR_OUT_OF_RANGE);
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(arena, &required_bytes));
    EXPECT_EQ(required_bytes, 192U);
    EXPECT_GSX_CODE(gsx_arena_reserve(arena, 256), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor_a));
    ASSERT_GSX_SUCCESS(gsx_arena_reserve(arena, 256));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_GE(info.capacity_bytes, 256U);

    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, ResetAndRewindRejectLiveTensorsThatWouldBecomeZombies)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t tensor_a = nullptr;
    gsx_tensor_t tensor_b = nullptr;
    gsx_tensor_desc tensor_desc{};
    gsx_arena_mark mark{};
    gsx_arena_info info{};

    desc.initial_capacity_bytes = 512;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));

    tensor_desc = make_f32_tensor_desc(arena, 8);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor_a, &tensor_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_get_mark(arena, &mark));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor_b, &tensor_desc));
    EXPECT_GSX_CODE(gsx_arena_rewind(arena, mark), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_arena_reset(arena), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor_b));
    ASSERT_GSX_SUCCESS(gsx_arena_rewind(arena, mark));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_EQ(info.active_tensor_count, 1U);

    EXPECT_GSX_CODE(gsx_arena_reset(arena), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor_a));
    ASSERT_GSX_SUCCESS(gsx_arena_reset(arena));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_EQ(info.used_bytes, 0U);
    EXPECT_EQ(info.peak_bytes, 0U);
    EXPECT_EQ(info.active_tensor_count, 0U);
    EXPECT_GSX_CODE(gsx_arena_rewind(arena, mark), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, ArenaAndBackendFreeRequireAllTensorHandlesToBeReleased)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};

    desc.initial_capacity_bytes = 256;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));

    tensor_desc = make_f32_tensor_desc(arena, 4);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));

    EXPECT_GSX_CODE(gsx_arena_free(arena), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_arena_reset(arena), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorGetDescRoundTripsRequestedAlignmentAndInfoReportsEffectiveAlignment)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    gsx_tensor_desc returned_desc{};
    gsx_tensor_info tensor_info{};

    arena_desc.initial_capacity_bytes = 256;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    tensor_desc = make_f32_tensor_desc(arena, 4, 0);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_desc(tensor, &returned_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &tensor_info));

    EXPECT_EQ(returned_desc.requested_alignment_bytes, 0U);
    EXPECT_EQ(returned_desc.data_type, GSX_DATA_TYPE_F32);
    EXPECT_EQ(returned_desc.arena, arena);
    EXPECT_EQ(returned_desc.shape[1], 0);
    EXPECT_EQ(returned_desc.shape[2], 0);
    EXPECT_EQ(returned_desc.shape[3], 0);
    EXPECT_EQ(tensor_info.effective_alignment_bytes, 64U);
    EXPECT_EQ(tensor_info.buffer_type, buffer_type);
    EXPECT_EQ(tensor_info.shape[1], 0);
    EXPECT_EQ(tensor_info.shape[2], 0);
    EXPECT_EQ(tensor_info.shape[3], 0);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, MultiRankTensorCopyWorksAcrossArenasOnSameBackend)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t src_arena = nullptr;
    gsx_arena_t dst_arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t src = nullptr;
    gsx_tensor_t dst = nullptr;
    gsx_tensor_desc src_desc{};
    gsx_tensor_desc dst_desc{};
    gsx_tensor_info src_info{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape = { 2, 3, 4, 0 };
    std::array<float, 24> values{};
    std::array<float, 24> roundtrip{};

    arena_desc.initial_capacity_bytes = 512;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&src_arena, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&dst_arena, buffer_type, &arena_desc));

    src_desc = make_f32_tensor_desc_with_shape(src_arena, shape, 3);
    dst_desc = make_f32_tensor_desc_with_shape(dst_arena, shape, 3);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&src, &src_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&dst, &dst_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(src, &src_info));
    EXPECT_EQ(src_info.rank, 3);
    EXPECT_EQ(src_info.shape[0], 2);
    EXPECT_EQ(src_info.shape[1], 3);
    EXPECT_EQ(src_info.shape[2], 4);
    EXPECT_EQ(src_info.shape[3], 0);
    EXPECT_EQ(src_info.size_bytes, sizeof(values));

    for(std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(index + 1);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(src, values.data(), sizeof(values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_copy(src, dst));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(dst, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, values);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(src));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(dst));
    ASSERT_GSX_SUCCESS(gsx_arena_free(src_arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(dst_arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorFreeDropsLiveStatsButDoesNotReclaimCursor)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t tensor_a = nullptr;
    gsx_tensor_t tensor_b = nullptr;
    gsx_tensor_desc tensor_desc_a{};
    gsx_tensor_desc tensor_desc_b{};
    gsx_arena_info info{};
    gsx_size_t required_bytes = 0;

    desc.initial_capacity_bytes = 64;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));

    tensor_desc_a = make_f32_tensor_desc(arena, 8);
    tensor_desc_b = make_f32_tensor_desc(arena, 10);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor_a, &tensor_desc_a));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor_a));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_EQ(info.used_bytes, 0U);
    EXPECT_EQ(info.active_tensor_count, 0U);

    EXPECT_GSX_CODE(gsx_tensor_init(&tensor_b, &tensor_desc_b), GSX_ERROR_OUT_OF_RANGE);
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(arena, &required_bytes));
    EXPECT_EQ(required_bytes, 104U);

    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorByteOpsAndFiniteCheckWorkForHostAndDeviceArenaBuffers)
{
    gsx_backend_t backend = create_cpu_backend();
    std::array<gsx_backend_buffer_type_class, 2> buffer_classes = {
        GSX_BACKEND_BUFFER_TYPE_HOST,
        GSX_BACKEND_BUFFER_TYPE_DEVICE,
    };

    for(gsx_backend_buffer_type_class buffer_class : buffer_classes) {
        gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, buffer_class);
        gsx_arena_t arena = nullptr;
        gsx_arena_desc arena_desc{};
        gsx_tensor_t src = nullptr;
        gsx_tensor_t dst = nullptr;
        gsx_tensor_desc tensor_desc{};
        bool is_finite = false;
        std::array<float, 4> upload_values = { 1.0f, 2.0f, 3.0f, 4.0f };
        std::array<float, 4> download_values = {};
        std::array<float, 4> zero_values = { 0.0f, 0.0f, 0.0f, 0.0f };
        std::array<float, 4> filled_values = { 1.5f, 1.5f, 1.5f, 1.5f };
        float fill_value = 1.5f;

        SCOPED_TRACE(testing::Message() << "buffer_class=" << static_cast<int>(buffer_class));

        arena_desc.initial_capacity_bytes = 256;
        arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
        ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

        tensor_desc = make_f32_tensor_desc(arena, 4);
        ASSERT_GSX_SUCCESS(gsx_tensor_init(&src, &tensor_desc));
        ASSERT_GSX_SUCCESS(gsx_tensor_init(&dst, &tensor_desc));

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(src, upload_values.data(), sizeof(upload_values)));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(src, download_values.data(), sizeof(download_values)));
        EXPECT_EQ(download_values, upload_values);

        ASSERT_GSX_SUCCESS(gsx_tensor_copy(src, dst));
        download_values.fill(0.0f);
        ASSERT_GSX_SUCCESS(gsx_tensor_download(dst, download_values.data(), sizeof(download_values)));
        EXPECT_EQ(download_values, upload_values);

        ASSERT_GSX_SUCCESS(gsx_tensor_set_zero(dst));
        download_values.fill(7.0f);
        ASSERT_GSX_SUCCESS(gsx_tensor_download(dst, download_values.data(), sizeof(download_values)));
        EXPECT_EQ(download_values, zero_values);

        ASSERT_GSX_SUCCESS(gsx_tensor_fill(dst, &fill_value, sizeof(fill_value)));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(dst, download_values.data(), sizeof(download_values)));
        EXPECT_EQ(download_values, filled_values);

        ASSERT_GSX_SUCCESS(gsx_tensor_check_finite(src, &is_finite));
        EXPECT_TRUE(is_finite);

        upload_values[1] = std::numeric_limits<float>::quiet_NaN();
        upload_values[3] = std::numeric_limits<float>::infinity();
        ASSERT_GSX_SUCCESS(gsx_tensor_upload(src, upload_values.data(), sizeof(upload_values)));
        ASSERT_GSX_SUCCESS(gsx_tensor_check_finite(src, &is_finite));
        EXPECT_FALSE(is_finite);

        ASSERT_GSX_SUCCESS(gsx_tensor_free(src));
        ASSERT_GSX_SUCCESS(gsx_tensor_free(dst));
        ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    }

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorNativeHandleExportsBackendPointerForLiveTensor)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    void *native_handle = nullptr;
    gsx_size_t offset_bytes = 0;

    arena_desc.initial_capacity_bytes = 256;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    tensor_desc = make_f32_tensor_desc(arena, 4);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_get_native_handle(tensor, &native_handle, &offset_bytes));
    EXPECT_NE(native_handle, nullptr);
    EXPECT_EQ(offset_bytes, 0U);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorNativeHandleRejectsInvalidArgumentsAndDryRunStorage)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_t dry_arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_arena_desc dry_desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_t dry_tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    void *native_handle = nullptr;
    gsx_size_t offset_bytes = 0;

    arena_desc.initial_capacity_bytes = 256;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    dry_desc.initial_capacity_bytes = 256;
    dry_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    dry_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&dry_arena, buffer_type, &dry_desc));

    tensor_desc = make_f32_tensor_desc(arena, 4);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));
    tensor_desc.arena = dry_arena;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&dry_tensor, &tensor_desc));

    EXPECT_GSX_CODE(gsx_tensor_get_native_handle(nullptr, &native_handle, &offset_bytes), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_get_native_handle(tensor, nullptr, &offset_bytes), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_get_native_handle(tensor, &native_handle, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_get_native_handle(dry_tensor, &native_handle, &offset_bytes), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(dry_tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(dry_arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorOpsRejectInvalidArgumentsAndIncompatibleCopies)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t src = nullptr;
    gsx_tensor_t dst = nullptr;
    gsx_tensor_t mismatched_shape = nullptr;
    gsx_tensor_t mismatched_format = nullptr;
    gsx_tensor_desc src_desc{};
    gsx_tensor_desc dst_desc{};
    gsx_tensor_desc mismatched_shape_desc{};
    gsx_tensor_desc mismatched_format_desc{};
    std::array<float, 4> values = { 1.0f, 2.0f, 3.0f, 4.0f };
    uint16_t wrong_fill_value = 7;
    bool is_finite = false;

    arena_desc.initial_capacity_bytes = 512;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    src_desc = make_f32_tensor_desc(arena, 4);
    dst_desc = make_f32_tensor_desc(arena, 4);
    mismatched_shape_desc = make_f32_tensor_desc(arena, 5);
    mismatched_format_desc = make_f32_tensor_desc(arena, 4);
    mismatched_format_desc.storage_format = GSX_STORAGE_FORMAT_HWC;

    ASSERT_GSX_SUCCESS(gsx_tensor_init(&src, &src_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&dst, &dst_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&mismatched_shape, &mismatched_shape_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&mismatched_format, &mismatched_format_desc));

    EXPECT_GSX_CODE(gsx_tensor_upload(src, nullptr, sizeof(values)), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_download(src, nullptr, sizeof(values)), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_upload(src, values.data(), sizeof(values) + sizeof(float)), GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(gsx_tensor_fill(src, &wrong_fill_value, sizeof(wrong_fill_value)), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_check_finite(src, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_copy(nullptr, dst), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_copy(src, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_copy(src, mismatched_shape), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_copy(src, mismatched_format), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(src, values.data(), sizeof(values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_check_finite(src, &is_finite));
    EXPECT_TRUE(is_finite);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(src));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(dst));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mismatched_shape));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mismatched_format));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorGatherResizeAndExpWorkOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x_gather = nullptr;
    gsx_tensor_t index_gather = nullptr;
    gsx_tensor_t out_gather = nullptr;
    gsx_tensor_t x_resize = nullptr;
    gsx_tensor_t out_resize = nullptr;
    gsx_tensor_t x_exp = nullptr;
    gsx_tensor_t out_exp = nullptr;
    gsx_tensor_desc x_gather_desc{};
    gsx_tensor_desc index_gather_desc{};
    gsx_tensor_desc out_gather_desc{};
    gsx_tensor_desc x_resize_desc{};
    gsx_tensor_desc out_resize_desc{};
    gsx_tensor_desc x_exp_desc{};
    gsx_tensor_desc out_exp_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_x_gather = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_out_gather = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_x_resize = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_out_resize = {};
    std::array<float, 10> x_gather_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    std::array<int32_t, 3> gather_indices = { 4, 1, 3 };
    std::array<float, 6> gathered_values = {};
    std::array<float, 6> expected_gathered_values = { 9.0f, 10.0f, 3.0f, 4.0f, 7.0f, 8.0f };
    std::array<float, 6> x_resize_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    std::array<float, 12> resized_values = {};
    std::array<float, 12> expected_resized_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 4> exp_inputs = { 0.0f, 1.0f, -1.0f, 2.0f };
    std::array<float, 4> exp_outputs = {};

    shape_x_gather[0] = 5;
    shape_x_gather[1] = 2;
    shape_out_gather[0] = 3;
    shape_out_gather[1] = 2;
    shape_x_resize[0] = 2;
    shape_x_resize[1] = 3;
    shape_out_resize[0] = 4;
    shape_out_resize[1] = 3;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    x_gather_desc = make_f32_tensor_desc_with_shape(arena, shape_x_gather, 2);
    out_gather_desc = make_f32_tensor_desc_with_shape(arena, shape_out_gather, 2);
    index_gather_desc = make_rank1_tensor_desc(arena, 3, GSX_DATA_TYPE_I32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x_gather, &x_gather_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&index_gather, &index_gather_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_gather, &out_gather_desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x_gather, x_gather_values.data(), sizeof(x_gather_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(index_gather, gather_indices.data(), sizeof(gather_indices)));
    ASSERT_GSX_SUCCESS(gsx_tensor_gather(x_gather, index_gather, out_gather));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out_gather, gathered_values.data(), sizeof(gathered_values)));
    EXPECT_EQ(gathered_values, expected_gathered_values);

    x_resize_desc = make_f32_tensor_desc_with_shape(arena, shape_x_resize, 2);
    out_resize_desc = make_f32_tensor_desc_with_shape(arena, shape_out_resize, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x_resize, &x_resize_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_resize, &out_resize_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x_resize, x_resize_values.data(), sizeof(x_resize_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_resize(x_resize, out_resize));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out_resize, resized_values.data(), sizeof(resized_values)));
    EXPECT_EQ(resized_values, expected_resized_values);

    x_exp_desc = make_f32_tensor_desc(arena, 4);
    out_exp_desc = make_f32_tensor_desc(arena, 4);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x_exp, &x_exp_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_exp, &out_exp_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x_exp, exp_inputs.data(), sizeof(exp_inputs)));
    ASSERT_GSX_SUCCESS(gsx_tensor_exp(x_exp, out_exp));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out_exp, exp_outputs.data(), sizeof(exp_outputs)));
    for(std::size_t i = 0; i < exp_outputs.size(); ++i) {
        EXPECT_NEAR(exp_outputs[i], std::exp(exp_inputs[i]), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_free(x_gather));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(index_gather));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_gather));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x_resize));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_resize));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x_exp));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_exp));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorUnaryOpsWorkOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_desc desc{};
    std::array<float, 4> input = { -2.0f, -0.5f, 0.5f, 2.0f };
    std::array<float, 4> output = {};

    arena_desc.initial_capacity_bytes = 2048;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    desc = make_f32_tensor_desc(arena, static_cast<gsx_index_t>(input.size()));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, input.data(), sizeof(input)));
    ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid(x, out));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        EXPECT_NEAR(output[i], sigmoid, 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid_derivative(x, out));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        EXPECT_NEAR(output[i], sigmoid * (1.0f - sigmoid), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_abs(x, out));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], std::fabs(input[i]), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, input.data(), sizeof(input)));
    ASSERT_GSX_SUCCESS(gsx_tensor_exp_inplace(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(x, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], std::exp(input[i]), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, input.data(), sizeof(input)));
    ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid_inplace(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(x, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        EXPECT_NEAR(output[i], sigmoid, 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, input.data(), sizeof(input)));
    ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid_derivative_inplace(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(x, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        EXPECT_NEAR(output[i], sigmoid * (1.0f - sigmoid), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, input.data(), sizeof(input)));
    ASSERT_GSX_SUCCESS(gsx_tensor_abs_inplace(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(x, output.data(), sizeof(output)));
    for(std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], std::fabs(input[i]), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorClampInplaceWorksOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t f32_tensor = nullptr;
    gsx_tensor_t i32_tensor = nullptr;
    gsx_tensor_t u8_tensor = nullptr;
    gsx_tensor_desc f32_desc{};
    gsx_tensor_desc i32_desc{};
    gsx_tensor_desc u8_desc{};
    std::array<float, 6> f32_values = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f };
    std::array<float, 6> f32_expected = { -0.5f, -0.5f, 0.0f, 1.0f, 1.5f, 1.5f };
    std::array<float, 6> f32_output = {};
    std::array<int32_t, 5> i32_values = { -10, -3, 0, 7, 20 };
    std::array<int32_t, 5> i32_expected = { -5, -3, 0, 5, 5 };
    std::array<int32_t, 5> i32_output = {};
    std::array<uint8_t, 6> u8_values = { 0U, 5U, 10U, 15U, 20U, 25U };
    std::array<uint8_t, 6> u8_expected = { 7U, 7U, 10U, 15U, 18U, 18U };
    std::array<uint8_t, 6> u8_output = {};
    float f32_min = -0.5f;
    float f32_max = 1.5f;
    int32_t i32_min = -5;
    int32_t i32_max = 5;
    uint8_t u8_min = 7U;
    uint8_t u8_max = 18U;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    f32_desc = make_f32_tensor_desc(arena, static_cast<gsx_index_t>(f32_values.size()));
    i32_desc = make_rank1_tensor_desc(arena, static_cast<gsx_index_t>(i32_values.size()), GSX_DATA_TYPE_I32);
    u8_desc = make_rank1_tensor_desc(arena, static_cast<gsx_index_t>(u8_values.size()), GSX_DATA_TYPE_U8);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&f32_tensor, &f32_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&i32_tensor, &i32_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&u8_tensor, &u8_desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(f32_tensor, f32_values.data(), sizeof(f32_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(i32_tensor, i32_values.data(), sizeof(i32_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(u8_tensor, u8_values.data(), sizeof(u8_values)));

    ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(f32_tensor, &f32_min, &f32_max));
    ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(i32_tensor, &i32_min, &i32_max));
    ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(u8_tensor, &u8_min, &u8_max));

    ASSERT_GSX_SUCCESS(gsx_tensor_download(f32_tensor, f32_output.data(), sizeof(f32_output)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(i32_tensor, i32_output.data(), sizeof(i32_output)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(u8_tensor, u8_output.data(), sizeof(u8_output)));
    EXPECT_EQ(f32_output, f32_expected);
    EXPECT_EQ(i32_output, i32_expected);
    EXPECT_EQ(u8_output, u8_expected);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(f32_tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(i32_tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(u8_tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorClampInplaceRejectsInvalidContracts)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    float min_value = -1.0f;
    float max_value = 1.0f;

    arena_desc.initial_capacity_bytes = 1024;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    tensor_desc = make_f32_tensor_desc(arena, 4);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));

    EXPECT_GSX_CODE(gsx_tensor_clamp_inplace(nullptr, &min_value, &max_value), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_clamp_inplace(tensor, nullptr, &max_value), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_clamp_inplace(tensor, &min_value, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    min_value = 2.0f;
    max_value = 1.0f;
    EXPECT_GSX_CODE(gsx_tensor_clamp_inplace(tensor, &min_value, &max_value), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorClampInplaceRejectsDryRunTensorStorage)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t dry_arena = nullptr;
    gsx_arena_desc dry_desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    float min_value = -1.0f;
    float max_value = 1.0f;

    dry_desc.initial_capacity_bytes = 1024;
    dry_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    dry_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&dry_arena, buffer_type, &dry_desc));

    tensor_desc = make_f32_tensor_desc(dry_arena, 4);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));

    EXPECT_GSX_CODE(gsx_tensor_clamp_inplace(tensor, &min_value, &max_value), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(dry_arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorGatherResizeAndExpRejectInvalidContracts)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_t index = nullptr;
    gsx_tensor_t out_mismatched_tail = nullptr;
    gsx_tensor_t out_mismatched_exp = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc out_desc{};
    gsx_tensor_desc index_desc{};
    gsx_tensor_desc out_mismatched_tail_desc{};
    gsx_tensor_desc out_mismatched_exp_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_x = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_out = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_out_mismatched_tail = {};
    std::array<float, 10> x_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    std::array<int32_t, 3> out_of_range_indices = { 0, 2, 5 };
    std::array<int32_t, 3> alias_indices = { 0, 2, 4 };

    shape_x[0] = 5;
    shape_x[1] = 2;
    shape_out[0] = 3;
    shape_out[1] = 2;
    shape_out_mismatched_tail[0] = 3;
    shape_out_mismatched_tail[1] = 3;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena, shape_x, 2);
    out_desc = make_f32_tensor_desc_with_shape(arena, shape_out, 2);
    index_desc = make_rank1_tensor_desc(arena, 3, GSX_DATA_TYPE_I32);
    out_mismatched_tail_desc = make_f32_tensor_desc_with_shape(arena, shape_out_mismatched_tail, 2);
    out_mismatched_exp_desc = make_f32_tensor_desc(arena, 3);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&index, &index_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_mismatched_tail, &out_mismatched_tail_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_mismatched_exp, &out_mismatched_exp_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, x_values.data(), sizeof(x_values)));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(index, out_of_range_indices.data(), sizeof(out_of_range_indices)));
    EXPECT_GSX_CODE(gsx_tensor_gather(x, index, out), GSX_ERROR_OUT_OF_RANGE);

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(index, alias_indices.data(), sizeof(alias_indices)));
    EXPECT_GSX_CODE(gsx_tensor_gather(x, index, x), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_gather(x, index, out_mismatched_tail), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_resize(x, out_mismatched_tail), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_exp(x, out_mismatched_exp), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(index));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_mismatched_tail));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_mismatched_exp));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorGatherResizeAndExpRejectDryRunTensorStorage)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t dry_arena = nullptr;
    gsx_arena_desc dry_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_t index = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc out_desc{};
    gsx_tensor_desc index_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape = {};

    shape[0] = 4;
    shape[1] = 2;

    dry_desc.initial_capacity_bytes = 1024;
    dry_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    dry_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&dry_arena, buffer_type, &dry_desc));

    x_desc = make_f32_tensor_desc_with_shape(dry_arena, shape, 2);
    out_desc = make_f32_tensor_desc_with_shape(dry_arena, shape, 2);
    index_desc = make_rank1_tensor_desc(dry_arena, 4, GSX_DATA_TYPE_I32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&index, &index_desc));

    EXPECT_GSX_CODE(gsx_tensor_gather(x, index, out), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_tensor_resize(x, out), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_tensor_exp(x, out), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(index));
    ASSERT_GSX_SUCCESS(gsx_arena_free(dry_arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorReduceUnaryAndBinaryWorkOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_t workspace_arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t unary_out = nullptr;
    gsx_tensor_t binary_out = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc target_desc{};
    gsx_tensor_desc unary_out_desc{};
    gsx_tensor_desc binary_out_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> x_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> out_shape = {};
    std::array<float, 24> x_values = {};
    std::array<float, 24> target_values = {};
    std::array<float, 2> out_values = {};

    for(std::size_t i = 0; i < x_values.size(); ++i) {
        x_values[i] = static_cast<float>(i + 1);
        target_values[i] = 1.0f;
    }
    x_shape[0] = 2;
    x_shape[1] = 3;
    x_shape[2] = 4;
    out_shape[0] = 2;
    out_shape[1] = 1;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    arena_desc.initial_capacity_bytes = 2048;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&workspace_arena, buffer_type, &arena_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 3);
    target_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 3);
    unary_out_desc = make_f32_tensor_desc_with_shape(arena, out_shape, 2);
    binary_out_desc = make_f32_tensor_desc_with_shape(arena, out_shape, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&target, &target_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&unary_out, &unary_out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&binary_out, &binary_out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, x_values.data(), sizeof(x_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(target, target_values.data(), sizeof(target_values)));

    ASSERT_GSX_SUCCESS(gsx_tensor_sum(workspace_arena, x, unary_out, 1));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(unary_out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 78.0f, 1e-5f);
    EXPECT_NEAR(out_values[1], 222.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_mean(workspace_arena, x, unary_out, 1));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(unary_out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 6.5f, 1e-5f);
    EXPECT_NEAR(out_values[1], 18.5f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_max(workspace_arena, x, unary_out, 1));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(unary_out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 12.0f, 1e-5f);
    EXPECT_NEAR(out_values[1], 24.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_mse(workspace_arena, x, target, binary_out, 1));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(binary_out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 42.166667f, 1e-4f);
    EXPECT_NEAR(out_values[1], 318.16666f, 1e-4f);

    ASSERT_GSX_SUCCESS(gsx_tensor_mae(workspace_arena, x, target, binary_out, 1));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(binary_out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 5.5f, 1e-5f);
    EXPECT_NEAR(out_values[1], 17.5f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(binary_out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(unary_out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(target));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_arena_free(workspace_arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorReduceSupportsStartAxisZeroAndDryRunNeedsNoWorkspaceOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_t workspace_arena = nullptr;
    gsx_arena_t workspace_dry = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_arena_desc workspace_dry_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc target_desc{};
    gsx_tensor_desc out_desc{};
    std::array<float, 6> x_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    std::array<float, 6> target_values = { 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 8.0f };
    std::array<float, 1> out_values = {};
    std::array<float, 1> out_before = { 17.0f };
    gsx_size_t required_before = 0;
    gsx_size_t required_after = 0;
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> x_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> out_shape = {};

    x_shape[0] = 2;
    x_shape[1] = 3;
    out_shape[0] = 1;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    workspace_dry_desc.initial_capacity_bytes = 0;
    workspace_dry_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    workspace_dry_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&workspace_arena, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&workspace_dry, buffer_type, &workspace_dry_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 2);
    target_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 2);
    out_desc = make_f32_tensor_desc_with_shape(arena, out_shape, 1);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&target, &target_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, x_values.data(), sizeof(x_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(target, target_values.data(), sizeof(target_values)));

    ASSERT_GSX_SUCCESS(gsx_tensor_sum(workspace_arena, x, out, 0));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 21.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_mean(workspace_arena, x, out, 0));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 3.5f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_max(workspace_arena, x, out, 0));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 6.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_mse(workspace_arena, x, target, out, 0));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 2.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_mae(workspace_arena, x, target, out, 0));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_values.data(), sizeof(out_values)));
    EXPECT_NEAR(out_values[0], 1.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(out, out_before.data(), sizeof(out_before)));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(workspace_dry, &required_before));
    ASSERT_GSX_SUCCESS(gsx_tensor_sum(workspace_dry, x, out, 0));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_values.data(), sizeof(out_values)));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(workspace_dry, &required_after));
    EXPECT_NEAR(out_values[0], out_before[0], 1e-5f);
    EXPECT_EQ(required_after, required_before);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(target));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_arena_free(workspace_dry));
    ASSERT_GSX_SUCCESS(gsx_arena_free(workspace_arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorReduceRejectsInvalidContracts)
{
    gsx_backend_t backend_a = create_cpu_backend();
    gsx_backend_t backend_b = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type_a = find_buffer_type(backend_a, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t buffer_type_b = find_buffer_type(backend_b, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena_a = nullptr;
    gsx_arena_t arena_b = nullptr;
    gsx_arena_t workspace_a = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t target_bad = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_t out_bad = nullptr;
    gsx_tensor_t out_other_backend = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc target_desc{};
    gsx_tensor_desc target_bad_desc{};
    gsx_tensor_desc out_desc{};
    gsx_tensor_desc out_bad_desc{};
    gsx_tensor_desc out_other_backend_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> x_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> out_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> out_bad_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> target_bad_shape = {};

    x_shape[0] = 2;
    x_shape[1] = 3;
    x_shape[2] = 4;
    out_shape[0] = 2;
    out_shape[1] = 1;
    out_bad_shape[0] = 2;
    out_bad_shape[1] = 2;
    target_bad_shape[0] = 2;
    target_bad_shape[1] = 3;
    target_bad_shape[2] = 2;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena_a, buffer_type_a, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&workspace_a, buffer_type_a, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena_b, buffer_type_b, &arena_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena_a, x_shape, 3);
    target_desc = make_f32_tensor_desc_with_shape(arena_a, x_shape, 3);
    target_bad_desc = make_f32_tensor_desc_with_shape(arena_a, target_bad_shape, 3);
    out_desc = make_f32_tensor_desc_with_shape(arena_a, out_shape, 2);
    out_bad_desc = make_f32_tensor_desc_with_shape(arena_a, out_bad_shape, 2);
    out_other_backend_desc = make_f32_tensor_desc_with_shape(arena_b, out_shape, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&target, &target_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&target_bad, &target_bad_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_bad, &out_bad_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_other_backend, &out_other_backend_desc));

    EXPECT_GSX_CODE(gsx_tensor_sum(workspace_a, x, out, 3), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_sum(workspace_a, x, out_bad, 1), GSX_ERROR_INVALID_ARGUMENT);

    EXPECT_GSX_CODE(gsx_tensor_mse(workspace_a, x, target_bad, out, 1), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_tensor_mae(workspace_a, x, target_bad, out, 1), GSX_ERROR_INVALID_ARGUMENT);

    EXPECT_GSX_CODE(gsx_tensor_sum(workspace_a, x, out_other_backend, 1), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_other_backend));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_bad));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(target_bad));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(target));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena_b));
    ASSERT_GSX_SUCCESS(gsx_arena_free(workspace_a));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena_a));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend_b));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend_a));
}

TEST(CoreRuntime, TensorReduceDryRunWorkspaceKeepsBytesUnchangedWithoutMutatingOutputOnCpu)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_t workspace_dry = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_arena_desc workspace_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc out_desc{};
    gsx_size_t required_before = 0;
    gsx_size_t required_after = 0;
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> x_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> out_shape = {};
    std::array<float, 24> x_values = {};
    std::array<float, 2> out_before = { -3.0f, 7.0f };
    std::array<float, 2> out_after = {};

    for(std::size_t i = 0; i < x_values.size(); ++i) {
        x_values[i] = static_cast<float>(i + 1);
    }
    x_shape[0] = 2;
    x_shape[1] = 3;
    x_shape[2] = 4;
    out_shape[0] = 2;
    out_shape[1] = 1;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    workspace_desc.initial_capacity_bytes = 0;
    workspace_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    workspace_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&workspace_dry, buffer_type, &workspace_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 3);
    out_desc = make_f32_tensor_desc_with_shape(arena, out_shape, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, x_values.data(), sizeof(x_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(out, out_before.data(), sizeof(out_before)));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(workspace_dry, &required_before));

    ASSERT_GSX_SUCCESS(gsx_tensor_sum(workspace_dry, x, out, 1));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out, out_after.data(), sizeof(out_after)));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(workspace_dry, &required_after));
    EXPECT_EQ(out_after, out_before);
    EXPECT_EQ(required_after, required_before);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_arena_free(workspace_dry));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorCopyIgnoresUnusedDescriptorShapeTail)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t src = nullptr;
    gsx_tensor_t dst = nullptr;
    gsx_tensor_desc src_desc{};
    gsx_tensor_desc dst_desc{};
    std::array<float, 4> values = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 4> roundtrip = {};

    desc.initial_capacity_bytes = 256;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));

    src_desc = make_f32_tensor_desc(arena, 4);
    dst_desc = make_f32_tensor_desc(arena, 4);
    src_desc.shape[1] = 111;
    src_desc.shape[2] = 222;
    src_desc.shape[3] = 333;
    dst_desc.shape[1] = 444;
    dst_desc.shape[2] = 555;
    dst_desc.shape[3] = 666;

    ASSERT_GSX_SUCCESS(gsx_tensor_init(&src, &src_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&dst, &dst_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(src, values.data(), sizeof(values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_copy(src, dst));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(dst, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, values);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(src));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(dst));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, TensorInitRejectsZeroExtentAndAcceptsScalarAsShapeOne)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc desc{};
    gsx_tensor_t scalar_tensor = nullptr;
    gsx_tensor_desc tensor_desc{};

    desc.initial_capacity_bytes = 256;
    desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &desc));

    tensor_desc = make_f32_tensor_desc(arena, 4);
    tensor_desc.rank = 0;
    EXPECT_GSX_CODE(gsx_tensor_init(&scalar_tensor, &tensor_desc), GSX_ERROR_INVALID_ARGUMENT);

    tensor_desc = make_f32_tensor_desc(arena, 4);
    tensor_desc.shape[0] = 0;
    EXPECT_GSX_CODE(gsx_tensor_init(&scalar_tensor, &tensor_desc), GSX_ERROR_INVALID_ARGUMENT);

    tensor_desc = make_f32_tensor_desc(arena, 1);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&scalar_tensor, &tensor_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(scalar_tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, DryRunGrowOnDemandReserveTracksCapacityWithoutBackingStorage)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_desc tensor_desc{};
    gsx_arena_info info{};
    gsx_size_t required_bytes = 0;

    arena_desc.initial_capacity_bytes = 0;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_EQ(info.capacity_bytes, 0U);
    EXPECT_TRUE(info.dry_run);

    tensor_desc = make_f32_tensor_desc(arena, 40);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor, &tensor_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(arena, &required_bytes));
    EXPECT_GE(info.capacity_bytes, sizeof(float) * 40U);
    EXPECT_EQ(required_bytes, sizeof(float) * 40U);
    EXPECT_EQ(info.active_tensor_count, 1U);
    EXPECT_TRUE(info.dry_run);
    EXPECT_GSX_CODE(gsx_tensor_set_zero(tensor), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_arena_reserve(arena, 512), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_reserve(arena, 512));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_GE(info.capacity_bytes, 512U);
    ASSERT_GSX_SUCCESS(gsx_arena_reserve(arena, 128));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_GE(info.capacity_bytes, 512U);

    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, DryRunGrowOnDemandTensorInitCanGrowWithLiveTensors)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t tensor_a = nullptr;
    gsx_tensor_t tensor_b = nullptr;
    gsx_tensor_desc desc_a{};
    gsx_tensor_desc desc_b{};
    gsx_arena_info info{};
    gsx_size_t required_bytes = 0;

    arena_desc.initial_capacity_bytes = 64;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    desc_a = make_f32_tensor_desc(arena, 32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor_a, &desc_a));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    EXPECT_EQ(info.active_tensor_count, 1U);
    EXPECT_GE(info.capacity_bytes, 128U);

    desc_b = make_f32_tensor_desc(arena, 16);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&tensor_b, &desc_b));
    ASSERT_GSX_SUCCESS(gsx_arena_get_info(arena, &info));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(arena, &required_bytes));
    EXPECT_EQ(info.active_tensor_count, 2U);
    EXPECT_GE(info.capacity_bytes, 192U);
    EXPECT_EQ(required_bytes, 192U);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor_a));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor_b));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GsLifecycleAndFieldQueryRespectAuxFlags)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_gs_info gs_info{};
    gsx_tensor_t tensor = nullptr;

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.arena = arena;
    gs_desc.count = 3;
    gs_desc.aux_flags = GSX_GS_AUX_GRAD_ACC;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.arena, arena);
    EXPECT_EQ(gs_info.count, 3U);
    EXPECT_EQ(gs_info.aux_flags, GSX_GS_AUX_GRAD_ACC);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &tensor));
    ASSERT_NE(tensor, nullptr);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_ACC, &tensor));
    ASSERT_NE(tensor, nullptr);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &tensor), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GsSetFieldZeroGradientsAndClampOpacityWork)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_tensor_t grad_mean3d = nullptr;
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t replacement = nullptr;
    gsx_tensor_desc replacement_desc{};
    std::array<float, 6> grad_values = { 1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.0f };
    std::array<float, 6> mean_values = { 10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f };
    std::array<float, 6> roundtrip = {};
    std::array<float, 6> zeros = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 2> opacity_values = { -0.2f, 0.8f };

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.arena = arena;
    gs_desc.count = 2;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &grad_mean3d));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(grad_mean3d, grad_values.data(), sizeof(grad_values)));
    ASSERT_GSX_SUCCESS(gsx_gs_zero_gradients(gs));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(grad_mean3d, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, zeros);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(opacity, opacity_values.data(), sizeof(opacity_values)));
    float low = 0.0f;
    float high = 0.3f;
    ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(opacity, &low, &high));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity, opacity_values.data(), sizeof(opacity_values)));
    EXPECT_FLOAT_EQ(opacity_values[0], 0.0f);
    EXPECT_FLOAT_EQ(opacity_values[1], 0.3f);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    replacement_desc = make_f32_tensor_desc_with_shape(arena, { 2, 3, 0, 0 }, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&replacement, &replacement_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(replacement, mean_values.data(), sizeof(mean_values)));
    ASSERT_GSX_SUCCESS(gsx_gs_set_field(gs, GSX_GS_FIELD_MEAN3D, replacement));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(mean3d, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, mean_values);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(replacement));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GsAuxMutationsStructuralOpsAndFiniteCheckWork)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_gs_info gs_info{};
    gsx_tensor_t grad_acc = nullptr;
    gsx_tensor_t visible_counter = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t permutation = nullptr;
    gsx_tensor_t gather_index = nullptr;
    gsx_tensor_desc permutation_desc{};
    gsx_tensor_desc gather_index_desc{};
    gsx_gs_finite_check_result finite_result{};
    std::array<float, 3> aux_values = { 1.0f, 2.0f, 3.0f };
    std::array<float, 3> opacity_values = { 1.0f, 2.0f, 3.0f };
    std::array<float, 3> roundtrip = {};
    std::array<float, 3> zeros = { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> permuted_opacity = { 3.0f, 1.0f, 2.0f };
    std::array<float, 9> mean_values = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    std::array<int32_t, 3> perm_values = { 2, 0, 1 };
    std::array<int32_t, 2> gather_values = { 0, 2 };
    std::array<int32_t, 2> gather_out_of_range_values = { 0, 3 };

    arena_desc.initial_capacity_bytes = 1U << 20;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.arena = arena;
    gs_desc.count = 3;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_set_aux_enabled(gs, GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER, true));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.aux_flags, GSX_GS_AUX_GRAD_ACC | GSX_GS_AUX_VISIBLE_COUNTER);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_ACC, &grad_acc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_VISIBLE_COUNTER, &visible_counter));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(grad_acc, aux_values.data(), sizeof(aux_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(visible_counter, aux_values.data(), sizeof(aux_values)));
    ASSERT_GSX_SUCCESS(gsx_gs_zero_aux_tensors(gs, GSX_GS_AUX_GRAD_ACC));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(grad_acc, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, zeros);
    ASSERT_GSX_SUCCESS(gsx_tensor_download(visible_counter, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, aux_values);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(opacity, opacity_values.data(), sizeof(opacity_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(mean3d, mean_values.data(), sizeof(mean_values)));

    permutation_desc = make_rank1_tensor_desc(arena, 3, GSX_DATA_TYPE_I32);
    gather_index_desc = make_rank1_tensor_desc(arena, 2, GSX_DATA_TYPE_I32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&permutation, &permutation_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&gather_index, &gather_index_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(permutation, perm_values.data(), sizeof(perm_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(gather_index, gather_values.data(), sizeof(gather_values)));

    ASSERT_GSX_SUCCESS(gsx_gs_permute(gs, permutation));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, permuted_opacity);

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(gather_index, gather_out_of_range_values.data(), sizeof(gather_out_of_range_values)));
    EXPECT_GSX_CODE(gsx_gs_gather(gs, gather_index), GSX_ERROR_OUT_OF_RANGE);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, permuted_opacity);

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(gather_index, gather_values.data(), sizeof(gather_values)));
    ASSERT_GSX_SUCCESS(gsx_gs_gather(gs, gather_index));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, 2U);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity, roundtrip.data(), sizeof(float) * 2));
    EXPECT_FLOAT_EQ(roundtrip[0], 3.0f);
    EXPECT_FLOAT_EQ(roundtrip[1], 2.0f);

    ASSERT_GSX_SUCCESS(gsx_gs_resize(gs, 3));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, 3U);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_FLOAT_EQ(roundtrip[0], 3.0f);
    EXPECT_FLOAT_EQ(roundtrip[1], 2.0f);
    EXPECT_FLOAT_EQ(roundtrip[2], 0.0f);

    ASSERT_GSX_SUCCESS(gsx_gs_check_finite(gs, &finite_result));
    EXPECT_TRUE(finite_result.is_finite);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(mean3d, mean_values.data(), sizeof(mean_values)));
    mean_values[1] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(mean3d, mean_values.data(), sizeof(mean_values)));
    ASSERT_GSX_SUCCESS(gsx_gs_check_finite(gs, &finite_result));
    EXPECT_FALSE(finite_result.is_finite);
    EXPECT_GT(finite_result.non_finite_count, 0U);

    ASSERT_GSX_SUCCESS(gsx_gs_set_aux_enabled(gs, GSX_GS_AUX_GRAD_ACC, false));
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_ACC, &grad_acc), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(permutation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(gather_index));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GsGatherRejectsDryRunIndexStorage)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_t dry_arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_arena_desc dry_arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_tensor_t index = nullptr;
    gsx_tensor_desc index_desc{};

    arena_desc.initial_capacity_bytes = 2048;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    dry_arena_desc.initial_capacity_bytes = 2048;
    dry_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    dry_arena_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&dry_arena, buffer_type, &dry_arena_desc));

    gs_desc.arena = arena;
    gs_desc.count = 3;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    index_desc = make_rank1_tensor_desc(dry_arena, 2, GSX_DATA_TYPE_I32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&index, &index_desc));

    EXPECT_GSX_CODE(gsx_gs_gather(gs, index), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(index));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(dry_arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GsSetAuxEnabledFailsWithoutCapacityAndKeepsState)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t sizing_arena = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_arena_desc sizing_arena_desc{};
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_gs_info gs_info{};
    gsx_tensor_t tensor = nullptr;
    gsx_size_t base_required_bytes = 0;

    sizing_arena_desc.initial_capacity_bytes = 0;
    sizing_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&sizing_arena, buffer_type, &sizing_arena_desc));
    gs_desc.arena = sizing_arena;
    gs_desc.count = 64;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(sizing_arena, &base_required_bytes));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    gs = nullptr;
    ASSERT_GSX_SUCCESS(gsx_arena_free(sizing_arena));
    sizing_arena = nullptr;

    arena_desc.initial_capacity_bytes = base_required_bytes;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    gs_desc.arena = arena;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));

    EXPECT_GSX_CODE(gsx_gs_set_aux_enabled(gs, GSX_GS_AUX_SH1, true), GSX_ERROR_OUT_OF_RANGE);
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.aux_flags, GSX_GS_AUX_NONE);
    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &tensor), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(CoreRuntime, GsResizeFailurePreservesCountAndData)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t sizing_arena = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_arena_desc sizing_arena_desc{};
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_gs_info gs_info{};
    gsx_tensor_t opacity = nullptr;
    gsx_size_t base_required_bytes = 0;
    std::array<float, 4> values = { 10.0f, 20.0f, 30.0f, 40.0f };
    std::array<float, 4> roundtrip = {};

    sizing_arena_desc.initial_capacity_bytes = 0;
    sizing_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&sizing_arena, buffer_type, &sizing_arena_desc));
    gs_desc.arena = sizing_arena;
    gs_desc.count = 4;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_get_required_bytes(sizing_arena, &base_required_bytes));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    gs = nullptr;
    ASSERT_GSX_SUCCESS(gsx_arena_free(sizing_arena));
    sizing_arena = nullptr;

    arena_desc.initial_capacity_bytes = base_required_bytes;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    gs_desc.arena = arena;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(opacity, values.data(), sizeof(values)));

    EXPECT_GSX_CODE(gsx_gs_resize(gs, 8), GSX_ERROR_OUT_OF_RANGE);
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, 4U);
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity, roundtrip.data(), sizeof(roundtrip)));
    EXPECT_EQ(roundtrip, values);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
