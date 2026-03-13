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

}  // namespace
