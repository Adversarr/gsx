#include "gsx/gsx.h"

#include "../gsx/src/gsx-impl.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

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

static bool has_metal_device()
{
    gsx_index_t metal_device_count = 0;
    gsx_error error = gsx_backend_registry_init();

    if(error.code != GSX_ERROR_SUCCESS) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_METAL, &metal_device_count);
    return error.code == GSX_ERROR_SUCCESS && metal_device_count > 0;
}

static gsx_backend_device_t get_metal_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    error = gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device);
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return backend_device;
}

static gsx_backend_t create_metal_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    backend_desc.device = get_metal_backend_device();
    if(backend_desc.device == nullptr) {
        return nullptr;
    }
    error = gsx_backend_init(&backend, &backend_desc);
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return backend;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_error error = gsx_backend_find_buffer_type(backend, type, &buffer_type);

    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return buffer_type;
}

static gsx_tensor_desc make_f32_tensor_desc_with_shape(gsx_arena_t arena, const std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> &shape, gsx_index_t rank)
{
    gsx_tensor_desc desc{};
    desc.arena = arena;
    desc.rank = rank;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    for(gsx_index_t i = 0; i < rank; ++i) {
        desc.shape[i] = shape[(std::size_t)i];
    }
    return desc;
}

static gsx_tensor_desc make_rank1_tensor_desc(gsx_arena_t arena, gsx_index_t length, gsx_data_type data_type)
{
    gsx_tensor_desc desc{};
    desc.arena = arena;
    desc.rank = 1;
    desc.shape[0] = length;
    desc.data_type = data_type;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    return desc;
}

class MetalBackendTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if(!has_metal_device()) {
            GTEST_SKIP() << "No Metal devices available";
        }
    }
};

TEST_F(MetalBackendTest, MetalDeviceDiscoveryExposesAtLeastOneDevice)
{
    gsx_index_t metal_device_count = 0;
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_device_info device_info{};

    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_METAL, &metal_device_count));
    ASSERT_GE(metal_device_count, 1);

    ASSERT_GSX_SUCCESS(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device));
    ASSERT_GSX_SUCCESS(gsx_backend_device_get_info(backend_device, &device_info));
    EXPECT_EQ(device_info.backend_type, GSX_BACKEND_TYPE_METAL);
    EXPECT_STREQ(device_info.backend_name, "metal");
    EXPECT_EQ(device_info.device_index, 0);
    EXPECT_NE(device_info.name, nullptr);
}

TEST_F(MetalBackendTest, MetalBackendExposesExpectedMetadataAndBufferTypes)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_device_t backend_device = get_metal_backend_device();
    gsx_backend_info backend_info{};
    gsx_backend_capabilities backend_capabilities{};
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_type_t host_pinned_buffer_type = nullptr;
    gsx_backend_buffer_type_t unified_buffer_type = nullptr;
    gsx_backend_buffer_type_info device_buffer_type_info{};
    gsx_backend_buffer_type_info host_pinned_buffer_type_info{};
    gsx_backend_buffer_type_info unified_buffer_type_info{};
    void *major_stream = nullptr;
    gsx_index_t buffer_type_count = 0;

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(backend_device, nullptr);

    ASSERT_GSX_SUCCESS(gsx_backend_get_info(backend, &backend_info));
    EXPECT_EQ(backend_info.backend_type, GSX_BACKEND_TYPE_METAL);
    EXPECT_EQ(backend_info.device, backend_device);

    ASSERT_GSX_SUCCESS(gsx_backend_get_capabilities(backend, &backend_capabilities));
    EXPECT_EQ(backend_capabilities.supported_data_types, GSX_DATA_TYPE_FLAG_F32 | GSX_DATA_TYPE_FLAG_I32);
    EXPECT_TRUE(backend_capabilities.supports_async_prefetch);

    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &major_stream));
    EXPECT_NE(major_stream, nullptr);

    ASSERT_GSX_SUCCESS(gsx_backend_count_buffer_types(backend, &buffer_type_count));
    EXPECT_EQ(buffer_type_count, 3);

    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 0, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 1, &host_pinned_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 2, &unified_buffer_type));

    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, &host_pinned_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED, &unified_buffer_type));

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &device_buffer_type), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_backend_get_buffer_type(backend, 3, &device_buffer_type), GSX_ERROR_OUT_OF_RANGE);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(device_buffer_type, &device_buffer_type_info));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(host_pinned_buffer_type, &host_pinned_buffer_type_info));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(unified_buffer_type, &unified_buffer_type_info));

    EXPECT_EQ(device_buffer_type_info.backend, backend);
    EXPECT_EQ(device_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    EXPECT_STREQ(device_buffer_type_info.name, "device");
    EXPECT_EQ(device_buffer_type_info.alignment_bytes, 256U);

    EXPECT_EQ(host_pinned_buffer_type_info.backend, backend);
    EXPECT_EQ(host_pinned_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED);
    EXPECT_STREQ(host_pinned_buffer_type_info.name, "host_pinned");
    EXPECT_EQ(host_pinned_buffer_type_info.alignment_bytes, 64U);

    EXPECT_EQ(unified_buffer_type_info.backend, backend);
    EXPECT_EQ(unified_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_UNIFIED);
    EXPECT_STREQ(unified_buffer_type_info.name, "unified");
    EXPECT_EQ(unified_buffer_type_info.alignment_bytes, 64U);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalBackendMajorStreamSyncSucceeds)
{
    gsx_backend_t backend = create_metal_backend();

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalHostVisibleBuffersSupportRoundtripAndZeroing)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t host_pinned_buffer_type = nullptr;
    gsx_backend_buffer_type_t unified_buffer_type = nullptr;
    std::array<std::uint8_t, 256> upload_bytes{};
    std::array<std::uint8_t, 256> download_bytes{};
    std::array<std::uint8_t, 256> zero_bytes{};

    for(std::size_t i = 0; i < upload_bytes.size(); ++i) {
        upload_bytes[i] = static_cast<std::uint8_t>(i);
        zero_bytes[i] = 0;
    }

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, &host_pinned_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED, &unified_buffer_type));

    for(gsx_backend_buffer_type_t buffer_type : { host_pinned_buffer_type, unified_buffer_type }) {
        gsx_backend_buffer_t buffer = nullptr;
        gsx_backend_buffer_desc buffer_desc{};

        buffer_desc.buffer_type = buffer_type;
        buffer_desc.size_bytes = upload_bytes.size();
        buffer_desc.alignment_bytes = 0;
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, upload_bytes.data(), upload_bytes.size()));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
        EXPECT_EQ(download_bytes, upload_bytes);

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_set_zero(buffer));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
        EXPECT_EQ(download_bytes, zero_bytes);

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    }

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalDeviceBufferByteOpsSubmitAsynchronousWork)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t device_buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    std::array<std::uint8_t, 256> upload_bytes{};
    std::array<std::uint8_t, 256> download_bytes{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = upload_bytes.size();
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&device_buffer, &buffer_desc));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(device_buffer, 0, upload_bytes.data(), upload_bytes.size()));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(device_buffer, 0, download_bytes.data(), download_bytes.size()));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_set_zero(device_buffer));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(device_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalBackendBuffersValidateAlignmentAndBounds)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t aligned_buffer = nullptr;
    gsx_backend_buffer_t rejected_buffer = nullptr;
    gsx_backend_buffer_desc aligned_buffer_desc{};
    gsx_backend_buffer_desc zero_size_buffer_desc{};
    gsx_backend_buffer_desc invalid_alignment_desc{};
    gsx_backend_buffer_info buffer_info{};
    std::array<std::uint8_t, 8> upload_bytes{};
    std::array<std::uint8_t, 8> download_bytes{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    aligned_buffer_desc.buffer_type = device_buffer_type;
    aligned_buffer_desc.size_bytes = 16;
    aligned_buffer_desc.alignment_bytes = 512;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&aligned_buffer, &aligned_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_get_info(aligned_buffer, &buffer_info));
    EXPECT_EQ(buffer_info.alignment_bytes, 512U);

    zero_size_buffer_desc.buffer_type = device_buffer_type;
    zero_size_buffer_desc.size_bytes = 0;
    zero_size_buffer_desc.alignment_bytes = 0;
    EXPECT_GSX_CODE(gsx_backend_buffer_init(&rejected_buffer, &zero_size_buffer_desc), GSX_ERROR_INVALID_ARGUMENT);

    invalid_alignment_desc.buffer_type = device_buffer_type;
    invalid_alignment_desc.size_bytes = 16;
    invalid_alignment_desc.alignment_bytes = 48;
    EXPECT_GSX_CODE(gsx_backend_buffer_init(&rejected_buffer, &invalid_alignment_desc), GSX_ERROR_INVALID_ARGUMENT);

    EXPECT_GSX_CODE(gsx_backend_buffer_upload(aligned_buffer, 12, upload_bytes.data(), upload_bytes.size()), GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(gsx_backend_buffer_download(aligned_buffer, 12, download_bytes.data(), download_bytes.size()), GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(gsx_backend_buffer_upload(aligned_buffer, 0, nullptr, 1), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_backend_buffer_download(aligned_buffer, 0, nullptr, 1), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(aligned_buffer, 0, nullptr, 0));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(aligned_buffer, 0, nullptr, 0));

    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(aligned_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalBackendCreationRejectsUnsupportedOptions)
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};
    int backend_option = 7;

    backend_desc.device = get_metal_backend_device();
    ASSERT_NE(backend_desc.device, nullptr);
    backend_desc.options = &backend_option;
    backend_desc.options_size_bytes = sizeof(backend_option);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_NOT_SUPPORTED);
}

TEST_F(MetalBackendTest, MetalHostVisibleBufferTensorOpsRoundtrip)
{
    gsx_backend_t backend = create_metal_backend();
    std::array<gsx_backend_buffer_type_class, 2> buffer_classes = {
        GSX_BACKEND_BUFFER_TYPE_HOST_PINNED,
        GSX_BACKEND_BUFFER_TYPE_UNIFIED,
    };

    ASSERT_NE(backend, nullptr);

    for(gsx_backend_buffer_type_class buffer_class : buffer_classes) {
        gsx_backend_buffer_type_t buffer_type = nullptr;
        gsx_backend_buffer_t src_buffer = nullptr;
        gsx_backend_buffer_t dst_buffer = nullptr;
        gsx_backend_buffer_desc buffer_desc{};
        gsx_backend_tensor_view src_view{};
        gsx_backend_tensor_view dst_view{};
        gsx_backend_tensor_view overlap_dst_view{};
        bool is_finite = false;
        std::array<float, 4> src_values = { 1.0f, 2.0f, 3.0f, 4.0f };
        std::array<float, 4> result_values{};
        std::array<float, 4> expected_filled = { -2.5f, -2.5f, -2.5f, -2.5f };
        std::array<float, 4> expected_zeroed_second = { -2.5f, 0.0f, -2.5f, -2.5f };
        float fill_value = -2.5f;

        SCOPED_TRACE(testing::Message() << "buffer_class=" << static_cast<int>(buffer_class));

        ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, buffer_class, &buffer_type));
        buffer_desc.buffer_type = buffer_type;
        buffer_desc.size_bytes = 64;
        buffer_desc.alignment_bytes = 0;
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&src_buffer, &buffer_desc));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&dst_buffer, &buffer_desc));

        src_view.buffer = src_buffer;
        src_view.offset_bytes = 16;
        src_view.size_bytes = sizeof(src_values);
        src_view.data_type = GSX_DATA_TYPE_F32;
        dst_view.buffer = dst_buffer;
        dst_view.offset_bytes = 32;
        dst_view.size_bytes = sizeof(src_values);
        dst_view.data_type = GSX_DATA_TYPE_F32;

        /* set_tensor / get_tensor roundtrip */
        ASSERT_GSX_SUCCESS(src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), 0, sizeof(src_values)));
        ASSERT_GSX_SUCCESS(src_buffer->iface->get_tensor(src_buffer, &src_view, result_values.data(), 0, sizeof(result_values)));
        EXPECT_EQ(result_values, src_values);

        /* copy_tensor between two host-visible buffers */
        ASSERT_GSX_SUCCESS(dst_buffer->iface->copy_tensor(dst_buffer, &src_view, &dst_view));
        result_values.fill(0.0f);
        ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, result_values.data(), 0, sizeof(result_values)));
        EXPECT_EQ(result_values, src_values);

        /* fill_tensor fills every element with the given value */
        ASSERT_GSX_SUCCESS(dst_buffer->iface->fill_tensor(dst_buffer, &dst_view, &fill_value, sizeof(fill_value)));
        ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, result_values.data(), 0, sizeof(result_values)));
        EXPECT_EQ(result_values, expected_filled);

        /* memset_tensor zeros one element without touching the rest */
        ASSERT_GSX_SUCCESS(dst_buffer->iface->memset_tensor(dst_buffer, &dst_view, 0, sizeof(float), sizeof(float)));
        ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, result_values.data(), 0, sizeof(result_values)));
        EXPECT_EQ(result_values, expected_zeroed_second);

        /* check_finite_tensor with all-finite data */
        ASSERT_GSX_SUCCESS(src_buffer->iface->check_finite_tensor(src_buffer, &src_view, &is_finite));
        EXPECT_TRUE(is_finite);

        /* check_finite_tensor detects NaN and Inf */
        src_values[1] = std::numeric_limits<float>::quiet_NaN();
        src_values[3] = std::numeric_limits<float>::infinity();
        ASSERT_GSX_SUCCESS(src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), 0, sizeof(src_values)));
        ASSERT_GSX_SUCCESS(src_buffer->iface->check_finite_tensor(src_buffer, &src_view, &is_finite));
        EXPECT_FALSE(is_finite);

        /* check_finite_tensor rejects unsupported element types */
        gsx_backend_tensor_view unsupported_view = src_view;
        unsupported_view.data_type = GSX_DATA_TYPE_U8;
        EXPECT_GSX_CODE(src_buffer->iface->check_finite_tensor(src_buffer, &unsupported_view, &is_finite), GSX_ERROR_NOT_SUPPORTED);

        /* copy_tensor rejects overlapping non-identical subranges */
        overlap_dst_view = src_view;
        overlap_dst_view.offset_bytes += sizeof(float);
        EXPECT_GSX_CODE(src_buffer->iface->copy_tensor(src_buffer, &src_view, &overlap_dst_view), GSX_ERROR_INVALID_ARGUMENT);

        /* set_tensor / memset_tensor out of range */
        EXPECT_GSX_CODE(
            src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), sizeof(float), sizeof(src_values)),
            GSX_ERROR_OUT_OF_RANGE
        );
        EXPECT_GSX_CODE(
            src_buffer->iface->memset_tensor(src_buffer, &src_view, 0, sizeof(src_values), 1),
            GSX_ERROR_OUT_OF_RANGE
        );

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(src_buffer));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(dst_buffer));
    }

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalDeviceBufferTensorOpsSubmitOrRejectAppropriately)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t src_buffer = nullptr;
    gsx_backend_buffer_t dst_buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    gsx_backend_tensor_view src_view{};
    gsx_backend_tensor_view dst_view{};
    std::array<float, 4> src_values = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 4> dst_values{};
    bool is_finite = false;
    float scalar_fill = 0.0f;
    uint8_t byte_fill = 0xAB;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = 64;
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&src_buffer, &buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&dst_buffer, &buffer_desc));

    src_view.buffer = src_buffer;
    src_view.offset_bytes = 0;
    src_view.size_bytes = sizeof(src_values);
    src_view.data_type = GSX_DATA_TYPE_F32;
    dst_view.buffer = dst_buffer;
    dst_view.offset_bytes = 0;
    dst_view.size_bytes = sizeof(src_values);
    dst_view.data_type = GSX_DATA_TYPE_F32;

    /* set_tensor and get_tensor submit async work without error */
    ASSERT_GSX_SUCCESS(src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), 0, sizeof(src_values)));
    ASSERT_GSX_SUCCESS(src_buffer->iface->get_tensor(src_buffer, &src_view, dst_values.data(), 0, sizeof(dst_values)));

    /* copy_tensor submits async GPU blit without error */
    ASSERT_GSX_SUCCESS(dst_buffer->iface->copy_tensor(dst_buffer, &src_view, &dst_view));

    /* memset_tensor submits async fill without error */
    ASSERT_GSX_SUCCESS(src_buffer->iface->memset_tensor(src_buffer, &src_view, 0, 0, sizeof(src_values)));

    /* fill_tensor with value_size_bytes == 1 submits async MTLFillBuffer without error */
    ASSERT_GSX_SUCCESS(dst_buffer->iface->fill_tensor(dst_buffer, &dst_view, &byte_fill, sizeof(byte_fill)));

    /* fill_tensor with value_size_bytes > 1 is not supported on device buffers without explicit synchronization */
    EXPECT_GSX_CODE(
        dst_buffer->iface->fill_tensor(dst_buffer, &dst_view, &scalar_fill, sizeof(scalar_fill)),
        GSX_ERROR_NOT_SUPPORTED
    );

    /* check_finite_tensor is not supported on device buffers without explicit synchronization */
    EXPECT_GSX_CODE(src_buffer->iface->check_finite_tensor(src_buffer, &src_view, &is_finite), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(src_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(dst_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalTensorGatherResizeAndExpWork)
{
    gsx_backend_t backend = create_metal_backend();
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

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);

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
    index_gather_desc = make_rank1_tensor_desc(arena, 3, GSX_DATA_TYPE_I32);
    out_gather_desc = make_f32_tensor_desc_with_shape(arena, shape_out_gather, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x_gather, &x_gather_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&index_gather, &index_gather_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_gather, &out_gather_desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x_gather, x_gather_values.data(), sizeof(x_gather_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(index_gather, gather_indices.data(), sizeof(gather_indices)));
    ASSERT_GSX_SUCCESS(gsx_tensor_gather(x_gather, index_gather, out_gather));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out_gather, gathered_values.data(), sizeof(gathered_values)));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    EXPECT_EQ(gathered_values, expected_gathered_values);

    x_resize_desc = make_f32_tensor_desc_with_shape(arena, shape_x_resize, 2);
    out_resize_desc = make_f32_tensor_desc_with_shape(arena, shape_out_resize, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x_resize, &x_resize_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_resize, &out_resize_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x_resize, x_resize_values.data(), sizeof(x_resize_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_resize(x_resize, out_resize));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out_resize, resized_values.data(), sizeof(resized_values)));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    EXPECT_EQ(resized_values, expected_resized_values);

    x_exp_desc = make_rank1_tensor_desc(arena, 4, GSX_DATA_TYPE_F32);
    out_exp_desc = make_rank1_tensor_desc(arena, 4, GSX_DATA_TYPE_F32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x_exp, &x_exp_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_exp, &out_exp_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x_exp, exp_inputs.data(), sizeof(exp_inputs)));
    ASSERT_GSX_SUCCESS(gsx_tensor_exp(x_exp, out_exp));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(out_exp, exp_outputs.data(), sizeof(exp_outputs)));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
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

TEST_F(MetalBackendTest, MetalTensorExpInplaceAndClampInplaceWork)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t exp_tensor = nullptr;
    gsx_tensor_t clamp_f32_tensor = nullptr;
    gsx_tensor_t clamp_i32_tensor = nullptr;
    gsx_tensor_desc exp_tensor_desc{};
    gsx_tensor_desc clamp_f32_tensor_desc{};
    gsx_tensor_desc clamp_i32_tensor_desc{};
    std::array<float, 4> exp_inputs = { 0.0f, 1.0f, -1.0f, 2.0f };
    std::array<float, 4> exp_outputs = {};
    std::array<float, 5> clamp_f32_inputs = { -3.0f, -0.5f, 0.2f, 2.0f, 5.0f };
    std::array<float, 5> clamp_f32_outputs = {};
    std::array<float, 5> expected_f32 = { -1.0f, -0.5f, 0.2f, 1.5f, 1.5f };
    std::array<int32_t, 5> clamp_i32_inputs = { -9, -1, 2, 8, 13 };
    std::array<int32_t, 5> clamp_i32_outputs = {};
    std::array<int32_t, 5> expected_i32 = { -2, -1, 2, 6, 6 };
    float f32_min = -1.0f;
    float f32_max = 1.5f;
    int32_t i32_min = -2;
    int32_t i32_max = 6;

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    exp_tensor_desc = make_rank1_tensor_desc(arena, 4, GSX_DATA_TYPE_F32);
    clamp_f32_tensor_desc = make_rank1_tensor_desc(arena, 5, GSX_DATA_TYPE_F32);
    clamp_i32_tensor_desc = make_rank1_tensor_desc(arena, 5, GSX_DATA_TYPE_I32);

    ASSERT_GSX_SUCCESS(gsx_tensor_init(&exp_tensor, &exp_tensor_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&clamp_f32_tensor, &clamp_f32_tensor_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&clamp_i32_tensor, &clamp_i32_tensor_desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(exp_tensor, exp_inputs.data(), sizeof(exp_inputs)));
    ASSERT_GSX_SUCCESS(gsx_tensor_exp_inplace(exp_tensor));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(exp_tensor, exp_outputs.data(), sizeof(exp_outputs)));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    for(std::size_t i = 0; i < exp_outputs.size(); ++i) {
        EXPECT_NEAR(exp_outputs[i], std::exp(exp_inputs[i]), 1e-6f);
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(clamp_f32_tensor, clamp_f32_inputs.data(), sizeof(clamp_f32_inputs)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(clamp_i32_tensor, clamp_i32_inputs.data(), sizeof(clamp_i32_inputs)));
    ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(clamp_f32_tensor, &f32_min, &f32_max));
    ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(clamp_i32_tensor, &i32_min, &i32_max));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(clamp_f32_tensor, clamp_f32_outputs.data(), sizeof(clamp_f32_outputs)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(clamp_i32_tensor, clamp_i32_outputs.data(), sizeof(clamp_i32_outputs)));
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    EXPECT_EQ(clamp_f32_outputs, expected_f32);
    EXPECT_EQ(clamp_i32_outputs, expected_i32);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(exp_tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(clamp_f32_tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(clamp_i32_tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalTensorUnaryAndClampWorkOnDeviceAndUnified)
{
    std::array<gsx_backend_buffer_type_class, 2> buffer_classes = {
        GSX_BACKEND_BUFFER_TYPE_DEVICE,
        GSX_BACKEND_BUFFER_TYPE_UNIFIED,
    };
    std::array<float, 4> unary_input = { -2.0f, -0.5f, 0.5f, 2.0f };
    std::array<float, 4> unary_output = {};
    std::array<float, 4> unary_expected = {};
    std::array<float, 5> clamp_f32_input = { -3.0f, -0.5f, 0.2f, 2.0f, 5.0f };
    std::array<float, 5> clamp_f32_output = {};
    std::array<float, 5> clamp_f32_expected = { -1.0f, -0.5f, 0.2f, 1.5f, 1.5f };
    std::array<int32_t, 5> clamp_i32_input = { -9, -1, 2, 8, 13 };
    std::array<int32_t, 5> clamp_i32_output = {};
    std::array<int32_t, 5> clamp_i32_expected = { -2, -1, 2, 6, 6 };
    float f32_min = -1.0f;
    float f32_max = 1.5f;
    int32_t i32_min = -2;
    int32_t i32_max = 6;

    for(gsx_backend_buffer_type_class buffer_class : buffer_classes) {
        gsx_backend_t backend = create_metal_backend();
        gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, buffer_class);
        gsx_arena_t arena = nullptr;
        gsx_arena_desc arena_desc{};
        gsx_tensor_t x = nullptr;
        gsx_tensor_t out = nullptr;
        gsx_tensor_t clamp_f32_tensor = nullptr;
        gsx_tensor_t clamp_i32_tensor = nullptr;
        gsx_tensor_desc x_desc{};
        gsx_tensor_desc out_desc{};
        gsx_tensor_desc clamp_f32_desc{};
        gsx_tensor_desc clamp_i32_desc{};

        SCOPED_TRACE(testing::Message() << "buffer_class=" << static_cast<int>(buffer_class));

        ASSERT_NE(backend, nullptr);
        ASSERT_NE(buffer_type, nullptr);

        arena_desc.initial_capacity_bytes = 4096;
        arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
        ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

        x_desc = make_rank1_tensor_desc(arena, 4, GSX_DATA_TYPE_F32);
        out_desc = make_rank1_tensor_desc(arena, 4, GSX_DATA_TYPE_F32);
        clamp_f32_desc = make_rank1_tensor_desc(arena, 5, GSX_DATA_TYPE_F32);
        clamp_i32_desc = make_rank1_tensor_desc(arena, 5, GSX_DATA_TYPE_I32);

        ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
        ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
        ASSERT_GSX_SUCCESS(gsx_tensor_init(&clamp_f32_tensor, &clamp_f32_desc));
        ASSERT_GSX_SUCCESS(gsx_tensor_init(&clamp_i32_tensor, &clamp_i32_desc));

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, unary_input.data(), sizeof(unary_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_exp(x, out));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(out, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            unary_expected[i] = std::exp(unary_input[i]);
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid(x, out));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(out, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            unary_expected[i] = 1.0f / (1.0f + std::exp(-unary_input[i]));
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid_derivative(x, out));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(out, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            const float sigmoid_value = 1.0f / (1.0f + std::exp(-unary_input[i]));

            unary_expected[i] = sigmoid_value * (1.0f - sigmoid_value);
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_abs(x, out));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(out, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            unary_expected[i] = std::fabs(unary_input[i]);
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, unary_input.data(), sizeof(unary_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_exp_inplace(x));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(x, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            unary_expected[i] = std::exp(unary_input[i]);
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, unary_input.data(), sizeof(unary_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid_inplace(x));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(x, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            unary_expected[i] = 1.0f / (1.0f + std::exp(-unary_input[i]));
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, unary_input.data(), sizeof(unary_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_sigmoid_derivative_inplace(x));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(x, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            const float sigmoid_value = 1.0f / (1.0f + std::exp(-unary_input[i]));

            unary_expected[i] = sigmoid_value * (1.0f - sigmoid_value);
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, unary_input.data(), sizeof(unary_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_abs_inplace(x));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(x, unary_output.data(), sizeof(unary_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            unary_expected[i] = std::fabs(unary_input[i]);
        }
        for(std::size_t i = 0; i < unary_output.size(); ++i) {
            EXPECT_NEAR(unary_output[i], unary_expected[i], 1e-6f);
        }

        ASSERT_GSX_SUCCESS(gsx_tensor_upload(clamp_f32_tensor, clamp_f32_input.data(), sizeof(clamp_f32_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_upload(clamp_i32_tensor, clamp_i32_input.data(), sizeof(clamp_i32_input)));
        ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(clamp_f32_tensor, &f32_min, &f32_max));
        ASSERT_GSX_SUCCESS(gsx_tensor_clamp_inplace(clamp_i32_tensor, &i32_min, &i32_max));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(clamp_f32_tensor, clamp_f32_output.data(), sizeof(clamp_f32_output)));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(clamp_i32_tensor, clamp_i32_output.data(), sizeof(clamp_i32_output)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        EXPECT_EQ(clamp_f32_output, clamp_f32_expected);
        EXPECT_EQ(clamp_i32_output, clamp_i32_expected);

        ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
        ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
        ASSERT_GSX_SUCCESS(gsx_tensor_free(clamp_f32_tensor));
        ASSERT_GSX_SUCCESS(gsx_tensor_free(clamp_i32_tensor));
        ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
        ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    }
}

TEST_F(MetalBackendTest, MetalTensorGatherResizeAndExpRejectInvalidContracts)
{
    gsx_backend_t backend = create_metal_backend();
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

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);

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
    out_mismatched_exp_desc = make_rank1_tensor_desc(arena, 3, GSX_DATA_TYPE_F32);
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

TEST_F(MetalBackendTest, MetalTensorGatherRejectsNegativeOutOfRangeIndices)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_t index = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc out_desc{};
    gsx_tensor_desc index_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_x = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> shape_out = {};
    std::array<float, 10> x_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    std::array<int32_t, 3> negative_indices = { -1, 2, 4 };

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);

    shape_x[0] = 5;
    shape_x[1] = 2;
    shape_out[0] = 3;
    shape_out[1] = 2;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena, shape_x, 2);
    out_desc = make_f32_tensor_desc_with_shape(arena, shape_out, 2);
    index_desc = make_rank1_tensor_desc(arena, 3, GSX_DATA_TYPE_I32);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&index, &index_desc));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(x, x_values.data(), sizeof(x_values)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(index, negative_indices.data(), sizeof(negative_indices)));
    EXPECT_GSX_CODE(gsx_tensor_gather(x, index, out), GSX_ERROR_OUT_OF_RANGE);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(index));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalTensorGatherResizeAndExpRejectDryRunTensorStorage)
{
    gsx_backend_t backend = create_metal_backend();
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

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);

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

TEST_F(MetalBackendTest, MetalTensorReduceApisReturnNotSupported)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_arena_t arena = nullptr;
    gsx_arena_t workspace_arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_t x = nullptr;
    gsx_tensor_t target = nullptr;
    gsx_tensor_t out = nullptr;
    gsx_tensor_desc x_desc{};
    gsx_tensor_desc target_desc{};
    gsx_tensor_desc out_desc{};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> x_shape = {};
    std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> out_shape = {};

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(buffer_type, nullptr);

    x_shape[0] = 2;
    x_shape[1] = 3;
    x_shape[2] = 4;
    out_shape[0] = 2;
    out_shape[1] = 1;

    arena_desc.initial_capacity_bytes = 4096;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));
    ASSERT_GSX_SUCCESS(gsx_arena_init(&workspace_arena, buffer_type, &arena_desc));

    x_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 3);
    target_desc = make_f32_tensor_desc_with_shape(arena, x_shape, 3);
    out_desc = make_f32_tensor_desc_with_shape(arena, out_shape, 2);
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&x, &x_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&target, &target_desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out, &out_desc));

    EXPECT_GSX_CODE(gsx_tensor_sum(workspace_arena, x, out, 1), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_tensor_mean(workspace_arena, x, out, 1), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_tensor_max(workspace_arena, x, out, 1), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_tensor_mse(workspace_arena, x, target, out, 1), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_tensor_mae(workspace_arena, x, target, out, 1), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(target));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(x));
    ASSERT_GSX_SUCCESS(gsx_arena_free(workspace_arena));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalDeviceBufferDownloadDeliversDataAfterSync)
{
    /* Regression guard: device buffer download is async. Data is only guaranteed
     * correct in the destination buffer after an explicit gsx_backend_major_stream_sync. */
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t device_buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    std::array<std::uint8_t, 256> upload_bytes{};
    std::array<std::uint8_t, 256> download_bytes{};

    download_bytes.fill(0);
    for(std::size_t i = 0; i < upload_bytes.size(); ++i) {
        upload_bytes[i] = static_cast<std::uint8_t>((i * 37u + 13u) & 0xFFu);
    }

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = upload_bytes.size();
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&device_buffer, &buffer_desc));

    /* Both upload and download are async on device buffers. */
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(device_buffer, 0, upload_bytes.data(), upload_bytes.size()));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(device_buffer, 0, download_bytes.data(), download_bytes.size()));

    /* After draining the major stream the completion handler must have fired. */
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    EXPECT_EQ(download_bytes, upload_bytes);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(device_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalBackendTest, MetalRendererForwardStagingReusedAcrossCallsWithStableResult)
{
    /* Verify that the context-owned staging buffers survive repeated forward calls
     * without crash or result drift — the key property of the lazy-grow reuse path. */
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t buffer_type = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_renderer_t renderer = nullptr;
    gsx_renderer_desc renderer_desc{};
    gsx_render_context_t context = nullptr;
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t rotation = nullptr;
    gsx_tensor_t logscale = nullptr;
    gsx_tensor_t sh0 = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t out_rgb = nullptr;
    gsx_tensor_desc desc{};
    gsx_render_forward_request forward_request{};
    gsx_camera_intrinsics intrinsics{};
    gsx_camera_pose pose{};
    /* 1 Gaussian centred in front of an 8x8 camera */
    std::array<float, 3> mean3d_values   = { 0.0f, 0.0f, 3.0f };
    std::array<float, 4> rotation_values = { 0.0f, 0.0f, 0.0f, 1.0f };
    std::array<float, 3> logscale_values = { -1.0f, -1.0f, -1.0f };
    std::array<float, 3> sh0_values      = { 0.282f, 0.0f, 0.0f };
    std::array<float, 1> opacity_values  = { 2.0f };
    std::array<float, 3 * 8 * 8> out_rgb_values{};
    std::array<float, 3 * 8 * 8> first_output{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &buffer_type));

    arena_desc.initial_capacity_bytes = 65536;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    renderer_desc.width = 8;
    renderer_desc.height = 8;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    renderer_desc.feature_flags = 0;
    ASSERT_GSX_SUCCESS(gsx_renderer_init(&renderer, backend, &renderer_desc));
    ASSERT_GSX_SUCCESS(gsx_render_context_init(&context, renderer));

    desc = {}; desc.rank = 2; desc.shape[0] = 1; desc.shape[1] = 3;
    desc.data_type = GSX_DATA_TYPE_F32; desc.storage_format = GSX_STORAGE_FORMAT_CHW; desc.arena = arena;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&mean3d, &desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(mean3d, mean3d_values.data(), sizeof(mean3d_values)));

    desc.shape[1] = 4;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&rotation, &desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(rotation, rotation_values.data(), sizeof(rotation_values)));

    desc.shape[1] = 3;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&logscale, &desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(logscale, logscale_values.data(), sizeof(logscale_values)));

    ASSERT_GSX_SUCCESS(gsx_tensor_init(&sh0, &desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(sh0, sh0_values.data(), sizeof(sh0_values)));

    desc.rank = 1; desc.shape[0] = 1; desc.shape[1] = 0;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&opacity, &desc));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(opacity, opacity_values.data(), sizeof(opacity_values)));

    desc.rank = 3; desc.shape[0] = 3; desc.shape[1] = 8; desc.shape[2] = 8;
    ASSERT_GSX_SUCCESS(gsx_tensor_init(&out_rgb, &desc));

    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.width = 8; intrinsics.height = 8;
    intrinsics.fx = 8.0f; intrinsics.fy = 8.0f; intrinsics.cx = 4.0f; intrinsics.cy = 4.0f;
    pose = {}; pose.rot.w = 1.0f;

    forward_request.intrinsics = &intrinsics;
    forward_request.pose = &pose;
    forward_request.near_plane = 0.1f;
    forward_request.far_plane = 100.0f;
    forward_request.background_color = gsx_vec3{ 0.0f, 0.0f, 0.0f };
    forward_request.precision = GSX_RENDER_PRECISION_FLOAT32;
    forward_request.sh_degree = 0;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    forward_request.gs_mean3d = mean3d;
    forward_request.gs_rotation = rotation;
    forward_request.gs_logscale = logscale;
    forward_request.gs_sh0 = sh0;
    forward_request.gs_opacity = opacity;
    forward_request.out_rgb = out_rgb;

    /* Three consecutive forwards: staging buffers must be reused and results stable. */
    for(int pass = 0; pass < 3; ++pass) {
        SCOPED_TRACE(testing::Message() << "forward pass=" << pass);
        ASSERT_GSX_SUCCESS(gsx_renderer_render(renderer, context, &forward_request));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        ASSERT_GSX_SUCCESS(gsx_tensor_download(out_rgb, out_rgb_values.data(), sizeof(out_rgb_values)));
        ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
        if(pass == 0) {
            first_output = out_rgb_values;
        } else {
            EXPECT_EQ(out_rgb_values, first_output);
        }
    }

    ASSERT_GSX_SUCCESS(gsx_tensor_free(out_rgb));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(opacity));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(sh0));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(logscale));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(rotation));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(mean3d));
    ASSERT_GSX_SUCCESS(gsx_render_context_free(context));
    ASSERT_GSX_SUCCESS(gsx_renderer_free(renderer));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} /* namespace */
