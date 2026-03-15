#include "gsx/gsx.h"

#include "../gsx/src/gsx-impl.h"

#include <gtest/gtest.h>

#include <array>
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
    EXPECT_EQ(backend_capabilities.supported_data_types, GSX_DATA_TYPE_FLAG_F32);
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

} /* namespace */
