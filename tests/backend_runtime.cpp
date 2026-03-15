#include "gsx/gsx.h"

#include "../gsx/src/gsx-impl.h"

#include <gtest/gtest.h>

#include <array>
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
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    error = gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device);
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return backend_device;
}

static gsx_backend_t create_cpu_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    backend_desc.device = get_cpu_backend_device();
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

TEST(BackendRegistryRuntime, ExplicitInitIsIdempotentAndExposesCpuBackendDevice)
{
    gsx_index_t backend_device_count = 0;
    gsx_index_t cpu_backend_device_count = 0;
    gsx_backend_device_t backend_device_by_type = nullptr;
    gsx_backend_device_info backend_device_info{};

    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices(&backend_device_count));
    ASSERT_GE(backend_device_count, 1);
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CPU, &cpu_backend_device_count));
    ASSERT_EQ(cpu_backend_device_count, 1);
    ASSERT_GSX_SUCCESS(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device_by_type));
    ASSERT_GSX_SUCCESS(gsx_backend_device_get_info(backend_device_by_type, &backend_device_info));
    EXPECT_EQ(backend_device_info.backend_type, GSX_BACKEND_TYPE_CPU);
    EXPECT_STREQ(backend_device_info.backend_name, "cpu");
    EXPECT_EQ(backend_device_info.device_index, 0);
    EXPECT_STREQ(backend_device_info.name, "cpu0");
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 1, &backend_device_by_type), GSX_ERROR_OUT_OF_RANGE);
}

TEST(BackendRegistryRuntime, CpuBackendDeviceInfoIsStable)
{
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_device_info backend_device_info{};
    gsx_index_t backend_device_count = 0;

    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CPU, &backend_device_count));
    ASSERT_EQ(backend_device_count, 1);
    ASSERT_GSX_SUCCESS(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device));
    ASSERT_GSX_SUCCESS(gsx_backend_device_get_info(backend_device, &backend_device_info));
    EXPECT_EQ(backend_device_info.backend_type, GSX_BACKEND_TYPE_CPU);
    EXPECT_STREQ(backend_device_info.backend_name, "cpu");
    EXPECT_EQ(backend_device_info.device_index, 0);
    EXPECT_STREQ(backend_device_info.name, "cpu0");
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 1, &backend_device), GSX_ERROR_OUT_OF_RANGE);
}

TEST(BackendRuntime, CpuBackendExposesExpectedMetadataAndBufferTypes)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_device_t backend_device = get_cpu_backend_device();
    gsx_backend_info backend_info{};
    gsx_backend_capabilities backend_capabilities{};
    gsx_backend_buffer_type_t host_buffer_type = nullptr;
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_type_info host_buffer_type_info{};
    gsx_backend_buffer_type_info device_buffer_type_info{};
    void *major_stream = reinterpret_cast<void *>(static_cast<std::uintptr_t>(1));
    gsx_index_t buffer_type_count = 0;

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(backend_device, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_get_info(backend, &backend_info));
    EXPECT_EQ(backend_info.backend_type, GSX_BACKEND_TYPE_CPU);
    EXPECT_EQ(backend_info.device, backend_device);

    ASSERT_GSX_SUCCESS(gsx_backend_get_capabilities(backend, &backend_capabilities));
    EXPECT_EQ(
        backend_capabilities.supported_data_types,
        GSX_DATA_TYPE_FLAG_F32 | GSX_DATA_TYPE_FLAG_U8 | GSX_DATA_TYPE_FLAG_I32
    );
    EXPECT_FALSE(backend_capabilities.supports_async_prefetch);

    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &major_stream));
    EXPECT_EQ(major_stream, nullptr);

    ASSERT_GSX_SUCCESS(gsx_backend_count_buffer_types(backend, &buffer_type_count));
    EXPECT_EQ(buffer_type_count, 2);
    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 0, &host_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 1, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, &host_buffer_type), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED, &host_buffer_type), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_backend_get_buffer_type(backend, 2, &host_buffer_type), GSX_ERROR_OUT_OF_RANGE);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(host_buffer_type, &host_buffer_type_info));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(device_buffer_type, &device_buffer_type_info));
    EXPECT_EQ(host_buffer_type_info.backend, backend);
    EXPECT_EQ(host_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_HOST);
    EXPECT_STREQ(host_buffer_type_info.name, "cpu-host");
    EXPECT_EQ(host_buffer_type_info.alignment_bytes, 64);
    EXPECT_EQ(host_buffer_type_info.max_allocation_size_bytes, 0);
    EXPECT_EQ(device_buffer_type_info.backend, backend);
    EXPECT_EQ(device_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    EXPECT_STREQ(device_buffer_type_info.name, "cpu-device");
    EXPECT_EQ(device_buffer_type_info.alignment_bytes, 64);
    EXPECT_EQ(device_buffer_type_info.max_allocation_size_bytes, 0);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendMajorStreamSyncIsSuccessfulNoOp)
{
    gsx_backend_t backend = create_cpu_backend();

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, BackendMajorStreamSyncRejectsNullBackend)
{
    EXPECT_GSX_CODE(gsx_backend_major_stream_sync(nullptr), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(BackendRuntime, CpuBackendBufferTypeRoundingMatchesContract)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_type_info buffer_type_info{};
    gsx_size_t alloc_size_bytes = 0;
    gsx_backend_t owner_backend = nullptr;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(device_buffer_type, &buffer_type_info));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_alloc_size(device_buffer_type, 0, &alloc_size_bytes));
    EXPECT_EQ(alloc_size_bytes, 0);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_alloc_size(device_buffer_type, 1, &alloc_size_bytes));
    EXPECT_EQ(alloc_size_bytes, 64);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_alloc_size(device_buffer_type, 65, &alloc_size_bytes));
    EXPECT_EQ(alloc_size_bytes, 128);
    EXPECT_GSX_CODE(
        gsx_backend_buffer_type_get_alloc_size(device_buffer_type, std::numeric_limits<gsx_size_t>::max(), &alloc_size_bytes),
        GSX_ERROR_OUT_OF_RANGE
    );
    EXPECT_EQ(buffer_type_info.alignment_bytes, 64);
    EXPECT_STREQ(buffer_type_info.name, "cpu-device");
    EXPECT_EQ(buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    EXPECT_EQ(buffer_type_info.max_allocation_size_bytes, 0U);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_backend(device_buffer_type, &owner_backend));
    EXPECT_EQ(owner_backend, backend);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendBuffersSupportRoundTripAndZeroing)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    gsx_backend_buffer_info buffer_info{};
    std::array<std::uint8_t, 16> upload_bytes{};
    std::array<std::uint8_t, 16> download_bytes{};
    std::array<std::uint8_t, 16> zero_bytes{};

    for(std::size_t index = 0; index < upload_bytes.size(); ++index) {
        upload_bytes[index] = static_cast<std::uint8_t>(index + 1);
        zero_bytes[index] = 0;
    }

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));
    buffer_desc.buffer_type = host_buffer_type;
    buffer_desc.size_bytes = upload_bytes.size();
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_get_info(buffer, &buffer_info));
    EXPECT_EQ(buffer_info.backend, backend);
    EXPECT_EQ(buffer_info.buffer_type, host_buffer_type);
    EXPECT_EQ(buffer_info.size_bytes, upload_bytes.size());
    EXPECT_EQ(buffer_info.alignment_bytes, 64);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, upload_bytes.data(), upload_bytes.size()));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
    EXPECT_EQ(download_bytes, upload_bytes);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_set_zero(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
    EXPECT_EQ(download_bytes, zero_bytes);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendBuffersValidateAlignmentLifetimeAndBounds)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t aligned_buffer = nullptr;
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
    aligned_buffer_desc.alignment_bytes = 256;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&aligned_buffer, &aligned_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_get_info(aligned_buffer, &buffer_info));
    EXPECT_EQ(buffer_info.alignment_bytes, 256);

    zero_size_buffer_desc.buffer_type = device_buffer_type;
    zero_size_buffer_desc.size_bytes = 0;
    zero_size_buffer_desc.alignment_bytes = 0;
    EXPECT_GSX_CODE(gsx_backend_buffer_init(&aligned_buffer, &zero_size_buffer_desc), GSX_ERROR_INVALID_ARGUMENT);

    invalid_alignment_desc.buffer_type = device_buffer_type;
    invalid_alignment_desc.size_bytes = 16;
    invalid_alignment_desc.alignment_bytes = 48;
    EXPECT_GSX_CODE(gsx_backend_buffer_init(&aligned_buffer, &invalid_alignment_desc), GSX_ERROR_INVALID_ARGUMENT);

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

TEST(BackendRuntime, CpuBackendBufferFreeReturnsInvalidStateOnLiveCountUnderflow)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));

    buffer_desc.buffer_type = host_buffer_type;
    buffer_desc.size_bytes = 16;
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    backend->live_buffer_count = 0;
    EXPECT_GSX_CODE(gsx_backend_buffer_free(buffer), GSX_ERROR_INVALID_STATE);

    backend->live_buffer_count = 1;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendCreationRejectsUnsupportedOptions)
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};
    int backend_option = 7;

    backend_desc.device = get_cpu_backend_device();
    ASSERT_NE(backend_desc.device, nullptr);
    backend_desc.options = &backend_option;
    backend_desc.options_size_bytes = sizeof(backend_option);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_NOT_SUPPORTED);
}

TEST(BackendRuntime, CpuBackendBufferUploadDownloadRoundtrip)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = nullptr;
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    auto run_roundtrip_test = [&](gsx_backend_buffer_type_t buf_type, gsx_size_t size_bytes, gsx_size_t offset_bytes) {
        SCOPED_TRACE(testing::Message() << "size=" << size_bytes << " offset=" << offset_bytes);
        
        std::vector<std::uint8_t> upload_data(size_bytes);
        std::vector<std::uint8_t> download_data(size_bytes);
        std::vector<std::uint8_t> expected_data(size_bytes);

        for(std::size_t i = 0; i < size_bytes; ++i) {
            upload_data[i] = static_cast<std::uint8_t>((i + 1) & 0xFF);
            expected_data[i] = upload_data[i];
        }

        buffer_desc.buffer_type = buf_type;
        buffer_desc.size_bytes = size_bytes + offset_bytes;
        buffer_desc.alignment_bytes = 0;

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

        if(size_bytes > 0) {
            ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, offset_bytes, upload_data.data(), size_bytes));
            ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, offset_bytes, download_data.data(), size_bytes));
            EXPECT_EQ(download_data, expected_data);
        } else {
            ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, nullptr, 0));
            ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, nullptr, 0));
        }

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
        buffer = nullptr;
    };

    run_roundtrip_test(host_buffer_type, 16, 0);
    run_roundtrip_test(device_buffer_type, 16, 0);
    run_roundtrip_test(host_buffer_type, 4096, 0);
    run_roundtrip_test(device_buffer_type, 4096, 0);
    run_roundtrip_test(host_buffer_type, 64, 64);
    run_roundtrip_test(device_buffer_type, 64, 128);
    run_roundtrip_test(host_buffer_type, 256, 256);

    {
        constexpr gsx_size_t large_size = 1024 * 1024;
        std::vector<std::uint8_t> large_upload(large_size);
        std::vector<std::uint8_t> large_download(large_size);

        for(std::size_t i = 0; i < large_size; ++i) {
            large_upload[i] = static_cast<std::uint8_t>((i & 1) ? 0xAA : 0x55);
        }

        buffer_desc.buffer_type = host_buffer_type;
        buffer_desc.size_bytes = large_size;
        buffer_desc.alignment_bytes = 0;
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, large_upload.data(), large_size));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, large_download.data(), large_size));
        EXPECT_EQ(large_download, large_upload);
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
        buffer = nullptr;
    }

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendBufferAlignmentVerification)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    gsx_backend_buffer_info buffer_info{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));

    auto verify_alignment = [&](gsx_size_t requested_alignment, gsx_size_t expected_alignment) {
        SCOPED_TRACE(testing::Message() << "requested=" << requested_alignment << " expected=" << expected_alignment);
        
        buffer_desc.buffer_type = host_buffer_type;
        buffer_desc.size_bytes = 128;
        buffer_desc.alignment_bytes = requested_alignment;

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_get_info(buffer, &buffer_info));
        EXPECT_EQ(buffer_info.alignment_bytes, expected_alignment);

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
        buffer = nullptr;
    };

    verify_alignment(0, 64);
    verify_alignment(64, 64);
    verify_alignment(128, 128);
    verify_alignment(256, 256);
    verify_alignment(512, 512);
    verify_alignment(1024, 1024);
    verify_alignment(4096, 4096);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendBufferLiveBufferSafety)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = nullptr;
    gsx_backend_buffer_t buffer1 = nullptr;
    gsx_backend_buffer_t buffer2 = nullptr;
    gsx_backend_buffer_t buffer3 = nullptr;
    gsx_backend_buffer_desc buffer_desc{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));

    buffer_desc.buffer_type = host_buffer_type;
    buffer_desc.size_bytes = 64;
    buffer_desc.alignment_bytes = 0;

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer1, &buffer_desc));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer1));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));

    backend = create_cpu_backend();
    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));
    buffer_desc.buffer_type = host_buffer_type;

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer1, &buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer2, &buffer_desc));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer1));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer2));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));

    backend = create_cpu_backend();
    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));
    buffer_desc.buffer_type = host_buffer_type;

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer1, &buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer2, &buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer3, &buffer_desc));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer2));
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer1));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer3));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));

    for(int cycle = 0; cycle < 5; ++cycle) {
        backend = create_cpu_backend();
        ASSERT_NE(backend, nullptr);
        ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &host_buffer_type));
        buffer_desc.buffer_type = host_buffer_type;

        gsx_backend_buffer_t buffers[5] = {};
        for(int i = 0; i < 5; ++i) {
            ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffers[i], &buffer_desc));
        }
        for(int i = 0; i < 5; ++i) {
            ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffers[i]));
        }
        ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    }
}

TEST(BackendRuntime, CpuBackendFreeRequiresArenaRelease)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    arena_desc.initial_capacity_bytes = 0;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    arena_desc.dry_run = true;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, device_buffer_type, &arena_desc));
    EXPECT_EQ(backend->live_arena_count, 1U);
    EXPECT_EQ(device_buffer_type->live_arena_count, 1U);
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    EXPECT_EQ(backend->live_arena_count, 0U);
    EXPECT_EQ(device_buffer_type->live_arena_count, 0U);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));

    backend = create_cpu_backend();
    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    arena_desc.initial_capacity_bytes = 128;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    arena_desc.dry_run = false;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, device_buffer_type, &arena_desc));
    EXPECT_EQ(backend->live_arena_count, 1U);
    EXPECT_EQ(device_buffer_type->live_arena_count, 1U);
    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    EXPECT_EQ(backend->live_arena_count, 0U);
    EXPECT_EQ(device_buffer_type->live_arena_count, 0U);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(BackendRuntime, CpuBackendTensorHooksSupportSubrangesCopyFillAndFiniteCheck)
{
    gsx_backend_t backend = create_cpu_backend();
    std::array<gsx_backend_buffer_type_class, 2> buffer_classes = {
        GSX_BACKEND_BUFFER_TYPE_HOST,
        GSX_BACKEND_BUFFER_TYPE_DEVICE,
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
        gsx_backend_tensor_view unsupported_view{};
        bool is_finite = false;
        std::array<float, 4> src_values = { 1.0f, 2.0f, 3.0f, 4.0f };
        std::array<float, 4> roundtrip_values = {};
        std::array<float, 4> filled_values = {};
        std::array<float, 4> expected_filled = { -2.5f, -2.5f, -2.5f, -2.5f };
        std::array<float, 4> expected_memset = { -2.5f, 0.0f, -2.5f, -2.5f };
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

        ASSERT_GSX_SUCCESS(src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), 0, sizeof(src_values)));
        ASSERT_GSX_SUCCESS(src_buffer->iface->get_tensor(src_buffer, &src_view, roundtrip_values.data(), 0, sizeof(roundtrip_values)));
        EXPECT_EQ(roundtrip_values, src_values);

        EXPECT_GSX_CODE(
            src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), sizeof(float), sizeof(src_values)),
            GSX_ERROR_OUT_OF_RANGE
        );
        EXPECT_GSX_CODE(
            src_buffer->iface->memset_tensor(src_buffer, &src_view, 0, sizeof(src_values), 1),
            GSX_ERROR_OUT_OF_RANGE
        );

        ASSERT_GSX_SUCCESS(dst_buffer->iface->copy_tensor(dst_buffer, &src_view, &dst_view));
        roundtrip_values.fill(0.0f);
        ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, roundtrip_values.data(), 0, sizeof(roundtrip_values)));
        EXPECT_EQ(roundtrip_values, src_values);

        ASSERT_GSX_SUCCESS(dst_buffer->iface->fill_tensor(dst_buffer, &dst_view, &fill_value, sizeof(fill_value)));
        ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, filled_values.data(), 0, sizeof(filled_values)));
        EXPECT_EQ(filled_values, expected_filled);

        ASSERT_GSX_SUCCESS(dst_buffer->iface->memset_tensor(dst_buffer, &dst_view, 0, sizeof(float), sizeof(float)));
        ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, filled_values.data(), 0, sizeof(filled_values)));
        EXPECT_EQ(filled_values, expected_memset);

        ASSERT_GSX_SUCCESS(src_buffer->iface->check_finite_tensor(src_buffer, &src_view, &is_finite));
        EXPECT_TRUE(is_finite);

        src_values[1] = std::numeric_limits<float>::quiet_NaN();
        src_values[3] = std::numeric_limits<float>::infinity();
        ASSERT_GSX_SUCCESS(src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), 0, sizeof(src_values)));
        ASSERT_GSX_SUCCESS(src_buffer->iface->check_finite_tensor(src_buffer, &src_view, &is_finite));
        EXPECT_FALSE(is_finite);

        unsupported_view = src_view;
        unsupported_view.data_type = GSX_DATA_TYPE_U8;
        EXPECT_GSX_CODE(src_buffer->iface->check_finite_tensor(src_buffer, &unsupported_view, &is_finite), GSX_ERROR_NOT_SUPPORTED);

        overlap_dst_view = src_view;
        overlap_dst_view.offset_bytes += sizeof(float);
        EXPECT_GSX_CODE(src_buffer->iface->copy_tensor(src_buffer, &src_view, &overlap_dst_view), GSX_ERROR_INVALID_ARGUMENT);
        ASSERT_GSX_SUCCESS(src_buffer->iface->copy_tensor(src_buffer, &src_view, &src_view));

        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(src_buffer));
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(dst_buffer));
    }

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
