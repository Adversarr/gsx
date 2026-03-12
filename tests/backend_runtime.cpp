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

TEST(BackendRegistryRuntime, ExplicitInitIsIdempotentAndExposesOneCpuBackendDevice)
{
    gsx_index_t backend_device_count = 0;
    gsx_index_t cpu_backend_device_count = 0;
    gsx_index_t cuda_backend_device_count = 0;
    gsx_backend_device_t backend_device_by_index = nullptr;
    gsx_backend_device_t backend_device_by_type = nullptr;
    gsx_backend_device_info backend_device_info{};

    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices(&backend_device_count));
    ASSERT_EQ(backend_device_count, 1);
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CPU, &cpu_backend_device_count));
    ASSERT_EQ(cpu_backend_device_count, 1);
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CUDA, &cuda_backend_device_count));
    ASSERT_EQ(cuda_backend_device_count, 0);
    ASSERT_GSX_SUCCESS(gsx_get_backend_device(0, &backend_device_by_index));
    ASSERT_GSX_SUCCESS(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &backend_device_by_type));
    ASSERT_EQ(backend_device_by_index, backend_device_by_type);
    ASSERT_GSX_SUCCESS(gsx_backend_device_get_info(backend_device_by_index, &backend_device_info));
    EXPECT_EQ(backend_device_info.backend_type, GSX_BACKEND_TYPE_CPU);
    EXPECT_STREQ(backend_device_info.backend_name, "cpu");
    EXPECT_EQ(backend_device_info.device_index, 0);
    EXPECT_STREQ(backend_device_info.name, "cpu0");
    EXPECT_GSX_CODE(gsx_get_backend_device(1, &backend_device_by_index), GSX_ERROR_OUT_OF_RANGE);
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
    EXPECT_EQ(backend_capabilities.supported_data_types, GSX_DATA_TYPE_FLAG_F32);
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

TEST(BackendRuntime, CpuBackendBufferTypeRoundingMatchesContract)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_size_t alloc_size_bytes = 0;
    gsx_size_t alignment_bytes = 0;
    const char *buffer_type_name = nullptr;
    gsx_backend_t owner_backend = nullptr;
    gsx_backend_buffer_type_class buffer_type_class = GSX_BACKEND_BUFFER_TYPE_HOST;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
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
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_alignment(device_buffer_type, &alignment_bytes));
    EXPECT_EQ(alignment_bytes, 64);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_name(device_buffer_type, &buffer_type_name));
    EXPECT_STREQ(buffer_type_name, "cpu-device");
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_backend(device_buffer_type, &owner_backend));
    EXPECT_EQ(owner_backend, backend);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_type(device_buffer_type, &buffer_type_class));
    EXPECT_EQ(buffer_type_class, GSX_BACKEND_BUFFER_TYPE_DEVICE);

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
    gsx_backend_buffer_t zero_size_buffer = nullptr;
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
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&zero_size_buffer, &zero_size_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_get_info(zero_size_buffer, &buffer_info));
    EXPECT_EQ(buffer_info.size_bytes, 0);
    EXPECT_EQ(buffer_info.alignment_bytes, 64);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(zero_size_buffer, 0, nullptr, 0));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(zero_size_buffer, 0, nullptr, 0));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_set_zero(zero_size_buffer));

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

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(zero_size_buffer));
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

}  // namespace
