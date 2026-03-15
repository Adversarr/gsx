#include "gsx/gsx.h"

#include "../gsx/src/gsx-impl.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

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

static bool has_cuda_device()
{
    gsx_index_t cuda_device_count = 0;
    gsx_error error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CUDA, &cuda_device_count);
    return error.code == GSX_ERROR_SUCCESS && cuda_device_count > 0;
}

static gsx_backend_device_t get_cuda_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    error = gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CUDA, 0, &backend_device);
    if(error.code != GSX_ERROR_SUCCESS) {
        ADD_FAILURE() << (error.message != nullptr ? error.message : "");
        return nullptr;
    }
    return backend_device;
}

static gsx_backend_t create_cuda_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    backend_desc.device = get_cuda_backend_device();
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

class CudaBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        if(!has_cuda_device()) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CudaBackendTest, CudaDeviceDiscoveryExposesAtLeastOneDevice)
{
    gsx_index_t cuda_device_count = 0;
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_device_info device_info{};

    ASSERT_GSX_SUCCESS(gsx_backend_registry_init());
    ASSERT_GSX_SUCCESS(gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CUDA, &cuda_device_count));
    ASSERT_GE(cuda_device_count, 1);

    ASSERT_GSX_SUCCESS(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CUDA, 0, &backend_device));
    ASSERT_GSX_SUCCESS(gsx_backend_device_get_info(backend_device, &device_info));
    EXPECT_EQ(device_info.backend_type, GSX_BACKEND_TYPE_CUDA);
    EXPECT_STREQ(device_info.backend_name, "cuda");
    EXPECT_EQ(device_info.device_index, 0);
    EXPECT_NE(device_info.name, nullptr);
    EXPECT_GT(device_info.total_memory_bytes, 0U);
}

TEST_F(CudaBackendTest, CudaBackendExposesExpectedMetadataAndBufferTypes)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_device_t backend_device = get_cuda_backend_device();
    gsx_backend_info backend_info{};
    gsx_backend_capabilities backend_capabilities{};
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_type_t host_pinned_buffer_type = nullptr;
    gsx_backend_buffer_type_info device_buffer_type_info{};
    gsx_backend_buffer_type_info host_pinned_buffer_type_info{};
    void *major_stream = nullptr;
    gsx_index_t buffer_type_count = 0;

    ASSERT_NE(backend, nullptr);
    ASSERT_NE(backend_device, nullptr);

    ASSERT_GSX_SUCCESS(gsx_backend_get_info(backend, &backend_info));
    EXPECT_EQ(backend_info.backend_type, GSX_BACKEND_TYPE_CUDA);
    EXPECT_EQ(backend_info.device, backend_device);

    ASSERT_GSX_SUCCESS(gsx_backend_get_capabilities(backend, &backend_capabilities));
    EXPECT_EQ(backend_capabilities.supported_data_types, GSX_DATA_TYPE_FLAG_F32);
    EXPECT_TRUE(backend_capabilities.supports_async_prefetch);

    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &major_stream));
    EXPECT_NE(major_stream, nullptr);

    ASSERT_GSX_SUCCESS(gsx_backend_count_buffer_types(backend, &buffer_type_count));
    EXPECT_EQ(buffer_type_count, 2);

    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 0, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_buffer_type(backend, 1, &host_pinned_buffer_type));

    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, &host_pinned_buffer_type));

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &device_buffer_type), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_UNIFIED, &device_buffer_type), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_GSX_CODE(gsx_backend_get_buffer_type(backend, 2, &device_buffer_type), GSX_ERROR_OUT_OF_RANGE);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(device_buffer_type, &device_buffer_type_info));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_type_get_info(host_pinned_buffer_type, &host_pinned_buffer_type_info));

    EXPECT_EQ(device_buffer_type_info.backend, backend);
    EXPECT_EQ(device_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    EXPECT_STREQ(device_buffer_type_info.name, "device");
    EXPECT_EQ(device_buffer_type_info.alignment_bytes, 256U);

    EXPECT_EQ(host_pinned_buffer_type_info.backend, backend);
    EXPECT_EQ(host_pinned_buffer_type_info.type, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED);
    EXPECT_STREQ(host_pinned_buffer_type_info.name, "host_pinned");
    EXPECT_EQ(host_pinned_buffer_type_info.alignment_bytes, 64U);

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendDeviceBufferUploadDownloadRoundtrip)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    std::array<std::uint8_t, 256> upload_bytes{};
    std::array<std::uint8_t, 256> download_bytes{};

    for(std::size_t i = 0; i < upload_bytes.size(); ++i) {
        upload_bytes[i] = static_cast<std::uint8_t>(i);
    }

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = upload_bytes.size();
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, upload_bytes.data(), upload_bytes.size()));

    void *stream = nullptr;
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    EXPECT_EQ(download_bytes, upload_bytes);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendHostPinnedBufferUploadDownloadRoundtrip)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t host_pinned_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    std::array<std::uint8_t, 256> upload_bytes{};
    std::array<std::uint8_t, 256> download_bytes{};

    for(std::size_t i = 0; i < upload_bytes.size(); ++i) {
        upload_bytes[i] = static_cast<std::uint8_t>(i);
    }

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, &host_pinned_buffer_type));

    buffer_desc.buffer_type = host_pinned_buffer_type;
    buffer_desc.size_bytes = upload_bytes.size();
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, upload_bytes.data(), upload_bytes.size()));

    void *stream = nullptr;
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    EXPECT_EQ(download_bytes, upload_bytes);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendBuffersValidateMetadataAndAlignment)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t aligned_buffer = nullptr;
    gsx_backend_buffer_t rejected_buffer = nullptr;
    gsx_backend_buffer_desc aligned_buffer_desc{};
    gsx_backend_buffer_desc zero_size_buffer_desc{};
    gsx_backend_buffer_desc invalid_alignment_desc{};
    gsx_backend_buffer_info buffer_info{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    aligned_buffer_desc.buffer_type = device_buffer_type;
    aligned_buffer_desc.size_bytes = 16;
    aligned_buffer_desc.alignment_bytes = 512;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&aligned_buffer, &aligned_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_get_info(aligned_buffer, &buffer_info));
    EXPECT_EQ(buffer_info.backend, backend);
    EXPECT_EQ(buffer_info.buffer_type, device_buffer_type);
    EXPECT_EQ(buffer_info.size_bytes, 16U);
    EXPECT_EQ(buffer_info.alignment_bytes, 512U);

    zero_size_buffer_desc.buffer_type = device_buffer_type;
    zero_size_buffer_desc.size_bytes = 0;
    zero_size_buffer_desc.alignment_bytes = 0;
    EXPECT_GSX_CODE(gsx_backend_buffer_init(&rejected_buffer, &zero_size_buffer_desc), GSX_ERROR_INVALID_ARGUMENT);

    invalid_alignment_desc.buffer_type = device_buffer_type;
    invalid_alignment_desc.size_bytes = 16;
    invalid_alignment_desc.alignment_bytes = 48;
    EXPECT_GSX_CODE(gsx_backend_buffer_init(&rejected_buffer, &invalid_alignment_desc), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(aligned_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendBufferSetZero)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    std::array<std::uint8_t, 256> upload_bytes{};
    std::array<std::uint8_t, 256> download_bytes{};
    std::array<std::uint8_t, 256> zero_bytes{};

    for(std::size_t i = 0; i < upload_bytes.size(); ++i) {
        upload_bytes[i] = static_cast<std::uint8_t>(i + 1);
    }

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = upload_bytes.size();
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(buffer, 0, upload_bytes.data(), upload_bytes.size()));

    void *stream = nullptr;
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_set_zero(buffer));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_download(buffer, 0, download_bytes.data(), download_bytes.size()));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    EXPECT_EQ(download_bytes, zero_bytes);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorFill)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    gsx_backend_tensor_view tensor_view{};
    std::array<float, 8> download_values{};
    float fill_value = 3.14159f;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = sizeof(float) * 8;
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    tensor_view.buffer = buffer;
    tensor_view.offset_bytes = 0;
    tensor_view.size_bytes = sizeof(float) * 8;
    tensor_view.data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(buffer->iface->fill_tensor(buffer, &tensor_view, &fill_value, sizeof(fill_value)));

    void *stream = nullptr;
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(buffer->iface->get_tensor(buffer, &tensor_view, download_values.data(), 0, sizeof(download_values)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    for(std::size_t i = 0; i < download_values.size(); ++i) {
        EXPECT_FLOAT_EQ(download_values[i], fill_value);
    }

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorCheckFinite)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    gsx_backend_tensor_view tensor_view{};
    std::array<float, 4> finite_values = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 4> non_finite_values = { 1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, std::numeric_limits<float>::infinity() };
    std::array<std::uint16_t, 4> finite_values_f16 = { 0x3C00u, 0x4000u, 0x4200u, 0x4400u };
    std::array<std::uint16_t, 4> non_finite_values_f16 = { 0x3C00u, 0x7C00u, 0x4200u, 0xFC00u };
    std::array<std::uint16_t, 4> finite_values_bf16 = { 0x3F80u, 0x4000u, 0x4040u, 0x4080u };
    std::array<std::uint16_t, 4> non_finite_values_bf16 = { 0x3F80u, 0x7F80u, 0x4040u, 0xFF80u };
    bool is_finite = false;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = sizeof(float) * 4;
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    tensor_view.buffer = buffer;
    tensor_view.offset_bytes = 0;
    tensor_view.size_bytes = sizeof(float) * 4;
    tensor_view.data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(buffer->iface->set_tensor(buffer, &tensor_view, finite_values.data(), 0, sizeof(finite_values)));

    void *stream = nullptr;
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(buffer->iface->check_finite_tensor(buffer, &tensor_view, &is_finite));
    EXPECT_TRUE(is_finite);

    ASSERT_GSX_SUCCESS(buffer->iface->set_tensor(buffer, &tensor_view, non_finite_values.data(), 0, sizeof(non_finite_values)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(buffer->iface->check_finite_tensor(buffer, &tensor_view, &is_finite));
    EXPECT_FALSE(is_finite);

    tensor_view.size_bytes = sizeof(finite_values_f16);
    tensor_view.data_type = GSX_DATA_TYPE_F16;
    ASSERT_GSX_SUCCESS(buffer->iface->set_tensor(buffer, &tensor_view, finite_values_f16.data(), 0, sizeof(finite_values_f16)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    ASSERT_GSX_SUCCESS(buffer->iface->check_finite_tensor(buffer, &tensor_view, &is_finite));
    EXPECT_TRUE(is_finite);

    ASSERT_GSX_SUCCESS(buffer->iface->set_tensor(buffer, &tensor_view, non_finite_values_f16.data(), 0, sizeof(non_finite_values_f16)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    ASSERT_GSX_SUCCESS(buffer->iface->check_finite_tensor(buffer, &tensor_view, &is_finite));
    EXPECT_FALSE(is_finite);

    tensor_view.size_bytes = sizeof(finite_values_bf16);
    tensor_view.data_type = GSX_DATA_TYPE_BF16;
    ASSERT_GSX_SUCCESS(buffer->iface->set_tensor(buffer, &tensor_view, finite_values_bf16.data(), 0, sizeof(finite_values_bf16)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    ASSERT_GSX_SUCCESS(buffer->iface->check_finite_tensor(buffer, &tensor_view, &is_finite));
    EXPECT_TRUE(is_finite);

    ASSERT_GSX_SUCCESS(buffer->iface->set_tensor(buffer, &tensor_view, non_finite_values_bf16.data(), 0, sizeof(non_finite_values_bf16)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    ASSERT_GSX_SUCCESS(buffer->iface->check_finite_tensor(buffer, &tensor_view, &is_finite));
    EXPECT_FALSE(is_finite);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorCopy)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t src_buffer = nullptr;
    gsx_backend_buffer_t dst_buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};
    gsx_backend_tensor_view src_view{};
    gsx_backend_tensor_view dst_view{};
    std::array<float, 4> src_values = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 4> dst_values = {};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = sizeof(float) * 4;
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&src_buffer, &buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&dst_buffer, &buffer_desc));

    src_view.buffer = src_buffer;
    src_view.offset_bytes = 0;
    src_view.size_bytes = sizeof(float) * 4;
    src_view.data_type = GSX_DATA_TYPE_F32;

    dst_view.buffer = dst_buffer;
    dst_view.offset_bytes = 0;
    dst_view.size_bytes = sizeof(float) * 4;
    dst_view.data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(src_buffer->iface->set_tensor(src_buffer, &src_view, src_values.data(), 0, sizeof(src_values)));

    void *stream = nullptr;
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(dst_buffer->iface->copy_tensor(dst_buffer, &src_view, &dst_view));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(dst_buffer->iface->get_tensor(dst_buffer, &dst_view, dst_values.data(), 0, sizeof(dst_values)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    for(std::size_t i = 0; i < src_values.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_values[i], src_values[i]);
    }

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(src_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(dst_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorCopyRejectsMismatchedSizes)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t src_buffer = nullptr;
    gsx_backend_buffer_t dst_buffer = nullptr;
    gsx_backend_buffer_desc src_buffer_desc{};
    gsx_backend_buffer_desc dst_buffer_desc{};
    gsx_backend_tensor_view src_view{};
    gsx_backend_tensor_view dst_view{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    src_buffer_desc.buffer_type = device_buffer_type;
    src_buffer_desc.size_bytes = 64;
    src_buffer_desc.alignment_bytes = 0;
    dst_buffer_desc.buffer_type = device_buffer_type;
    dst_buffer_desc.size_bytes = 32;
    dst_buffer_desc.alignment_bytes = 0;

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&src_buffer, &src_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&dst_buffer, &dst_buffer_desc));

    src_view.buffer = src_buffer;
    src_view.offset_bytes = 0;
    src_view.size_bytes = 64;
    src_view.data_type = GSX_DATA_TYPE_F32;

    dst_view.buffer = dst_buffer;
    dst_view.offset_bytes = 0;
    dst_view.size_bytes = 32;
    dst_view.data_type = GSX_DATA_TYPE_F32;

    EXPECT_GSX_CODE(dst_buffer->iface->copy_tensor(dst_buffer, &src_view, &dst_view), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(src_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(dst_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorGatherDeviceBuffer)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t x_buffer = nullptr;
    gsx_backend_buffer_t index_buffer = nullptr;
    gsx_backend_buffer_t out_buffer = nullptr;
    gsx_backend_buffer_desc x_buffer_desc{};
    gsx_backend_buffer_desc index_buffer_desc{};
    gsx_backend_buffer_desc out_buffer_desc{};
    gsx_backend_tensor_view x_view{};
    gsx_backend_tensor_view index_view{};
    gsx_backend_tensor_view out_view{};
    std::array<float, 10> x_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    std::array<int32_t, 3> indices = { 4, 1, 3 };
    std::array<float, 6> gathered_values = {};
    std::array<float, 6> expected_values = { 9.0f, 10.0f, 3.0f, 4.0f, 7.0f, 8.0f };
    std::array<gsx_index_t, 2> x_shape = { 5, 2 };
    std::array<gsx_index_t, 2> out_shape = { 3, 2 };
    void *stream = nullptr;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));

    x_buffer_desc.buffer_type = device_buffer_type;
    x_buffer_desc.size_bytes = sizeof(x_values);
    x_buffer_desc.alignment_bytes = 0;
    index_buffer_desc.buffer_type = device_buffer_type;
    index_buffer_desc.size_bytes = sizeof(indices);
    index_buffer_desc.alignment_bytes = 0;
    out_buffer_desc.buffer_type = device_buffer_type;
    out_buffer_desc.size_bytes = sizeof(gathered_values);
    out_buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&x_buffer, &x_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&index_buffer, &index_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&out_buffer, &out_buffer_desc));

    x_view.buffer = x_buffer;
    x_view.offset_bytes = 0;
    x_view.size_bytes = sizeof(x_values);
    x_view.data_type = GSX_DATA_TYPE_F32;
    index_view.buffer = index_buffer;
    index_view.offset_bytes = 0;
    index_view.size_bytes = sizeof(indices);
    index_view.data_type = GSX_DATA_TYPE_I32;
    out_view.buffer = out_buffer;
    out_view.offset_bytes = 0;
    out_view.size_bytes = sizeof(gathered_values);
    out_view.data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(x_buffer->iface->set_tensor(x_buffer, &x_view, x_values.data(), 0, sizeof(x_values)));
    ASSERT_GSX_SUCCESS(index_buffer->iface->set_tensor(index_buffer, &index_view, indices.data(), 0, sizeof(indices)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(
        out_buffer->iface->gather_tensor(out_buffer, &x_view, &index_view, &out_view, 2, x_shape.data(), 2, out_shape.data())
    );
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(out_buffer->iface->get_tensor(out_buffer, &out_view, gathered_values.data(), 0, sizeof(gathered_values)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    EXPECT_EQ(gathered_values, expected_values);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(x_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(index_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(out_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorGatherHostPinnedBuffer)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t host_pinned_buffer_type = nullptr;
    gsx_backend_buffer_t x_buffer = nullptr;
    gsx_backend_buffer_t index_buffer = nullptr;
    gsx_backend_buffer_t out_buffer = nullptr;
    gsx_backend_buffer_desc x_buffer_desc{};
    gsx_backend_buffer_desc index_buffer_desc{};
    gsx_backend_buffer_desc out_buffer_desc{};
    gsx_backend_tensor_view x_view{};
    gsx_backend_tensor_view index_view{};
    gsx_backend_tensor_view out_view{};
    std::array<float, 10> x_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    std::array<int32_t, 3> indices = { 4, 1, 3 };
    std::array<float, 6> gathered_values = {};
    std::array<float, 6> expected_values = { 9.0f, 10.0f, 3.0f, 4.0f, 7.0f, 8.0f };
    std::array<gsx_index_t, 2> x_shape = { 5, 2 };
    std::array<gsx_index_t, 2> out_shape = { 3, 2 };
    void *stream = nullptr;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED, &host_pinned_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));

    x_buffer_desc.buffer_type = host_pinned_buffer_type;
    x_buffer_desc.size_bytes = sizeof(x_values);
    x_buffer_desc.alignment_bytes = 0;
    index_buffer_desc.buffer_type = host_pinned_buffer_type;
    index_buffer_desc.size_bytes = sizeof(indices);
    index_buffer_desc.alignment_bytes = 0;
    out_buffer_desc.buffer_type = host_pinned_buffer_type;
    out_buffer_desc.size_bytes = sizeof(gathered_values);
    out_buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&x_buffer, &x_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&index_buffer, &index_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&out_buffer, &out_buffer_desc));

    x_view.buffer = x_buffer;
    x_view.offset_bytes = 0;
    x_view.size_bytes = sizeof(x_values);
    x_view.data_type = GSX_DATA_TYPE_F32;
    index_view.buffer = index_buffer;
    index_view.offset_bytes = 0;
    index_view.size_bytes = sizeof(indices);
    index_view.data_type = GSX_DATA_TYPE_I32;
    out_view.buffer = out_buffer;
    out_view.offset_bytes = 0;
    out_view.size_bytes = sizeof(gathered_values);
    out_view.data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(x_buffer->iface->set_tensor(x_buffer, &x_view, x_values.data(), 0, sizeof(x_values)));
    ASSERT_GSX_SUCCESS(index_buffer->iface->set_tensor(index_buffer, &index_view, indices.data(), 0, sizeof(indices)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    ASSERT_GSX_SUCCESS(
        out_buffer->iface->gather_tensor(out_buffer, &x_view, &index_view, &out_view, 2, x_shape.data(), 2, out_shape.data())
    );

    ASSERT_GSX_SUCCESS(out_buffer->iface->get_tensor(out_buffer, &out_view, gathered_values.data(), 0, sizeof(gathered_values)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    EXPECT_EQ(gathered_values, expected_values);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(x_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(index_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(out_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendTensorGatherRejectsInvalidContracts)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t x_buffer = nullptr;
    gsx_backend_buffer_t index_buffer = nullptr;
    gsx_backend_buffer_t out_buffer = nullptr;
    gsx_backend_buffer_t out_mismatched_tail_buffer = nullptr;
    gsx_backend_buffer_desc x_buffer_desc{};
    gsx_backend_buffer_desc index_buffer_desc{};
    gsx_backend_buffer_desc out_buffer_desc{};
    gsx_backend_buffer_desc out_mismatched_tail_desc{};
    gsx_backend_tensor_view x_view{};
    gsx_backend_tensor_view index_view{};
    gsx_backend_tensor_view out_view{};
    gsx_backend_tensor_view out_mismatched_tail_view{};
    std::array<float, 10> x_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    std::array<int32_t, 3> out_of_range_indices = { 0, 2, 5 };
    std::array<int32_t, 3> in_range_indices = { 0, 2, 4 };
    std::array<gsx_index_t, 2> x_shape = { 5, 2 };
    std::array<gsx_index_t, 2> out_shape = { 3, 2 };
    std::array<gsx_index_t, 2> out_mismatched_tail_shape = { 3, 3 };
    void *stream = nullptr;

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));
    ASSERT_GSX_SUCCESS(gsx_backend_get_major_stream(backend, &stream));

    x_buffer_desc.buffer_type = device_buffer_type;
    x_buffer_desc.size_bytes = sizeof(x_values);
    x_buffer_desc.alignment_bytes = 0;
    index_buffer_desc.buffer_type = device_buffer_type;
    index_buffer_desc.size_bytes = sizeof(out_of_range_indices);
    index_buffer_desc.alignment_bytes = 0;
    out_buffer_desc.buffer_type = device_buffer_type;
    out_buffer_desc.size_bytes = sizeof(float) * 6;
    out_buffer_desc.alignment_bytes = 0;
    out_mismatched_tail_desc.buffer_type = device_buffer_type;
    out_mismatched_tail_desc.size_bytes = sizeof(float) * 9;
    out_mismatched_tail_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&x_buffer, &x_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&index_buffer, &index_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&out_buffer, &out_buffer_desc));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&out_mismatched_tail_buffer, &out_mismatched_tail_desc));

    x_view.buffer = x_buffer;
    x_view.offset_bytes = 0;
    x_view.size_bytes = sizeof(x_values);
    x_view.data_type = GSX_DATA_TYPE_F32;
    index_view.buffer = index_buffer;
    index_view.offset_bytes = 0;
    index_view.size_bytes = sizeof(out_of_range_indices);
    index_view.data_type = GSX_DATA_TYPE_I32;
    out_view.buffer = out_buffer;
    out_view.offset_bytes = 0;
    out_view.size_bytes = sizeof(float) * 6;
    out_view.data_type = GSX_DATA_TYPE_F32;
    out_mismatched_tail_view.buffer = out_mismatched_tail_buffer;
    out_mismatched_tail_view.offset_bytes = 0;
    out_mismatched_tail_view.size_bytes = sizeof(float) * 9;
    out_mismatched_tail_view.data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(x_buffer->iface->set_tensor(x_buffer, &x_view, x_values.data(), 0, sizeof(x_values)));
    ASSERT_GSX_SUCCESS(index_buffer->iface->set_tensor(index_buffer, &index_view, out_of_range_indices.data(), 0, sizeof(out_of_range_indices)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    EXPECT_GSX_CODE(
        out_buffer->iface->gather_tensor(out_buffer, &x_view, &index_view, &out_view, 2, x_shape.data(), 2, out_shape.data()),
        GSX_ERROR_OUT_OF_RANGE
    );

    ASSERT_GSX_SUCCESS(index_buffer->iface->set_tensor(index_buffer, &index_view, in_range_indices.data(), 0, sizeof(in_range_indices)));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));

    EXPECT_GSX_CODE(
        out_mismatched_tail_buffer->iface->gather_tensor(
            out_mismatched_tail_buffer,
            &x_view,
            &index_view,
            &out_mismatched_tail_view,
            2,
            x_shape.data(),
            2,
            out_mismatched_tail_shape.data()
        ),
        GSX_ERROR_INVALID_ARGUMENT
    );

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(x_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(index_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(out_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(out_mismatched_tail_buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(CudaBackendTest, CudaBackendCreationRejectsUnsupportedOptions)
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};
    int backend_option = 7;

    backend_desc.device = get_cuda_backend_device();
    ASSERT_NE(backend_desc.device, nullptr);
    backend_desc.options = &backend_option;
    backend_desc.options_size_bytes = sizeof(backend_option);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_NOT_SUPPORTED);

    backend_desc.options = nullptr;
    backend_desc.options_size_bytes = sizeof(backend_option);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_INVALID_ARGUMENT);
}

TEST_F(CudaBackendTest, CudaBackendBufferLifetimeValidation)
{
    gsx_backend_t backend = create_cuda_backend();
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};

    ASSERT_NE(backend, nullptr);
    ASSERT_GSX_SUCCESS(gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type));

    buffer_desc.buffer_type = device_buffer_type;
    buffer_desc.size_bytes = 64;
    buffer_desc.alignment_bytes = 0;
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&buffer, &buffer_desc));

    EXPECT_GSX_CODE(gsx_backend_free(backend), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(buffer));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
