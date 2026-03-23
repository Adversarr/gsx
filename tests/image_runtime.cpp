#include "gsx/gsx.h"
#include "gsx/extra/gsx-image.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
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

static gsx_backend_t create_backend_by_type(gsx_backend_type backend_type)
{
    gsx_backend_t backend = nullptr;
    gsx_backend_device_t device = nullptr;
    gsx_backend_desc desc{};

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    if(gsx_get_backend_device_by_type(backend_type, 0, &device).code != GSX_ERROR_SUCCESS) {
        return nullptr;
    }
    desc.device = device;
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &desc), GSX_ERROR_SUCCESS);
    return backend;
}

static bool has_backend_device(gsx_backend_type backend_type)
{
    gsx_index_t device_count = 0;
    if(gsx_backend_registry_init().code != GSX_ERROR_SUCCESS) {
        return false;
    }
    return gsx_count_backend_devices_by_type(backend_type, &device_count).code == GSX_ERROR_SUCCESS && device_count > 0;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;
    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, type, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static bool backend_supports_u8(gsx_backend_t backend)
{
    gsx_backend_capabilities capabilities{};
    if(gsx_backend_get_capabilities(backend, &capabilities).code != GSX_ERROR_SUCCESS) {
        return false;
    }
    return (capabilities.supported_data_types & GSX_DATA_TYPE_FLAG_U8) != 0;
}

static gsx_tensor_t make_image_tensor(
    gsx_backend_buffer_type_t buffer_type,
    gsx_storage_format storage_format,
    gsx_data_type data_type,
    gsx_index_t channels,
    gsx_index_t height,
    gsx_index_t width)
{
    gsx_arena_t arena = nullptr;
    gsx_tensor_t tensor = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_tensor_desc tensor_desc{};

    if(gsx_arena_init(&arena, buffer_type, &arena_desc).code != GSX_ERROR_SUCCESS) {
        return nullptr;
    }
    tensor_desc.rank = 3;
    tensor_desc.data_type = data_type;
    tensor_desc.storage_format = storage_format;
    tensor_desc.arena = arena;
    if(storage_format == GSX_STORAGE_FORMAT_CHW) {
        tensor_desc.shape[0] = channels;
        tensor_desc.shape[1] = height;
        tensor_desc.shape[2] = width;
    } else {
        tensor_desc.shape[0] = height;
        tensor_desc.shape[1] = width;
        tensor_desc.shape[2] = channels;
    }
    if(gsx_tensor_init(&tensor, &tensor_desc).code != GSX_ERROR_SUCCESS) {
        (void)gsx_arena_free(arena);
        return nullptr;
    }
    return tensor;
}

static void free_tensor_and_arena(gsx_tensor_t tensor)
{
    gsx_tensor_desc desc{};
    gsx_arena_t arena = nullptr;

    ASSERT_NE(tensor, nullptr);
    ASSERT_GSX_SUCCESS(gsx_tensor_get_desc(tensor, &desc));
    arena = desc.arena;
    ASSERT_GSX_SUCCESS(gsx_tensor_free(tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
}

template <typename T>
static std::vector<T> download_tensor_values(gsx_backend_t backend, gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    std::vector<T> values;

    if(gsx_backend_major_stream_sync(backend).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    if(gsx_tensor_get_info(tensor, &info).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    values.resize((size_t)(info.size_bytes / sizeof(T)));
    if(gsx_tensor_download(tensor, values.data(), info.size_bytes).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    if(gsx_backend_major_stream_sync(backend).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    return values;
}

static std::vector<float> make_linear_fixture(gsx_storage_format storage_format)
{
    constexpr gsx_index_t channels = 3;
    constexpr gsx_index_t height = 2;
    constexpr gsx_index_t width = 3;
    std::vector<float> values((size_t)(channels * height * width));
    const std::array<float, 18> seed = {
        0.0f, 0.0031308f, 0.018f, 0.2f, 0.5f, 1.0f,
        0.9f, 0.7f, 0.4f, 0.25f, 0.1f, 0.8f,
        0.6f, 0.05f, 0.95f, 0.33f, 0.66f, 0.12f
    };

    if(storage_format == GSX_STORAGE_FORMAT_CHW) {
        for(size_t i = 0; i < seed.size(); ++i) {
            values[i] = seed[i];
        }
        return values;
    }
    for(gsx_index_t c = 0; c < channels; ++c) {
        for(gsx_index_t y = 0; y < height; ++y) {
            for(gsx_index_t x = 0; x < width; ++x) {
                size_t src = (size_t)(((c * height) + y) * width + x);
                size_t dst = (size_t)(((y * width) + x) * channels + c);
                values[dst] = seed[src];
            }
        }
    }
    return values;
}

template <typename T>
static void expect_vector_eq(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
    ASSERT_EQ(lhs.size(), rhs.size());
    EXPECT_EQ(lhs, rhs);
}

static void expect_vector_near(const std::vector<float> &lhs, const std::vector<float> &rhs, float tol)
{
    ASSERT_EQ(lhs.size(), rhs.size());
    for(size_t i = 0; i < lhs.size(); ++i) {
        EXPECT_NEAR(lhs[i], rhs[i], tol) << "index=" << i;
    }
}

struct ImageParityOutputs {
    std::vector<float> linear_to_srgb;
    std::vector<float> srgb_to_linear;
    std::vector<uint8_t> f32_to_u8;
    std::vector<float> u8_to_f32;
    std::vector<float> chw_to_hwc;
    std::vector<float> hwc_to_chw;
};

static ImageParityOutputs run_image_ops(gsx_backend_t backend, gsx_backend_buffer_type_class buffer_type_class, bool supports_u8)
{
    constexpr gsx_index_t channels = 3;
    constexpr gsx_index_t height = 2;
    constexpr gsx_index_t width = 3;

    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, buffer_type_class);
    gsx_tensor_t linear_chw = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_F32, channels, height, width);
    gsx_tensor_t srgb_chw = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_F32, channels, height, width);
    gsx_tensor_t linear_roundtrip_chw = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_F32, channels, height, width);
    gsx_tensor_t u8_chw = supports_u8 ? make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_U8, channels, height, width) : nullptr;
    gsx_tensor_t f32_from_u8_chw = supports_u8 ? make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_F32, channels, height, width) : nullptr;
    gsx_tensor_t hwc_f32 = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_HWC, GSX_DATA_TYPE_F32, channels, height, width);
    gsx_tensor_t chw_from_hwc = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_F32, channels, height, width);
    ImageParityOutputs outputs;
    std::vector<float> linear_fixture = make_linear_fixture(GSX_STORAGE_FORMAT_CHW);

    EXPECT_GSX_CODE(gsx_tensor_upload(linear_chw, linear_fixture.data(), linear_fixture.size() * sizeof(float)), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_tensor_image_convert_colorspace(srgb_chw, GSX_IMAGE_COLOR_SPACE_SRGB, linear_chw, GSX_IMAGE_COLOR_SPACE_LINEAR), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_tensor_image_convert_colorspace(linear_roundtrip_chw, GSX_IMAGE_COLOR_SPACE_LINEAR, srgb_chw, GSX_IMAGE_COLOR_SPACE_SRGB), GSX_ERROR_SUCCESS);
    if(supports_u8) {
        EXPECT_GSX_CODE(gsx_tensor_image_convert_data_type(u8_chw, linear_chw), GSX_ERROR_SUCCESS);
        EXPECT_GSX_CODE(gsx_tensor_image_convert_data_type(f32_from_u8_chw, u8_chw), GSX_ERROR_SUCCESS);
    }
    EXPECT_GSX_CODE(gsx_tensor_image_convert_storage_format(hwc_f32, linear_chw), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_tensor_image_convert_storage_format(chw_from_hwc, hwc_f32), GSX_ERROR_SUCCESS);

    outputs.linear_to_srgb = download_tensor_values<float>(backend, srgb_chw);
    outputs.srgb_to_linear = download_tensor_values<float>(backend, linear_roundtrip_chw);
    if(supports_u8) {
        outputs.f32_to_u8 = download_tensor_values<uint8_t>(backend, u8_chw);
        outputs.u8_to_f32 = download_tensor_values<float>(backend, f32_from_u8_chw);
    }
    outputs.chw_to_hwc = download_tensor_values<float>(backend, hwc_f32);
    outputs.hwc_to_chw = download_tensor_values<float>(backend, chw_from_hwc);
    EXPECT_GSX_CODE(gsx_backend_major_stream_sync(backend), GSX_ERROR_SUCCESS);

    free_tensor_and_arena(chw_from_hwc);
    free_tensor_and_arena(hwc_f32);
    if(f32_from_u8_chw != nullptr) {
        free_tensor_and_arena(f32_from_u8_chw);
    }
    if(u8_chw != nullptr) {
        free_tensor_and_arena(u8_chw);
    }
    free_tensor_and_arena(linear_roundtrip_chw);
    free_tensor_and_arena(srgb_chw);
    free_tensor_and_arena(linear_chw);
    return outputs;
}

TEST(ImageRuntime, CpuImageOpsProduceExpectedRoundTrips)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    ImageParityOutputs outputs = run_image_ops(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, true);
    std::vector<float> linear_fixture = make_linear_fixture(GSX_STORAGE_FORMAT_CHW);
    std::vector<float> hwc_fixture = make_linear_fixture(GSX_STORAGE_FORMAT_HWC);

    expect_vector_near(outputs.srgb_to_linear, linear_fixture, 1e-5f);
    expect_vector_eq(outputs.chw_to_hwc, hwc_fixture);
    expect_vector_eq(outputs.hwc_to_chw, linear_fixture);
    ASSERT_FALSE(outputs.f32_to_u8.empty());
    ASSERT_EQ(outputs.u8_to_f32.size(), linear_fixture.size());

    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(ImageRuntime, ConvertColorspaceRejectsMismatchedDataType)
{
    gsx_backend_t backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_tensor_t src = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_F32, 3, 2, 2);
    gsx_tensor_t dst = make_image_tensor(buffer_type, GSX_STORAGE_FORMAT_CHW, GSX_DATA_TYPE_U8, 3, 2, 2);

    EXPECT_GSX_CODE(
        gsx_tensor_image_convert_colorspace(dst, GSX_IMAGE_COLOR_SPACE_SRGB, src, GSX_IMAGE_COLOR_SPACE_LINEAR),
        GSX_ERROR_INVALID_ARGUMENT);

    free_tensor_and_arena(dst);
    free_tensor_and_arena(src);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(ImageRuntime, OptionalGpuBackendsMatchCpuGroundTruth)
{
    constexpr std::array<gsx_backend_type, 2> optional_backends = { GSX_BACKEND_TYPE_CUDA, GSX_BACKEND_TYPE_METAL };
    ImageParityOutputs cpu_outputs{};
    gsx_backend_t cpu_backend = create_backend_by_type(GSX_BACKEND_TYPE_CPU);

    cpu_outputs = run_image_ops(cpu_backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, true);
    ASSERT_GSX_SUCCESS(gsx_backend_free(cpu_backend));

    for(gsx_backend_type backend_type : optional_backends) {
        if(!has_backend_device(backend_type)) {
            continue;
        }

        gsx_backend_t gpu_backend = create_backend_by_type(backend_type);
        gsx_backend_buffer_type_class buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
        ImageParityOutputs gpu_outputs = run_image_ops(gpu_backend, buffer_type_class, backend_supports_u8(gpu_backend));

        expect_vector_near(gpu_outputs.linear_to_srgb, cpu_outputs.linear_to_srgb, 1e-5f);
        expect_vector_near(gpu_outputs.srgb_to_linear, cpu_outputs.srgb_to_linear, 1e-5f);
        if(!gpu_outputs.f32_to_u8.empty()) {
            expect_vector_eq(gpu_outputs.f32_to_u8, cpu_outputs.f32_to_u8);
            expect_vector_near(gpu_outputs.u8_to_f32, cpu_outputs.u8_to_f32, 1e-6f);
        }
        expect_vector_near(gpu_outputs.chw_to_hwc, cpu_outputs.chw_to_hwc, 1e-6f);
        expect_vector_near(gpu_outputs.hwc_to_chw, cpu_outputs.hwc_to_chw, 1e-6f);

        ASSERT_GSX_SUCCESS(gsx_backend_free(gpu_backend));
    }
}

} // namespace
