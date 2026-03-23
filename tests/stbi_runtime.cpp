#include "gsx/extra/gsx-stbi.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
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

static std::string make_temp_image_path(const char *suffix)
{
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("gsx_stbi_runtime_" + std::to_string(now) + suffix);
    return path.string();
}

TEST(StbiRuntime, LoadRequiresEmptyDestinationPointer)
{
    gsx_image image{};
    uint8_t placeholder = 0;

    image.pixels = &placeholder;
    EXPECT_GSX_CODE(
        gsx_image_load(
            &image,
            "nonexistent_path.png",
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC),
        GSX_ERROR_INVALID_STATE);
}

TEST(StbiRuntime, WriteRejectsUnsupportedTypeAndFormat)
{
    const std::vector<uint8_t> pixels = {0, 64, 255, 255, 64, 0};
    const std::string png_path = make_temp_image_path(".png");

    EXPECT_GSX_CODE(
        gsx_image_write_png(
            png_path.c_str(),
            pixels.data(),
            2,
            1,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_TILED_CHW),
        GSX_ERROR_NOT_SUPPORTED);
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WritePngFromF32HwcAndLoadAsU8Hwc)
{
    const std::vector<float> pixels = {0.0f, 0.5f, 1.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), INFINITY};
    const std::vector<uint8_t> expected = {0, 128, 255, 0, 0, 255};
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            pixels.data(),
            2,
            1,
            3,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(
            &image,
            png_path.c_str(),
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_NE(image.pixels, nullptr);
    EXPECT_EQ(image.width, 2);
    EXPECT_EQ(image.height, 1);
    EXPECT_EQ(image.channels, 3);
    EXPECT_EQ(image.data_type, GSX_DATA_TYPE_U8);
    EXPECT_EQ(image.storage_format, GSX_STORAGE_FORMAT_HWC);
    const uint8_t *out = static_cast<const uint8_t *>(image.pixels);
    for (gsx_index_t i = 0; i < 6; ++i) {
        EXPECT_EQ(out[i], expected[i]) << "quantization mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WriteJpgFromU8ChwAndLoadAsF32Chw)
{
    const std::vector<uint8_t> chw_pixels = {
        255, 0,
        0, 255,
        0, 0
    };
    const std::string jpg_path = make_temp_image_path(".jpg");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_jpg(
            jpg_path.c_str(),
            chw_pixels.data(),
            2,
            1,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_CHW,
            95));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(
            &image,
            jpg_path.c_str(),
            3,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_CHW));
    ASSERT_NE(image.pixels, nullptr);
    EXPECT_EQ(image.width, 2);
    EXPECT_EQ(image.height, 1);
    EXPECT_EQ(image.channels, 3);
    EXPECT_EQ(image.data_type, GSX_DATA_TYPE_F32);
    EXPECT_EQ(image.storage_format, GSX_STORAGE_FORMAT_CHW);
    const float *out = static_cast<const float *>(image.pixels);
    const float eps = 5e-2f;
    for (gsx_index_t i = 0; i < 6; ++i) {
        const float expected = chw_pixels[i] / 255.0f;
        EXPECT_NEAR(out[i], expected, eps) << "roundtrip mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(jpg_path.c_str());
}

TEST(StbiRuntime, FreeRejectsNull)
{
    EXPECT_GSX_CODE(gsx_image_free(NULL), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(StbiRuntime, FreeAlreadyFreedSucceeds)
{
    gsx_image image{};
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    ASSERT_EQ(image.pixels, nullptr);
}

TEST(StbiRuntime, LoadRejectsNonexistentFile)
{
    gsx_image image{};
    EXPECT_GSX_CODE(
        gsx_image_load(
            &image,
            "nonexistent_path_12345.png",
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC),
        GSX_ERROR_IO);
}

TEST(StbiRuntime, LoadRejectsInvalidChannelCount)
{
    gsx_image image{};
    const std::string png_path = make_temp_image_path(".png");
    const std::vector<uint8_t> black_pixel = {0, 0, 0};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            black_pixel.data(),
            1,
            1,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC));

    EXPECT_GSX_CODE(
        gsx_image_load(&image, png_path.c_str(), 0, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC),
        GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(
        gsx_image_load(&image, png_path.c_str(), 5, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC),
        GSX_ERROR_OUT_OF_RANGE);
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, LoadRejectsUnsupportedOutputType)
{
    gsx_image image{};
    const std::string png_path = make_temp_image_path(".png");
    const std::vector<uint8_t> black_pixel = {0, 0, 0};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            black_pixel.data(),
            1,
            1,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC));

    EXPECT_GSX_CODE(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_TILED_CHW),
        GSX_ERROR_NOT_SUPPORTED);
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WritePngFromU8HwcPreservesPixelValues)
{
    const std::vector<uint8_t> pixels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            pixels.data(),
            2,
            2,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.data_type, GSX_DATA_TYPE_U8);
    ASSERT_EQ(image.storage_format, GSX_STORAGE_FORMAT_HWC);
    const uint8_t *out = static_cast<const uint8_t *>(image.pixels);
    for (gsx_index_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out[i], pixels[i]) << "pixel mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WritePngFromU8ChwPreservesPixelValues)
{
    const std::vector<uint8_t> chw_pixels = {
        0,  3,  6,  9,
        1,  4,  7,  10,
        2,  5,  8,  11
    };
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            chw_pixels.data(),
            2,
            2,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_CHW));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_CHW));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.data_type, GSX_DATA_TYPE_U8);
    ASSERT_EQ(image.storage_format, GSX_STORAGE_FORMAT_CHW);
    const uint8_t *out = static_cast<const uint8_t *>(image.pixels);
    for (gsx_index_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out[i], chw_pixels[i]) << "pixel mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WritePngFromU8ChwAndLoadAsHwcProducesCorrectHwcLayout)
{
    const std::vector<uint8_t> chw_pixels = {
        0,  3,  6,  9,
        1,  4,  7,  10,
        2,  5,  8,  11
    };
    const std::vector<uint8_t> expected_hwc = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11
    };
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            chw_pixels.data(),
            2,
            2,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_CHW));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.storage_format, GSX_STORAGE_FORMAT_HWC);
    const uint8_t *out = static_cast<const uint8_t *>(image.pixels);
    for (gsx_index_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out[i], expected_hwc[i]) << "pixel mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WritePngFromU8HwcAndLoadAsChwProducesCorrectChwLayout)
{
    const std::vector<uint8_t> hwc_pixels = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11
    };
    const std::vector<uint8_t> expected_chw = {
        0,  3,  6,  9,
        1,  4,  7,  10,
        2,  5,  8,  11
    };
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            hwc_pixels.data(),
            2,
            2,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_CHW));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.storage_format, GSX_STORAGE_FORMAT_CHW);
    const uint8_t *out = static_cast<const uint8_t *>(image.pixels);
    for (gsx_index_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out[i], expected_chw[i]) << "pixel mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WritePngFromF32HwcAndLoadAsF32HwcPreservesValues)
{
    const std::vector<float> pixels = {0.0f, 0.5f, 1.0f, 0.25f, 0.75f, 1.0f};
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            pixels.data(),
            2,
            1,
            3,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_HWC));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.data_type, GSX_DATA_TYPE_F32);
    ASSERT_EQ(image.storage_format, GSX_STORAGE_FORMAT_HWC);
    const float *out = static_cast<const float *>(image.pixels);
    const float eps = 1.0f / 255.0f;
    for (gsx_index_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(out[i], pixels[i], eps) << "pixel mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, F32QuantizationEdgeCases)
{
    const std::vector<float> input_f32 = {
        0.0f,
        0.5f,
        1.0f,
        -0.5f,
        -1.0f,
        1.5f,
        2.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };
    const std::vector<uint8_t> expected_u8 = {
        0,
        128,
        255,
        0,
        0
    };
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            input_f32.data(),
            4,
            4,
            3,
            GSX_DATA_TYPE_F32,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.width, 4);
    ASSERT_EQ(image.height, 4);
    const uint8_t *out = static_cast<const uint8_t *>(image.pixels);
    for (gsx_index_t i = 0; i < 5; ++i) {
        EXPECT_EQ(out[i], expected_u8[i]) << "quantization mismatch at index " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, U8ToF32RoundtripPreservesValues)
{
    const std::vector<uint8_t> u8_pixels = {0, 64, 128, 191, 255};
    const std::string png_path = make_temp_image_path(".png");
    gsx_image image{};

    ASSERT_GSX_SUCCESS(
        gsx_image_write_png(
            png_path.c_str(),
            u8_pixels.data(),
            1,
            1,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC));
    ASSERT_GSX_SUCCESS(
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_HWC));

    ASSERT_NE(image.pixels, nullptr);
    ASSERT_EQ(image.data_type, GSX_DATA_TYPE_F32);
    const float *out = static_cast<const float *>(image.pixels);
    const float eps = 1.0f / 255.0f;
    for (gsx_index_t i = 0; i < 3; ++i) {
        const float expected = u8_pixels[i] / 255.0f;
        EXPECT_NEAR(out[i], expected, eps) << "roundtrip mismatch at channel " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(png_path.c_str());
}

TEST(StbiRuntime, WriteJpgQualityAffectsFileSize)
{
    const std::vector<uint8_t> pixels(300, 128);
    const std::string jpg_path_low = make_temp_image_path("_q1.jpg");
    const std::string jpg_path_high = make_temp_image_path("_q100.jpg");

    ASSERT_GSX_SUCCESS(
        gsx_image_write_jpg(
            jpg_path_low.c_str(),
            pixels.data(),
            10,
            10,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC,
            1));
    ASSERT_GSX_SUCCESS(
        gsx_image_write_jpg(
            jpg_path_high.c_str(),
            pixels.data(),
            10,
            10,
            3,
            GSX_DATA_TYPE_U8,
            GSX_STORAGE_FORMAT_HWC,
            100));

    auto size_low = std::filesystem::file_size(jpg_path_low);
    auto size_high = std::filesystem::file_size(jpg_path_high);
    EXPECT_LT(size_low, size_high) << "low quality should produce smaller file than high quality";

    std::remove(jpg_path_low.c_str());
    std::remove(jpg_path_high.c_str());
}

TEST(StbiRuntime, ResizeRejectsNullOutput)
{
    gsx_image input{};
    input.pixels = nullptr;
    EXPECT_GSX_CODE(
        gsx_image_resize(NULL, &input, 10, 10),
        GSX_ERROR_INVALID_ARGUMENT);
}

TEST(StbiRuntime, ResizeRejectsNullInput)
{
    gsx_image output{};
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, NULL, 10, 10),
        GSX_ERROR_INVALID_ARGUMENT);
}

TEST(StbiRuntime, ResizeRejectsNonNullOutputPixels)
{
    gsx_image input{};
    input.pixels = nullptr;
    gsx_image output{};
    output.pixels = reinterpret_cast<void*>(1);
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 10, 10),
        GSX_ERROR_INVALID_STATE);
}

TEST(StbiRuntime, ResizeRejectsNonPositiveOutputDimensions)
{
    gsx_image input{};
    input.pixels = nullptr;
    gsx_image output{};
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 0, 10),
        GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 10, 0),
        GSX_ERROR_OUT_OF_RANGE);
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, -1, 10),
        GSX_ERROR_OUT_OF_RANGE);
}

TEST(StbiRuntime, ResizeRejectsUnsupportedStorageFormat)
{
    gsx_image input{};
    input.pixels = nullptr;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_TILED_CHW;
    gsx_image output{};
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 10, 10),
        GSX_ERROR_NOT_SUPPORTED);
}

TEST(StbiRuntime, ResizeRejectsNullInputPixels)
{
    gsx_image input{};
    input.pixels = nullptr;
    input.width = 10;
    input.height = 10;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;
    gsx_image output{};
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 5, 5),
        GSX_ERROR_INVALID_STATE);
}

TEST(StbiRuntime, ResizeU8HwcDownscale)
{
    const std::vector<uint8_t> hwc_pixels = {
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        255, 255, 0
    };
    gsx_image input{};
    input.pixels = const_cast<uint8_t*>(hwc_pixels.data());
    input.width = 4;
    input.height = 1;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 1));

    ASSERT_NE(output.pixels, nullptr);
    EXPECT_EQ(output.width, 2);
    EXPECT_EQ(output.height, 1);
    EXPECT_EQ(output.channels, 3);
    EXPECT_EQ(output.data_type, GSX_DATA_TYPE_U8);
    EXPECT_EQ(output.storage_format, GSX_STORAGE_FORMAT_HWC);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeU8HwcUpscale)
{
    const std::vector<uint8_t> hwc_pixels = {
        255, 0, 0,
        0, 255, 0
    };
    gsx_image input{};
    input.pixels = const_cast<uint8_t*>(hwc_pixels.data());
    input.width = 2;
    input.height = 1;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 4, 1));

    ASSERT_NE(output.pixels, nullptr);
    EXPECT_EQ(output.width, 4);
    EXPECT_EQ(output.height, 1);
    EXPECT_EQ(output.channels, 3);
    EXPECT_EQ(output.data_type, GSX_DATA_TYPE_U8);
    EXPECT_EQ(output.storage_format, GSX_STORAGE_FORMAT_HWC);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeU8ChwDownscale)
{
    const std::vector<uint8_t> chw_pixels = {
        255, 0,
        0, 255,
        0, 0
    };
    gsx_image input{};
    input.pixels = const_cast<uint8_t*>(chw_pixels.data());
    input.width = 2;
    input.height = 1;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_CHW;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 1, 1));

    ASSERT_NE(output.pixels, nullptr);
    EXPECT_EQ(output.width, 1);
    EXPECT_EQ(output.height, 1);
    EXPECT_EQ(output.channels, 3);
    EXPECT_EQ(output.data_type, GSX_DATA_TYPE_U8);
    EXPECT_EQ(output.storage_format, GSX_STORAGE_FORMAT_CHW);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeF32HwcDownscale)
{
    const std::vector<float> hwc_pixels = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f
    };
    gsx_image input{};
    input.pixels = const_cast<float*>(hwc_pixels.data());
    input.width = 4;
    input.height = 1;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_F32;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 1));

    ASSERT_NE(output.pixels, nullptr);
    EXPECT_EQ(output.width, 2);
    EXPECT_EQ(output.height, 1);
    EXPECT_EQ(output.channels, 3);
    EXPECT_EQ(output.data_type, GSX_DATA_TYPE_F32);
    EXPECT_EQ(output.storage_format, GSX_STORAGE_FORMAT_HWC);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeF32ChwUpscale)
{
    const std::vector<float> chw_pixels = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f
    };
    gsx_image input{};
    input.pixels = const_cast<float*>(chw_pixels.data());
    input.width = 2;
    input.height = 1;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_F32;
    input.storage_format = GSX_STORAGE_FORMAT_CHW;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 4, 1));

    ASSERT_NE(output.pixels, nullptr);
    EXPECT_EQ(output.width, 4);
    EXPECT_EQ(output.height, 1);
    EXPECT_EQ(output.channels, 3);
    EXPECT_EQ(output.data_type, GSX_DATA_TYPE_F32);
    EXPECT_EQ(output.storage_format, GSX_STORAGE_FORMAT_CHW);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizePreservesChannelCount)
{
    const std::vector<uint8_t> single_channel = {255, 128, 64, 32};
    gsx_image input{};
    input.pixels = const_cast<uint8_t*>(single_channel.data());
    input.width = 4;
    input.height = 1;
    input.channels = 1;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 1));

    EXPECT_EQ(output.channels, 1);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, Resize4ChannelImage)
{
    const std::vector<uint8_t> rgba_pixels = {
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
        255, 255, 0, 255
    };
    gsx_image input{};
    input.pixels = const_cast<uint8_t*>(rgba_pixels.data());
    input.width = 4;
    input.height = 1;
    input.channels = 4;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 1));

    EXPECT_EQ(output.channels, 4);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeRejectsInvalidChannelCount)
{
    gsx_image input{};
    input.pixels = nullptr;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;
    input.channels = 5;
    gsx_image output{};
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 10, 10),
        GSX_ERROR_OUT_OF_RANGE);

    input.channels = 0;
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 10, 10),
        GSX_ERROR_OUT_OF_RANGE);

    input.channels = -1;
    EXPECT_GSX_CODE(
        gsx_image_resize(&output, &input, 10, 10),
        GSX_ERROR_OUT_OF_RANGE);
}

TEST(StbiRuntime, ResizeU8SolidColorPreservesValues)
{
    const std::vector<uint8_t> solid_red(4 * 4 * 3, 0);
    std::vector<uint8_t> pixels = solid_red;
    for(size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i] = 200;
    }
    gsx_image input{};
    input.pixels = pixels.data();
    input.width = 4;
    input.height = 4;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 2));

    const uint8_t* out_pixels = static_cast<const uint8_t*>(output.pixels);
    for(gsx_index_t i = 0; i < 2 * 2 * 3; i += 3) {
        EXPECT_NEAR(out_pixels[i], 200, 2) << "R channel at pixel " << i/3;
        EXPECT_NEAR(out_pixels[i + 1], 0, 2) << "G channel at pixel " << i/3;
        EXPECT_NEAR(out_pixels[i + 2], 0, 2) << "B channel at pixel " << i/3;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeF32SolidColorPreservesValues)
{
    std::vector<float> pixels(4 * 4 * 3, 0.0f);
    for(size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i] = 0.75f;
    }
    gsx_image input{};
    input.pixels = pixels.data();
    input.width = 4;
    input.height = 4;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_F32;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 2));

    const float* out_pixels = static_cast<const float*>(output.pixels);
    for(gsx_index_t i = 0; i < 2 * 2 * 3; i += 3) {
        EXPECT_NEAR(out_pixels[i], 0.75f, 0.01f) << "R channel at pixel " << i/3;
        EXPECT_NEAR(out_pixels[i + 1], 0.0f, 0.01f) << "G channel at pixel " << i/3;
        EXPECT_NEAR(out_pixels[i + 2], 0.0f, 0.01f) << "B channel at pixel " << i/3;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeU8DownscalePreservesRegions)
{
    std::vector<uint8_t> pixels = {
        100, 100, 200, 200,
        100, 100, 200, 200,
        100, 100, 200, 200,
        100, 100, 200, 200
    };
    gsx_image input{};
    input.pixels = pixels.data();
    input.width = 4;
    input.height = 4;
    input.channels = 1;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 2));

    const uint8_t* out_pixels = static_cast<const uint8_t*>(output.pixels);
    EXPECT_NEAR(out_pixels[0], 100, 15);
    EXPECT_NEAR(out_pixels[1], 200, 15);
    EXPECT_NEAR(out_pixels[2], 100, 15);
    EXPECT_NEAR(out_pixels[3], 200, 15);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeF32DownscalePreservesRegions)
{
    std::vector<float> pixels = {
        0.4f, 0.4f, 0.8f, 0.8f,
        0.4f, 0.4f, 0.8f, 0.8f,
        0.4f, 0.4f, 0.8f, 0.8f,
        0.4f, 0.4f, 0.8f, 0.8f
    };
    gsx_image input{};
    input.pixels = pixels.data();
    input.width = 4;
    input.height = 4;
    input.channels = 1;
    input.data_type = GSX_DATA_TYPE_F32;
    input.storage_format = GSX_STORAGE_FORMAT_HWC;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 2));

    const float* out_pixels = static_cast<const float*>(output.pixels);
    EXPECT_NEAR(out_pixels[0], 0.4f, 0.1f);
    EXPECT_NEAR(out_pixels[1], 0.8f, 0.1f);
    EXPECT_NEAR(out_pixels[2], 0.4f, 0.1f);
    EXPECT_NEAR(out_pixels[3], 0.8f, 0.1f);
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

TEST(StbiRuntime, ResizeU8ChwPreservesValues)
{
    std::vector<uint8_t> chw_pixels = {
        200, 200, 200, 200,
        0, 0, 0, 0,
        50, 50, 50, 50
    };
    gsx_image input{};
    input.pixels = chw_pixels.data();
    input.width = 2;
    input.height = 2;
    input.channels = 3;
    input.data_type = GSX_DATA_TYPE_U8;
    input.storage_format = GSX_STORAGE_FORMAT_CHW;

    gsx_image output{};
    ASSERT_GSX_SUCCESS(gsx_image_resize(&output, &input, 2, 2));

    const uint8_t* out_pixels = static_cast<const uint8_t*>(output.pixels);
    EXPECT_EQ(output.storage_format, GSX_STORAGE_FORMAT_CHW);
    for(int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_pixels[i], 200, 2) << "R channel pixel " << i;
    }
    for(int i = 4; i < 8; ++i) {
        EXPECT_NEAR(out_pixels[i], 0, 2) << "G channel pixel " << i;
    }
    for(int i = 8; i < 12; ++i) {
        EXPECT_NEAR(out_pixels[i], 50, 2) << "B channel pixel " << i;
    }
    ASSERT_GSX_SUCCESS(gsx_image_free(&output));
}

} // namespace
