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
            GSX_DATA_TYPE_I16,
            GSX_STORAGE_FORMAT_HWC),
        GSX_ERROR_NOT_SUPPORTED);
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
        gsx_image_load(&image, png_path.c_str(), 3, GSX_DATA_TYPE_I16, GSX_STORAGE_FORMAT_HWC),
        GSX_ERROR_NOT_SUPPORTED);
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

} // namespace
