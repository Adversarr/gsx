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
    ASSERT_GSX_SUCCESS(gsx_image_free(&image));
    std::remove(jpg_path.c_str());
}

} // namespace
