#include "gsx/extra/gsx-io-ply.h"
#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

static std::string make_temp_ply_path()
{
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("gsx_io_ply_runtime_" + std::to_string(now) + ".ply");
    return path.string();
}

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

static gsx_gs_t create_gs_with_count(gsx_backend_buffer_type_t buffer_type, gsx_size_t count, gsx_gs_aux_flags aux_flags = GSX_GS_AUX_DEFAULT)
{
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_arena_desc arena_desc{};

    arena_desc.initial_capacity_bytes = 262144;

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = count;
    gs_desc.aux_flags = aux_flags;

    gsx_error error = gsx_gs_init(&gs, &gs_desc);
    if(!gsx_error_is_success(error)) {
        return nullptr;
    }
    return gs;
}

TEST(IoPlyRuntime, ReadRejectsNullOutGsPointer)
{
    EXPECT_GSX_CODE(gsx_read_ply(nullptr, "dummy.ply"), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(IoPlyRuntime, ReadRejectsNullFilename)
{
    gsx_gs_t gs = nullptr;
    EXPECT_GSX_CODE(gsx_read_ply(&gs, nullptr), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(IoPlyRuntime, ReadRequiresPreinitializedGsHandle)
{
    gsx_gs_t gs = nullptr;
    EXPECT_GSX_CODE(gsx_read_ply(&gs, "nonexistent.ply"), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(IoPlyRuntime, ReadReturnsIoErrorForNonexistentFile)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 1, GSX_GS_AUX_NONE);

    ASSERT_NE(gs, nullptr);
    EXPECT_GSX_CODE(gsx_read_ply(&gs, "/nonexistent/path/file.ply"), GSX_ERROR_IO);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(IoPlyRuntime, ReadReturnsIoErrorForInvalidPlyContent)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 1, GSX_GS_AUX_NONE);
    const std::string ply_path = make_temp_ply_path();

    ASSERT_NE(gs, nullptr);

    std::ofstream ofs(ply_path);
    ofs << "ply\nformat ascii 1.0\n";
    ofs.close();

    EXPECT_GSX_CODE(gsx_read_ply(&gs, ply_path.c_str()), GSX_ERROR_IO);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    std::remove(ply_path.c_str());
}

TEST(IoPlyRuntime, WriteRejectsNullGs)
{
    EXPECT_GSX_CODE(gsx_write_ply(nullptr, "dummy.ply"), GSX_ERROR_INVALID_ARGUMENT);
}

TEST(IoPlyRuntime, WriteRejectsNullFilename)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 1, GSX_GS_AUX_NONE);

    ASSERT_NE(gs, nullptr);
    EXPECT_GSX_CODE(gsx_write_ply(gs, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(IoPlyRuntime, ReadEmptyPlyFileReturnsErrorForZeroCount)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 1, GSX_GS_AUX_NONE);
    const std::string ply_path = make_temp_ply_path();

    ASSERT_NE(gs, nullptr);

    std::ofstream ofs(ply_path);
    ofs << "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n";
    ofs.close();

    EXPECT_GSX_CODE(gsx_read_ply(&gs, ply_path.c_str()), GSX_ERROR_IO);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    std::remove(ply_path.c_str());
}

TEST(IoPlyRuntime, RoundtripPreservesMean3dAndOpacity)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_gs_info gs_info{};
    gsx_tensor_t mean3d_tensor = nullptr;
    gsx_tensor_t opacity_tensor = nullptr;
    gsx_tensor_t logscale_tensor = nullptr;
    gsx_tensor_t rotation_tensor = nullptr;
    gsx_tensor_t sh0_tensor = nullptr;
    const std::string ply_path = make_temp_ply_path();
    const gsx_size_t gaussian_count = 3;
    std::vector<float> mean3d_values(gaussian_count * 3);
    std::vector<float> opacity_values(gaussian_count);
    std::vector<float> logscale_values(gaussian_count * 3, 0.0f);
    std::vector<float> rotation_values(gaussian_count * 4, 0.0f);
    std::vector<float> sh0_values(gaussian_count * 3, 0.0f);
    std::vector<float> mean3d_roundtrip(gaussian_count * 3);
    std::vector<float> opacity_roundtrip(gaussian_count);

    mean3d_values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    opacity_values = { 0.1f, 0.5f, 0.9f };
    for(gsx_size_t i = 0; i < gaussian_count; ++i) {
        rotation_values[i * 4 + 0] = 1.0f;
    }

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc.initial_capacity_bytes = 65536;
    gs_desc.count = gaussian_count;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, gaussian_count);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d_tensor));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity_tensor));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &logscale_tensor));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &rotation_tensor));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &sh0_tensor));

    ASSERT_GSX_SUCCESS(gsx_tensor_upload(mean3d_tensor, mean3d_values.data(), mean3d_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(opacity_tensor, opacity_values.data(), opacity_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(logscale_tensor, logscale_values.data(), logscale_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(rotation_tensor, rotation_values.data(), rotation_values.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(sh0_tensor, sh0_values.data(), sh0_values.size() * sizeof(float)));

    ASSERT_GSX_SUCCESS(gsx_write_ply(gs, ply_path.c_str()));

    gsx_gs_t gs_read = nullptr;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs_read, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_read_ply(&gs_read, ply_path.c_str()));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs_read, &gs_info));
    EXPECT_EQ(gs_info.count, gaussian_count);

    gsx_tensor_t mean3d_read = nullptr;
    gsx_tensor_t opacity_read = nullptr;
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_MEAN3D, &mean3d_read));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_OPACITY, &opacity_read));

    ASSERT_GSX_SUCCESS(gsx_tensor_download(mean3d_read, mean3d_roundtrip.data(), mean3d_roundtrip.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(opacity_read, opacity_roundtrip.data(), opacity_roundtrip.size() * sizeof(float)));

    for(std::size_t i = 0; i < mean3d_values.size(); ++i) {
        EXPECT_NEAR(mean3d_roundtrip[i], mean3d_values[i], 1e-5f);
    }
    for(std::size_t i = 0; i < opacity_values.size(); ++i) {
        EXPECT_NEAR(opacity_roundtrip[i], opacity_values[i], 1e-5f);
    }

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs_read));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    std::remove(ply_path.c_str());
}

TEST(IoPlyRuntime, RoundtripPreservesAllFields)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 2, GSX_GS_AUX_DEFAULT);
    gsx_gs_t gs_read = create_gs_with_count(buffer_type, 2, GSX_GS_AUX_DEFAULT);
    gsx_gs_info gs_info{};
    const std::string ply_path = make_temp_ply_path();
    const gsx_size_t gaussian_count = 2;

    ASSERT_NE(gs, nullptr);
    ASSERT_NE(gs_read, nullptr);

    std::array<float, 6> mean3d = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    std::array<float, 6> logscale = { -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f };
    std::array<float, 8> rotation = { 1.0f, 0.0f, 0.0f, 0.0f, 0.707f, 0.0f, 0.707f, 0.0f };
    std::array<float, 2> opacity = { 0.25f, 0.75f };
    std::array<float, 6> sh0 = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    std::array<float, 18> sh1 = {};
    std::array<float, 30> sh2 = {};
    std::array<float, 42> sh3 = {};

    for(std::size_t i = 0; i < sh1.size(); ++i) sh1[i] = static_cast<float>(i) * 0.01f;
    for(std::size_t i = 0; i < sh2.size(); ++i) sh2[i] = static_cast<float>(i) * 0.02f;
    for(std::size_t i = 0; i < sh3.size(); ++i) sh3[i] = static_cast<float>(i) * 0.03f;

    gsx_tensor_t tensor = nullptr;

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, mean3d.data(), mean3d.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, logscale.data(), logscale.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, rotation.data(), rotation.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, opacity.data(), opacity.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, sh0.data(), sh0.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, sh1.data(), sh1.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, sh2.data(), sh2.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, sh3.data(), sh3.size() * sizeof(float)));

    ASSERT_GSX_SUCCESS(gsx_write_ply(gs, ply_path.c_str()));

    ASSERT_GSX_SUCCESS(gsx_read_ply(&gs_read, ply_path.c_str()));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs_read, &gs_info));
    EXPECT_EQ(gs_info.count, gaussian_count);

    std::array<float, 6> mean3d_out = {};
    std::array<float, 6> logscale_out = {};
    std::array<float, 8> rotation_out = {};
    std::array<float, 2> opacity_out = {};
    std::array<float, 6> sh0_out = {};
    std::array<float, 18> sh1_out = {};
    std::array<float, 30> sh2_out = {};
    std::array<float, 42> sh3_out = {};

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_MEAN3D, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, mean3d_out.data(), mean3d_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_LOGSCALE, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, logscale_out.data(), logscale_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_ROTATION, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, rotation_out.data(), rotation_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_OPACITY, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, opacity_out.data(), opacity_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_SH0, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, sh0_out.data(), sh0_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_SH1, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, sh1_out.data(), sh1_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_SH2, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, sh2_out.data(), sh2_out.size() * sizeof(float)));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs_read, GSX_GS_FIELD_SH3, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(tensor, sh3_out.data(), sh3_out.size() * sizeof(float)));

    for(std::size_t i = 0; i < mean3d.size(); ++i) EXPECT_NEAR(mean3d_out[i], mean3d[i], 1e-5f);
    for(std::size_t i = 0; i < logscale.size(); ++i) EXPECT_NEAR(logscale_out[i], logscale[i], 1e-5f);
    for(std::size_t i = 0; i < rotation.size(); ++i) EXPECT_NEAR(rotation_out[i], rotation[i], 1e-5f);
    for(std::size_t i = 0; i < opacity.size(); ++i) EXPECT_NEAR(opacity_out[i], opacity[i], 1e-5f);
    for(std::size_t i = 0; i < sh0.size(); ++i) EXPECT_NEAR(sh0_out[i], sh0[i], 1e-5f);
    for(std::size_t i = 0; i < sh1.size(); ++i) EXPECT_NEAR(sh1_out[i], sh1[i], 1e-5f);
    for(std::size_t i = 0; i < sh2.size(); ++i) EXPECT_NEAR(sh2_out[i], sh2[i], 1e-5f);
    for(std::size_t i = 0; i < sh3.size(); ++i) EXPECT_NEAR(sh3_out[i], sh3[i], 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs_read));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    std::remove(ply_path.c_str());
}

TEST(IoPlyRuntime, ReadHandlesPlyWithVertexColorsWhenNoFdc)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 2, GSX_GS_AUX_NONE);
    gsx_gs_info gs_info{};
    const std::string ply_path = make_temp_ply_path();

    ASSERT_NE(gs, nullptr);

    std::ofstream ofs(ply_path, std::ios::binary);
    ofs << "ply\nformat binary_little_endian 1.0\nelement vertex 2\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "property float opacity\n";
    ofs << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    ofs << "property float rot_w\nproperty float rot_x\nproperty float rot_y\nproperty float rot_z\n";
    ofs << "end_header\n";

    for(int i = 0; i < 2; ++i) {
        float x = static_cast<float>(i);
        float y = static_cast<float>(i + 1);
        float z = static_cast<float>(i + 2);
        uint8_t r = 128;
        uint8_t g = 64;
        uint8_t b = 192;
        float opacity = 0.5f;
        float scale[3] = { 0.1f, 0.2f, 0.3f };
        float rot[4] = { 1.0f, 0.0f, 0.0f, 0.0f };

        ofs.write(reinterpret_cast<char*>(&x), sizeof(float));
        ofs.write(reinterpret_cast<char*>(&y), sizeof(float));
        ofs.write(reinterpret_cast<char*>(&z), sizeof(float));
        ofs.write(reinterpret_cast<char*>(&r), sizeof(uint8_t));
        ofs.write(reinterpret_cast<char*>(&g), sizeof(uint8_t));
        ofs.write(reinterpret_cast<char*>(&b), sizeof(uint8_t));
        ofs.write(reinterpret_cast<char*>(&opacity), sizeof(float));
        ofs.write(reinterpret_cast<char*>(scale), 3 * sizeof(float));
        ofs.write(reinterpret_cast<char*>(rot), 4 * sizeof(float));
    }
    ofs.close();

    ASSERT_GSX_SUCCESS(gsx_read_ply(&gs, ply_path.c_str()));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, 2u);

    gsx_tensor_t mean3d_tensor = nullptr;
    std::array<float, 6> mean3d_out = {};
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d_tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(mean3d_tensor, mean3d_out.data(), mean3d_out.size() * sizeof(float)));

    EXPECT_NEAR(mean3d_out[0], 0.0f, 1e-5f);
    EXPECT_NEAR(mean3d_out[1], 1.0f, 1e-5f);
    EXPECT_NEAR(mean3d_out[2], 2.0f, 1e-5f);
    EXPECT_NEAR(mean3d_out[3], 1.0f, 1e-5f);
    EXPECT_NEAR(mean3d_out[4], 2.0f, 1e-5f);
    EXPECT_NEAR(mean3d_out[5], 3.0f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    std::remove(ply_path.c_str());
}

TEST(IoPlyRuntime, ReadHandlesPlyWithDoublePrecisionVertices)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = create_gs_with_count(buffer_type, 1, GSX_GS_AUX_NONE);
    gsx_gs_info gs_info{};
    const std::string ply_path = make_temp_ply_path();

    ASSERT_NE(gs, nullptr);

    std::ofstream ofs(ply_path, std::ios::binary);
    ofs << "ply\nformat binary_little_endian 1.0\nelement vertex 1\n";
    ofs << "property double x\nproperty double y\nproperty double z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";

    double x = 1.5;
    double y = 2.5;
    double z = 3.5;
    uint8_t rgb[3] = { 255, 128, 64 };

    ofs.write(reinterpret_cast<char*>(&x), sizeof(double));
    ofs.write(reinterpret_cast<char*>(&y), sizeof(double));
    ofs.write(reinterpret_cast<char*>(&z), sizeof(double));
    ofs.write(reinterpret_cast<char*>(rgb), 3 * sizeof(uint8_t));
    ofs.close();

    ASSERT_GSX_SUCCESS(gsx_read_ply(&gs, ply_path.c_str()));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, 1u);

    gsx_tensor_t mean3d_tensor = nullptr;
    std::array<float, 3> mean3d_out = {};
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d_tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_download(mean3d_tensor, mean3d_out.data(), mean3d_out.size() * sizeof(float)));

    EXPECT_NEAR(mean3d_out[0], 1.5f, 1e-5f);
    EXPECT_NEAR(mean3d_out[1], 2.5f, 1e-5f);
    EXPECT_NEAR(mean3d_out[2], 3.5f, 1e-5f);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    std::remove(ply_path.c_str());
}

TEST(IoPlyRuntime, FieldShapesMatchExpectedDimensions)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_arena_t arena = nullptr;
    gsx_arena_desc arena_desc{};
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};
    gsx_gs_info gs_info{};
    gsx_tensor_t tensor = nullptr;
    gsx_tensor_info info{};
    const gsx_size_t gaussian_count = 10;

    arena_desc.initial_capacity_bytes = 131072;
    ASSERT_GSX_SUCCESS(gsx_arena_init(&arena, buffer_type, &arena_desc));

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc = arena_desc;
    gs_desc.count = gaussian_count;
    gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
    ASSERT_GSX_SUCCESS(gsx_gs_init(&gs, &gs_desc));
    ASSERT_GSX_SUCCESS(gsx_gs_get_info(gs, &gs_info));
    EXPECT_EQ(gs_info.count, gaussian_count);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 2);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 3);
    EXPECT_EQ(info.data_type, GSX_DATA_TYPE_F32);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 2);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 3);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 2);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 4);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 1);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 2);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 3);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 3);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 3);
    EXPECT_EQ(info.shape[2], 3);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 3);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 5);
    EXPECT_EQ(info.shape[2], 3);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &tensor));
    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.rank, 3);
    EXPECT_EQ(info.shape[0], static_cast<gsx_index_t>(gaussian_count));
    EXPECT_EQ(info.shape[1], 7);
    EXPECT_EQ(info.shape[2], 3);

    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_arena_free(arena));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
