#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
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

static gsx_backend_device_t get_backend_device(gsx_backend_type backend_type)
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    if(gsx_get_backend_device_by_type(backend_type, 0, &backend_device).code != GSX_ERROR_SUCCESS) {
        return nullptr;
    }
    return backend_device;
}

static gsx_backend_t create_backend(gsx_backend_type backend_type)
{
    gsx_backend_device_t backend_device = get_backend_device(backend_type);
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};

    if(backend_device == nullptr) {
        return nullptr;
    }

    backend_desc.device = backend_device;
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

struct TestImage {
    bool present = false;
    gsx_data_type data_type = GSX_DATA_TYPE_U8;
    gsx_index_t width = 0;
    gsx_index_t height = 0;
    gsx_index_t channel_count = 0;
    gsx_size_t row_stride_bytes = 0;
    std::vector<uint8_t> u8;
    std::vector<float> f32;
    std::vector<int32_t> i32;
};

static TestImage make_u8_image(
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    const std::vector<uint8_t> &values,
    gsx_size_t row_padding_bytes = 0)
{
    TestImage image{};
    gsx_size_t packed_row_bytes = 0;

    image.present = true;
    image.data_type = GSX_DATA_TYPE_U8;
    image.width = width;
    image.height = height;
    image.channel_count = channel_count;
    packed_row_bytes = (gsx_size_t)width * (gsx_size_t)channel_count;
    image.row_stride_bytes = packed_row_bytes + row_padding_bytes;
    image.u8.assign((size_t)(image.row_stride_bytes * (gsx_size_t)height), 0u);
    for(gsx_index_t y = 0; y < height; ++y) {
        std::memcpy(
            image.u8.data() + (size_t)((gsx_size_t)y * image.row_stride_bytes),
            values.data() + (size_t)((gsx_size_t)y * packed_row_bytes),
            (size_t)packed_row_bytes);
    }
    return image;
}

static TestImage make_f32_image(
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    const std::vector<float> &values,
    gsx_size_t row_padding_bytes = 0)
{
    TestImage image{};
    gsx_size_t packed_row_bytes = 0;
    gsx_size_t packed_row_elements = 0;
    gsx_size_t padded_row_elements = 0;

    image.present = true;
    image.data_type = GSX_DATA_TYPE_F32;
    image.width = width;
    image.height = height;
    image.channel_count = channel_count;
    packed_row_bytes = (gsx_size_t)width * (gsx_size_t)channel_count * sizeof(float);
    image.row_stride_bytes = packed_row_bytes + row_padding_bytes;
    packed_row_elements = packed_row_bytes / sizeof(float);
    padded_row_elements = image.row_stride_bytes / sizeof(float);
    image.f32.assign((size_t)(padded_row_elements * (gsx_size_t)height), 0.0f);
    for(gsx_index_t y = 0; y < height; ++y) {
        std::memcpy(
            image.f32.data() + (size_t)((gsx_size_t)y * padded_row_elements),
            values.data() + (size_t)((gsx_size_t)y * packed_row_elements),
            (size_t)packed_row_bytes);
    }
    return image;
}

static TestImage make_i32_image(
    gsx_index_t width,
    gsx_index_t height,
    gsx_index_t channel_count,
    const std::vector<int32_t> &values)
{
    TestImage image{};

    image.present = true;
    image.data_type = GSX_DATA_TYPE_I32;
    image.width = width;
    image.height = height;
    image.channel_count = channel_count;
    image.row_stride_bytes = (gsx_size_t)width * (gsx_size_t)channel_count * sizeof(int32_t);
    image.i32 = values;
    return image;
}

static gsx_cpu_image_view make_view(const TestImage &image)
{
    gsx_cpu_image_view view{};

    if(!image.present) {
        return view;
    }

    view.data_type = image.data_type;
    view.width = image.width;
    view.height = image.height;
    view.channel_count = image.channel_count;
    view.row_stride_bytes = image.row_stride_bytes;
    switch(image.data_type) {
    case GSX_DATA_TYPE_U8:
        view.data = image.u8.data();
        break;
    case GSX_DATA_TYPE_F32:
        view.data = image.f32.data();
        break;
    case GSX_DATA_TYPE_I32:
        view.data = image.i32.data();
        break;
    default:
        view.data = nullptr;
        break;
    }
    return view;
}

struct TestSample {
    gsx_camera_intrinsics intrinsics{};
    gsx_camera_pose pose{};
    TestImage rgb;
    TestImage alpha;
    TestImage invdepth;
    bool has_stable_sample_id = false;
    gsx_id_t stable_sample_id = 0;
    void *release_token = nullptr;
};

struct TestDatasetObject {
    std::vector<TestSample> samples;
    gsx_size_t get_length_calls = 0;
    gsx_size_t get_sample_calls = 0;
    gsx_size_t release_calls = 0;
    std::vector<gsx_size_t> fetched_indices;
    std::vector<void *> released_tokens;
    bool force_zero_length = false;
};

static gsx_error test_dataset_get_length(void *object, gsx_size_t *out_length)
{
    auto *dataset = static_cast<TestDatasetObject *>(object);

    dataset->get_length_calls += 1;
    *out_length = dataset->force_zero_length ? 0 : (gsx_size_t)dataset->samples.size();
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error test_dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
    auto *dataset = static_cast<TestDatasetObject *>(object);
    const TestSample *sample = nullptr;

    dataset->get_sample_calls += 1;
    dataset->fetched_indices.push_back(sample_index);
    if(sample_index >= dataset->samples.size()) {
        return gsx_error{ GSX_ERROR_OUT_OF_RANGE, "sample_index is out of range" };
    }
    sample = &dataset->samples[(size_t)sample_index];

    std::memset(out_sample, 0, sizeof(*out_sample));
    out_sample->intrinsics = sample->intrinsics;
    out_sample->pose = sample->pose;
    out_sample->rgb = make_view(sample->rgb);
    out_sample->alpha = make_view(sample->alpha);
    out_sample->invdepth = make_view(sample->invdepth);
    out_sample->has_stable_sample_id = sample->has_stable_sample_id;
    out_sample->stable_sample_id = sample->stable_sample_id;
    out_sample->release_token = sample->release_token;
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static void test_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    auto *dataset = static_cast<TestDatasetObject *>(object);

    dataset->release_calls += 1;
    dataset->released_tokens.push_back(sample->release_token);
}

static std::vector<float> download_f32_tensor(gsx_backend_t backend, gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    std::vector<float> values;
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    error = gsx_tensor_get_info(tensor, &info);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return values;
    }
    EXPECT_EQ(info.size_bytes % sizeof(float), 0U);
    if(info.size_bytes % sizeof(float) != 0U) {
        return values;
    }
    values.resize((size_t)(info.size_bytes / sizeof(float)));
    error = gsx_backend_major_stream_sync(backend);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return values;
    }
    error = gsx_tensor_download(tensor, values.data(), info.size_bytes);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return values;
    }
    error = gsx_backend_major_stream_sync(backend);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    return values;
}

static std::vector<uint8_t> download_u8_tensor(gsx_backend_t backend, gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    std::vector<uint8_t> values;
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    error = gsx_tensor_get_info(tensor, &info);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return values;
    }
    values.resize((size_t)info.size_bytes);
    error = gsx_backend_major_stream_sync(backend);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return values;
    }
    error = gsx_tensor_download(tensor, values.data(), info.size_bytes);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return values;
    }
    error = gsx_backend_major_stream_sync(backend);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    return values;
}

static void expect_tensor_shape(
    gsx_tensor_t tensor,
    gsx_data_type data_type,
    gsx_storage_format storage_format,
    const std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> &shape,
    gsx_index_t rank)
{
    gsx_tensor_info info{};

    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.data_type, data_type);
    EXPECT_EQ(info.storage_format, storage_format);
    EXPECT_EQ(info.rank, rank);
    for(gsx_index_t dim = 0; dim < rank; ++dim) {
        EXPECT_EQ(info.shape[dim], shape[(size_t)dim]);
    }
}

static TestSample make_sample_with_u8_rgb(
    gsx_index_t width,
    gsx_index_t height,
    const std::vector<uint8_t> &rgb_values,
    const std::vector<uint8_t> &alpha_values,
    const std::vector<uint8_t> &invdepth_values)
{
    TestSample sample{};

    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.fx = 4.0f;
    sample.intrinsics.fy = 6.0f;
    sample.intrinsics.cx = 0.5f;
    sample.intrinsics.cy = 0.5f;
    sample.intrinsics.camera_id = 7;
    sample.intrinsics.width = width;
    sample.intrinsics.height = height;
    sample.pose.rot.w = 1.0f;
    sample.pose.camera_id = 7;
    sample.pose.frame_id = 11;
    sample.rgb = make_u8_image(width, height, 3, rgb_values, 2);
    sample.alpha = make_u8_image(width, height, 1, alpha_values, 1);
    sample.invdepth = make_u8_image(width, height, 1, invdepth_values, 3);
    sample.has_stable_sample_id = true;
    sample.stable_sample_id = 99;
    sample.release_token = reinterpret_cast<void *>(uintptr_t{ 0x1234 });
    return sample;
}

TEST(DataRuntime, DatasetInitAndValidation)
{
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc desc{};
    gsx_dataset_info info{};

    desc.object = &dataset_object;
    EXPECT_GSX_CODE(gsx_dataset_init(&dataset, &desc), GSX_ERROR_INVALID_ARGUMENT);

    dataset_object.force_zero_length = true;
    desc.get_length = test_dataset_get_length;
    desc.get_sample = test_dataset_get_sample;
    desc.release_sample = test_dataset_release_sample;
    EXPECT_GSX_CODE(gsx_dataset_init(&dataset, &desc), GSX_ERROR_INVALID_ARGUMENT);

    dataset_object.force_zero_length = false;
    dataset_object.samples.push_back(make_sample_with_u8_rgb(
        2,
        2,
        { 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110 },
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 }));
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &desc));
    ASSERT_GSX_SUCCESS(gsx_dataset_get_info(dataset, &info));
    EXPECT_EQ(info.length, 1U);
    EXPECT_EQ(dataset_object.get_length_calls, 2U);
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
}

TEST(DataRuntime, DataloaderInitRejectsUnsupportedAndLiveDatasetMisuse)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc dataset_desc{};
    gsx_dataloader_desc dataloader_desc{};
    gsx_dataloader_t loader = nullptr;

    ASSERT_NE(backend, nullptr);

    dataset_object.samples.push_back(make_sample_with_u8_rgb(
        2,
        2,
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 },
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 }));
    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataloader_desc.output_width = 2;
    dataloader_desc.output_height = 2;

    dataloader_desc.enable_async_prefetch = true;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);
    dataloader_desc.enable_async_prefetch = false;

    dataloader_desc.prefetch_count = 1;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);
    dataloader_desc.prefetch_count = 0;

    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_TILED_CHW;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;

    dataloader_desc.image_data_type = GSX_DATA_TYPE_BF16;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);
    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;

    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc));
    EXPECT_GSX_CODE(gsx_dataset_free(dataset), GSX_ERROR_INVALID_STATE);

    gsx_dataloader_t second_loader = nullptr;
    EXPECT_GSX_CODE(gsx_dataloader_init(&second_loader, backend, dataset, &dataloader_desc), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(loader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, NextExProducesCHWBilinearOutputAndResizedIntrinsics)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc dataset_desc{};
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    gsx_dataloader_result result{};

    ASSERT_NE(backend, nullptr);

    dataset_object.samples.push_back(make_sample_with_u8_rgb(
        2,
        2,
        { 0, 10, 20, 40, 50, 60, 80, 90, 100, 120, 130, 140 },
        { 10, 30, 50, 70 },
        { 20, 40, 60, 80 }));
    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataloader_desc.output_width = 1;
    dataloader_desc.output_height = 1;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.boundary_flags, GSX_DATALOADER_BOUNDARY_NEW_EPOCH | GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION);
    EXPECT_EQ(result.epoch_index, 0U);
    EXPECT_EQ(result.stable_sample_index, 0U);
    EXPECT_TRUE(result.has_stable_sample_id);
    EXPECT_EQ(result.stable_sample_id, 99U);
    EXPECT_EQ(result.intrinsics.width, 1);
    EXPECT_EQ(result.intrinsics.height, 1);
    EXPECT_FLOAT_EQ(result.intrinsics.fx, 2.0f);
    EXPECT_FLOAT_EQ(result.intrinsics.fy, 3.0f);
    EXPECT_FLOAT_EQ(result.intrinsics.cx, 0.0f);
    EXPECT_FLOAT_EQ(result.intrinsics.cy, 0.0f);

    expect_tensor_shape(
        result.rgb_image, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 3, 1, 1, 0 }, 3);
    expect_tensor_shape(
        result.alpha_image, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 1, 1, 0 }, 3);
    expect_tensor_shape(
        result.invdepth_image,
        GSX_DATA_TYPE_F32,
        GSX_STORAGE_FORMAT_CHW,
        std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 1, 1, 0 },
        3);
    EXPECT_EQ(download_f32_tensor(backend, result.rgb_image), (std::vector<float>{ 60.0f, 70.0f, 80.0f }));
    EXPECT_EQ(download_f32_tensor(backend, result.alpha_image), (std::vector<float>{ 40.0f }));
    EXPECT_EQ(download_f32_tensor(backend, result.invdepth_image), (std::vector<float>{ 50.0f }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.boundary_flags, GSX_DATALOADER_BOUNDARY_NEW_EPOCH);
    EXPECT_EQ(result.epoch_index, 1U);
    EXPECT_EQ(dataset_object.release_calls, 2U);
    EXPECT_EQ(dataset_object.released_tokens.size(), 2U);
    EXPECT_EQ(dataset_object.released_tokens[0], reinterpret_cast<void *>(uintptr_t{ 0x1234 }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, NextExProducesHWCU8OutputWithExplicitSingleChannelShape)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc dataset_desc{};
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    gsx_dataloader_result result{};
    TestSample sample{};

    ASSERT_NE(backend, nullptr);

    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.width = 2;
    sample.intrinsics.height = 1;
    sample.intrinsics.camera_id = 3;
    sample.pose.rot.w = 1.0f;
    sample.pose.camera_id = 3;
    sample.rgb = make_f32_image(2, 1, 3, { 1.0f, 2.0f, 3.0f, 254.7f, 255.0f, 100.4f });
    sample.alpha = make_f32_image(2, 1, 1, { 4.0f, 5.0f });
    sample.invdepth = make_f32_image(2, 1, 1, { 6.0f, 7.0f });
    dataset_object.samples.push_back(sample);

    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_U8;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_HWC;
    dataloader_desc.output_width = 2;
    dataloader_desc.output_height = 1;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));
    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));

    expect_tensor_shape(
        result.rgb_image, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 2, 3, 0 }, 3);
    expect_tensor_shape(
        result.alpha_image, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 2, 1, 0 }, 3);
    expect_tensor_shape(
        result.invdepth_image, GSX_DATA_TYPE_U8, GSX_STORAGE_FORMAT_HWC, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 2, 1, 0 }, 3);
    EXPECT_EQ(download_u8_tensor(backend, result.rgb_image), (std::vector<uint8_t>{ 1, 2, 3, 255, 255, 100 }));
    EXPECT_EQ(download_u8_tensor(backend, result.alpha_image), (std::vector<uint8_t>{ 4, 5 }));
    EXPECT_EQ(download_u8_tensor(backend, result.invdepth_image), (std::vector<uint8_t>{ 6, 7 }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, UnsupportedInputImageDtypeFailsAndStillReleasesSample)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc dataset_desc{};
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    TestSample sample{};

    ASSERT_NE(backend, nullptr);

    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.width = 1;
    sample.intrinsics.height = 1;
    sample.intrinsics.camera_id = 5;
    sample.pose.rot.w = 1.0f;
    sample.pose.camera_id = 5;
    sample.rgb = make_i32_image(1, 1, 3, { 1, 2, 3 });
    dataset_object.samples.push_back(sample);

    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    dataloader_desc.output_width = 1;
    dataloader_desc.output_height = 1;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));
    EXPECT_GSX_CODE(gsx_dataloader_next_ex(dataloader, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    gsx_dataloader_result result{};
    EXPECT_GSX_CODE(gsx_dataloader_next_ex(dataloader, &result), GSX_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(dataset_object.release_calls, 1U);

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, SetOutputShapeRecreatesNewGeometry)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc dataset_desc{};
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc desc{};
    gsx_dataloader_result result{};

    ASSERT_NE(backend, nullptr);

    dataset_object.samples.push_back(make_sample_with_u8_rgb(
        2,
        2,
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 }));
    dataset_desc.object = &dataset_object;
    dataset_desc.get_length = test_dataset_get_length;
    dataset_desc.get_sample = test_dataset_get_sample;
    dataset_desc.release_sample = test_dataset_release_sample;
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

    desc.image_data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.output_width = 2;
    desc.output_height = 2;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &desc));
    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    expect_tensor_shape(
        result.rgb_image, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 3, 2, 2, 0 }, 3);

    ASSERT_GSX_SUCCESS(gsx_dataloader_set_output_shape(dataloader, 1, 1));
    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    expect_tensor_shape(
        result.rgb_image, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 3, 1, 1, 0 }, 3);

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, OptionalGpuBackendsReturnBackendOwnedTensorsWhenAvailable)
{
    const std::array<gsx_backend_type, 2> optional_types = { GSX_BACKEND_TYPE_CUDA, GSX_BACKEND_TYPE_METAL };

    for(const gsx_backend_type backend_type : optional_types) {
        gsx_backend_t backend = create_backend(backend_type);
        TestDatasetObject dataset_object{};
        gsx_dataset_t dataset = nullptr;
        gsx_dataset_desc dataset_desc{};
        gsx_dataloader_t dataloader = nullptr;
        gsx_dataloader_desc desc{};
        gsx_dataloader_result result{};
        gsx_tensor_info info{};
        gsx_backend_t tensor_backend = nullptr;

        if(backend == nullptr) {
            continue;
        }

        dataset_object.samples.push_back(make_sample_with_u8_rgb(
            1,
            1,
            { 9, 19, 29 },
            { 3 },
            { 4 }));
        dataset_desc.object = &dataset_object;
        dataset_desc.get_length = test_dataset_get_length;
        dataset_desc.get_sample = test_dataset_get_sample;
        dataset_desc.release_sample = test_dataset_release_sample;
        ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &dataset_desc));

        desc.image_data_type = GSX_DATA_TYPE_F32;
        desc.storage_format = GSX_STORAGE_FORMAT_CHW;
        desc.output_width = 1;
        desc.output_height = 1;
        ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &desc));
        ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
        ASSERT_GSX_SUCCESS(gsx_tensor_get_info(result.rgb_image, &info));
        ASSERT_GSX_SUCCESS(gsx_arena_get_backend(info.arena, &tensor_backend));
        EXPECT_EQ(tensor_backend, backend);
        EXPECT_EQ(download_f32_tensor(backend, result.rgb_image), (std::vector<float>{ 9.0f, 19.0f, 29.0f }));

        ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
        ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
        ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
    }
}

}  // namespace
