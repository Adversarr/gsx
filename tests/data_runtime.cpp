#include "gsx/gsx.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
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

struct TestSample {
    gsx_camera_intrinsics intrinsics{};
    gsx_camera_pose pose{};
    std::vector<uint8_t> rgb_u8;
    std::vector<float> rgb_f32;
    std::vector<uint8_t> alpha_u8;
    std::vector<float> alpha_f32;
    std::vector<uint8_t> invdepth_u8;
    std::vector<float> invdepth_f32;
    bool has_stable_sample_id = false;
    gsx_id_t stable_sample_id = 0;
    void *release_token = nullptr;
    bool omit_rgb_data = false;
    bool omit_alpha_data = false;
    bool omit_invdepth_data = false;
};

struct TestDatasetObject {
    std::vector<TestSample> samples;
    gsx_size_t get_length_calls = 0;
    gsx_size_t get_sample_calls = 0;
    gsx_size_t release_calls = 0;
    std::vector<gsx_size_t> fetched_indices;
    std::vector<void *> released_tokens;
    bool force_zero_length = false;
    gsx_data_type source_data_type = GSX_DATA_TYPE_U8;
    bool has_alpha = false;
    bool has_invdepth = false;
    std::thread::id caller_thread_id{};
    std::thread::id first_callback_thread_id{};
    bool saw_non_caller_thread = false;
    gsx_size_t outstanding_samples = 0;
    gsx_size_t max_outstanding_samples = 0;
    std::mutex mutex;
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

    if(dataset == nullptr || out_sample == nullptr) {
        return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_sample must be non-null" };
    }
    if(sample_index >= dataset->samples.size()) {
        return gsx_error{ GSX_ERROR_OUT_OF_RANGE, "sample_index is out of range" };
    }

    {
        std::lock_guard<std::mutex> lock(dataset->mutex);
        if(dataset->first_callback_thread_id == std::thread::id{}) {
            dataset->first_callback_thread_id = std::this_thread::get_id();
        }
        if(std::this_thread::get_id() != dataset->caller_thread_id) {
            dataset->saw_non_caller_thread = true;
        }
        dataset->outstanding_samples += 1;
        if(dataset->outstanding_samples > dataset->max_outstanding_samples) {
            dataset->max_outstanding_samples = dataset->outstanding_samples;
        }
    }

    sample = &dataset->samples[(size_t)sample_index];
    dataset->get_sample_calls += 1;
    dataset->fetched_indices.push_back(sample_index);

    std::memset(out_sample, 0, sizeof(*out_sample));
    out_sample->intrinsics = sample->intrinsics;
    out_sample->pose = sample->pose;
    if(dataset->source_data_type == GSX_DATA_TYPE_U8) {
        out_sample->rgb_data = sample->omit_rgb_data ? nullptr : sample->rgb_u8.data();
        out_sample->alpha_data = sample->omit_alpha_data ? nullptr : (dataset->has_alpha ? sample->alpha_u8.data() : nullptr);
        out_sample->invdepth_data = sample->omit_invdepth_data ? nullptr : (dataset->has_invdepth ? sample->invdepth_u8.data() : nullptr);
    } else {
        out_sample->rgb_data = sample->omit_rgb_data ? nullptr : sample->rgb_f32.data();
        out_sample->alpha_data = sample->omit_alpha_data ? nullptr : (dataset->has_alpha ? sample->alpha_f32.data() : nullptr);
        out_sample->invdepth_data = sample->omit_invdepth_data ? nullptr : (dataset->has_invdepth ? sample->invdepth_f32.data() : nullptr);
    }
    out_sample->stable_sample_id = sample->stable_sample_id;
    out_sample->has_stable_sample_id = sample->has_stable_sample_id;
    out_sample->release_token = sample->release_token;
    return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static void test_dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
    auto *dataset = static_cast<TestDatasetObject *>(object);

    if(dataset == nullptr || sample == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(dataset->mutex);
        if(dataset->outstanding_samples != 0) {
            dataset->outstanding_samples -= 1;
        }
    }
    dataset->release_calls += 1;
    dataset->released_tokens.push_back(sample->release_token);
}

static gsx_dataset_desc make_dataset_desc(TestDatasetObject *dataset, gsx_data_type source_data_type, gsx_index_t width, gsx_index_t height)
{
    gsx_dataset_desc desc{};

    dataset->source_data_type = source_data_type;
    desc.object = dataset;
    desc.image_data_type = source_data_type;
    desc.width = width;
    desc.height = height;
    desc.has_rgb = true;
    desc.has_alpha = dataset->has_alpha;
    desc.has_invdepth = dataset->has_invdepth;
    desc.get_length = test_dataset_get_length;
    desc.get_sample = test_dataset_get_sample;
    desc.release_sample = test_dataset_release_sample;
    return desc;
}

static gsx_dataset_t init_dataset(TestDatasetObject *dataset, gsx_data_type source_data_type, gsx_index_t width, gsx_index_t height)
{
    gsx_dataset_t handle = nullptr;
    gsx_dataset_desc desc = make_dataset_desc(dataset, source_data_type, width, height);

    EXPECT_GSX_CODE(gsx_dataset_init(&handle, &desc), GSX_ERROR_SUCCESS);
    return handle;
}

static std::vector<float> download_f32_tensor(gsx_backend_t backend, gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    std::vector<float> values;

    if(gsx_tensor_get_info(tensor, &info).code != GSX_ERROR_SUCCESS) {
        return values;
    }
    if(info.size_bytes % sizeof(float) != 0U) {
        return values;
    }
    values.resize((size_t)(info.size_bytes / sizeof(float)));
    if(gsx_backend_major_stream_sync(backend).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    if(gsx_tensor_download(tensor, values.data(), info.size_bytes).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    if(gsx_backend_major_stream_sync(backend).code != GSX_ERROR_SUCCESS) {
        return {};
    }
    return values;
}

static void expect_tensor_shape(
    gsx_tensor_t tensor,
    gsx_data_type data_type,
    const std::array<gsx_index_t, GSX_TENSOR_MAX_DIM> &shape,
    gsx_index_t rank)
{
    gsx_tensor_info info{};

    ASSERT_GSX_SUCCESS(gsx_tensor_get_info(tensor, &info));
    EXPECT_EQ(info.data_type, data_type);
    EXPECT_EQ(info.storage_format, GSX_STORAGE_FORMAT_CHW);
    EXPECT_EQ(info.rank, rank);
    for(gsx_index_t dim = 0; dim < rank; ++dim) {
        EXPECT_EQ(info.shape[dim], shape[(size_t)dim]);
    }
}

static TestSample make_u8_sample(
    gsx_index_t width,
    gsx_index_t height,
    const std::vector<uint8_t> &rgb_values,
    const std::vector<uint8_t> &alpha_values = {},
    const std::vector<uint8_t> &invdepth_values = {})
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
    sample.rgb_u8 = rgb_values;
    sample.alpha_u8 = alpha_values;
    sample.invdepth_u8 = invdepth_values;
    sample.has_stable_sample_id = true;
    sample.stable_sample_id = 99;
    sample.release_token = reinterpret_cast<void *>(uintptr_t{ 0x1234 });
    return sample;
}

static TestSample make_f32_sample(
    gsx_index_t width,
    gsx_index_t height,
    const std::vector<float> &rgb_values,
    const std::vector<float> &alpha_values = {},
    const std::vector<float> &invdepth_values = {})
{
    TestSample sample{};

    sample.intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    sample.intrinsics.fx = 1.0f;
    sample.intrinsics.fy = 1.0f;
    sample.intrinsics.cx = 0.5f;
    sample.intrinsics.cy = 0.5f;
    sample.intrinsics.camera_id = 3;
    sample.intrinsics.width = width;
    sample.intrinsics.height = height;
    sample.pose.rot.w = 1.0f;
    sample.pose.camera_id = 3;
    sample.rgb_f32 = rgb_values;
    sample.alpha_f32 = alpha_values;
    sample.invdepth_f32 = invdepth_values;
    return sample;
}

TEST(DataRuntime, DatasetInitAndInfoExposeFixedMetadata)
{
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc desc{};
    gsx_dataset_info info{};

    desc.object = &dataset_object;
    EXPECT_GSX_CODE(gsx_dataset_init(&dataset, &desc), GSX_ERROR_INVALID_ARGUMENT);

    dataset_object.force_zero_length = true;
    desc = make_dataset_desc(&dataset_object, GSX_DATA_TYPE_U8, 2, 2);
    EXPECT_GSX_CODE(gsx_dataset_init(&dataset, &desc), GSX_ERROR_INVALID_ARGUMENT);

    dataset_object.force_zero_length = false;
    dataset_object.has_alpha = true;
    dataset_object.has_invdepth = true;
    dataset_object.samples.push_back(make_u8_sample(
        2,
        2,
        { 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110 },
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 }));
    desc = make_dataset_desc(&dataset_object, GSX_DATA_TYPE_U8, 2, 2);
    ASSERT_GSX_SUCCESS(gsx_dataset_init(&dataset, &desc));
    ASSERT_GSX_SUCCESS(gsx_dataset_get_info(dataset, &info));
    EXPECT_EQ(info.length, 1U);
    EXPECT_EQ(info.image_data_type, GSX_DATA_TYPE_U8);
    EXPECT_EQ(info.width, 2);
    EXPECT_EQ(info.height, 2);
    EXPECT_TRUE(info.has_rgb);
    EXPECT_TRUE(info.has_alpha);
    EXPECT_TRUE(info.has_invdepth);
    EXPECT_EQ(dataset_object.get_length_calls, 2U);
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
}

TEST(DataRuntime, DatasetInitRejectsUnsupportedSourceType)
{
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataset_desc desc{};

    dataset_object.samples.push_back(make_u8_sample(1, 1, { 1, 2, 3 }));
    desc = make_dataset_desc(&dataset_object, GSX_DATA_TYPE_I32, 1, 1);
    EXPECT_GSX_CODE(gsx_dataset_init(&dataset, &desc), GSX_ERROR_NOT_SUPPORTED);
}

TEST(DataRuntime, DataloaderInitRejectsInvalidAsyncAndDatasetReuse)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t loader = nullptr;
    gsx_dataloader_t second_loader = nullptr;
    gsx_dataloader_desc dataloader_desc{};

    ASSERT_NE(backend, nullptr);

    dataset_object.samples.push_back(make_u8_sample(2, 2, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }));
    dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_U8, 2, 2);
    ASSERT_NE(dataset, nullptr);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.enable_async_prefetch = true;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_INVALID_ARGUMENT);

    dataloader_desc.enable_async_prefetch = false;
    dataloader_desc.prefetch_count = 1;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_INVALID_ARGUMENT);

    dataloader_desc.prefetch_count = 0;
    dataloader_desc.image_data_type = GSX_DATA_TYPE_U8;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F16;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_BF16;
    EXPECT_GSX_CODE(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&loader, backend, dataset, &dataloader_desc));
    EXPECT_GSX_CODE(gsx_dataset_free(dataset), GSX_ERROR_INVALID_STATE);
    EXPECT_GSX_CODE(gsx_dataloader_init(&second_loader, backend, dataset, &dataloader_desc), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(loader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, NextExProducesFixedCHWOutputAndPreservesMetadata)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    gsx_dataloader_result result{};

    ASSERT_NE(backend, nullptr);

    dataset_object.has_alpha = true;
    dataset_object.has_invdepth = true;
    dataset_object.samples.push_back(make_u8_sample(
        2,
        2,
        { 0, 10, 20, 40, 50, 60, 80, 90, 100, 120, 130, 140 },
        { 10, 30, 50, 70 },
        { 20, 40, 60, 80 }));
    dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_U8, 2, 2);
    ASSERT_NE(dataset, nullptr);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));
    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));

    EXPECT_EQ(result.boundary_flags, GSX_DATALOADER_BOUNDARY_NEW_EPOCH | GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION);
    EXPECT_EQ(result.epoch_index, 0U);
    EXPECT_EQ(result.stable_sample_index, 0U);
    EXPECT_TRUE(result.has_stable_sample_id);
    EXPECT_EQ(result.stable_sample_id, 99U);
    EXPECT_EQ(result.intrinsics.width, 2);
    EXPECT_EQ(result.intrinsics.height, 2);

    expect_tensor_shape(
        result.rgb_image, GSX_DATA_TYPE_F32, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 3, 2, 2, 0 }, 3);
    expect_tensor_shape(
        result.alpha_image, GSX_DATA_TYPE_F32, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 2, 2, 0 }, 3);
    expect_tensor_shape(
        result.invdepth_image, GSX_DATA_TYPE_F32, std::array<gsx_index_t, GSX_TENSOR_MAX_DIM>{ 1, 2, 2, 0 }, 3);
    EXPECT_EQ(download_f32_tensor(backend, result.rgb_image), (std::vector<float>{ 0.0f, 40.0f, 80.0f, 120.0f, 10.0f, 50.0f, 90.0f, 130.0f, 20.0f, 60.0f, 100.0f, 140.0f }));
    EXPECT_EQ(download_f32_tensor(backend, result.alpha_image), (std::vector<float>{ 10.0f, 30.0f, 50.0f, 70.0f }));
    EXPECT_EQ(download_f32_tensor(backend, result.invdepth_image), (std::vector<float>{ 20.0f, 40.0f, 60.0f, 80.0f }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.boundary_flags, GSX_DATALOADER_BOUNDARY_NEW_EPOCH);
    EXPECT_EQ(result.epoch_index, 1U);
    EXPECT_EQ(dataset_object.release_calls, 2U);
    EXPECT_EQ(dataset_object.released_tokens[0], reinterpret_cast<void *>(uintptr_t{ 0x1234 }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, DataloaderRejectsNonF32OutputEvenForFloatSource)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};

    ASSERT_NE(backend, nullptr);

    dataset_object.has_alpha = true;
    dataset_object.has_invdepth = true;
    dataset_object.samples.push_back(make_f32_sample(
        2,
        1,
        { 1.0f, 2.0f, 3.0f, 254.7f, 255.0f, 100.4f },
        { 4.0f, 5.0f },
        { 6.0f, 7.0f }));
    dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_F32, 2, 1);
    ASSERT_NE(dataset, nullptr);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_U8;
    EXPECT_GSX_CODE(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, NextExRejectsMissingRequiredPointers)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    TestSample sample = make_u8_sample(1, 1, { 1, 2, 3 });

    ASSERT_NE(backend, nullptr);

    sample.omit_rgb_data = true;
    dataset_object.samples.push_back(sample);
    dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_U8, 1, 1);
    ASSERT_NE(dataset, nullptr);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    {
        gsx_dataloader_result result{};
        EXPECT_GSX_CODE(gsx_dataloader_next_ex(dataloader, &result), GSX_ERROR_INVALID_ARGUMENT);
    }
    EXPECT_EQ(dataset_object.release_calls, 1U);

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, NextExRejectsIntrinsicGeometryMismatch)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    TestSample sample = make_u8_sample(1, 1, { 1, 2, 3 });

    ASSERT_NE(backend, nullptr);

    sample.intrinsics.width = 2;
    dataset_object.samples.push_back(sample);
    dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_U8, 1, 1);
    ASSERT_NE(dataset, nullptr);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    {
        gsx_dataloader_result result{};
        EXPECT_GSX_CODE(gsx_dataloader_next_ex(dataloader, &result), GSX_ERROR_INVALID_ARGUMENT);
    }
    EXPECT_EQ(dataset_object.release_calls, 1U);

    ASSERT_GSX_SUCCESS(gsx_dataloader_free(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataset_free(dataset));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(DataRuntime, AsyncPrefetchUsesWorkerThreadAndPreservesOrder)
{
    gsx_backend_t backend = create_backend(GSX_BACKEND_TYPE_CPU);
    TestDatasetObject dataset_object{};
    gsx_dataset_t dataset = nullptr;
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc dataloader_desc{};
    gsx_dataloader_result result{};

    ASSERT_NE(backend, nullptr);

    dataset_object.caller_thread_id = std::this_thread::get_id();
    dataset_object.samples.push_back(make_u8_sample(1, 1, { 1, 2, 3 }));
    dataset_object.samples.push_back(make_u8_sample(1, 1, { 4, 5, 6 }));
    dataset_object.samples.push_back(make_u8_sample(1, 1, { 7, 8, 9 }));
    dataset_object.samples.push_back(make_u8_sample(1, 1, { 10, 11, 12 }));
    dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_U8, 1, 1);
    ASSERT_NE(dataset, nullptr);

    dataloader_desc.image_data_type = GSX_DATA_TYPE_F32;
    dataloader_desc.enable_async_prefetch = true;
    dataloader_desc.prefetch_count = 2;
    ASSERT_GSX_SUCCESS(gsx_dataloader_init(&dataloader, backend, dataset, &dataloader_desc));

    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.stable_sample_index, 0U);
    EXPECT_EQ(download_f32_tensor(backend, result.rgb_image), (std::vector<float>{ 1.0f, 2.0f, 3.0f }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.stable_sample_index, 1U);
    EXPECT_EQ(download_f32_tensor(backend, result.rgb_image), (std::vector<float>{ 4.0f, 5.0f, 6.0f }));

    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.stable_sample_index, 2U);
    EXPECT_EQ(download_f32_tensor(backend, result.rgb_image), (std::vector<float>{ 7.0f, 8.0f, 9.0f }));

    EXPECT_TRUE(dataset_object.saw_non_caller_thread);
    EXPECT_EQ(dataset_object.max_outstanding_samples, 1U);
    EXPECT_EQ(dataset_object.release_calls, 4U);

    ASSERT_GSX_SUCCESS(gsx_dataloader_reset(dataloader));
    ASSERT_GSX_SUCCESS(gsx_dataloader_next_ex(dataloader, &result));
    EXPECT_EQ(result.stable_sample_index, 0U);

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
        gsx_dataloader_t dataloader = nullptr;
        gsx_dataloader_desc desc{};
        gsx_dataloader_result result{};
        gsx_tensor_info info{};
        gsx_backend_t tensor_backend = nullptr;

        if(backend == nullptr) {
            continue;
        }

        dataset_object.samples.push_back(make_u8_sample(1, 1, { 9, 19, 29 }));
        dataset = init_dataset(&dataset_object, GSX_DATA_TYPE_U8, 1, 1);
        ASSERT_NE(dataset, nullptr);

        desc.image_data_type = GSX_DATA_TYPE_F32;
        desc.enable_async_prefetch = true;
        desc.prefetch_count = 1;
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
