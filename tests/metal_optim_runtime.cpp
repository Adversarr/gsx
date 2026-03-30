extern "C" {
#include "../gsx/src/gsx-impl.h"
}

#include <gtest/gtest.h>

#include <array>
#include <cmath>
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

struct ManualTensor {
    gsx_backend_buffer_t buffer = nullptr;
    gsx_tensor tensor{};
};

struct AdamRefGroup {
    std::vector<float> params;
    std::vector<float> grads;
    std::vector<float> m;
    std::vector<float> v;
    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float weight_decay = 0.0f;
    float epsilon = 0.0f;
    float max_grad = 0.0f;
    gsx_size_t step = 0;
};

static bool has_metal_device()
{
    gsx_index_t metal_device_count = 0;
    gsx_error error = gsx_backend_registry_init();
    if(error.code != GSX_ERROR_SUCCESS) {
        return false;
    }
    error = gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_METAL, &metal_device_count);
    return error.code == GSX_ERROR_SUCCESS && metal_device_count > 0;
}

static gsx_backend_device_t get_metal_backend_device()
{
    gsx_backend_device_t backend_device = nullptr;

    EXPECT_GSX_CODE(gsx_backend_registry_init(), GSX_ERROR_SUCCESS);
    EXPECT_GSX_CODE(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device), GSX_ERROR_SUCCESS);
    return backend_device;
}

static gsx_backend_t create_metal_backend()
{
    gsx_backend_t backend = nullptr;
    gsx_backend_desc backend_desc{};

    backend_desc.device = get_metal_backend_device();
    EXPECT_NE(backend_desc.device, nullptr);
    EXPECT_GSX_CODE(gsx_backend_init(&backend, &backend_desc), GSX_ERROR_SUCCESS);
    return backend;
}

static void sync_backend(gsx_backend_t backend)
{
    ASSERT_GSX_SUCCESS(gsx_backend_major_stream_sync(backend));
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type)
{
    gsx_backend_buffer_type_t buffer_type = nullptr;

    EXPECT_GSX_CODE(gsx_backend_find_buffer_type(backend, type, &buffer_type), GSX_ERROR_SUCCESS);
    return buffer_type;
}

static gsx_size_t gsx_element_size_bytes(gsx_data_type data_type)
{
    switch(data_type) {
    case GSX_DATA_TYPE_F32:
    case GSX_DATA_TYPE_I32:
        return 4;
    case GSX_DATA_TYPE_U8:
        return 1;
    default:
        return 0;
    }
}

static void configure_rank1_tensor(ManualTensor *manual_tensor, gsx_data_type data_type, gsx_index_t length)
{
    std::memset(&manual_tensor->tensor, 0, sizeof(manual_tensor->tensor));
    manual_tensor->tensor.backing_buffer = manual_tensor->buffer;
    manual_tensor->tensor.offset_bytes = 0;
    manual_tensor->tensor.size_bytes = (gsx_size_t)length * gsx_element_size_bytes(data_type);
    manual_tensor->tensor.alloc_span_bytes = manual_tensor->tensor.size_bytes;
    manual_tensor->tensor.alloc_start_bytes = 0;
    manual_tensor->tensor.alloc_end_bytes = manual_tensor->tensor.size_bytes;
    manual_tensor->tensor.rank = 1;
    manual_tensor->tensor.shape[0] = length;
    manual_tensor->tensor.data_type = data_type;
    manual_tensor->tensor.storage_format = GSX_STORAGE_FORMAT_CHW;
}

template <typename T>
static ManualTensor make_rank1_tensor(gsx_backend_buffer_type_t buffer_type, gsx_data_type data_type, const std::vector<T> &values)
{
    ManualTensor manual_tensor{};
    gsx_backend_buffer_desc buffer_desc{};
    gsx_error error{};

    buffer_desc.buffer_type = buffer_type;
    buffer_desc.size_bytes = (gsx_size_t)values.size() * sizeof(T);
    error = gsx_backend_buffer_init(&manual_tensor.buffer, &buffer_desc);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return {};
    }
    configure_rank1_tensor(&manual_tensor, data_type, (gsx_index_t)values.size());
    if(!values.empty()) {
        error = gsx_backend_buffer_upload(manual_tensor.buffer, 0, values.data(), buffer_desc.size_bytes);
        EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
        if(error.code != GSX_ERROR_SUCCESS) {
            gsx_backend_buffer_free(manual_tensor.buffer);
            return {};
        }
    }
    return manual_tensor;
}

template <typename T>
static void rebind_rank1_tensor(ManualTensor *manual_tensor, gsx_backend_buffer_type_t buffer_type, gsx_data_type data_type, const std::vector<T> &values)
{
    gsx_backend_buffer_t new_buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};

    buffer_desc.buffer_type = buffer_type;
    buffer_desc.size_bytes = (gsx_size_t)values.size() * sizeof(T);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&new_buffer, &buffer_desc));
    if(!values.empty()) {
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(new_buffer, 0, values.data(), buffer_desc.size_bytes));
    }
    if(manual_tensor->buffer != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(manual_tensor->buffer));
    }
    manual_tensor->buffer = new_buffer;
    configure_rank1_tensor(manual_tensor, data_type, (gsx_index_t)values.size());
}

template <typename T>
static std::vector<T> download_rank1_tensor(gsx_backend_t backend, const ManualTensor &manual_tensor)
{
    const gsx_size_t byte_count = manual_tensor.tensor.size_bytes;
    std::vector<T> values((std::size_t)manual_tensor.tensor.shape[0]);

    if(byte_count != 0) {
        EXPECT_GSX_CODE(gsx_backend_buffer_download(manual_tensor.buffer, 0, values.data(), byte_count), GSX_ERROR_SUCCESS);
        sync_backend(backend);
    }
    return values;
}

static void destroy_manual_tensor(ManualTensor *manual_tensor)
{
    if(manual_tensor->buffer != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(manual_tensor->buffer));
        manual_tensor->buffer = nullptr;
    }
}

static gsx_optim_param_group_desc make_param_group_desc(
    gsx_optim_param_role role,
    ManualTensor *parameter,
    ManualTensor *gradient,
    float learning_rate,
    float beta1,
    float beta2,
    float weight_decay,
    float epsilon,
    float max_grad
)
{
    gsx_optim_param_group_desc desc{};

    desc.role = role;
    desc.parameter = &parameter->tensor;
    desc.gradient = &gradient->tensor;
    desc.learning_rate = learning_rate;
    desc.beta1 = beta1;
    desc.beta2 = beta2;
    desc.weight_decay = weight_decay;
    desc.epsilon = epsilon;
    desc.max_grad = max_grad;
    return desc;
}

static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 2e-4f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index=" << i;
    }
}

static float adam_clamp_gradient(const AdamRefGroup *group, float gradient)
{
    if(group->max_grad <= 0.0f) {
        return gradient;
    }
    if(gradient > group->max_grad) {
        return group->max_grad;
    }
    if(gradient < -group->max_grad) {
        return -group->max_grad;
    }
    return gradient;
}

static void adam_step(AdamRefGroup *group)
{
    group->step += 1;

    {
        double beta1_correction = 1.0 - std::pow((double)group->beta1, (double)group->step);
        double beta2_correction = 1.0 - std::pow((double)group->beta2, (double)group->step);
        for(std::size_t i = 0; i < group->params.size(); ++i) {
            float gradient = adam_clamp_gradient(group, group->grads[i]);
            float first_moment = group->beta1 * group->m[i] + (1.0f - group->beta1) * gradient;
            float second_moment = group->beta2 * group->v[i] + (1.0f - group->beta2) * gradient * gradient;
            float parameter = group->params[i];

            group->m[i] = first_moment;
            group->v[i] = second_moment;
            if(group->weight_decay > 0.0f) {
                parameter -= group->learning_rate * group->weight_decay * parameter;
            }
            parameter -=
                group->learning_rate
                * ((float)((double)first_moment / beta1_correction)
                    / (std::sqrt((float)((double)second_moment / beta2_correction)) + group->epsilon));
            group->params[i] = parameter;
        }
    }
}

static void reset_group_state(AdamRefGroup *group)
{
    std::fill(group->m.begin(), group->m.end(), 0.0f);
    std::fill(group->v.begin(), group->v.end(), 0.0f);
    group->step = 0;
}

static void permute_group(AdamRefGroup *group, const std::vector<int32_t> &permutation)
{
    std::vector<float> params(group->params.size());
    std::vector<float> grads(group->grads.size());
    std::vector<float> m(group->m.size());
    std::vector<float> v(group->v.size());

    for(std::size_t i = 0; i < permutation.size(); ++i) {
        params[i] = group->params[(std::size_t)permutation[i]];
        grads[i] = group->grads[(std::size_t)permutation[i]];
        m[i] = group->m[(std::size_t)permutation[i]];
        v[i] = group->v[(std::size_t)permutation[i]];
    }
    group->params = std::move(params);
    group->grads = std::move(grads);
    group->m = std::move(m);
    group->v = std::move(v);
}

static void gather_group(AdamRefGroup *group, const std::vector<int32_t> &indices)
{
    std::vector<float> params(indices.size());
    std::vector<float> grads(indices.size());
    std::vector<float> m(indices.size());
    std::vector<float> v(indices.size());

    for(std::size_t i = 0; i < indices.size(); ++i) {
        params[i] = group->params[(std::size_t)indices[i]];
        grads[i] = group->grads[(std::size_t)indices[i]];
        m[i] = group->m[(std::size_t)indices[i]];
        v[i] = group->v[(std::size_t)indices[i]];
    }
    group->params = std::move(params);
    group->grads = std::move(grads);
    group->m = std::move(m);
    group->v = std::move(v);
}

static void resize_group(AdamRefGroup *group, const std::vector<float> &new_params, const std::vector<float> &new_grads)
{
    for(float value : new_params) {
        group->params.push_back(value);
        group->m.push_back(0.0f);
        group->v.push_back(0.0f);
    }
    for(float value : new_grads) {
        group->grads.push_back(value);
    }
}

class MetalOptimRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if(!has_metal_device()) {
            GTEST_SKIP() << "No Metal devices available";
        }
    }
};

TEST_F(MetalOptimRuntimeTest, InitRejectsHostPinnedStateBufferTypeForMetal)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t host_pinned_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST_PINNED);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;

    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    EXPECT_EQ(optim->state_buffer_type, device_buffer_type);
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));

    optim_desc.state_buffer_type = host_pinned_buffer_type;
    EXPECT_GSX_CODE(gsx_optim_init(&optim, backend, &optim_desc), GSX_ERROR_NOT_SUPPORTED);

    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, StepSelectionResetAndLearningRateMatchReferenceAdam)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter_a = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, -2.0f });
    ManualTensor gradient_a = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.2f, -0.4f });
    ManualTensor parameter_b = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.5f });
    ManualTensor gradient_b = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.3f });
    gsx_optim_param_group_desc groups[2]{};
    gsx_optim_desc optim_desc{};
    gsx_optim_step_request request{};
    gsx_optim_param_group_desc desc_by_role{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref_a{};
    AdamRefGroup ref_b{};

    groups[0] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_MEAN3D,
        &parameter_a,
        &gradient_a,
        0.1f,
        0.9f,
        0.99f,
        0.01f,
        1e-6f,
        0.0f
    );
    groups[1] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_CUSTOM,
        &parameter_b,
        &gradient_b,
        0.2f,
        0.8f,
        0.95f,
        0.0f,
        1e-6f,
        0.0f
    );

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 2;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref_a.params = { 1.0f, -2.0f };
    ref_a.grads = { 0.2f, -0.4f };
    ref_a.m = { 0.0f, 0.0f };
    ref_a.v = { 0.0f, 0.0f };
    ref_a.learning_rate = 0.1f;
    ref_a.beta1 = 0.9f;
    ref_a.beta2 = 0.99f;
    ref_a.weight_decay = 0.01f;
    ref_a.epsilon = 1e-6f;

    ref_b.params = { 0.5f, -1.5f };
    ref_b.grads = { 0.1f, 0.3f };
    ref_b.m = { 0.0f, 0.0f };
    ref_b.v = { 0.0f, 0.0f };
    ref_b.learning_rate = 0.2f;
    ref_b.beta1 = 0.8f;
    ref_b.beta2 = 0.95f;
    ref_b.weight_decay = 0.0f;
    ref_b.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref_a);
    adam_step(&ref_b);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter_b), ref_b.params);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_a.buffer, 0, std::array<float, 2>{ 0.3f, 0.1f }.data(), sizeof(float) * 2));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_b.buffer, 0, std::array<float, 2>{ -0.7f, 0.8f }.data(), sizeof(float) * 2));
    ref_a.grads = { 0.3f, 0.1f };
    ref_b.grads = { -0.7f, 0.8f };
    request.role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D;
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, &request));
    adam_step(&ref_a);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter_b), ref_b.params);

    ASSERT_GSX_SUCCESS(gsx_optim_set_learning_rate_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, 0.05f));
    ASSERT_GSX_SUCCESS(gsx_optim_get_param_group_desc_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &desc_by_role));
    EXPECT_NEAR(desc_by_role.learning_rate, 0.05f, 1e-7f);
    ref_a.learning_rate = 0.05f;

    ASSERT_GSX_SUCCESS(gsx_optim_reset(optim));
    reset_group_state(&ref_a);
    reset_group_state(&ref_b);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_a.buffer, 0, std::array<float, 2>{ -0.2f, 0.6f }.data(), sizeof(float) * 2));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_b.buffer, 0, std::array<float, 2>{ 0.4f, -0.5f }.data(), sizeof(float) * 2));
    ref_a.grads = { -0.2f, 0.6f };
    ref_b.grads = { 0.4f, -0.5f };

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref_a);
    adam_step(&ref_b);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter_b), ref_b.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter_a);
    destroy_manual_tensor(&gradient_a);
    destroy_manual_tensor(&parameter_b);
    destroy_manual_tensor(&gradient_b);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, ElementClampMatchesReferenceAdam)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, -1.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 4.0f });
    gsx_optim_param_group_desc group{};
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref{};

    group = make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 2.5f);
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, -1.0f };
    ref.grads = { 3.0f, 4.0f };
    ref.m = { 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f };
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;
    ref.max_grad = 2.5f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(backend, gradient), ref.grads, 3e-4f);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter), ref.params, 3e-4f);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, AdamVectorizedPathHandlesTailElements)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.25f, -0.5f, 0.75f, -1.0f, 1.25f });
    gsx_optim_param_group_desc group{};
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref{};

    group = make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.05f, 0.9f, 0.999f, 0.01f, 1e-6f, 0.0f);
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f };
    ref.grads = { 0.25f, -0.5f, 0.75f, -1.0f, 1.25f };
    ref.m = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    ref.learning_rate = 0.05f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.999f;
    ref.weight_decay = 0.01f;
    ref.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter), ref.params, 3e-4f);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, PermuteGatherResizeKeepStateAcrossSteps)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f, 1.5f });
    ManualTensor permutation = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
    ManualTensor gather_indices = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 2 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 1.0f, 2.0f });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { -0.5f, 0.25f, 0.75f });
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation.tensor));

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 2.0f });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.0f });
    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, &gather_indices.tensor));

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 2.0f, 9.0f });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.0f, 0.2f });
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 3));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation);
    destroy_manual_tensor(&gather_indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, PermuteRejectsInvalidControlAndKeepsAdamState)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor invalid_permutation = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 0, 1 });
    ManualTensor permutation = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref{};
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, 2.0f, 3.0f };
    ref.grads = { 1.0f, 2.0f, 3.0f };
    ref.m = { 0.0f, 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f, 0.0f };
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);

    EXPECT_GSX_CODE(gsx_optim_permute(optim, &invalid_permutation.tensor), GSX_ERROR_INVALID_ARGUMENT);

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { ref.params[2], ref.params[0], ref.params[1] });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.0f, 2.0f });
    permute_group(&ref, { 2, 0, 1 });
    ref.grads = { 0.5f, -1.0f, 2.0f };

    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&invalid_permutation);
    destroy_manual_tensor(&permutation);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, GatherRejectsMismatchedIndicesAndKeepsAdamState)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f, 1.5f });
    ManualTensor invalid_indices = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 1, 2 });
    ManualTensor gather_indices = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 2 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref{};
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, 2.0f, 3.0f };
    ref.grads = { 0.5f, 1.0f, 1.5f };
    ref.m = { 0.0f, 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f, 0.0f };
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { ref.params[0], ref.params[2] });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { 1.25f, -0.5f });
    EXPECT_GSX_CODE(gsx_optim_gather(optim, &invalid_indices.tensor), GSX_ERROR_INVALID_ARGUMENT);

    gather_group(&ref, { 0, 2 });
    ref.grads = { 1.25f, -0.5f };
    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, &gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&invalid_indices);
    destroy_manual_tensor(&gather_indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, ResizeRejectsMismatchedCountAndZeroInitializesNewStateRows)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref{};
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, 2.0f };
    ref.grads = { 0.5f, 1.0f };
    ref.m = { 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f };
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { ref.params[0], ref.params[1], 10.0f });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { 0.25f, -0.75f, 1.5f });
    EXPECT_GSX_CODE(gsx_optim_resize(optim, 4), GSX_ERROR_INVALID_STATE);

    resize_group(&ref, { 10.0f }, { 1.5f });
    ref.grads[0] = 0.25f;
    ref.grads[1] = -0.75f;
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 3));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(backend, parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, ConsecutivePermutesReuseScratchAndFreeAfterCommit)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor permutation_a = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
    ManualTensor permutation_b = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 1, 2, 0 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 1.0f, 2.0f });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { -0.5f, 0.4f, 0.3f });
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation_a.tensor));

    rebind_rank1_tensor<float>(&parameter, device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    rebind_rank1_tensor<float>(&gradient, device_buffer_type, GSX_DATA_TYPE_F32, { 0.25f, -0.75f, 0.5f });
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation_b.tensor));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation_a);
    destroy_manual_tensor(&permutation_b);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST_F(MetalOptimRuntimeTest, HandleRemainsValidAfterMutationValidationFailures)
{
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    ManualTensor parameter = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor wrong_length = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 1 });
    ManualTensor wrong_type = make_rank1_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32, { 2.0f, 0.0f, 1.0f });
    ManualTensor duplicate_perm = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 0, 1 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    gsx_optim_info info{};
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    EXPECT_GSX_CODE(gsx_optim_permute(optim, &wrong_length.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    EXPECT_GSX_CODE(gsx_optim_permute(optim, &wrong_type.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    EXPECT_GSX_CODE(gsx_optim_permute(optim, &duplicate_perm.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    EXPECT_GSX_CODE(gsx_optim_resize(optim, 2), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient.buffer, 0, std::array<float, 3>{ -0.3f, 0.4f, 0.2f }.data(), sizeof(float) * 3));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&wrong_length);
    destroy_manual_tensor(&wrong_type);
    destroy_manual_tensor(&duplicate_perm);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

template <typename T>
static ManualTensor make_rank2_tensor(gsx_backend_buffer_type_t buffer_type, gsx_data_type data_type, const std::vector<T> &values, gsx_index_t rows, gsx_index_t cols)
{
    ManualTensor manual_tensor{};
    gsx_backend_buffer_desc buffer_desc{};
    gsx_error error{};

    buffer_desc.buffer_type = buffer_type;
    buffer_desc.size_bytes = (gsx_size_t)values.size() * sizeof(T);
    error = gsx_backend_buffer_init(&manual_tensor.buffer, &buffer_desc);
    EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
    if(error.code != GSX_ERROR_SUCCESS) {
        return {};
    }
    std::memset(&manual_tensor.tensor, 0, sizeof(manual_tensor.tensor));
    manual_tensor.tensor.backing_buffer = manual_tensor.buffer;
    manual_tensor.tensor.size_bytes = buffer_desc.size_bytes;
    manual_tensor.tensor.alloc_span_bytes = buffer_desc.size_bytes;
    manual_tensor.tensor.alloc_end_bytes = buffer_desc.size_bytes;
    manual_tensor.tensor.rank = 2;
    manual_tensor.tensor.shape[0] = rows;
    manual_tensor.tensor.shape[1] = cols;
    manual_tensor.tensor.data_type = data_type;
    manual_tensor.tensor.storage_format = GSX_STORAGE_FORMAT_CHW;
    if(!values.empty()) {
        error = gsx_backend_buffer_upload(manual_tensor.buffer, 0, values.data(), buffer_desc.size_bytes);
        EXPECT_EQ(error.code, GSX_ERROR_SUCCESS) << (error.message != nullptr ? error.message : "");
        if(error.code != GSX_ERROR_SUCCESS) {
            gsx_backend_buffer_free(manual_tensor.buffer);
            return {};
        }
    }
    return manual_tensor;
}

template <typename T>
static void rebind_rank2_tensor(
    ManualTensor *manual_tensor,
    gsx_backend_buffer_type_t buffer_type,
    gsx_data_type data_type,
    const std::vector<T> &values,
    gsx_index_t rows,
    gsx_index_t cols
)
{
    gsx_backend_buffer_t new_buffer = nullptr;
    gsx_backend_buffer_desc buffer_desc{};

    buffer_desc.buffer_type = buffer_type;
    buffer_desc.size_bytes = (gsx_size_t)values.size() * sizeof(T);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_init(&new_buffer, &buffer_desc));
    if(!values.empty()) {
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(new_buffer, 0, values.data(), buffer_desc.size_bytes));
    }
    if(manual_tensor->buffer != nullptr) {
        ASSERT_GSX_SUCCESS(gsx_backend_buffer_free(manual_tensor->buffer));
    }

    manual_tensor->buffer = new_buffer;
    std::memset(&manual_tensor->tensor, 0, sizeof(manual_tensor->tensor));
    manual_tensor->tensor.backing_buffer = manual_tensor->buffer;
    manual_tensor->tensor.size_bytes = buffer_desc.size_bytes;
    manual_tensor->tensor.alloc_span_bytes = buffer_desc.size_bytes;
    manual_tensor->tensor.alloc_end_bytes = buffer_desc.size_bytes;
    manual_tensor->tensor.rank = 2;
    manual_tensor->tensor.shape[0] = rows;
    manual_tensor->tensor.shape[1] = cols;
    manual_tensor->tensor.data_type = data_type;
    manual_tensor->tensor.storage_format = GSX_STORAGE_FORMAT_CHW;
}

/* Verify that the row gather GPU kernel handles row_floats > 1 correctly by using a
 * rank-2 parameter tensor (shape [3, 4] = 3 rows of 4 floats) and comparing against
 * a CPU reference after permute + step. */
TEST_F(MetalOptimRuntimeTest, GpuGatherKernelPermutesMultiFloatRowsCorrectly)
{
    constexpr gsx_index_t N = 3, M = 4; /* rows x floats-per-row */
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);

    /* Row 0: {1,2,3,4}, Row 1: {5,6,7,8}, Row 2: {9,10,11,12} */
    ManualTensor parameter = make_rank2_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32,
        { 1.f,2.f,3.f,4.f, 5.f,6.f,7.f,8.f, 9.f,10.f,11.f,12.f }, N, M);
    ManualTensor gradient = make_rank2_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32,
        { .1f,.2f,.3f,.4f, .5f,.6f,.7f,.8f, .9f,1.f,1.1f,1.2f }, N, M);
    /* permutation {2, 0, 1}: new row 0 ← old row 2, new row 1 ← old row 0, new row 2 ← old row 1 */
    ManualTensor permutation = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

/* Verify that the row compact GPU kernel handles row_floats > 1 correctly by using a
 * rank-2 parameter tensor (shape [4, 3]) and comparing against a CPU reference after
 * gather + step. */
TEST_F(MetalOptimRuntimeTest, GpuGatherKernelCompactsMultiFloatRowsCorrectly)
{
    constexpr gsx_index_t N = 4, M = 3; /* rows x floats-per-row */
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);

    ManualTensor parameter = make_rank2_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32,
        { 1.f,2.f,3.f, 4.f,5.f,6.f, 7.f,8.f,9.f, 10.f,11.f,12.f }, N, M);
    ManualTensor gradient = make_rank2_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32,
        { .1f,.2f,.3f, .4f,.5f,.6f, .7f,.8f,.9f, 1.f,1.1f,1.2f }, N, M);
    /* keep rows 0, 2, 3 */
    ManualTensor gather_indices = make_rank1_tensor<int32_t>(device_buffer_type, GSX_DATA_TYPE_I32, { 0, 2, 3 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    rebind_rank2_tensor<float>(
        &parameter,
        device_buffer_type,
        GSX_DATA_TYPE_F32,
        { 1.f,2.f,3.f, 7.f,8.f,9.f, 10.f,11.f,12.f },
        3,
        M
    );
    rebind_rank2_tensor<float>(
        &gradient,
        device_buffer_type,
        GSX_DATA_TYPE_F32,
        { .1f,.2f,.3f, .7f,.8f,.9f, 1.f,1.1f,1.2f },
        3,
        M
    );
    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, &gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&gather_indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

/* Verify that the resize path copies existing moment rows and zero-initialises new rows
 * by using a rank-2 tensor (shape [2, 4] resized to [4, 4]) and comparing against a CPU
 * reference after resize + step. */
TEST_F(MetalOptimRuntimeTest, GpuResizePreservesOldMomentsAndZeroesNewRows)
{
    constexpr gsx_index_t N_OLD = 2, M = 4;
    gsx_backend_t backend = create_metal_backend();
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);

    ManualTensor parameter = make_rank2_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32,
        { 1.f,2.f,3.f,4.f, 5.f,6.f,7.f,8.f }, N_OLD, M);
    ManualTensor gradient = make_rank2_tensor<float>(device_buffer_type, GSX_DATA_TYPE_F32,
        { .1f,.2f,.3f,.4f, .5f,.6f,.7f,.8f }, N_OLD, M);
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    rebind_rank2_tensor<float>(
        &parameter,
        device_buffer_type,
        GSX_DATA_TYPE_F32,
        { 1.f,2.f,3.f,4.f, 5.f,6.f,7.f,8.f, 9.f,10.f,11.f,12.f, 13.f,14.f,15.f,16.f },
        4,
        M
    );
    rebind_rank2_tensor<float>(
        &gradient,
        device_buffer_type,
        GSX_DATA_TYPE_F32,
        { .1f,.2f,.3f,.4f, .5f,.6f,.7f,.8f, .9f,1.f,1.1f,1.2f, 1.3f,1.4f,1.5f,1.6f },
        4,
        M
    );
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 4));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

} // namespace
