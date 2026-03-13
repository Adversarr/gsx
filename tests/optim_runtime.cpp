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
    float max_grad_norm = 0.0f;
    gsx_size_t step = 0;
};

struct CpuOptimLayout {
    gsx_optim base;
    gsx_size_t *step_counts = nullptr;
    gsx_tensor_t *first_moments = nullptr;
    gsx_tensor_t *second_moments = nullptr;
    gsx_tensor_t *scratch_first_moments = nullptr;
    gsx_tensor_t *scratch_second_moments = nullptr;
    gsx_arena_t state_arena = nullptr;
    gsx_arena_t scratch_arena = nullptr;
};

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
static std::vector<T> download_rank1_tensor(const ManualTensor &manual_tensor)
{
    const gsx_size_t byte_count = manual_tensor.tensor.size_bytes;
    std::vector<T> values((std::size_t)manual_tensor.tensor.shape[0]);

    if(byte_count != 0) {
        EXPECT_GSX_CODE(gsx_backend_buffer_download(manual_tensor.buffer, 0, values.data(), byte_count), GSX_ERROR_SUCCESS);
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
    float max_grad_norm
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
    desc.max_grad_norm = max_grad_norm;
    return desc;
}

/*
 * Default tolerance for CPU-backend optimizer comparisons. The optimizer uses
 * float32 arithmetic with double-precision intermediate beta-correction terms.
 * 1e-5f covers all floating-point rounding variation across compilers and
 * optimization levels on CPU. This is intentionally stricter than what would be
 * required for cross-backend comparisons, where looser thresholds are needed to
 * account for differing FMA behavior or instruction ordering on other hardware.
 */
static void expect_near_vectors(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1e-5f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index=" << i;
    }
}

static void adam_apply_group_clip(AdamRefGroup *group)
{
    double norm_sq = 0.0;

    if(group->max_grad_norm <= 0.0f) {
        return;
    }
    for(float value : group->grads) {
        norm_sq += (double)value * (double)value;
    }

    const double norm = std::sqrt(norm_sq);
    if(norm == 0.0 || norm <= (double)group->max_grad_norm) {
        return;
    }

    const float scale = (float)((double)group->max_grad_norm / norm);
    for(float &value : group->grads) {
        value *= scale;
    }
}

static void adam_step(AdamRefGroup *group)
{
    adam_apply_group_clip(group);
    group->step += 1;

    const double beta1_correction = 1.0 - std::pow((double)group->beta1, (double)group->step);
    const double beta2_correction = 1.0 - std::pow((double)group->beta2, (double)group->step);
    for(std::size_t i = 0; i < group->params.size(); ++i) {
        const float gradient = group->grads[i];
        const float first_moment = group->beta1 * group->m[i] + (1.0f - group->beta1) * gradient;
        const float second_moment = group->beta2 * group->v[i] + (1.0f - group->beta2) * gradient * gradient;
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

static void prune_group(AdamRefGroup *group, const std::vector<uint8_t> &keep_mask)
{
    std::vector<float> params;
    std::vector<float> grads;
    std::vector<float> m;
    std::vector<float> v;

    for(std::size_t i = 0; i < keep_mask.size(); ++i) {
        if(keep_mask[i] == 0) {
            continue;
        }
        params.push_back(group->params[i]);
        grads.push_back(group->grads[i]);
        m.push_back(group->m[i]);
        v.push_back(group->v[i]);
    }
    group->params = std::move(params);
    group->grads = std::move(grads);
    group->m = std::move(m);
    group->v = std::move(v);
}

static void grow_group(AdamRefGroup *group, const std::vector<float> &new_params, const std::vector<float> &new_grads)
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

TEST(OptimRuntime, InitValidatesDescriptorsAndResolvesStateBufferType)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_t other_backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_backend_buffer_type_t device_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_backend_buffer_type_t foreign_host_buffer_type = find_buffer_type(other_backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor bad_gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f });
    gsx_optim_param_group_desc groups[2]{};
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    groups[0] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_MEAN3D,
        &parameter,
        &gradient,
        0.1f,
        0.9f,
        0.99f,
        0.01f,
        1e-6f,
        0.0f
    );

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 1;

    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    EXPECT_EQ(optim->state_buffer_type, device_buffer_type);
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));

    optim_desc.state_buffer_type = host_buffer_type;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    EXPECT_EQ(optim->state_buffer_type, host_buffer_type);
    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));

    optim_desc.state_buffer_type = foreign_host_buffer_type;
    EXPECT_GSX_CODE(gsx_optim_init(&optim, backend, &optim_desc), GSX_ERROR_INVALID_ARGUMENT);

    optim_desc.state_buffer_type = nullptr;
    groups[0].gradient = &bad_gradient.tensor;
    EXPECT_GSX_CODE(gsx_optim_init(&optim, backend, &optim_desc), GSX_ERROR_INVALID_ARGUMENT);

    groups[0].gradient = &gradient.tensor;
    groups[1] = groups[0];
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 2;
    EXPECT_GSX_CODE(gsx_optim_init(&optim, backend, &optim_desc), GSX_ERROR_INVALID_ARGUMENT);

    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&bad_gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(other_backend));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, StepSelectionResetAndLearningRateUpdatesMatchReferenceAdam)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter_a = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, -2.0f });
    ManualTensor gradient_a = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.2f, -0.4f });
    ManualTensor parameter_b = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.5f });
    ManualTensor gradient_b = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.3f });
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
    expect_near_vectors(download_rank1_tensor<float>(parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(parameter_b), ref_b.params);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_a.buffer, 0, std::array<float, 2>{ 0.3f, 0.1f }.data(), sizeof(float) * 2));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_b.buffer, 0, std::array<float, 2>{ -0.7f, 0.8f }.data(), sizeof(float) * 2));
    ref_a.grads = { 0.3f, 0.1f };
    ref_b.grads = { -0.7f, 0.8f };
    request.role_flags = GSX_OPTIM_PARAM_ROLE_FLAG_MEAN3D;
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, &request));
    adam_step(&ref_a);
    expect_near_vectors(download_rank1_tensor<float>(parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(parameter_b), ref_b.params);

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
    expect_near_vectors(download_rank1_tensor<float>(parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(parameter_b), ref_b.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter_a);
    destroy_manual_tensor(&gradient_a);
    destroy_manual_tensor(&parameter_b);
    destroy_manual_tensor(&gradient_b);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, StepAppliesPerGroupClipInsideFusedAdam)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, -1.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 4.0f });
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
    ref.max_grad_norm = 2.5f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(gradient), ref.grads);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, PermuteRejectsInvalidControlAndKeepsAdamState)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor invalid_permutation = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 0, 1 });
    ManualTensor permutation = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
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

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { ref.params[2], ref.params[0], ref.params[1] });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.0f, 2.0f });
    permute_group(&ref, { 2, 0, 1 });
    ref.grads = { 0.5f, -1.0f, 2.0f };

    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&invalid_permutation);
    destroy_manual_tensor(&permutation);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, PruneRejectsMismatchedMaskAndKeepsAdamState)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f, 1.5f });
    ManualTensor invalid_mask = make_rank1_tensor<uint8_t>(host_buffer_type, GSX_DATA_TYPE_U8, { 1, 0, 0 });
    ManualTensor keep_mask = make_rank1_tensor<uint8_t>(host_buffer_type, GSX_DATA_TYPE_U8, { 1, 0, 1 });
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

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { ref.params[0], ref.params[2] });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 1.25f, -0.5f });
    EXPECT_GSX_CODE(gsx_optim_prune(optim, &invalid_mask.tensor), GSX_ERROR_INVALID_STATE);

    prune_group(&ref, { 1, 0, 1 });
    ref.grads = { 1.25f, -0.5f };
    ASSERT_GSX_SUCCESS(gsx_optim_prune(optim, &keep_mask.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&invalid_mask);
    destroy_manual_tensor(&keep_mask);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, GrowRejectsMismatchedCountAndZeroInitializesNewStateRows)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f });
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

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { ref.params[0], ref.params[1], 10.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.25f, -0.75f, 1.5f });
    EXPECT_GSX_CODE(gsx_optim_grow(optim, 2), GSX_ERROR_INVALID_STATE);

    grow_group(&ref, { 10.0f }, { 1.5f });
    ref.grads[0] = 0.25f;
    ref.grads[1] = -0.75f;
    ASSERT_GSX_SUCCESS(gsx_optim_grow(optim, 1));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, ConsecutivePermutesReuseScratchAndFreeAfterCommit)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor permutation_a = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
    ManualTensor permutation_b = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 1, 2, 0 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 1.0f, 2.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { -0.5f, 0.4f, 0.3f });
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation_a.tensor));

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.25f, -0.75f, 0.5f });
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation_b.tensor));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation_a);
    destroy_manual_tensor(&permutation_b);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, PermuteFailureAfterPrepareStillAllowsNextPermuteAndStep)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor permutation = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 2, 0, 1 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    CpuOptimLayout *cpu_optim = nullptr;
    gsx_size_t corrupted_size = 0;
    gsx_size_t parameter_size = 0;
    gsx_size_t gradient_size = 0;
    gsx_size_t first_size = 0;
    gsx_size_t second_size = 0;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    cpu_optim = reinterpret_cast<CpuOptimLayout *>(optim);
    ASSERT_NE(cpu_optim, nullptr);
    ASSERT_NE(cpu_optim->first_moments, nullptr);
    ASSERT_NE(cpu_optim->second_moments, nullptr);
    ASSERT_NE(cpu_optim->first_moments[0], nullptr);
    ASSERT_NE(cpu_optim->second_moments[0], nullptr);

    parameter_size = parameter.tensor.size_bytes;
    gradient_size = gradient.tensor.size_bytes;
    first_size = cpu_optim->first_moments[0]->size_bytes;
    second_size = cpu_optim->second_moments[0]->size_bytes;
    corrupted_size = parameter_size - 2;
    ASSERT_GT(corrupted_size, (gsx_size_t)0);
    ASSERT_NE(corrupted_size % (gsx_size_t)parameter.tensor.shape[0], (gsx_size_t)0);

    parameter.tensor.size_bytes = corrupted_size;
    gradient.tensor.size_bytes = corrupted_size;
    cpu_optim->first_moments[0]->size_bytes = corrupted_size;
    cpu_optim->second_moments[0]->size_bytes = corrupted_size;
    EXPECT_GSX_CODE(gsx_optim_permute(optim, &permutation.tensor), GSX_ERROR_INVALID_STATE);

    parameter.tensor.size_bytes = parameter_size;
    gradient.tensor.size_bytes = gradient_size;
    cpu_optim->first_moments[0]->size_bytes = first_size;
    cpu_optim->second_moments[0]->size_bytes = second_size;

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 1.0f, 2.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { -0.3f, 0.5f, 0.7f });
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, HandleRemainsValidAfterMutationValidationFailures)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor wrong_length = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 1 });
    ManualTensor wrong_type = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 2.0f, 0.0f, 1.0f });
    ManualTensor duplicate_perm = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 0, 1 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    gsx_optim_info info{};
    AdamRefGroup ref{};

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, 2.0f, 3.0f };
    ref.grads = { 0.1f, 0.2f, 0.3f };
    ref.m = { 0.0f, 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f, 0.0f };
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);

    // Wrong-length permutation: must fail, handle must remain usable.
    EXPECT_GSX_CODE(gsx_optim_permute(optim, &wrong_length.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    // Wrong element type for permutation tensor.
    EXPECT_GSX_CODE(gsx_optim_permute(optim, &wrong_type.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    // Duplicate entries in permutation.
    EXPECT_GSX_CODE(gsx_optim_permute(optim, &duplicate_perm.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    // grow mismatch: parameter tensor not yet grown, so growth_count check fails.
    EXPECT_GSX_CODE(gsx_optim_grow(optim, 2), GSX_ERROR_INVALID_STATE);
    ASSERT_GSX_SUCCESS(gsx_optim_get_info(optim, &info));
    EXPECT_EQ(info.param_group_count, 1);

    // Handle must still step and free cleanly after all the failures above.
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient.buffer, 0, std::array<float, 3>{ -0.3f, 0.4f, 0.2f }.data(), sizeof(float) * 3));
    ref.grads = { -0.3f, 0.4f, 0.2f };
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&wrong_length);
    destroy_manual_tensor(&wrong_type);
    destroy_manual_tensor(&duplicate_perm);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, StructuralMutationSequencePreservesAdamState)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f, 4.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f, 0.4f });
    ManualTensor permutation = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 3, 0, 1, 2 });
    ManualTensor keep_mask = make_rank1_tensor<uint8_t>(host_buffer_type, GSX_DATA_TYPE_U8, { 1, 0, 1, 1 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.01f, 0.9f, 0.999f, 0.0f, 1e-8f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    AdamRefGroup ref{};

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref.params = { 1.0f, 2.0f, 3.0f, 4.0f };
    ref.grads = { 0.1f, 0.2f, 0.3f, 0.4f };
    ref.m.assign(4, 0.0f);
    ref.v.assign(4, 0.0f);
    ref.learning_rate = 0.01f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.999f;
    ref.epsilon = 1e-8f;

    // Step 1: accumulate non-zero moment state.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    // Permute [3,0,1,2]: caller declares element src[k] is now at dst[i] where perm[i]=k.
    // Rebind the parameter tensor to the permuted layout before calling permute on the optimizer.
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[3], ref.params[0], ref.params[1], ref.params[2] });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { -0.5f, 0.3f, 0.1f, -0.2f });
    permute_group(&ref, { 3, 0, 1, 2 });
    ref.grads = { -0.5f, 0.3f, 0.1f, -0.2f };
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &permutation.tensor));

    // Step 2: moment state must reflect the permuted layout.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    // Prune [1,0,1,1]: keep positions 0, 2, 3; drop position 1.
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[0], ref.params[2], ref.params[3] });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.7f, -0.4f, 0.2f });
    prune_group(&ref, { 1, 0, 1, 1 });
    ref.grads = { 0.7f, -0.4f, 0.2f };
    ASSERT_GSX_SUCCESS(gsx_optim_prune(optim, &keep_mask.tensor));

    // Step 3: moment state must reflect the pruned layout.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    // Grow by 2: new entries must have zero moment state.
    grow_group(&ref, { 10.0f, 20.0f }, { -0.1f, 0.1f });
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[0], ref.params[1], ref.params[2], 10.0f, 20.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.3f, -0.6f, 0.4f, -0.1f, 0.1f });
    ref.grads = { 0.3f, -0.6f, 0.4f, -0.1f, 0.1f };
    ASSERT_GSX_SUCCESS(gsx_optim_grow(optim, 2));

    // Step 4: all five entries including the new zeroed ones must match reference.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation);
    destroy_manual_tensor(&keep_mask);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, ResetAfterGrowZeroInitializesAllMoments)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f });
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
    ref.m.assign(2, 0.0f);
    ref.v.assign(2, 0.0f);
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;

    // Two steps to accumulate non-trivial moment values.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient.buffer, 0, std::array<float, 2>{ -0.3f, 0.8f }.data(), sizeof(float) * 2));
    ref.grads = { -0.3f, 0.8f };
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);

    // Grow by 2; new entries in the reference start with zero moments.
    grow_group(&ref, { 5.0f, 6.0f }, { 0.2f, -0.4f });
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[0], ref.params[1], 5.0f, 6.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { -0.3f, 0.8f, 0.2f, -0.4f });
    ASSERT_GSX_SUCCESS(gsx_optim_grow(optim, 2));

    // Reset clears ALL moments including those for the newly grown entries.
    reset_group_state(&ref);
    ASSERT_GSX_SUCCESS(gsx_optim_reset(optim));

    // Step from all-zero moments on the 4-element layout must match fresh reference.
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient.buffer, 0, std::array<float, 4>{ 0.1f, -0.1f, 0.3f, -0.3f }.data(), sizeof(float) * 4));
    ref.grads = { 0.1f, -0.1f, 0.3f, -0.3f };
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
