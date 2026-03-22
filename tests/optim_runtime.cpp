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

struct ArenaTensor {
    gsx_arena_t arena = nullptr;
    gsx_tensor_t tensor = nullptr;
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

static gsx_gs_t make_runtime_gs(gsx_backend_buffer_type_t buffer_type, gsx_size_t count)
{
    gsx_gs_t gs = nullptr;
    gsx_gs_desc gs_desc{};

    gs_desc.buffer_type = buffer_type;
    gs_desc.arena_desc.initial_capacity_bytes = 1U << 20;
    gs_desc.count = count;
    gs_desc.aux_flags = GSX_GS_AUX_NONE;
    EXPECT_GSX_CODE(gsx_gs_init(&gs, &gs_desc), GSX_ERROR_SUCCESS);
    return gs;
}

static void upload_gs_field_f32(gsx_gs_t gs, gsx_gs_field field, const std::vector<float> &values)
{
    gsx_tensor_t tensor = nullptr;

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, field, &tensor));
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(tensor->data_type, GSX_DATA_TYPE_F32);
    ASSERT_EQ(tensor->size_bytes, (gsx_size_t)values.size() * sizeof(float));
    ASSERT_GSX_SUCCESS(gsx_tensor_upload(tensor, values.data(), tensor->size_bytes));
}

static ArenaTensor make_gs_index_tensor(gsx_gs_t gs, const std::vector<int32_t> &indices)
{
    gsx_tensor_t mean3d = nullptr;
    gsx_backend_buffer_type_t buffer_type = nullptr;
    ArenaTensor arena_tensor{};
    gsx_arena_desc arena_desc{};
    gsx_tensor_desc desc{};

    EXPECT_GSX_CODE(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d), GSX_ERROR_SUCCESS);
    if(mean3d == nullptr) {
        return {};
    }
    EXPECT_GSX_CODE(gsx_arena_get_buffer_type(mean3d->arena, &buffer_type), GSX_ERROR_SUCCESS);
    if(buffer_type == nullptr) {
        return {};
    }

    arena_desc.initial_capacity_bytes = 1024;
    EXPECT_GSX_CODE(gsx_arena_init(&arena_tensor.arena, buffer_type, &arena_desc), GSX_ERROR_SUCCESS);
    if(arena_tensor.arena == nullptr) {
        return {};
    }

    desc.rank = 1;
    desc.shape[0] = (gsx_index_t)indices.size();
    desc.data_type = GSX_DATA_TYPE_I32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena_tensor.arena;
    EXPECT_GSX_CODE(gsx_tensor_init(&arena_tensor.tensor, &desc), GSX_ERROR_SUCCESS);
    if(arena_tensor.tensor == nullptr) {
        if(arena_tensor.arena != nullptr) {
            (void)gsx_arena_free(arena_tensor.arena);
        }
        return {};
    }
    if(!indices.empty()) {
        EXPECT_GSX_CODE(
            gsx_tensor_upload(arena_tensor.tensor, indices.data(), (gsx_size_t)indices.size() * sizeof(int32_t)),
            GSX_ERROR_SUCCESS);
    }
    return arena_tensor;
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

    const double beta1_correction = 1.0 - std::pow((double)group->beta1, (double)group->step);
    const double beta2_correction = 1.0 - std::pow((double)group->beta2, (double)group->step);
    for(std::size_t i = 0; i < group->params.size(); ++i) {
        const float gradient = adam_clamp_gradient(group, group->grads[i]);
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

TEST(OptimRuntime, StepAppliesPerGroupElementClampInsideFusedAdam)
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
    ref.max_grad = 2.5f;

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

TEST(OptimRuntime, GatherRejectsMismatchedIndicesAndKeepsAdamState)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, 1.0f, 1.5f });
    ManualTensor invalid_indices = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 1, 2 });
    ManualTensor gather_indices = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 2 });
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
    EXPECT_GSX_CODE(gsx_optim_gather(optim, &invalid_indices.tensor), GSX_ERROR_INVALID_ARGUMENT);

    gather_group(&ref, { 0, 2 });
    ref.grads = { 1.25f, -0.5f };
    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, &gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&invalid_indices);
    destroy_manual_tensor(&gather_indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, ResizeRejectsMismatchedCountAndZeroInitializesNewStateRows)
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
    EXPECT_GSX_CODE(gsx_optim_resize(optim, 4), GSX_ERROR_INVALID_STATE);

    resize_group(&ref, { 10.0f }, { 1.5f });
    ref.grads[0] = 0.25f;
    ref.grads[1] = -0.75f;
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 3));
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

    // resize mismatch: parameter tensor shape is still 3, but requested target extent is 5.
    EXPECT_GSX_CODE(gsx_optim_resize(optim, 5), GSX_ERROR_INVALID_STATE);
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
    ManualTensor gather_indices = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 2, 3 });
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

    // Gather [0,2,3]: keep positions 0, 2, 3 in that order.
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[0], ref.params[2], ref.params[3] });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.7f, -0.4f, 0.2f });
    gather_group(&ref, { 0, 2, 3 });
    ref.grads = { 0.7f, -0.4f, 0.2f };
    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, &gather_indices.tensor));

    // Step 3: moment state must reflect the gathered layout.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    // Resize to 5 by adding 2 entries: new entries must have zero moment state.
    resize_group(&ref, { 10.0f, 20.0f }, { -0.1f, 0.1f });
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[0], ref.params[1], ref.params[2], 10.0f, 20.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.3f, -0.6f, 0.4f, -0.1f, 0.1f });
    ref.grads = { 0.3f, -0.6f, 0.4f, -0.1f, 0.1f };
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 5));

    // Step 4: all five entries including the new zeroed ones must match reference.
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&permutation);
    destroy_manual_tensor(&gather_indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, ResetAfterResizeZeroInitializesAllMoments)
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

    // Resize to 4 by adding 2; new entries in the reference start with zero moments.
    resize_group(&ref, { 5.0f, 6.0f }, { 0.2f, -0.4f });
    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32,
        { ref.params[0], ref.params[1], 5.0f, 6.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { -0.3f, 0.8f, 0.2f, -0.4f });
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 4));

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

TEST(OptimRuntime, StepRejectsStepCounterOverflowAndRecovers)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, -2.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.2f, -0.4f });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;
    CpuOptimLayout *cpu_optim = nullptr;
    AdamRefGroup ref{};

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    cpu_optim = reinterpret_cast<CpuOptimLayout *>(optim);
    ASSERT_NE(cpu_optim, nullptr);
    ASSERT_NE(cpu_optim->step_counts, nullptr);
    cpu_optim->step_counts[0] = UINT64_MAX;

    EXPECT_GSX_CODE(gsx_optim_step(optim, nullptr), GSX_ERROR_OUT_OF_RANGE);

    cpu_optim->step_counts[0] = 0;
    ref.params = { 1.0f, -2.0f };
    ref.grads = { 0.2f, -0.4f };
    ref.m = { 0.0f, 0.0f };
    ref.v = { 0.0f, 0.0f };
    ref.learning_rate = 0.1f;
    ref.beta1 = 0.9f;
    ref.beta2 = 0.99f;
    ref.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref);
    expect_near_vectors(download_rank1_tensor<float>(parameter), ref.params);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, PermuteRejectsNegativeAndOutOfRangeEntries)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor negative_perm = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { -1, 0, 1 });
    ManualTensor out_of_range_perm = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 1, 3 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    EXPECT_GSX_CODE(gsx_optim_permute(optim, &negative_perm.tensor), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_optim_permute(optim, &out_of_range_perm.tensor), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&negative_perm);
    destroy_manual_tensor(&out_of_range_perm);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, GatherRejectsNegativeAndOutOfRangeEntries)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor negative_indices = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { -1, 0 });
    ManualTensor out_of_range_indices = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 3 });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_MEAN3D, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    rebind_rank1_tensor<float>(&parameter, host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 3.0f });
    rebind_rank1_tensor<float>(&gradient, host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.3f });

    EXPECT_GSX_CODE(gsx_optim_gather(optim, &negative_indices.tensor), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_optim_gather(optim, &out_of_range_indices.tensor), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    destroy_manual_tensor(&negative_indices);
    destroy_manual_tensor(&out_of_range_indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, ResetParamGroupByIndexAndRoleAffectsOnlyTargetGroup)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter_a = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f });
    ManualTensor gradient_a = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.5f, -1.0f });
    ManualTensor parameter_b = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 3.0f, 4.0f });
    ManualTensor gradient_b = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { -0.25f, 0.75f });
    gsx_optim_param_group_desc groups[2]{};
    gsx_optim_desc optim_desc{};
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
        0.0f,
        1e-6f,
        0.0f
    );
    groups[1] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_OPACITY,
        &parameter_b,
        &gradient_b,
        0.1f,
        0.9f,
        0.99f,
        0.0f,
        1e-6f,
        0.0f
    );
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 2;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ref_a.params = { 1.0f, 2.0f };
    ref_a.grads = { 0.5f, -1.0f };
    ref_a.m = { 0.0f, 0.0f };
    ref_a.v = { 0.0f, 0.0f };
    ref_a.learning_rate = 0.1f;
    ref_a.beta1 = 0.9f;
    ref_a.beta2 = 0.99f;
    ref_a.epsilon = 1e-6f;
    ref_b.params = { 3.0f, 4.0f };
    ref_b.grads = { -0.25f, 0.75f };
    ref_b.m = { 0.0f, 0.0f };
    ref_b.v = { 0.0f, 0.0f };
    ref_b.learning_rate = 0.1f;
    ref_b.beta1 = 0.9f;
    ref_b.beta2 = 0.99f;
    ref_b.epsilon = 1e-6f;

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref_a);
    adam_step(&ref_b);
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref_a);
    adam_step(&ref_b);

    ASSERT_GSX_SUCCESS(gsx_optim_reset_param_group_by_index(optim, 0));
    reset_group_state(&ref_a);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_a.buffer, 0, std::array<float, 2>{ 0.2f, 0.1f }.data(), sizeof(float) * 2));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_b.buffer, 0, std::array<float, 2>{ -0.1f, -0.2f }.data(), sizeof(float) * 2));
    ref_a.grads = { 0.2f, 0.1f };
    ref_b.grads = { -0.1f, -0.2f };
    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    adam_step(&ref_a);
    adam_step(&ref_b);
    expect_near_vectors(download_rank1_tensor<float>(parameter_a), ref_a.params);
    expect_near_vectors(download_rank1_tensor<float>(parameter_b), ref_b.params);

    ASSERT_GSX_SUCCESS(gsx_optim_reset_param_group_by_role(optim, GSX_OPTIM_PARAM_ROLE_OPACITY));
    reset_group_state(&ref_b);

    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_a.buffer, 0, std::array<float, 2>{ -0.2f, 0.3f }.data(), sizeof(float) * 2));
    ASSERT_GSX_SUCCESS(gsx_backend_buffer_upload(gradient_b.buffer, 0, std::array<float, 2>{ 0.6f, -0.4f }.data(), sizeof(float) * 2));
    ref_a.grads = { -0.2f, 0.3f };
    ref_b.grads = { 0.6f, -0.4f };
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

TEST(OptimRuntime, MultiGroupGatherRejectsLeadingExtentMismatch)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor parameter_a = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor gradient_a = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ManualTensor parameter_b = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { 4.0f, 5.0f, 6.0f });
    ManualTensor gradient_b = make_rank1_tensor<float>(host_buffer_type, GSX_DATA_TYPE_F32, { -0.1f, -0.2f, -0.3f });
    ManualTensor indices = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0, 1 });
    gsx_optim_param_group_desc groups[2]{};
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    groups[0] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_MEAN3D,
        &parameter_a,
        &gradient_a,
        0.1f,
        0.9f,
        0.99f,
        0.0f,
        1e-6f,
        0.0f
    );
    groups[1] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_CUSTOM,
        &parameter_b,
        &gradient_b,
        0.1f,
        0.9f,
        0.99f,
        0.0f,
        1e-6f,
        0.0f
    );
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 2;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    rebind_rank1_tensor<float>(&parameter_a, host_buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 3.0f });
    rebind_rank1_tensor<float>(&gradient_a, host_buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.3f });
    EXPECT_GSX_CODE(gsx_optim_gather(optim, &indices.tensor), GSX_ERROR_INVALID_STATE);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter_a);
    destroy_manual_tensor(&gradient_a);
    destroy_manual_tensor(&parameter_b);
    destroy_manual_tensor(&gradient_b);
    destroy_manual_tensor(&indices);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, RebindParamGroupsFromGsRefreshesBuiltInTensorHandles)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_gs_t gs = make_runtime_gs(buffer_type, 3);
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t grad_mean3d = nullptr;
    gsx_tensor_t opacity = nullptr;
    gsx_tensor_t grad_opacity = nullptr;
    gsx_tensor_t gathered_mean3d = nullptr;
    gsx_tensor_t gathered_grad_mean3d = nullptr;
    gsx_tensor_t gathered_opacity = nullptr;
    gsx_tensor_t gathered_grad_opacity = nullptr;
    gsx_optim_param_group_desc groups[2]{};
    gsx_optim_desc optim_desc{};
    gsx_optim_param_group_desc desc_mean{};
    gsx_optim_param_group_desc desc_opacity{};
    ArenaTensor gather_indices = make_gs_index_tensor(gs, { 2, 0 });
    gsx_optim_t optim = nullptr;

    ASSERT_NE(gather_indices.arena, nullptr);
    ASSERT_NE(gather_indices.tensor, nullptr);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &grad_mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_OPACITY, &grad_opacity));
    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_MEAN3D, { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_OPACITY, { 0.1f, 0.2f, 0.3f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_OPACITY, { 0.4f, 0.5f, 0.6f });

    groups[0].role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    groups[0].parameter = mean3d;
    groups[0].gradient = grad_mean3d;
    groups[0].learning_rate = 0.01f;
    groups[0].beta1 = 0.9f;
    groups[0].beta2 = 0.99f;
    groups[0].epsilon = 1e-6f;
    groups[1].role = GSX_OPTIM_PARAM_ROLE_OPACITY;
    groups[1].parameter = opacity;
    groups[1].gradient = grad_opacity;
    groups[1].learning_rate = 0.02f;
    groups[1].beta1 = 0.9f;
    groups[1].beta2 = 0.99f;
    groups[1].epsilon = 1e-6f;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 2;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_gather(gs, gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &gathered_mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &gathered_grad_mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &gathered_opacity));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_OPACITY, &gathered_grad_opacity));

    ASSERT_GSX_SUCCESS(gsx_optim_rebind_param_groups_from_gs(optim, gs));
    ASSERT_GSX_SUCCESS(gsx_optim_get_param_group_desc_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &desc_mean));
    ASSERT_GSX_SUCCESS(gsx_optim_get_param_group_desc_by_role(optim, GSX_OPTIM_PARAM_ROLE_OPACITY, &desc_opacity));
    EXPECT_EQ(desc_mean.parameter, gathered_mean3d);
    EXPECT_EQ(desc_mean.gradient, gathered_grad_mean3d);
    EXPECT_EQ(desc_opacity.parameter, gathered_opacity);
    EXPECT_EQ(desc_opacity.gradient, gathered_grad_opacity);

    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, gather_indices.tensor));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    ASSERT_GSX_SUCCESS(gsx_tensor_free(gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(gather_indices.arena));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, RebindParamGroupsFromGsRejectsCustomRole)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    gsx_gs_t gs = make_runtime_gs(buffer_type, 2);
    ManualTensor parameter = make_rank1_tensor<float>(buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f });
    ManualTensor gradient = make_rank1_tensor<float>(buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f });
    gsx_optim_param_group_desc group =
        make_param_group_desc(GSX_OPTIM_PARAM_ROLE_CUSTOM, &parameter, &gradient, 0.1f, 0.9f, 0.99f, 0.0f, 1e-6f, 0.0f);
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = &group;
    optim_desc.param_group_count = 1;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    EXPECT_GSX_CODE(gsx_optim_rebind_param_groups_from_gs(optim, gs), GSX_ERROR_NOT_SUPPORTED);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&parameter);
    destroy_manual_tensor(&gradient);
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, RebindParamGroupsFromGsIsAtomicOnFailure)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_DEVICE);
    gsx_gs_t gs = make_runtime_gs(buffer_type, 3);
    gsx_tensor_t mean3d = nullptr;
    gsx_tensor_t grad_mean3d = nullptr;
    gsx_tensor_t gathered_mean3d = nullptr;
    gsx_optim_param_group_desc groups[2]{};
    gsx_optim_desc optim_desc{};
    gsx_optim_param_group_desc before{};
    gsx_optim_param_group_desc after{};
    ManualTensor custom_param = make_rank1_tensor<float>(buffer_type, GSX_DATA_TYPE_F32, { 1.0f, 2.0f, 3.0f });
    ManualTensor custom_grad = make_rank1_tensor<float>(buffer_type, GSX_DATA_TYPE_F32, { 0.1f, 0.2f, 0.3f });
    ArenaTensor gather_indices = make_gs_index_tensor(gs, { 2, 1 });
    gsx_optim_t optim = nullptr;

    ASSERT_NE(gather_indices.arena, nullptr);
    ASSERT_NE(gather_indices.tensor, nullptr);

    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean3d));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_GRAD_MEAN3D, &grad_mean3d));
    upload_gs_field_f32(gs, GSX_GS_FIELD_MEAN3D, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
    upload_gs_field_f32(gs, GSX_GS_FIELD_GRAD_MEAN3D, { 0.3f, 0.2f, 0.1f, 0.6f, 0.5f, 0.4f, 0.9f, 0.8f, 0.7f });

    groups[0].role = GSX_OPTIM_PARAM_ROLE_MEAN3D;
    groups[0].parameter = mean3d;
    groups[0].gradient = grad_mean3d;
    groups[0].learning_rate = 0.01f;
    groups[0].beta1 = 0.9f;
    groups[0].beta2 = 0.99f;
    groups[0].epsilon = 1e-6f;
    groups[1] = make_param_group_desc(
        GSX_OPTIM_PARAM_ROLE_CUSTOM,
        &custom_param,
        &custom_grad,
        0.02f,
        0.9f,
        0.99f,
        0.0f,
        1e-6f,
        0.0f
    );
    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = groups;
    optim_desc.param_group_count = 2;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ASSERT_GSX_SUCCESS(gsx_gs_gather(gs, gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &gathered_mean3d));
    ASSERT_GSX_SUCCESS(gsx_optim_get_param_group_desc_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &before));
    EXPECT_GSX_CODE(gsx_optim_rebind_param_groups_from_gs(optim, gs), GSX_ERROR_NOT_SUPPORTED);
    ASSERT_GSX_SUCCESS(gsx_optim_get_param_group_desc_by_role(optim, GSX_OPTIM_PARAM_ROLE_MEAN3D, &after));
    EXPECT_EQ(after.parameter, before.parameter);
    EXPECT_EQ(after.gradient, before.gradient);
    EXPECT_NE(after.parameter, gathered_mean3d);

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&custom_param);
    destroy_manual_tensor(&custom_grad);
    ASSERT_GSX_SUCCESS(gsx_tensor_free(gather_indices.tensor));
    ASSERT_GSX_SUCCESS(gsx_arena_free(gather_indices.arena));
    ASSERT_GSX_SUCCESS(gsx_gs_free(gs));
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

TEST(OptimRuntime, EmptyOptimizerAllowsNoOpMutations)
{
    gsx_backend_t backend = create_cpu_backend();
    gsx_backend_buffer_type_t host_buffer_type = find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST);
    ManualTensor control = make_rank1_tensor<int32_t>(host_buffer_type, GSX_DATA_TYPE_I32, { 0 });
    gsx_optim_desc optim_desc{};
    gsx_optim_t optim = nullptr;

    optim_desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
    optim_desc.param_groups = nullptr;
    optim_desc.param_group_count = 0;
    ASSERT_GSX_SUCCESS(gsx_optim_init(&optim, backend, &optim_desc));

    ASSERT_GSX_SUCCESS(gsx_optim_step(optim, nullptr));
    ASSERT_GSX_SUCCESS(gsx_optim_permute(optim, &control.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_gather(optim, &control.tensor));
    ASSERT_GSX_SUCCESS(gsx_optim_resize(optim, 0));
    ASSERT_GSX_SUCCESS(gsx_optim_reset(optim));

    ASSERT_GSX_SUCCESS(gsx_optim_free(optim));
    destroy_manual_tensor(&control);
    ASSERT_GSX_SUCCESS(gsx_backend_free(backend));
}

}  // namespace
