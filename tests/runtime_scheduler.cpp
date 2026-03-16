extern "C" {
#include "../gsx/src/gsx-impl.h"
}

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

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

static gsx_scheduler_desc make_constant_desc()
{
    gsx_scheduler_desc desc{};

    desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
    desc.initial_learning_rate = 0.1f;
    desc.final_learning_rate = 0.01f;
    desc.delay_steps = 0;
    desc.delay_multiplier = 1.0f;
    desc.decay_begin_step = 0;
    desc.decay_end_step = 0;
    return desc;
}

static gsx_scheduler_desc make_delayed_exponential_desc()
{
    gsx_scheduler_desc desc{};

    desc.algorithm = GSX_SCHEDULER_ALGORITHM_DELAYED_EXPONENTIAL;
    desc.initial_learning_rate = 0.2f;
    desc.final_learning_rate = 0.02f;
    desc.delay_steps = 10;
    desc.delay_multiplier = 0.25f;
    desc.decay_begin_step = 5;
    desc.decay_end_step = 35;
    return desc;
}

static double clamp01(double value)
{
    if(value < 0.0) {
        return 0.0;
    }
    if(value > 1.0) {
        return 1.0;
    }
    return value;
}

static double expected_delayed_exponential(const gsx_scheduler_desc &desc, gsx_size_t step)
{
    double t = 0.0;
    double base_lr = 0.0;
    double delay_rate = 1.0;
    const double pi = 3.14159265358979323846;
    const double initial_lr = static_cast<double>(desc.initial_learning_rate);
    const double final_lr = static_cast<double>(desc.final_learning_rate);

    if(desc.decay_end_step > desc.decay_begin_step) {
        t = (static_cast<double>(step) - static_cast<double>(desc.decay_begin_step))
            / (static_cast<double>(desc.decay_end_step) - static_cast<double>(desc.decay_begin_step));
    } else if(step >= desc.decay_end_step) {
        t = 1.0;
    }
    t = clamp01(t);

    if(initial_lr == 0.0 && final_lr == 0.0) {
        base_lr = 0.0;
    } else if(initial_lr > 0.0 && final_lr > 0.0) {
        base_lr = std::exp((1.0 - t) * std::log(initial_lr) + t * std::log(final_lr));
    } else {
        base_lr = initial_lr + (final_lr - initial_lr) * t;
        if(base_lr < 0.0) {
            base_lr = 0.0;
        }
    }

    if(desc.delay_steps != 0 && step < desc.delay_steps) {
        double ratio = static_cast<double>(step) / static_cast<double>(desc.delay_steps);
        ratio = clamp01(ratio);
        delay_rate = static_cast<double>(desc.delay_multiplier)
            + (1.0 - static_cast<double>(desc.delay_multiplier)) * std::sin(0.5 * pi * ratio);
    }

    return base_lr * delay_rate;
}

TEST(SchedulerRuntime, InitRejectsInvalidArgumentsAndInvalidDescriptorValues)
{
    gsx_scheduler_desc desc = make_constant_desc();
    gsx_scheduler_t scheduler = nullptr;

    EXPECT_GSX_CODE(gsx_scheduler_init(nullptr, &desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    desc.algorithm = static_cast<gsx_scheduler_algorithm>(999);
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_OUT_OF_RANGE);

    desc = make_constant_desc();
    desc.initial_learning_rate = -0.01f;
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc = make_constant_desc();
    desc.final_learning_rate = -0.01f;
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc = make_constant_desc();
    desc.delay_multiplier = -0.1f;
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc = make_constant_desc();
    desc.initial_learning_rate = std::numeric_limits<float>::quiet_NaN();
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc = make_constant_desc();
    desc.delay_multiplier = std::numeric_limits<float>::infinity();
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_INVALID_ARGUMENT);

    desc = make_constant_desc();
    desc.decay_begin_step = 8;
    desc.decay_end_step = 3;
    EXPECT_GSX_CODE(gsx_scheduler_init(&scheduler, &desc), GSX_ERROR_OUT_OF_RANGE);
}

TEST(SchedulerRuntime, InitGetDescAndFreeRoundTrip)
{
    gsx_scheduler_desc desc = make_delayed_exponential_desc();
    gsx_scheduler_desc queried{};
    gsx_scheduler_t scheduler = nullptr;

    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &desc));
    ASSERT_NE(scheduler, nullptr);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_desc(scheduler, &queried));
    EXPECT_EQ(queried.algorithm, desc.algorithm);
    EXPECT_FLOAT_EQ(queried.initial_learning_rate, desc.initial_learning_rate);
    EXPECT_FLOAT_EQ(queried.final_learning_rate, desc.final_learning_rate);
    EXPECT_EQ(queried.delay_steps, desc.delay_steps);
    EXPECT_FLOAT_EQ(queried.delay_multiplier, desc.delay_multiplier);
    EXPECT_EQ(queried.decay_begin_step, desc.decay_begin_step);
    EXPECT_EQ(queried.decay_end_step, desc.decay_end_step);
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
}

TEST(SchedulerRuntime, ConstantScheduleStepAndGetLearningRateAreDeterministic)
{
    gsx_scheduler_desc desc = make_constant_desc();
    gsx_scheduler_t scheduler = nullptr;
    gsx_scheduler_state state{};
    gsx_float_t lr = 0.0f;

    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &desc));

    ASSERT_GSX_SUCCESS(gsx_scheduler_step(scheduler, 0, &lr));
    EXPECT_NEAR(lr, 0.1f, 1e-7f);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_state(scheduler, &state));
    EXPECT_EQ(state.current_step, 0u);
    EXPECT_NEAR(state.current_learning_rate, 0.1f, 1e-7f);

    ASSERT_GSX_SUCCESS(gsx_scheduler_step(scheduler, 123, &lr));
    EXPECT_NEAR(lr, 0.1f, 1e-7f);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_learning_rate(scheduler, &lr));
    EXPECT_NEAR(lr, 0.1f, 1e-7f);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_state(scheduler, &state));
    EXPECT_EQ(state.current_step, 123u);
    EXPECT_NEAR(state.current_learning_rate, 0.1f, 1e-7f);

    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
}

TEST(SchedulerRuntime, ResetRestoresInitialStateAfterSteps)
{
    gsx_scheduler_desc desc = make_delayed_exponential_desc();
    gsx_scheduler_t scheduler = nullptr;
    gsx_scheduler_state state{};
    gsx_float_t lr = 0.0f;

    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &desc));
    ASSERT_GSX_SUCCESS(gsx_scheduler_step(scheduler, 17, &lr));
    ASSERT_GSX_SUCCESS(gsx_scheduler_step(scheduler, 42, &lr));

    ASSERT_GSX_SUCCESS(gsx_scheduler_reset(scheduler));
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_state(scheduler, &state));
    EXPECT_EQ(state.current_step, 0u);
    EXPECT_NEAR(state.current_learning_rate, desc.initial_learning_rate, 1e-7f);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_learning_rate(scheduler, &lr));
    EXPECT_NEAR(lr, desc.initial_learning_rate, 1e-7f);
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
}

TEST(SchedulerRuntime, StateRoundTripAndValidation)
{
    gsx_scheduler_desc desc = make_constant_desc();
    gsx_scheduler_t scheduler = nullptr;
    gsx_scheduler_state state{};
    gsx_scheduler_state queried{};
    gsx_float_t lr = 0.0f;

    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &desc));

    EXPECT_GSX_CODE(gsx_scheduler_set_state(nullptr, &state), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_set_state(scheduler, nullptr), GSX_ERROR_INVALID_ARGUMENT);

    state.current_step = 77;
    state.current_learning_rate = 0.033f;
    ASSERT_GSX_SUCCESS(gsx_scheduler_set_state(scheduler, &state));
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_state(scheduler, &queried));
    EXPECT_EQ(queried.current_step, 77u);
    EXPECT_NEAR(queried.current_learning_rate, 0.033f, 1e-7f);
    ASSERT_GSX_SUCCESS(gsx_scheduler_get_learning_rate(scheduler, &lr));
    EXPECT_NEAR(lr, 0.033f, 1e-7f);

    state.current_learning_rate = -0.2f;
    EXPECT_GSX_CODE(gsx_scheduler_set_state(scheduler, &state), GSX_ERROR_INVALID_ARGUMENT);

    state.current_learning_rate = std::numeric_limits<float>::infinity();
    EXPECT_GSX_CODE(gsx_scheduler_set_state(scheduler, &state), GSX_ERROR_INVALID_ARGUMENT);

    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
}

TEST(SchedulerRuntime, DelayedExponentialScheduleMatchesReferenceAtBoundaryAndIntermediateSteps)
{
    gsx_scheduler_desc desc = make_delayed_exponential_desc();
    gsx_scheduler_t scheduler = nullptr;
    gsx_float_t lr = 0.0f;
    const gsx_size_t steps[] = { 0, 3, 9, 10, 5, 20, 35, 60 };

    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &desc));

    for(gsx_size_t step : steps) {
        const double expected = expected_delayed_exponential(desc, step);
        ASSERT_GSX_SUCCESS(gsx_scheduler_step(scheduler, step, &lr));
        EXPECT_NEAR(static_cast<double>(lr), expected, 1e-6);
    }

    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
}

TEST(SchedulerRuntime, ApiRejectsNullInputsForAllEntryPoints)
{
    gsx_scheduler_desc desc = make_constant_desc();
    gsx_scheduler_t scheduler = nullptr;
    gsx_scheduler_state state{};
    gsx_float_t lr = 0.0f;

    ASSERT_GSX_SUCCESS(gsx_scheduler_init(&scheduler, &desc));
    EXPECT_GSX_CODE(gsx_scheduler_get_desc(nullptr, &desc), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_get_desc(scheduler, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_reset(nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_get_state(nullptr, &state), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_get_state(scheduler, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_step(nullptr, 0, &lr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_step(scheduler, 0, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_get_learning_rate(nullptr, &lr), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_GSX_CODE(gsx_scheduler_get_learning_rate(scheduler, nullptr), GSX_ERROR_INVALID_ARGUMENT);
    ASSERT_GSX_SUCCESS(gsx_scheduler_free(scheduler));
    EXPECT_GSX_CODE(gsx_scheduler_free(nullptr), GSX_ERROR_INVALID_ARGUMENT);
}

}  // namespace
