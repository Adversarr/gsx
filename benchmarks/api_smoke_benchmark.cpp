#include "gsx/gsx-base.h"

#include <benchmark/benchmark.h>

namespace {

struct gsx_error_benchmark_fixture {
    gsx_error success = { GSX_ERROR_SUCCESS, nullptr };
    gsx_error failure = { GSX_ERROR_INVALID_ARGUMENT, "invalid argument" };
};

/*
 * This benchmark only validates the benchmark harness wiring. Meaningful
 * performance coverage should be added once GSX exposes executable runtime code.
 */
static void BM_Base_ErrorIsSuccess(benchmark::State &state)
{
    gsx_error_benchmark_fixture fixture;

    for(auto _ : state) {
        benchmark::DoNotOptimize(gsx_error_is_success(fixture.success));
        benchmark::DoNotOptimize(gsx_error_is_success(fixture.failure));
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * 2);
}

BENCHMARK(BM_Base_ErrorIsSuccess);

}  // namespace

BENCHMARK_MAIN();
