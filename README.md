# gsx

GSX is an experimental multiplatform toolbox for 3D Gaussian Splatting.

The current repository now exposes a stable public header contract that is
meant to guide the implementation work that follows.

## Public API layers

- `gsx/include/gsx/gsx-base.h`
  Shared ABI rules, versioning, error types, opaque handle declarations, and
  plain value types.
- `gsx/include/gsx/gsx-core.h`
  Core compute objects: arenas, tensors, Gaussian state, and health checks.
- `gsx/include/gsx/gsx-backend.h`
  Device enumeration, backend construction, backend buffer-type discovery,
  capability queries, and backend major-stream access.
- `gsx/include/gsx/gsx-render.h`
  Camera semantics and explicit forward/backward renderer requests.
- `gsx/include/gsx/gsx-data.h`
  Callback-backed dataset contracts and policy-only dataloader interfaces.
- `gsx/include/gsx/gsx-loss.h`
  Differentiable loss objects and scalar quality metrics.
- `gsx/include/gsx/gsx-optim.h`
  Optimizer objects, GS-centric role-based parameter-group descriptors, LR
  control, and gradient norm utilities.
- `gsx/include/gsx/gsx-adc.h`
  Automatic density control policies with explicit request/result contracts.
- `gsx/include/gsx/gsx-runtime.h`
  Replay-critical runtime objects: scheduler, session, and checkpoint I/O.

`gsx/include/gsx/gsx.h` is the umbrella header for the full stable surface.

## Stable ABI rules

- Operational objects are opaque handles.
- Backend buffer types are immutable backend-owned borrowed handles used to pick
  allocation placement.
- Public descriptor, state, and result structs are plain POD values in v0.
- Callers are expected to zero-initialize input structs before filling fields.
- Query functions return `gsx_error` and write results through out parameters.
- Runtime/session functionality remains thin and replay-focused. Tooling and
  product-specific policy stay out of the core ABI.

## Execution Model

- Public backend-bound calls are ordered on one backend-owned major stream or
  command queue.
- Callers must dispatch backend-bound public calls from one main thread, or
  externally serialize them to the same effect.
- GSX does not support public overlap of render, optimizer, ADC, or tensor
  transfer work on the same backend.
- Implementations may use private helper threads or streams internally for
  dataloader prefetch only.
- Tensors and samples returned through the public API are ready for use on the
  backend major stream when the call returns.

## Thread Safety

- Unless an API explicitly documents a type as an immutable value, GSX public
  handles are not safe for concurrent calls from multiple threads.
- Callers should treat backend, arena, tensor, GS, renderer, dataloader, loss,
  metric, optimizer, ADC, scheduler, and session operations as a single
  externally serialized stream per backend.
- Borrowed handles and output pointers must not outlive their owning object,
  and object destruction must not race with any use of borrowed state.
- Copying plain public value structs by value is always safe.

## Validation

The repository standardizes validation around CTest, GoogleTest, and Google
Benchmark:

- standalone umbrella-header inclusion smoke tests in both C and C++
- one pure-C API contract executable for ABI-oriented `_Static_assert`,
  `_Generic`, layout, and callback checks
- one C++ GoogleTest binary for richer public API contract checks grouped by
  concern
- one opt-in Google Benchmark smoke executable for benchmark harness validation

When dependencies are installed outside CMake's default search path, provide the
package prefix at configure time:

```sh
cmake -S . -B build [-DCMAKE_PREFIX_PATH=<install-prefix>]
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA backend validation is opt-in and enabled during configure:

```sh
cmake -S . -B build-cuda [-DCMAKE_PREFIX_PATH=<install-prefix>] -DGSX_USE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

When `GSX_USE_CUDA=ON`, CUDA runtime tests are built and executed only when
CUDA toolkit discovery succeeds.

Benchmarks are intentionally separate from the normal test pass:

```sh
cmake -S . -B build-bench [-DCMAKE_PREFIX_PATH=<install-prefix>] -DGSX_BUILD_BENCHMARKS=ON
cmake --build build-bench
./build-bench/gsx_benchmark_smoke
```
