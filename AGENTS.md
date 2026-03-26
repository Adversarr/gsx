# GSX Agent Guide
GSX is an experimental cross-platform C library for 3D gaussian splatting.
Public APIs assume one caller-visible main thread and ordered backend dispatch on one caller-visible major stream or command queue per backend. Do not introduce caller-visible concurrency, extra public streams, or backend-specific behavior into the public mental model unless the repo already does so.

## Repo Map
- `gsx/include/gsx/`: stable public C API; `gsx.h` is the umbrella header and the header layering is `gsx-base.h` -> `gsx-core.h` -> subsystem headers -> `gsx-runtime.h`.
- `gsx/src/`: backend-neutral implementation layer; this is the main place to extend public behavior, validation, dispatch, and shared lifecycle rules.
- `gsx/src/gsx-impl.h`: shared internal structs, vtables, lifetime counters, and layering rules; treat it as the central internal contract.
- `gsx/src/extra/`: non-core helpers and bundled integrations such as image helpers, stb glue, PLY I/O, and FLANN integration.
- `gsx/src/gsx-cpu/`: always-built CPU backend.
- `gsx/src/gsx-cuda/`: optional CUDA backend, kernels, and async dataloader support.
- `gsx/src/gsx-metal/`: optional Metal backend, Objective-C bridging, embedded kernels, and renderer internals.
- `apps/`: CLI tools and examples used to inspect capabilities and exercise end-to-end flows.
- `tests/`: C API contract tests, GoogleTest runtime tests, smoke tests, and standalone-header compile tests.
- `benchmarks/`: Google Benchmark targets.
- `cmake/`: CMake helpers and coverage helpers.

## `gsx/src` Codemap
- `gsx-backend.c`: backend registry, provider/device enumeration, backend creation, buffer type discovery, backend metadata, and major-stream level operations.
- `gsx-core.c`: arenas, backend buffer-backed tensor storage, tensor lifetime, gaussian-state containers, tensor validation, and core memory/layout rules.
- `gsx-render.c`: backend-neutral renderer and render-context entrypoints, request validation, and dispatch into backend renderer implementations.
- `gsx-loss.c`: loss object/context setup, loss request validation, metric-style helpers, and dispatch for L1/MSE/SSIM-style operations.
- `gsx-optim.c`: optimizer descriptors, parameter groups, learning-rate state, step/reset behavior, and backend dispatch for optimizer work.
- `gsx-adc.c`: adaptive density control object lifecycle, policy validation, and transactional ADC dispatch.
- `gsx-data.c`: datasets, dataloaders, resize/shuffle/prefetch plumbing, callback integration, and image/sample transport rules.
- `gsx-runtime.c`: scheduler/session/checkpoint logic and replay-critical runtime state; this sits at the top of the internal layering.
- `gsx-random.c`: RNG helpers and random tensor utilities used by core/runtime flows.
- `extra/gsx-image.c`, `extra/gsx-stbi.c`: image loading/storage helpers and stb-backed functionality.
- `extra/gsx-io-ply.cpp`, `extra/gsx-flann.cpp`: PLY import and nearest-neighbor support used by higher-level tooling and data paths.

## Backend Codemap
- `gsx/src/gsx-cpu/core.c`: CPU backend buffer/tensor primitives and CPU-side data movement.
- `gsx/src/gsx-cpu/loss.c`, `gsx/src/gsx-cpu/optim.c`, `gsx/src/gsx-cpu/adc.c`, `gsx/src/gsx-cpu/render.c`: CPU implementations of the main compute subsystems.
- `gsx/src/gsx-cpu/adc/default.c`, `gsx/src/gsx-cpu/adc/mcmc.c`: CPU ADC policy variants.
- `gsx/src/gsx-cpu/async_dl.cpp`: private async dataloader implementation; keep this internal and invisible to the public threading model.
- `gsx/src/gsx-cuda/backend.c`, `buffer.c`, `shared.c`: CUDA backend/device setup, shared helpers, and buffer management.
- `gsx/src/gsx-cuda/render.c`, `optim.c`, `loss.c`, `adc.cu`: CUDA subsystem entrypoints layered over CUDA kernels.
- `gsx/src/gsx-cuda/render-kernels.cu`, `reduce-kernels.cu`, `loss-pointwise-kernels.cu`, `loss-ssim-kernels.cu`, `gsx-cuda-kernels.cu`: device kernels and low-level CUDA compute helpers.
- `gsx/src/gsx-cuda/async_dl.cu`: CUDA async dataloader support; same public-threading caveat as CPU/Metal async loaders.
- `gsx/src/gsx-metal/backend.m`, `buffer.m`, `shared.m`: Metal device/queue setup, buffer handling, and shared Objective-C helpers.
- `gsx/src/gsx-metal/render.c`, `loss.c`, `optim.c`, `adc.c`: Metal-facing subsystem entrypoints.
- `gsx/src/gsx-metal/renderer/`: renderer internals split into context, forward/backward, and sorting helpers.
- `gsx/src/gsx-metal/kernels/`: Metal shader sources compiled and embedded by CMake; do not treat these as public API.

## Build Defaults
- Trust `CMakeLists.txt` over `README.md` when they disagree.
- Current defaults: `GSX_USE_CUDA=OFF`, `GSX_USE_METAL=OFF`, `GSX_ENABLE_COVERAGE=OFF`, `GSX_BUILD_APP_EXTRAS=OFF`, `GSX_BUILD_TESTS=ON`, `GSX_BUILD_BENCHMARKS=OFF`.
- Metal is Apple-only.
- Coverage is host-only and incompatible with CUDA.
- Non-MSVC builds use `-Wall -Wextra -Wpedantic` through `gsx_configure_target()`.

## Configure, Build, And Test
- Default build: `cmake -S . -B build && cmake --build build -j8`.
- CUDA build: `cmake -S . -B build-cuda -DGSX_USE_CUDA=ON && cmake --build build-cuda -j8`.
- Metal build: `cmake -S . -B build-metal -DGSX_USE_METAL=ON && cmake --build build-metal -j8`.
- Optional switches: `GSX_BUILD_BENCHMARKS=ON`, `GSX_BUILD_APP_EXTRAS=ON`, `GSX_ENABLE_COVERAGE=ON`, `-DCMAKE_PREFIX_PATH=/path/to/prefix`.
- Build one target with `cmake --build build --target <target>`.

## Test Workflow
- Tests are enabled by default in the current top-level CMake.
- GTest is required; configure with `-DCMAKE_PREFIX_PATH=/path/to/prefix` if CMake cannot find it.
- Many C++ tests use `gtest_discover_tests()`, so CTest names are often individual GoogleTest case names rather than executable names.

- Run all tests: `ctest --test-dir build --output-on-failure`.
- Run labels: `ctest --test-dir build -L contract --output-on-failure` or `ctest --test-dir build -L unit --output-on-failure`.
- Run one CTest test: `ctest --test-dir build -R '^gsx_api_contract_c$' --output-on-failure`.
- Run one discovered GoogleTest case: `ctest --test-dir build -R '^BackendRuntime\.CpuBackendExposesExpectedMetadataAndBufferTypes$' --output-on-failure`.
- Run one executable directly: `./build/tests/gsx_backend_runtime_cpp --gtest_filter='BackendRuntime.CpuBackendExposesExpectedMetadataAndBufferTypes'`.
- Backend-specific runs use separate build dirs such as `build-cuda` and `build-metal`.

## Coverage, Benchmarks, Lint
- Coverage: `cmake -S . -B build-cov -DGSX_ENABLE_COVERAGE=ON && cmake --build build-cov -j8 --target gsx_coverage`.
- Coverage requires `gcovr`; Clang also requires `llvm-cov`.
- Coverage reports land in `build-*/coverage/` as TXT, Markdown, XML, and JSON.
- Agent coverage helper: `.agents/skills/test-coverage/scripts/generate_coverage.sh`.
- Benchmarks require Google Benchmark and `GSX_BUILD_BENCHMARKS=ON`.
- Main app targets: `gsx-info`, `banana-optim`, `ssim-numerical-diff`, `image-fit-demo`, `multi-gaussian-render`, `render-pcd`.
- Extra apps behind `GSX_BUILD_APP_EXTRAS=ON`: `train-preprocessed`, `ply-viewer`.
- No repo-standard lint or format command is configured.
- No `clang-format`, `clang-tidy`, or CMake `lint`/`format` target is wired in the checked-out repo.
- Match surrounding formatting; do not introduce a new formatter style in isolated edits.

## Where To Look For Patterns
- Public API layering and caller-visible contracts are documented best in `gsx/include/gsx/`, especially `gsx/include/gsx/gsx.h` and `gsx/include/gsx/gsx-base.h`.
- Internal layering, object shapes, lifetime counters, and vtable structure are centralized in `gsx/src/gsx-impl.h`.
- If you are changing a subsystem, read its backend-neutral file in `gsx/src/` first, then the matching backend files.
- For concrete style and organization cues, the most representative files are `gsx/src/gsx-core.c`, `gsx/src/gsx-loss.c`, `gsx/src/gsx-render.c`, and `gsx/src/gsx-runtime.c`.

## Conventions You Will See
- Public headers are intentionally self-contained, layered, and heavily documented; they are the best source for caller-facing semantics.
- Backend-neutral policy usually lives in `gsx/src/`, while backend mechanics usually live in `gsx/src/gsx-cpu/`, `gsx/src/gsx-cuda/`, or `gsx/src/gsx-metal/`.
- Naming is consistently namespaced: public APIs use `gsx_` and `GSX_`, internal structs often use `struct gsx_xxx`, handles use `gsx_xxx_t`, and internal interfaces/vtables use `gsx_xxx_i`.
- Public ABI-sensitive code tends to use GSX typedefs such as `gsx_index_t`, `gsx_size_t`, `gsx_id_t`, `gsx_float_t`, `gsx_flags32_t`, and `gsx_flags64_t`.
- Error-heavy paths typically use `gsx_error`, early validation, explicit cleanup, and ownership-aware messages, but the exact local shape should follow the surrounding file.
- Comments are usually reserved for contracts, ownership, ordering, synchronization, and rollback behavior rather than line-by-line narration.

## Testing Map
- `tests/api_contract.c` and `tests/api_contract.cpp` are the main references for ABI/signature/shape expectations.
- `tests/standalone_header_compile.c` and `tests/standalone_header_compile.cpp` show the self-contained-header expectation.
- `tests/backend_runtime.cpp`, `tests/core_runtime.cpp`, `tests/render_runtime.cpp`, `tests/loss_runtime.cpp`, `tests/optim_runtime.cpp`, `tests/adc_runtime.cpp`, `tests/data_runtime.cpp`, `tests/runtime_scheduler.cpp`, and `tests/runtime_session.cpp` cover runtime behavior by subsystem.
- Backend-specific runtime coverage lives in the CUDA and Metal test files and is easiest to spot through `tests/CMakeLists.txt`.
- `tests/CMakeLists.txt` is the best place to answer "what test target exists?", "what labels exist?", and "what exact CTest name should I run?".

## Practical Editing Hints
- When changing public surface area, inspect the corresponding header, neutral implementation file, tests, and any backend implementations that dispatch that feature.
- When changing runtime orchestration, also inspect lower layers because `gsx-runtime.c` sits above backend/render/data/loss/optim/adc layering.
- When changing ownership or lifecycle behavior, check both the header docs and `gsx/src/gsx-impl.h` because many invariants are expressed in both places.
- When uncertain about style, match the nearest well-maintained neighboring code instead of treating this file as a strict style manual.

## Review Lens
- Review in context rather than by isolated diff hunk: related headers, callers, callees, tests, and enabled backends often matter.
- The most valuable checks here are usually correctness, lifetime safety, portability, completeness across backends, and whether tests still reflect the contract.

## Editor And Agent Rules
- No Cursor rules were found in `.cursor/rules/` or `.cursorrules`.
- No Copilot instructions were found in `.github/copilot-instructions.md`.
- Repo-local agent guidance also exists in `.agents/skills/code-review/SKILL.md` and `.agents/skills/test-coverage/SKILL.md`; use them as supplemental workflow guidance, not as replacements for this file.
