---
name: kernel-tuning
description: |
  Optimize backend execution speed in GSX-style C/C++/CUDA/Metal codebases while preserving correctness. Use this skill whenever the user asks to tune kernels, speed up a backend path or full pipeline, analyze timing breakdowns, compare backend-vs-CPU numerics, or improve benchmark results after tests, even if they do not explicitly say "kernel tuning."
compatibility:
  tools: [cmake>=3.20, ctest, rg]
  os: [Linux, macOS]
---

# Kernel Tuning Skill

What it does
- Optimizes speed across any backend hot path, not just rendering kernels.
- Treats tests and correctness checks as non-negotiable gates.
- Defaults to end-to-end verification with `bench_dataset` when that app is available.
- Keeps the workflow parameterized for `<backend>`, `<build_dir>`, and backend-specific debug env vars.

When to reach for this skill
- The user wants better throughput or lower step time in a backend path or full pipeline.
- The task mentions kernels, timing breakdowns, launch geometry, occupancy, synchronization, fusion, batching, or dispatch overhead.
- The user wants backend-vs-CPU numerical validation before or after a speed change.

Inputs to pin down early
- `backend`: backend name passed to the app, for example `<backend>`.
- `build_dir`: build directory for that backend, for example `build-<backend>`.
- `dump_env_var`: backend-specific forward-dump env var, for example `GSX_<BACKEND_UPPER>_FORWARD_DUMP`.
- `benchmark_cmd`: benchmark command if the user already has one.
- `dataset_root`: default `data/garden` when present.
- `ply_path`: default `data/points.ply` when present.

Workflow

1. Find the real hot path
- Use benchmark stage output, timing breakdowns, or profiler evidence.
- Do not assume the bottleneck is one render kernel or that it is memory bound.
- Read the neutral entrypoint, backend implementation, and relevant tests before editing.

2. Change one speed lever at a time
- Let evidence drive the choice: launch shape, instruction mix, synchronization, occupancy, memory traffic, divergence, fusion, or host overhead.
- Preserve public behavior and backend-neutral contracts.
- Keep diffs attributable unless a larger rewrite is clearly justified.

3. Prove correctness before claiming speed
- Start with the full build and test pass:
```bash
cmake --build build -j --target all && ctest build/
```
- Then inspect the failing tail if needed:
```bash
ctest --test-dir build --output-on-failure | tail -n 30
```
- Treat any correctness regression as a blocker. Fast but wrong does not count.

4. Run numerical comparison sweeps
- Use the backend-specific dump env var and compare against CPU reference across a size sweep before trusting benchmark numbers.
```bash
for n in 4 16 32 128 512 1024 1536 2048; do
  echo "--- n=$n"
  <dump_env_var>=1 <build_dir>/apps/multi-gaussian-render \
    --backend <backend> \
    --compare-with-cpu true \
    --gaussian-count "$n" \
    --width 489 \
    --height 328 2>&1 | rg "(cpu reference compare)|<BACKEND_UPPER>|FAILED|PASSED"
done && <build_dir>/apps/multi-gaussian-render \
  --backend <backend> \
  --numerical-diff true \
  --gaussian-count 24
```
- Replace `<BACKEND_UPPER>` with the uppercase backend token that appears in logs.
- If the numerical sweep regresses, stop and explain the failure mode before making more performance changes.

5. Benchmark only after correctness is stable
- Default to the end-to-end dataset benchmark, not the microbenchmarks under `benchmarks/`.
- Treat `apps/CMakeLists.txt` as the source of truth for whether `bench_dataset` exists and remember it requires `GSX_BUILD_APP_EXTRAS=ON`.
- If `data/garden` and `data/points.ply` are present in the active workspace, use them by default:
```bash
cmake -S . -B <build_dir> -DGSX_BUILD_APP_EXTRAS=ON [backend flags...]
cmake --build <build_dir> -j8 --target bench_dataset
<build_dir>/apps/bench_dataset --dataset-root data/garden --ply data/points.ply --backend <backend>
```
- If the user already supplied a benchmark command, use that instead.
- If the dataset or PLY file is unavailable, stop and ask the user for the dataset location instead of silently switching to another benchmark.
- Only use binaries under `benchmarks/` when the user explicitly asks for those microbenchmarks.
- If possible, collect before/after numbers using the same command, inputs, and build settings.
- Use stage timing output to identify which parts moved.
- Report whether the win comes from kernel time, launch overhead, scaling, overlap, or a combination.

What to report
- The tuning hypothesis and the specific hot path or kernel set changed.
- Correctness status from build/tests.
- Numerical comparison status across the sweep, including any failing sizes.
- Benchmark result before vs after, or a note that benchmarking was skipped because correctness was not yet restored.
- Risks, follow-up ideas, and any remaining uncertainty.

Useful habits
- Keep diffs small enough that regressions are attributable.
- Re-read the surrounding backend-neutral contract after touching backend code; many performance bugs are really contract bugs.
- If a tuning idea increases complexity, explain why the speedup justifies it.
- Prefer reproducible commands and fixed problem sizes so later comparisons are meaningful.
- Prefer `bench_dataset` for default performance claims because it captures full training-step behavior and stage timing, while `benchmarks/` only samples narrower kernels.
- Do not lock onto a single diagnosis too early. Revisit the benchmark breakdown after each meaningful change.

Example placeholders
- `backend=<backend>`
- `build_dir=build-<backend>`
- `dump_env_var=GSX_<BACKEND_UPPER>_FORWARD_DUMP`
- `dataset_root=data/garden`
- `ply_path=data/points.ply`

Default response shape
- `Target`: hot path or kernel set being tuned and the tuning hypothesis.
- `Changes`: concise explanation of what changed and why.
- `Correctness`: build/test and numerical check results.
- `Performance`: benchmark results or why they were deferred.
- `Next steps`: the most promising follow-up experiments.
