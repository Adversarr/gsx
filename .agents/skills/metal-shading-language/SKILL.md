---
name: metal-shading-language
description: Use this skill whenever working on Metal Shading Language, Apple GPU shaders, `.metal` code, compute or render pipeline shader semantics, threadgroup or device memory, texture or sampler usage, Metal standard library functions, numerical behavior, or Metal Performance Primitives. Consult it proactively for spec-grounded explanations, code review, bug fixing, API usage constraints, and writing new MSL that must match Apple's documented rules.
---

# Metal Shading Language

Use this skill to answer Metal Shading Language questions with chapter-aware, spec-grounded guidance.

## What this skill does

- Help write, explain, review, debug, and refactor MSL code.
- Point to the right part of the bundled spec before making strong claims.
- Surface version, platform, GPU-family, and address-space constraints when they matter.
- Prefer the bundled markdown references over memory when exact rules or edge cases matter.

## Working approach

1. Identify the user's actual task: language syntax, shader entry-point design, resource binding, memory model, built-ins, numerical behavior, or MPP/tensor ops.
2. Read only the relevant chapter files from `references/chapters/` first; do not load everything unless the task is broad.
3. Give the answer in practical terms, but mention chapter or section numbers when they help the user verify the rule.
4. Call out uncertainty if the OCR text looks malformed or if the rule clearly depends on Metal version, OS release, or hardware tier.
5. When writing code, keep it aligned with the documented constraints rather than generic C++ intuition.

## Reference map

- `references/chapters/01-introduction.md` covers language model, compiler flags, versioning, and coordinate systems.
- `references/chapters/02-data-types.md` covers scalar/vector/matrix/resource types, layout, alignment, and binding-facing data rules.
- `references/chapters/03-operators.md` covers operator semantics and expression edge cases.
- `references/chapters/04-address-spaces.md` covers `device`, `constant`, `thread`, `threadgroup`, and newer address spaces.
- `references/chapters/05-function-and-variable-declarations.md` covers shader entry points, attributes, stage I/O, and resource bindings.
- `references/chapters/06-metal-standard-library.md` covers standard library built-ins, precision variants, synchronization, and SIMD-group helpers.
- `references/chapters/07-metal-performance-primitives.md` covers MPP tensor ops and cooperative tensor workflows.
- `references/chapters/08-numerical-compliance.md` covers IEEE-754 deviations, accuracy, NaN/INF behavior, and texture conversion rules.
- `references/chapters/09-appendix.md` covers Metal 3.2 additions and where to look for them.

## Notes

- These files come from OCR output, so occasional formatting glitches are expected.
- Prefer consulting only the chapter you need instead of loading every file.

## Chapter summaries

### Chapter 1 - Introduction

Metal Shading Language is a unified, C++-based GPU language for graphics and compute shaders, and this chapter sets the model for what MSL is and is not. It establishes Metal 4's C++17 baseline, core restrictions, compiler controls, and the coordinate systems shaders target.

- MSL is unified across graphics and compute and many features depend on Metal language version and GPU family support.
- Metal 4 is C++17-based, but MSL forbids or limits features like RTTI, `new`/`delete`, exceptions, virtual functions, derived classes, and the standard C++ library.
- Overloading is supported, but shader entry functions themselves cannot be overloaded.
- Pointer arguments on shader entry points must declare Metal address spaces such as `device`, `constant`, or `threadgroup`.
- Compiler settings like `-std=...`, `-ffast-math`, and macros such as `__METAL_VERSION__` materially affect behavior.
- Coordinate systems matter: clip space, left-handed NDC, top-left-origin viewport space, and normalized texture coordinates all have explicit rules.

Best used for: deciding language compatibility, compiler settings, and core shader-model assumptions before writing or reviewing MSL.

### Chapter 2 - Data Types

This is the main reference for what MSL values and resources look like in memory and which type declarations are legal. It covers scalar, vector, matrix, packed, atomic, texture, sampler, argument-buffer, and newer advanced resource and tensor-facing types.

- Scalars, vectors, and matrices have explicit size and alignment rules, including common padding behavior for 3-component vectors.
- Swizzles are powerful but constrained; some mixed-component or duplicate-write forms are illegal or undefined.
- Packed types are for storage layout, not a free substitute for compute-friendly types.
- Textures, samplers, sparse resources, and other resource types have strict access-mode and platform/version restrictions.
- Arrays, structs, and argument buffers containing resources follow binding and nesting rules that affect legality.
- Advanced types like `uniform<T>`, visible function tables, ray-tracing types, per-vertex values, mesh types, and tensor types add specialized constraints.

Best used for: checking type legality, memory layout, alignment, and resource declaration rules.

### Chapter 3 - Operators

This chapter explains how operators behave across scalars, vectors, and matrices, especially where Metal diverges from naive C++ expectations. It highlights undefined or unspecified cases around integer overflow, division, modulus, and NaN comparisons.

- Scalar/vector operators use normal arithmetic conversions plus scalar widening for componentwise vector math.
- `bfloat` and `half` cannot be mixed implicitly in expressions.
- Integer divide-by-zero, signed overflow, and some modulus cases are unspecified or undefined.
- Vector relational operators return componentwise boolean vectors.
- Comparisons with `NaN` behave per floating-point rules rather than user intuition.
- Matrix multiplication is true linear algebra multiplication, not componentwise multiplication.

Best used for: checking exact expression semantics and edge cases in MSL code.

### Chapter 4 - Address Spaces

Address spaces define memory region, lifetime, visibility, and mutability, and they are mandatory on pointer-like shader interfaces. This chapter covers the standard spaces plus newer specialized ones and explains coherency behavior.

- Pointer and reference arguments must declare an address space; omitting one is a compile-time error.
- `device` is read/write device-backed buffer memory, while textures still use texture APIs instead of direct element access.
- `constant` is read-only, initialized at declaration time, and tied to constant-expression rules.
- `thread` is private per-thread storage and `threadgroup` is shared per-threadgroup storage.
- Specialized spaces such as `threadgroup_imageblock`, `ray_data`, and `object_data` are stage- and feature-specific.
- Coherency rules differ by address space, and newer `coherent(device)`-style behavior is version-gated.

Best used for: choosing correct address spaces, memory lifetime, and visibility rules.

### Chapter 5 - Function and Variable Declarations

This chapter defines how Metal exposes shader functions, stage inputs/outputs, and bound resources. It is the key reference for legal shader signatures, attributes, entry-point qualifiers, and pipeline-facing declarations.

- Qualified functions use attributes like `[[vertex]]`, `[[fragment]]`, `[[kernel]]`, `[[mesh]]`, `[[object]]`, `[[visible]]`, and `[[intersection(... )]]`.
- Entry points have strict return-type and stage-specific signature constraints.
- Binding attributes like `[[buffer(i)]]`, `[[texture(i)]]`, `[[sampler(i)]]`, and `[[threadgroup(i)]]` must be unique within resource classes.
- Vertex, tessellation, mesh, fragment, and compute interfaces use dedicated built-in attributes and type restrictions.
- Host-visible names, explicit template instantiation, user annotations, and raster-order-group semantics are also defined here.
- Aliasing rules matter: some resource arguments to the same function are not allowed to alias.

Best used for: validating shader entry points, resource bindings, and attribute usage.

### Chapter 6 - Metal Standard Library

This chapter catalogs the built-ins and helper facilities in the Metal Standard Library, from common math to synchronization and SIMD-group operations. It matters both for correctness and for performance-sensitive compute and graphics code.

- Standard-library functions and enums live in `metal`, with umbrella access through `<metal_stdlib>` and more specific headers for domains like math or SIMD-group work.
- Many functions have documented input constraints; outside them, behavior may be undefined.
- Single-precision math often has `fast` and `precise` variants with different NaN handling and accuracy guarantees.
- The library spans integer bit ops, geometry, constants, matrices, synchronization, barriers, reductions, and SIMD-group communication.
- SIMD-group and matrix helpers have operational constraints around active lanes, barriers, and memory ordering.
- Precision and performance tradeoffs are often controlled by `-ffast-math` or explicit `metal::fast` / `metal::precise` usage.

Best used for: selecting built-ins and checking correctness, precision, synchronization, and SIMD-group usage constraints.

### Chapter 7 - Metal Performance Primitives

This chapter covers Apple-silicon-focused tensor primitives under `mpp`, especially `tensor_ops` for matrix multiplication and convolution. It is a specialized performance reference for high-throughput tensor workflows in Metal 4.

- Execution scope is fundamental: thread, SIMD-group, or multi-SIMD-group scope changes how TensorOps must be called.
- All participating threads in the declared scope must reach `run()` or behavior is undefined.
- Device and threadgroup tensor outputs often require matching barriers before readback.
- `matmul2d` supports generalized GEMM-style workflows, configurable transpose and precision modes, and several type combinations.
- `cooperative_tensor` supports distributed accumulation, reductions, and iterator-based post-processing.
- `convolution2d` currently targets specific execution-scope and tensor-layout constraints.

Best used for: implementing or reviewing high-performance tensor operations in MSL.

### Chapter 8 - Numerical Compliance

Metal does not implement the full IEEE 754 model, and this chapter explains the practical consequences. It defines NaN/INF handling, denormals, rounding, accuracy guarantees, fast-math tradeoffs, and texture conversion behavior.

- INF is supported for major floating types, but NaN support depends on fast-math settings and signaling NaNs are unsupported.
- Denormals may flush to zero and floating-point exceptions are not supported.
- Float-to-int conversion uses round-toward-zero and NaN-to-int becomes `0`.
- Accuracy guarantees are specified in ULPs for many operations and differ across type and mode.
- Fast math can reassociate expressions and weaken assumptions about signed zero, NaN, INF, underflow, and overflow.
- Texture sampling, reading, writing, and pixel-format conversion rules are tightly specified, including sRGB and normalized integer behavior.

Best used for: checking numerical guarantees, edge cases, and texture conversion behavior.

### Chapter 9 - Appendix

The appendix is a compact version-delta reference focused on Metal 3.2 additions. It points to the main sections for details rather than replacing them.

- Covers Metal 3.2 additions such as Relaxed Math.
- Notes new intersection-result-reference support.
- Points to updated memory coherency, thread scope, and fence behavior.
- Mentions global bindings and logging additions.
- Serves as a navigation map into the main spec for newer features.

Best used for: quickly identifying Metal 3.2 additions and where to find their full rules.

## Output guidance

- For direct questions, answer briefly and cite the relevant chapter file when useful.
- For code review or debugging, explain which spec rule is being violated or relied on.
- For new shader code, mention any version or platform assumptions that the code depends on.
- If the bundled OCR text is clearly malformed, say so and avoid over-claiming on the exact wording.
