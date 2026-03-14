# GSX - AGENTS.md

GSX is an experimental C library for 3D gaussian splatting, targeting at high performance and cross-platform compatibility (under prototyping, APIs will change frequently).
GSX is designed around one caller-visible main CPU thread that dispatches backend-bound public API work in program order onto one caller-visible major GPU stream or command queue per backend. Public APIs assumes user only use in a single thread, and do not expose concurrent compute/transfer lanes on the same backend; any extra threads or streams are internal-only implementation details used for things like dataloader prefetch.

## Code Map for GSX

Repository layout:

- `gsx/include/gsx/`: public C API headers. Keep caller-visible types, enums and function declarations here.
- `gsx/src/`: core implementation for public API entrypoints and shared backend/runtime logic.
- `gsx/src/gsx-impl.h`: internal backend-agnostic declarations shared across implementation units.
- `gsx/src/gsx-cpu/`: CPU backend implementation.
- `gsx/src/gsx-cuda/`: CUDA backend implementation, device buffers and kernels.
- `apps/`: small executable tools and examples used to inspect or exercise the library.
- `tests/`: API contract, runtime and backend behavior coverage.
- `benchmarks/`: benchmark targets for API and runtime performance checks.
- `CMakeLists.txt`, `gsx/CMakeLists.txt`, `apps/CMakeLists.txt`, `tests/CMakeLists.txt`, `benchmarks/CMakeLists.txt`: build graph entrypoints and target wiring.

## Development

### General Guidelines

1. Write clear and concise code that is easy to read and understand. Use comments to explain complex logic, important decisions and declarations, but avoid over-commenting trivial code logic.
2. Use meaningful variable and function names that accurately describe their purpose.
3. Function declarations in ONE line, with the return type, function name, and argument list.
4. Code as Document: We do NOT rely on any markdown documents to document the code, instead, we use the code and comments as the document. They should be self-explanatory and consistent.
5. Preserve the public mental model in code structure: public APIs are single-threaded from the caller side, ordered, and backend dispatch must not imply extra caller-visible streams or queues.
6. Prefer small backend-agnostic entrypoints in `gsx/src/` and keep backend-specific behavior inside `gsx/src/gsx-<backend>/`.
7. When adding a new public API, update the matching public header, core implementation path, and tests in the same change.

### Coding Style Guide for GSX C API

Naming convention for OOP in C:

1. For types, use the `gsx_` prefix. Use `struct gsx_xxx` for the concrete struct definition, `gsx_xxx_t` for the typedef, and `gsx_xxx_i` for an interface or vtable type.
2. For functions, use `gsx_xxx_operation` for methods operating on an object and `gsx_operation_xxx` for class-level queries or helpers. Keep names explicit about backend, resource, or operation scope.
3. For enums, use `enum gsx_yyy : gsx_index_t { ... }` for the definition and `GSX_YYY_XXX` for enum values.
4. Keep public API names stable and backend-neutral. Backend-specific helpers should stay internal unless there is a clear caller-visible abstraction.
5. Match file names to the subsystem they implement, for example `gsx-backend.c`, `gsx-core.c`, `gsx-loss.c`, `gsx-optim.c`.

Use `gsx_error` for error handling:

- Functions should return `gsx_error` unless there is a strong reason not to.
- The function takes the pointer to the object as the first argument when operating on an object instance.
- Check every operation result or propagate it upward. No silent failure, partial success without status, or ignored backend errors.
- Prefer early returns on failure and keep cleanup paths explicit.

Implementation expectations:

- Keep public headers minimal and self-contained. `tests/standalone_header_compile.c` and `tests/standalone_header_compile.cpp` should remain valid after header changes.
- Validate assumptions at the API boundary, then keep internal hot paths simple.
- Avoid mixing backend-independent policy with backend-specific mechanics in the same function when a thin dispatch layer is sufficient.
- Add brief comments only where ordering, ownership, lifetime, or backend synchronization rules are not obvious from the code.

