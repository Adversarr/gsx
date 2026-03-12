# GSX - AGENTS.md
GSX is an experimental C library for 3D gaussian splatting, targeting at high performance and cross-platform compatibility (under prototyping, APIs will change frequently).

## Code Map for GSX

- `gsx/`: source for core compute, session and runtime API:
  `gsx/include/gsx/gsx*.h` contains the public API, struct definitions.

## Code Style

### General Guidelines
1. Write clear and concise code that is easy to read and understand. Use comments to explain complex logic, important decisions and declarations, but avoid over-commenting trivial code logic.
2. Use meaningful variable and function names that accurately describe their purpose.
3. Function declarations in ONE line, with the return type, function name, and argument list.

### Coding Style Guide for GSX C API

naming convention for OOP in C:
1. for types, use `gsx_` prefix, e.g. `struct gsx_xxx` for the struct definition, and `gsx_xxx_t` for the typedef of pointer to the struct, and `gsx_xxx_i` to indicate the interface (vtable).
2. for functions, use `gsx_xxx_operation` for operations on the object and `gsx_operation_xxx` for the operations that queries the class info, e.g. `gsx_backend_init` to create a backend object, and `gsx_count_devices` to query the number of available devices available.
3. for enums, use `enum gsx_yyy : gsx_index_t{...}` for the enum definition, and `GSX_YYY_XXX` for the enum values.

Use `gsx_error` for error handling: the function takes the pointer to the object as the first argument, and returns `gsx_error` to indicate the status of the operation. Check the operations' status or propagate errors, NO silent failure. For example:
```c
gsx_error gsx_create_xxx(gsx_xxx_t out_xxx, gsx_yyy_t yyy);
```
