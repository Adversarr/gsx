find_package(benchmark CONFIG QUIET)
if(NOT benchmark_FOUND)
    message(
        FATAL_ERROR
        "Could not find benchmark. "
        "Install the system package (e.g., libbenchmark-dev on Ubuntu, benchmark via vcpkg, or benchmark via Homebrew) "
        "and ensure CMake can discover it. If installed in a non-standard location, "
        "configure with -DCMAKE_PREFIX_PATH=<install-prefix>."
    )
endif()

add_executable(gsx_benchmark_smoke api_loss_benchmark.cpp)
target_link_libraries(gsx_benchmark_smoke PRIVATE gsx::gsx benchmark::benchmark)
target_compile_definitions(gsx_benchmark_smoke PRIVATE GSX_HAS_CUDA=$<BOOL:${GSX_USE_CUDA}>)
gsx_configure_target(gsx_benchmark_smoke CXX)

if(GSX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(gsx_benchmark_smoke PRIVATE CUDA::cudart)
endif()

if(GSX_USE_METAL)
    add_executable(gsx_benchmark_metal metal_kernel_bench.cpp)
    target_link_libraries(gsx_benchmark_metal PRIVATE gsx::gsx benchmark::benchmark)
    target_compile_definitions(gsx_benchmark_metal PRIVATE GSX_HAS_METAL=$<BOOL:${GSX_USE_METAL}>)
    gsx_configure_target(gsx_benchmark_metal CXX)
endif()
