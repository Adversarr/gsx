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

add_executable(gsx_benchmark_smoke api_smoke_benchmark.cpp)
target_link_libraries(gsx_benchmark_smoke PRIVATE gsx::headers benchmark::benchmark)
gsx_configure_target(gsx_benchmark_smoke CXX)
