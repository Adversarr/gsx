find_package(benchmark CONFIG QUIET)
if(NOT benchmark_FOUND)
    message(
        FATAL_ERROR
        "Could not find benchmark via CMake package configuration. "
        "Install the system package and make sure CMake can discover it. "
        "If you installed it with Homebrew, configure with "
        "-DCMAKE_PREFIX_PATH=/opt/homebrew."
    )
endif()

add_executable(gsx_benchmark_smoke api_smoke_benchmark.cpp)
target_link_libraries(gsx_benchmark_smoke PRIVATE gsx::headers benchmark::benchmark)
gsx_configure_target(gsx_benchmark_smoke CXX)
