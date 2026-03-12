cmake_minimum_required(VERSION 3.20)

project(gsx VERSION 2.0.0 LANGUAGES C CXX)

option(GSX_BUILD_TESTS "Build compile-time API contract tests." ON)
option(GSX_BUILD_BENCHMARKS "Build Google Benchmark smoke benchmarks." OFF)

add_library(gsx_headers INTERFACE)
add_library(gsx::headers ALIAS gsx_headers)

add_library(
    gsx
    STATIC
        gsx/src/gsx-backend.c
        gsx/src/gsx-cpu/gsx-cpu.c
)
add_library(gsx::gsx ALIAS gsx)

target_include_directories(
    gsx_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gsx/include>
)

target_compile_features(gsx_headers INTERFACE c_std_11 cxx_std_17)
target_link_libraries(gsx PUBLIC gsx::headers)
target_include_directories(
    gsx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gsx/include>
)

function(gsx_configure_target target_name language_name)
    if(language_name STREQUAL "C")
        set_target_properties(
            ${target_name}
            PROPERTIES
                C_STANDARD 11
                C_STANDARD_REQUIRED YES
                C_EXTENSIONS NO
        )
    elseif(language_name STREQUAL "CXX")
        set_target_properties(
            ${target_name}
            PROPERTIES
                CXX_STANDARD 17
                CXX_STANDARD_REQUIRED YES
                CXX_EXTENSIONS NO
        )
    else()
        message(FATAL_ERROR "Unsupported target language: ${language_name}")
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

gsx_configure_target(gsx C)

if(GSX_BUILD_TESTS)
    include(CTest)
    enable_testing()
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        message(
            FATAL_ERROR
            "Could not find GTest via CMake package configuration. "
            "Install the system package and make sure CMake can discover it. "
            "If you installed it with Homebrew, configure with "
            "-DCMAKE_PREFIX_PATH=/opt/homebrew."
        )
    endif()
    include(GoogleTest)

    add_executable(gsx_header_umbrella_c tests/standalone_header_compile.c)
    target_link_libraries(gsx_header_umbrella_c PRIVATE gsx::headers)
    target_compile_definitions(gsx_header_umbrella_c PRIVATE GSX_TEST_HEADER="gsx/gsx.h")
    gsx_configure_target(gsx_header_umbrella_c C)
    add_test(NAME gsx_header_umbrella_c COMMAND gsx_header_umbrella_c)
    set_tests_properties(gsx_header_umbrella_c PROPERTIES LABELS "contract")

    add_executable(gsx_header_umbrella_cpp tests/standalone_header_compile.cpp)
    target_link_libraries(gsx_header_umbrella_cpp PRIVATE gsx::headers)
    target_compile_definitions(gsx_header_umbrella_cpp PRIVATE GSX_TEST_HEADER="gsx/gsx.h")
    gsx_configure_target(gsx_header_umbrella_cpp CXX)
    add_test(NAME gsx_header_umbrella_cpp COMMAND gsx_header_umbrella_cpp)
    set_tests_properties(gsx_header_umbrella_cpp PROPERTIES LABELS "contract")

    add_executable(gsx_api_contract_c tests/api_contract.c)
    target_link_libraries(gsx_api_contract_c PRIVATE gsx::headers)
    gsx_configure_target(gsx_api_contract_c C)
    add_test(NAME gsx_api_contract_c COMMAND gsx_api_contract_c)
    set_tests_properties(gsx_api_contract_c PROPERTIES LABELS "contract")

    add_executable(gsx_api_contract_cpp tests/api_contract.cpp)
    target_link_libraries(gsx_api_contract_cpp PRIVATE gsx::headers GTest::gtest_main)
    gsx_configure_target(gsx_api_contract_cpp CXX)
    gtest_discover_tests(
        gsx_api_contract_cpp
        PROPERTIES
            LABELS "unit"
    )

    add_executable(gsx_backend_preinit_runtime_c tests/backend_preinit_runtime.c)
    target_link_libraries(gsx_backend_preinit_runtime_c PRIVATE gsx::gsx)
    gsx_configure_target(gsx_backend_preinit_runtime_c C)
    add_test(NAME gsx_backend_preinit_runtime_c COMMAND gsx_backend_preinit_runtime_c)
    set_tests_properties(gsx_backend_preinit_runtime_c PROPERTIES LABELS "unit")

    add_executable(gsx_backend_runtime_cpp tests/backend_runtime.cpp)
    target_link_libraries(gsx_backend_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
    gsx_configure_target(gsx_backend_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_backend_runtime_cpp
        PROPERTIES
            LABELS "unit"
    )

    add_executable(gsx_backend_registry_topology_cpp tests/backend_registry_topology.cpp)
    target_link_libraries(gsx_backend_registry_topology_cpp PRIVATE gsx::gsx GTest::gtest_main)
    gsx_configure_target(gsx_backend_registry_topology_cpp CXX)
    gtest_discover_tests(
        gsx_backend_registry_topology_cpp
        PROPERTIES
            LABELS "unit"
    )
endif()

if(GSX_BUILD_BENCHMARKS)
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

    add_executable(gsx_benchmark_smoke benchmarks/api_smoke_benchmark.cpp)
    target_link_libraries(gsx_benchmark_smoke PRIVATE gsx::headers benchmark::benchmark)
    gsx_configure_target(gsx_benchmark_smoke CXX)
endif()
