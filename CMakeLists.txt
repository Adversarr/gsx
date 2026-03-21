cmake_minimum_required(VERSION 3.20)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(GSX_USE_CUDA "Enable CUDA backend support." OFF)
option(GSX_USE_METAL "Enable Metal backend support." OFF)
option(GSX_ENABLE_COVERAGE "Enable gcov-compatible host coverage instrumentation and report targets." OFF)
option(GSX_BUILD_APP_EXTRAS "Build extra applications." OFF)
option(GSX_BUILD_TESTS "Build tests." ON)
option(GSX_BUILD_BENCHMARKS "Build benchmarks." OFF)
if(GSX_USE_METAL AND NOT APPLE)
    message(FATAL_ERROR "GSX_USE_METAL is only supported on Apple platforms (macOS/iOS). Disable it or switch to an Apple host.")
endif()

if(GSX_USE_CUDA)
    project(gsx VERSION 0.0.1 LANGUAGES C CXX CUDA)
elseif(GSX_USE_METAL)
    project(gsx VERSION 0.0.1 LANGUAGES C CXX OBJC OBJCXX)
else()
    project(gsx VERSION 0.0.1 LANGUAGES C CXX)
endif()

if(GSX_ENABLE_COVERAGE)
    if(MSVC)
        message(FATAL_ERROR "GSX_ENABLE_COVERAGE is not supported with MSVC.")
    endif()
    if(GSX_USE_CUDA)
        message(FATAL_ERROR "GSX_ENABLE_COVERAGE is currently supported only for host-only builds. Disable GSX_USE_CUDA.")
    endif()
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        message(FATAL_ERROR "GSX_ENABLE_COVERAGE requires a GNU-compatible coverage toolchain (GCC or Clang).")
    endif()

    find_program(GSX_GCOVR_EXECUTABLE NAMES gcovr)
    if(GSX_GCOVR_EXECUTABLE STREQUAL "GSX_GCOVR_EXECUTABLE-NOTFOUND")
        message(FATAL_ERROR "GSX_ENABLE_COVERAGE requires gcovr to be installed and available on PATH.")
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        find_program(GSX_LLVM_COV_EXECUTABLE NAMES llvm-cov)
        if(GSX_LLVM_COV_EXECUTABLE STREQUAL "GSX_LLVM_COV_EXECUTABLE-NOTFOUND")
            message(FATAL_ERROR "GSX_ENABLE_COVERAGE with Clang requires llvm-cov to be installed and available on PATH.")
        endif()
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gsx-llvm-gcov.sh.in
            ${CMAKE_CURRENT_BINARY_DIR}/gsx-llvm-gcov.sh
            @ONLY
        )
        file(
            CHMOD
            ${CMAKE_CURRENT_BINARY_DIR}/gsx-llvm-gcov.sh
            PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        )
        set(GSX_COVERAGE_GCOV_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/gsx-llvm-gcov.sh)
    else()
        find_program(GSX_COVERAGE_GCOV_EXECUTABLE NAMES gcov)
        if(GSX_COVERAGE_GCOV_EXECUTABLE STREQUAL "GSX_COVERAGE_GCOV_EXECUTABLE-NOTFOUND")
            message(FATAL_ERROR "GSX_ENABLE_COVERAGE requires gcov to be installed and available on PATH.")
        endif()
    endif()

    add_library(gsx_coverage_flags INTERFACE)
    target_compile_options(
        gsx_coverage_flags
        INTERFACE
            $<$<COMPILE_LANGUAGE:C>:--coverage;-O0;-g>
            $<$<COMPILE_LANGUAGE:CXX>:--coverage;-O0;-g>
            $<$<COMPILE_LANGUAGE:OBJC>:--coverage;-O0;-g>
            $<$<COMPILE_LANGUAGE:OBJCXX>:--coverage;-O0;-g>
    )
    target_link_options(gsx_coverage_flags INTERFACE --coverage)
endif()


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

add_subdirectory(gsx)
add_subdirectory(apps)

if(GSX_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GSX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
