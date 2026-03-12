cmake_minimum_required(VERSION 3.20)

project(gsx VERSION 2.0.0 LANGUAGES C CXX)

option(GSX_BUILD_TESTS "Build compile-time API contract tests." ON)
option(GSX_BUILD_BENCHMARKS "Build Google Benchmark smoke benchmarks." OFF)

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
