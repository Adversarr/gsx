cmake_minimum_required(VERSION 3.20)

project(gsx VERSION 2.0.0 LANGUAGES C CXX)

option(GSX_BUILD_TESTS "Build compile-time API contract tests." ON)

add_library(gsx_headers INTERFACE)
add_library(gsx::headers ALIAS gsx_headers)

target_include_directories(
    gsx_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gsx/include>
)

target_compile_features(gsx_headers INTERFACE c_std_11 cxx_std_17)

if(GSX_BUILD_TESTS)
    include(CTest)
    enable_testing()

    function(gsx_add_header_compile_target language target_name header_name)
        if(language STREQUAL "C")
            add_executable(${target_name} tests/standalone_header_compile.c)
            set_target_properties(
                ${target_name}
                PROPERTIES
                    C_STANDARD 11
                    C_STANDARD_REQUIRED YES
                    C_EXTENSIONS NO
            )
        elseif(language STREQUAL "CXX")
            add_executable(${target_name} tests/standalone_header_compile.cpp)
            set_target_properties(
                ${target_name}
                PROPERTIES
                    CXX_STANDARD 17
                    CXX_STANDARD_REQUIRED YES
                    CXX_EXTENSIONS NO
            )
        else()
            message(FATAL_ERROR "Unsupported language for header compile target: ${language}")
        endif()

        target_link_libraries(${target_name} PRIVATE gsx::headers)
        target_compile_definitions(${target_name} PRIVATE GSX_TEST_HEADER=\"${header_name}\")
        add_test(NAME ${target_name} COMMAND ${target_name})
    endfunction()

    function(gsx_add_contract_target language target_name source_file)
        add_executable(${target_name} ${source_file})
        target_link_libraries(${target_name} PRIVATE gsx::headers)

        if(language STREQUAL "C")
            set_target_properties(
                ${target_name}
                PROPERTIES
                    C_STANDARD 11
                    C_STANDARD_REQUIRED YES
                    C_EXTENSIONS NO
            )
        elseif(language STREQUAL "CXX")
            set_target_properties(
                ${target_name}
                PROPERTIES
                    CXX_STANDARD 17
                    CXX_STANDARD_REQUIRED YES
                    CXX_EXTENSIONS NO
            )
        else()
            message(FATAL_ERROR "Unsupported language for API contract target: ${language}")
        endif()

        add_test(NAME ${target_name} COMMAND ${target_name})
    endfunction()

    gsx_add_header_compile_target(C   gsx_header_base_c      "gsx/gsx-base.h")
    gsx_add_header_compile_target(C   gsx_header_core_c      "gsx/gsx-core.h")
    gsx_add_header_compile_target(C   gsx_header_backend_c   "gsx/gsx-backend.h")
    gsx_add_header_compile_target(C   gsx_header_render_c    "gsx/gsx-render.h")
    gsx_add_header_compile_target(C   gsx_header_data_c      "gsx/gsx-data.h")
    gsx_add_header_compile_target(C   gsx_header_loss_c      "gsx/gsx-loss.h")
    gsx_add_header_compile_target(C   gsx_header_optim_c     "gsx/gsx-optim.h")
    gsx_add_header_compile_target(C   gsx_header_adc_c       "gsx/gsx-adc.h")
    gsx_add_header_compile_target(C   gsx_header_runtime_c   "gsx/gsx-runtime.h")
    gsx_add_header_compile_target(C   gsx_header_umbrella_c  "gsx/gsx.h")

    gsx_add_header_compile_target(CXX gsx_header_base_cpp     "gsx/gsx-base.h")
    gsx_add_header_compile_target(CXX gsx_header_core_cpp     "gsx/gsx-core.h")
    gsx_add_header_compile_target(CXX gsx_header_backend_cpp  "gsx/gsx-backend.h")
    gsx_add_header_compile_target(CXX gsx_header_render_cpp   "gsx/gsx-render.h")
    gsx_add_header_compile_target(CXX gsx_header_data_cpp     "gsx/gsx-data.h")
    gsx_add_header_compile_target(CXX gsx_header_loss_cpp     "gsx/gsx-loss.h")
    gsx_add_header_compile_target(CXX gsx_header_optim_cpp    "gsx/gsx-optim.h")
    gsx_add_header_compile_target(CXX gsx_header_adc_cpp      "gsx/gsx-adc.h")
    gsx_add_header_compile_target(CXX gsx_header_runtime_cpp  "gsx/gsx-runtime.h")
    gsx_add_header_compile_target(CXX gsx_header_umbrella_cpp "gsx/gsx.h")

    gsx_add_contract_target(C   gsx_api_contract_c   tests/api_contract.c)
    gsx_add_contract_target(CXX gsx_api_contract_cpp tests/api_contract.cpp)
endif()
