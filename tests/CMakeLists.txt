find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(
        FATAL_ERROR
        "Could not find GTest. "
        "Install the system package (e.g., libgtest-dev on Ubuntu, gtest via vcpkg, or gtest via Homebrew) "
        "and ensure CMake can discover it. If installed in a non-standard location, "
        "configure with -DCMAKE_PREFIX_PATH=<install-prefix>."
    )
endif()
include(GoogleTest)

add_executable(gsx_header_umbrella_c standalone_header_compile.c)
target_link_libraries(gsx_header_umbrella_c PRIVATE gsx::headers)
target_compile_definitions(gsx_header_umbrella_c PRIVATE GSX_TEST_HEADER="gsx/gsx.h")
gsx_configure_target(gsx_header_umbrella_c C)
add_test(NAME gsx_header_umbrella_c COMMAND gsx_header_umbrella_c)
set_tests_properties(gsx_header_umbrella_c PROPERTIES LABELS "contract")

add_executable(gsx_header_umbrella_cpp standalone_header_compile.cpp)
target_link_libraries(gsx_header_umbrella_cpp PRIVATE gsx::headers)
target_compile_definitions(gsx_header_umbrella_cpp PRIVATE GSX_TEST_HEADER="gsx/gsx.h")
gsx_configure_target(gsx_header_umbrella_cpp CXX)
add_test(NAME gsx_header_umbrella_cpp COMMAND gsx_header_umbrella_cpp)
set_tests_properties(gsx_header_umbrella_cpp PROPERTIES LABELS "contract")

add_executable(gsx_api_contract_c api_contract.c)
target_link_libraries(gsx_api_contract_c PRIVATE gsx::headers)
gsx_configure_target(gsx_api_contract_c C)
add_test(NAME gsx_api_contract_c COMMAND gsx_api_contract_c)
set_tests_properties(gsx_api_contract_c PROPERTIES LABELS "contract")

add_executable(gsx_api_contract_cpp api_contract.cpp)
target_link_libraries(gsx_api_contract_cpp PRIVATE gsx::headers GTest::gtest_main)
gsx_configure_target(gsx_api_contract_cpp CXX)
gtest_discover_tests(
    gsx_api_contract_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_backend_preinit_runtime_c backend_preinit_runtime.c)
target_link_libraries(gsx_backend_preinit_runtime_c PRIVATE gsx::gsx)
gsx_configure_target(gsx_backend_preinit_runtime_c C)
add_test(NAME gsx_backend_preinit_runtime_c COMMAND gsx_backend_preinit_runtime_c)
set_tests_properties(gsx_backend_preinit_runtime_c PROPERTIES LABELS "unit")

add_executable(gsx_backend_runtime_cpp backend_runtime.cpp)
target_link_libraries(gsx_backend_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_backend_runtime_cpp CXX)
gtest_discover_tests(
    gsx_backend_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_core_runtime_cpp core_runtime.cpp)
target_link_libraries(gsx_core_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_core_runtime_cpp CXX)
gtest_discover_tests(
    gsx_core_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_optim_runtime_cpp optim_runtime.cpp)
target_link_libraries(gsx_optim_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_optim_runtime_cpp CXX)
gtest_discover_tests(
    gsx_optim_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_adc_runtime_cpp adc_runtime.cpp)
target_link_libraries(gsx_adc_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_adc_runtime_cpp CXX)
gtest_discover_tests(
    gsx_adc_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_loss_runtime_cpp loss_runtime.cpp)
target_link_libraries(gsx_loss_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_loss_runtime_cpp CXX)
gtest_discover_tests(
    gsx_loss_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_backend_registry_topology_cpp backend_registry_topology.cpp)
target_link_libraries(gsx_backend_registry_topology_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_backend_registry_topology_cpp CXX)
gtest_discover_tests(
    gsx_backend_registry_topology_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_render_runtime_cpp render_runtime.cpp)
target_link_libraries(gsx_render_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_render_runtime_cpp CXX)
gtest_discover_tests(
    gsx_render_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_data_runtime_cpp data_runtime.cpp)
target_link_libraries(gsx_data_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_data_runtime_cpp CXX)
gtest_discover_tests(
    gsx_data_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_runtime_scheduler_cpp runtime_scheduler.cpp)
target_link_libraries(gsx_runtime_scheduler_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_runtime_scheduler_cpp CXX)
gtest_discover_tests(
    gsx_runtime_scheduler_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_runtime_session_cpp runtime_session.cpp)
target_link_libraries(gsx_runtime_session_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_runtime_session_cpp CXX)
gtest_discover_tests(
    gsx_runtime_session_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_stbi_runtime_cpp stbi_runtime.cpp)
target_link_libraries(gsx_stbi_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_stbi_runtime_cpp CXX)
gtest_discover_tests(
    gsx_stbi_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

add_executable(gsx_io_ply_runtime_cpp io_ply_runtime.cpp)
target_link_libraries(gsx_io_ply_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_io_ply_runtime_cpp CXX)
gtest_discover_tests(
    gsx_io_ply_runtime_cpp
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

if(GSX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(gsx_cuda_backend_runtime_cpp cuda_backend_runtime.cpp)
    target_link_libraries(gsx_cuda_backend_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main CUDA::cudart)
    gsx_configure_target(gsx_cuda_backend_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_cuda_backend_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_cuda_optim_runtime_cpp cuda_optim_runtime.cpp)
    target_link_libraries(gsx_cuda_optim_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main CUDA::cudart)
    gsx_configure_target(gsx_cuda_optim_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_cuda_optim_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_cuda_loss_runtime_cpp cuda_loss_runtime.cpp)
    target_link_libraries(gsx_cuda_loss_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main CUDA::cudart)
    gsx_configure_target(gsx_cuda_loss_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_cuda_loss_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_cuda_render_runtime_cpp cuda_render_runtime.cpp)
    target_link_libraries(gsx_cuda_render_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main CUDA::cudart)
    gsx_configure_target(gsx_cuda_render_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_cuda_render_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_cuda_adc_runtime_cpp cuda_adc_runtime.cpp)
    target_link_libraries(gsx_cuda_adc_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main CUDA::cudart)
    gsx_configure_target(gsx_cuda_adc_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_cuda_adc_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )
endif()

if(GSX_USE_METAL)
    add_executable(gsx_metal_backend_runtime_cpp metal_backend_runtime.cpp)
    target_link_libraries(gsx_metal_backend_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
    gsx_configure_target(gsx_metal_backend_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_metal_backend_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_metal_optim_runtime_cpp metal_optim_runtime.cpp)
    target_link_libraries(gsx_metal_optim_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
    gsx_configure_target(gsx_metal_optim_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_metal_optim_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_metal_loss_runtime_cpp metal_loss_runtime.cpp)
    target_link_libraries(gsx_metal_loss_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
    gsx_configure_target(gsx_metal_loss_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_metal_loss_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )

    add_executable(gsx_metal_adc_runtime_cpp metal_adc_runtime.cpp)
    target_link_libraries(gsx_metal_adc_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
    gsx_configure_target(gsx_metal_adc_runtime_cpp CXX)
    gtest_discover_tests(
        gsx_metal_adc_runtime_cpp
        PROPERTIES
            LABELS "unit"
        DISCOVERY_TIMEOUT 30
    )
endif()
