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
)

add_executable(gsx_core_runtime_cpp core_runtime.cpp)
target_link_libraries(gsx_core_runtime_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_core_runtime_cpp CXX)
gtest_discover_tests(
    gsx_core_runtime_cpp
    PROPERTIES
        LABELS "unit"
)

add_executable(gsx_backend_registry_topology_cpp backend_registry_topology.cpp)
target_link_libraries(gsx_backend_registry_topology_cpp PRIVATE gsx::gsx GTest::gtest_main)
gsx_configure_target(gsx_backend_registry_topology_cpp CXX)
gtest_discover_tests(
    gsx_backend_registry_topology_cpp
    PROPERTIES
        LABELS "unit"
)
