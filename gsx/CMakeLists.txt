add_library(
    gsx
    STATIC
        src/gsx-core.c
        src/gsx-backend.c
        src/gsx-cpu/gsx-cpu.c
)
add_library(gsx::gsx ALIAS gsx)

add_library(gsx_headers INTERFACE)
add_library(gsx::headers ALIAS gsx_headers)

target_include_directories(
    gsx_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(gsx_headers INTERFACE c_std_11 cxx_std_17)
target_link_libraries(gsx PUBLIC gsx::headers)
target_include_directories(
    gsx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

gsx_configure_target(gsx C)
