set(
    GSX_SOURCES
        src/gsx-core.c
        src/gsx-backend.c
        src/gsx-cpu/gsx-cpu.c
)

if(GSX_USE_CUDA)
    list(
        APPEND
        GSX_SOURCES
            src/gsx-cuda/gsx-cuda.c
            src/gsx-cuda/gsx-cuda-kernels.cu
    )
endif()

add_library(
    gsx
    STATIC
        ${GSX_SOURCES}
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

target_compile_definitions(gsx PRIVATE GSX_HAS_CUDA=$<BOOL:${GSX_USE_CUDA}>)

target_include_directories(
    gsx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

gsx_configure_target(gsx C)

target_include_directories(
    gsx
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(GSX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(gsx PRIVATE CUDA::cudart)
    set_target_properties(gsx PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
