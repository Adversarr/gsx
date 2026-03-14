set(
    GSX_SOURCES
        src/gsx-loss.c
        src/gsx-optim.c
        src/gsx-core.c
        src/gsx-backend.c
        src/gsx-render.c
        src/extra/gsx-stbi.c
        src/gsx-cpu/core.c
        src/gsx-cpu/loss.c
        src/gsx-cpu/optim.c
        src/gsx-cpu/render.c
        src/stb/stb_impl.c
)

if(GSX_USE_CUDA)
    list(
        APPEND
        GSX_SOURCES
            src/gsx-cuda/shared.c
            src/gsx-cuda/backend.c
            src/gsx-cuda/buffer.c
            src/gsx-cuda/loss.c
            src/gsx-cuda/optim.c
            src/gsx-cuda/render.c
            src/gsx-cuda/render-kernels.cu
            src/gsx-cuda/fastgs/forward.cu
            src/gsx-cuda/fastgs/backward.cu
            src/gsx-cuda/gsx-cuda-kernels.cu
            src/gsx-cuda/loss-pointwise-kernels.cu
            src/gsx-cuda/loss-ssim-kernels.cu
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

if(NOT MSVC)
    target_link_libraries(gsx PUBLIC m)
endif()

target_include_directories(
    gsx
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/stb
)

if(GSX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(gsx PRIVATE CUDA::cudart)
    set_target_properties(gsx PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(gsx PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endif()
