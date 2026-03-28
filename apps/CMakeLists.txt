add_executable(gsx-info gsx-info.c)
target_link_libraries(gsx-info PRIVATE gsx::gsx)
gsx_configure_target(gsx-info C)

add_executable(banana-optim banana-optim.c)
target_link_libraries(banana-optim PRIVATE gsx::gsx)
gsx_configure_target(banana-optim C)

add_executable(ssim-numerical-diff ssim-numerical-diff.c)
target_link_libraries(ssim-numerical-diff PRIVATE gsx::gsx)
gsx_configure_target(ssim-numerical-diff C)

add_executable(image-fit-demo image_fit_demo.c)
target_link_libraries(image-fit-demo PRIVATE gsx::gsx)
gsx_configure_target(image-fit-demo C)

add_executable(multi-gaussian-render multi-gaussian-render.c)
target_link_libraries(multi-gaussian-render PRIVATE gsx::gsx)
gsx_configure_target(multi-gaussian-render C)

add_executable(render-pcd render_pcd.c)
target_link_libraries(render-pcd PRIVATE gsx::gsx)
gsx_configure_target(render-pcd C)

if(GSX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(banana-optim PRIVATE CUDA::cudart)
endif()

if(GSX_BUILD_APP_EXTRAS)
    include(../cmake/CPM.cmake)
    CPMAddPackage("gh:nlohmann/json@3.11.0")
    CPMAddPackage("gh:jarro2783/cxxopts@3.3.1")
    CPMAddPackage("gh:ocornut/imgui@1.92.6")

    add_executable(bench_dataset bench_dataset.cpp)
    target_link_libraries(bench_dataset PRIVATE gsx::gsx nlohmann_json::nlohmann_json)
    target_compile_definitions(bench_dataset PRIVATE GSX_HAS_CUDA=$<BOOL:${GSX_USE_CUDA}>)
    gsx_configure_target(bench_dataset CXX)

    if(GSX_USE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(bench_dataset PRIVATE CUDA::cudart)
    endif()

    add_executable(train-preprocessed train_preprocessed.cpp)
    target_link_libraries(train-preprocessed PRIVATE gsx::gsx nlohmann_json::nlohmann_json cxxopts::cxxopts)
    gsx_configure_target(train-preprocessed CXX)

    # With GUI
    find_package(SDL3 QUIET)
    if(SDL3_FOUND)
        add_executable(ply-viewer ply_viewer.cpp)
        target_sources(ply-viewer PRIVATE
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp)
        target_include_directories(ply-viewer PRIVATE
            ${imgui_SOURCE_DIR}
            ${imgui_SOURCE_DIR}/backends)
        target_link_libraries(ply-viewer PRIVATE gsx::gsx SDL3::SDL3 nlohmann_json::nlohmann_json cxxopts::cxxopts)
        gsx_configure_target(ply-viewer CXX)
    else()
        message(STATUS "SDL not found, skipping.")
    endif()
endif()
