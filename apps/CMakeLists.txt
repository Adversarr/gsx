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
    
endif()
