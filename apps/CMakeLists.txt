add_executable(gsx-info gsx-info.c)
target_link_libraries(gsx-info PRIVATE gsx::gsx)
gsx_configure_target(gsx-info C)

add_executable(banana-optim banana-optim.c)
target_link_libraries(banana-optim PRIVATE gsx::gsx)
gsx_configure_target(banana-optim C)

if(GSX_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(banana-optim PRIVATE CUDA::cudart)
endif()
