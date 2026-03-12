add_executable(gsx-info gsx-info.c)
target_link_libraries(gsx-info PRIVATE gsx::gsx)
gsx_configure_target(gsx-info C)
