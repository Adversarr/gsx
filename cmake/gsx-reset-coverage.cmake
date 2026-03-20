if(NOT DEFINED GSX_COVERAGE_BUILD_DIR)
    message(FATAL_ERROR "GSX_COVERAGE_BUILD_DIR must be provided.")
endif()

file(
    GLOB_RECURSE
    gsx_coverage_runtime_files
    LIST_DIRECTORIES FALSE
    "${GSX_COVERAGE_BUILD_DIR}/*.gcda"
    "${GSX_COVERAGE_BUILD_DIR}/*.gcov"
)

if(gsx_coverage_runtime_files)
    file(REMOVE ${gsx_coverage_runtime_files})
endif()
