cmake_minimum_required(VERSION 3.18)
project(P3miniapps LANGUAGES CXX)

# Add cmake helpers
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

add_subdirectory(lib)

set(APPLICATION AUTO CACHE STRING "CHOICE OF APPLICATION")
set(PROGRAMMING_MODEL AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

include(CTest)
if(BUILD_TESTING AND (NOT (PROGRAMMING_MODEL STREQUAL "KOKKOS")))
    add_subdirectory(tests)
endif()

# Mini applications
add_subdirectory(miniapps)
