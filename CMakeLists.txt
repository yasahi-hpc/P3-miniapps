cmake_minimum_required(VERSION 3.16)
project(P3miniapps LANGUAGES CXX)

# Add cmake helpers
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

# Find libraries 
find_package(mdspan CONFIG)
if(NOT mdspan_FOUND)
    add_subdirectory(ext_lib/mdspan)
endif()

set(PROGRAMMING_MODEL AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
set(APPLICATION AUTO CACHE STRING "CHOICE OF APPLICATION")

add_subdirectory(lib)

include(CTest)
if(BUILD_TESTING AND (NOT (PROGRAMMING_MODEL STREQUAL "KOKKOS")))
    find_package(GTest CONFIG)
    if(NOT GTest_FOUND)
        add_subdirectory(ext_lib/googletest)
    endif()
    add_subdirectory(tests)
endif()

# Mini applications
add_subdirectory(miniapps)
