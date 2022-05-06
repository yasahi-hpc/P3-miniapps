cmake_minimum_required(VERSION 3.18)
project(P3miniapps LANGUAGES CXX)

# Add cmake helpers
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

add_subdirectory(lib)

set(APPLICATION AUTO CACHE STRING "CHOICE OF APPLICATION")

if(APPLICATION STREQUAL "heat3D")
    add_subdirectory(heat3D)
elseif(APPLICATION STREQUAL "heat3d_mpi")
    add_subdirectory(heat3d_mpi)
elseif(APPLICATION STREQUAL "vlp4d")
    add_subdirectory(vlp4d)
elseif(APPLICATION STREQUAL "vlp4d_mpi")
    add_subdirectory(vlp4d_mpi)
else()
    # Default is heat3D
    add_subdirectory(heat3D)
endif()
