file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(heat3d_mpi ${SOURCES})

# Compiler version
target_compile_features(heat3d_mpi PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d_mpi PUBLIC MPI::MPI_CXX math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3d_mpi PUBLIC -O3 -Minfo=accel)
    target_link_libraries(heat3d_mpi PUBLIC MPI::MPI_CXX math_lib)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3d_mpi PUBLIC -O3)
    target_link_libraries(heat3d_mpi PUBLIC MPI::MPI_CXX math_lib)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()
