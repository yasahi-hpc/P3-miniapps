file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(heat3d_mpi ${SOURCES})

# Compiler version
target_compile_features(heat3d_mpi PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d_mpi PUBLIC Kokkos::kokkos MPI::MPI_CXX math_lib)
target_compile_definitions(heat3d_mpi PUBLIC SIMD)
