file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(heat3d_mpi ${SOURCES})

# Compiler version
target_compile_features(heat3d_mpi PUBLIC cxx_std_17)

# Link libraries
target_link_libraries(heat3d_mpi PUBLIC MPI::MPI_CXX math_lib)

# Macro
target_compile_definitions(heat3d_mpi PUBLIC SIMD)
