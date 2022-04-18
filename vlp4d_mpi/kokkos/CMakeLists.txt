file(GLOB SOURCES "*.cpp")

# Compile option

# Find libraries
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(vlp4d_mpi ${SOURCES})

# Compiler version
target_compile_features(vlp4d_mpi PUBLIC cxx_std_17)

# Link Kokkos and math lib
set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_libraries(vlp4d_mpi PUBLIC Kokkos::kokkos MPI::MPI_CXX math_lib)
    target_compile_definitions(vlp4d_mpi PUBLIC SIMD CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 CUDA_LAUNCH_BLOCKING=1)
elseif(BACKEND STREQUAL "HIP")
    target_link_libraries(vlp4d_mpi PUBLIC Kokkos::kokkos MPI::MPI_CXX math_lib)
    target_compile_definitions(vlp4d_mpi PUBLIC SIMD)
elseif(BACKEND STREQUAL "OPENMP")
    target_link_libraries(vlp4d_mpi PUBLIC Kokkos::kokkos MPI::MPI_CXX OpenMP::OpenMP_CXX math_lib)
    target_compile_definitions(vlp4d_mpi PUBLIC NO_SCATTER_VIEWS LAYOUT_LEFT SIMD)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Compiler options
target_compile_options(vlp4d_mpi PUBLIC -O3)
