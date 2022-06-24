file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(vlp4d_mpi ${SOURCES})

# Compiler version
target_compile_features(vlp4d_mpi PUBLIC cxx_std_17)

# Link Kokkos and math lib
set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_compile_definitions(vlp4d_mpi PUBLIC SIMD)
elseif(BACKEND STREQUAL "HIP")
    target_compile_definitions(vlp4d_mpi PUBLIC NO_SCATTER_VIEWS LOOP_1D)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(vlp4d_mpi PUBLIC NO_SCATTER_VIEWS LAYOUT_LEFT SIMD)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Compiler options
target_compile_options(vlp4d_mpi PUBLIC -O3)

# Libraries
target_link_libraries(vlp4d_mpi PUBLIC MPI::MPI_CXX math_lib)

# Macro
target_compile_definitions(vlp4d_mpi PUBLIC $<$<BOOL:${NO_ERROR_CHECK}>: NO_ERROR_CHECK>)
