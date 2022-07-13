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
    target_compile_options(heat3d_mpi PUBLIC -stdpar=gpu)
    target_link_options(heat3d_mpi PUBLIC -stdpar=gpu)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3d_mpi PUBLIC -stdpar=multicore)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Macro
target_compile_definitions(heat3d_mpi PUBLIC $<$<BOOL:${ACCESS_VIA_RAW_POINTERS}>: ACCESS_VIA_RAW_POINTERS>)
target_compile_definitions(heat3d_mpi PUBLIC $<$<BOOL:${RANGE_POLICY_1D}>: RANGE_POLICY_1D>)
