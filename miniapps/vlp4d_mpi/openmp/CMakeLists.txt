file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(vlp4d_mpi ${SOURCES})

target_compile_features(vlp4d_mpi PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(vlp4d_mpi PUBLIC -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast)
    target_compile_definitions(vlp4d_mpi PUBLIC ENABLE_OPENMP_OFFLOAD LONG_ENOUGH_BUFFER)
elseif(BACKEND STREQUAL "HIP")
    target_compile_definitions(vlp4d_mpi PUBLIC ENABLE_OPENMP_OFFLOAD LONG_ENOUGH_BUFFER)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(vlp4d_mpi PUBLIC -O3)
    target_compile_definitions(vlp4d_mpi PUBLIC LAYOUT_LEFT SIMD LONG_ENOUGH_BUFFER)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

target_link_libraries(vlp4d_mpi PUBLIC MPI::MPI_CXX math_lib)

# Macro
target_compile_definitions(vlp4d_mpi PUBLIC $<$<BOOL:${NO_ERROR_CHECK}>: NO_ERROR_CHECK>)
