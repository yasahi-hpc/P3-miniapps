file(GLOB SOURCES "*.cpp")

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(vlp4d_mpi ${SOURCES})

target_compile_features(vlp4d_mpi PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_compile_options(vlp4d_mpi PUBLIC -O3 -Minfo=accel)
else()
    target_compile_options(vlp4d_mpi PUBLIC -O3)
endif()

target_link_libraries(vlp4d_mpi PUBLIC MPI::MPI_CXX math_lib)
target_compile_definitions(vlp4d_mpi PUBLIC LONG_ENOUGH_BUFFER)

# Macro
target_compile_definitions(vlp4d_mpi PUBLIC $<$<BOOL:${NO_ERROR_CHECK}>: NO_ERROR_CHECK>)
