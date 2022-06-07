file(GLOB SOURCES "*.cpp")

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(vlp4d_mpi ${SOURCES})

target_compile_features(vlp4d_mpi PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    find_package(OpenACC REQUIRED)
    target_link_libraries(vlp4d_mpi PUBLIC OpenACC::OpenACC_CXX MPI::MPI_CXX math_lib)
    target_compile_options(vlp4d_mpi PUBLIC -O3 -Minfo=accel)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(vlp4d_mpi PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX math_lib)
    target_compile_options(vlp4d_mpi PUBLIC -O3)
endif()

target_compile_definitions(vlp4d_mpi PUBLIC LONG_ENOUGH_BUFFER)
