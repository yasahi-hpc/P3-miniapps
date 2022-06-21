file(GLOB SOURCES "*.cpp")

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(heat3d_mpi ${SOURCES})

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    target_compile_features(heat3d_mpi PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(heat3d_mpi PUBLIC --expt-relaxed-constexpr --extended-lambda)
elseif(BACKEND STREQUAL "HIP")
    target_compile_features(heat3d_mpi PUBLIC cxx_std_17)
elseif(BACKEND STREQUAL "OPENMP")
    enable_language(CUDA)
    target_compile_features(heat3d_mpi PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

    target_compile_options(heat3d_mpi PUBLIC --expt-relaxed-constexpr --extended-lambda ${OpenMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP and OPENMP must be On.")
endif()

target_link_libraries(heat3d_mpi PUBLIC MPI::MPI_CXX math_lib)

# Macro
target_compile_definitions(heat3d_mpi PUBLIC $<$<BOOL:${ACCESS_VIA_RAW_POINTERS}>: ACCESS_VIA_RAW_POINTERS>)
