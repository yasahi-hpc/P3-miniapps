add_library(math_lib INTERFACE)

# Find libraries
find_package(Kokkos REQUIRED)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED COMPONENTS cufft cublas)
    target_compile_definitions(math_lib INTERFACE ENABLE_CUDA)
    target_link_libraries(math_lib INTERFACE Kokkos::kokkos CUDA::cufft CUDA::cublas)
elseif(BACKEND STREQUAL "HIP")
    find_package(rocfft REQUIRED CONFIG PATHS)
    find_package(rocblas REQUIRED CONFIG PATHS)
    target_compile_definitions(math_lib INTERFACE ENABLE_HIP)
    target_link_libraries(math_lib INTERFACE Kokkos::kokkos roc::rocfft roc::rocblas)
elseif(BACKEND STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    target_link_libraries(math_lib INTERFACE Kokkos::kokkos OpenMP::OpenMP_CXX fftw3)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Compiler versions
target_compile_features(math_lib INTERFACE cxx_std_17)

# Enable macro
target_compile_definitions(math_lib INTERFACE ENABLE_KOKKOS)

# Linking libraries
#target_link_libraries(math_lib INTERFACE Kokkos::kokkos
#    $<$<BOOL:${ENABLE_OPENMP}>: fftw3>
#    $<$<BOOL:${ENABLE_HIP}>: roc::rocfft>
#    $<$<BOOL:${ENABLE_CUDA}>: CUDA::cufft>
#    )

target_include_directories(math_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
