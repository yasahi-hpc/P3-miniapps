add_library(math_lib INTERFACE)

# Find libraries
find_package(OpenMP REQUIRED)

if(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit COMPONENTS cufft cublas)
    if(CUDAToolkit_FOUND)
        target_link_libraries(math_lib INTERFACE -mp=gpu std::mdspan CUDA::cufft CUDA::cublas)
        target_compile_options(math_lib INTERFACE -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast)
    else()
        target_link_libraries(math_lib INTERFACE -mp=gpu std::mdspan -cudalib=cufft,cublas)
        target_compile_options(math_lib INTERFACE -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast -cudalib=cufft,cublas)
    endif()
    target_compile_definitions(math_lib INTERFACE ENABLE_CUDA ENABLE_OPENMP_OFFLOAD)
elseif(BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    find_package(rocblas REQUIRED CONFIG PATHS)
    find_package(rocfft REQUIRED CONFIG PATHS)
    #target_link_libraries(math_lib INTERFACE std::mdspan roc::rocfft roc::rocblas hip::amdhip64)
    # For some reason, this does not work
    target_link_libraries(math_lib INTERFACE std::mdspan roc::rocfft hip::amdhip64 -I${rocblas_INCLUDE_DIR} -L${rocblas_INCLUDE_DIR}/../lib -lrocblas)
    target_compile_options(math_lib INTERFACE -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${CMAKE_HIP_ARCHITECTURES})
    target_link_options(math_lib INTERFACE -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${CMAKE_HIP_ARCHITECTURES})
    target_compile_definitions(math_lib INTERFACE ENABLE_HIP ENABLE_OPENMP_OFFLOAD -D__HIP_PLATFORM_AMD__)
elseif(BACKEND STREQUAL "OPENMP")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
        target_compile_options(math_lib INTERFACE -ipo -xHOST)
    endif()
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    target_compile_options(math_lib INTERFACE -O3)
    target_link_libraries(math_lib INTERFACE std::mdspan OpenMP::OpenMP_CXX fftw3)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Compiler options
target_compile_options(math_lib INTERFACE -O3)

# Compiler versions
target_compile_features(math_lib INTERFACE cxx_std_17)

target_include_directories(math_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
