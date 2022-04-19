add_library(math_lib INTERFACE)

# Find libraries
find_package(layout_contiguous 0.1 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit COMPONENTS cufft cublas)
    if(CUDAToolkit_FOUND)
        target_link_libraries(math_lib INTERFACE layout_contiguous::layout_contiguous -mp=gpu CUDA::cufft CUDA::cublas)
        target_compile_options(math_lib INTERFACE -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast)
    else()
        target_link_libraries(math_lib INTERFACE layout_contiguous::layout_contiguous -mp=gpu -cudalib=cufft,cublas)
        target_compile_options(math_lib INTERFACE -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast -cudalib=cufft,cublas)
    endif()
    target_compile_definitions(math_lib INTERFACE ENABLE_CUDA)
elseif(BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    find_package(rocthrust REQUIRED CONFIG PATHS)
    target_link_libraries(math_lib INTERFACE std::mdspan layout_contiguous::layout_contiguous hip::device roc::rocthrust OpenMP::OpenMP_CXX)
    target_compile_definitions(math_lib INTERFACE ENABLE_HIP)
elseif(BACKEND STREQUAL "OPENMP")
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    target_compile_options(math_lib INTERFACE -O3)
    target_link_libraries(math_lib INTERFACE std::mdspan layout_contiguous::layout_contiguous OpenMP::OpenMP_CXX fftw3)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Compiler options
target_compile_options(math_lib INTERFACE -O3)

# Compiler versions
target_compile_features(math_lib INTERFACE cxx_std_17)

target_include_directories(math_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
