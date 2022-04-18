add_library(math_lib INTERFACE)

# Find libraries
find_package(mdspan CONFIG REQUIRED)
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED COMPONENTS cufft)
    target_link_libraries(math_lib INTERFACE std::mdspan layout_contiguous::layout_contiguous CUDA::cufft)
    target_compile_definitions(math_lib INTERFACE ENABLE_CUDA)
elseif(BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    find_package(rocthrust REQUIRED CONFIG PATHS)
    target_link_libraries(math_lib INTERFACE std::mdspan layout_contiguous::layout_contiguous hip::device roc::rocthrust)
    target_compile_features(math_lib INTERFACE cxx_std_17)
    target_compile_definitions(math_lib INTERFACE ENABLE_HIP)
elseif(BACKEND STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    target_link_libraries(math_lib INTERFACE std::mdspan layout_contiguous::layout_contiguous OpenMP::OpenMP_CXX fftw3)
    target_compile_features(math_lib INTERFACE cxx_std_17)
    target_compile_definitions(math_lib INTERFACE ENABLE_OPENMP)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

if(PROGRAMMING_MODEL STREQUAL "THRUST")
    target_compile_definitions(math_lib INTERFACE ENABLE_THRUST)
endif()

# Compiler options
target_compile_options(math_lib INTERFACE -O3)

target_include_directories(math_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
