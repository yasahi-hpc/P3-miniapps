file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(layout_contiguous 0.1 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(vlp4d ${SOURCES})

# Compiler version
target_compile_features(vlp4d PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(vlp4d PUBLIC -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast)
    target_link_libraries(vlp4d PUBLIC layout_contiguous::layout_contiguous math_lib -mp=gpu)
    target_compile_definitions(vlp4d PUBLIC ENABLE_OPENMP_OFFLOAD)
elseif(BACKEND STREQUAL "HIP")
    target_compile_definitions(vlp4d PUBLIC ENABLE_OPENMP_OFFLOAD)
    target_link_libraries(vlp4d PUBLIC layout_contiguous::layout_contiguous OpenMP::OpenMP_CXX math_lib)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(vlp4d PUBLIC -O3 -mcmodel=medium -fast)
    target_link_libraries(vlp4d PUBLIC layout_contiguous::layout_contiguous OpenMP::OpenMP_CXX math_lib)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()
