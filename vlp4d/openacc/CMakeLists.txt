file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(vlp4d ${SOURCES})

# Compiler version
target_compile_features(vlp4d PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    find_package(OpenACC REQUIRED)
    target_link_libraries(vlp4d PUBLIC layout_contiguous::layout_contiguous OpenACC::OpenACC_CXX math_lib)
    target_compile_options(vlp4d PUBLIC -O3 -Minfo=accel)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(vlp4d PUBLIC layout_contiguous::layout_contiguous OpenMP::OpenMP_CXX math_lib)
    target_compile_options(vlp4d PUBLIC -O3)
endif()
