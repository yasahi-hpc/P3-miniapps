file(GLOB SOURCES "*.cpp")

add_executable(heat3d ${SOURCES})

# Compiler version
target_compile_features(heat3d PUBLIC cxx_std_17)

# Link libraries
target_link_libraries(heat3d PUBLIC math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3d PUBLIC -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast)
    target_compile_definitions(heat3d PUBLIC ENABLE_OPENMP_OFFLOAD)
elseif(BACKEND STREQUAL "HIP")
    target_compile_options(heat3d PUBLIC -O3)
    target_link_libraries(heat3d PUBLIC math_lib)
    target_compile_definitions(heat3d PUBLIC ENABLE_OPENMP_OFFLOAD)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3d PUBLIC -O3)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP and OPENMP must be On.")
endif()
