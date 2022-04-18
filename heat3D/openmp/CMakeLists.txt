file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(heat3D ${SOURCES})

target_compile_features(heat3D PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3D PUBLIC -O3 -Minfo=accel -mp=gpu -mcmodel=medium -fast)
    target_compile_definitions(heat3D PUBLIC ENABLE_OPENMP_OFFLOAD)
    target_link_libraries(heat3D PUBLIC layout_contiguous::layout_contiguous math_lib -mp=gpu)
elseif(BACKEND STREQUAL "HIP")
    find_package(rocthrust REQUIRED CONFIG PATHS)
    target_compile_definitions(heat3D PUBLIC ENABLE_OPENMP_OFFLOAD)
    target_link_libraries(heat3D PUBLIC layout_contiguous::layout_contiguous math_lib)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3D PUBLIC -O3)
    target_link_libraries(heat3D PUBLIC layout_contiguous::layout_contiguous math_lib)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

target_compile_definitions(heat3D PUBLIC
    $<$<BOOL:${ENABLE_LOOP_3D}>: ENABLE_LOOP_3D>
)
