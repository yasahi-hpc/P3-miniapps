file(GLOB SOURCES "*.cpp")

find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(heat3D ${SOURCES})

target_compile_features(heat3D PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    find_package(OpenACC REQUIRED)
    target_link_libraries(heat3D PUBLIC layout_contiguous::layout_contiguous OpenACC::OpenACC_CXX)
    target_compile_definitions(heat3D PUBLIC ENABLE_OPENACC)
    target_compile_options(heat3D PUBLIC -O3 -Minfo=accel)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(heat3D PUBLIC layout_contiguous::layout_contiguous OpenMP::OpenMP_CXX)
    target_compile_options(heat3D PUBLIC -O3)
endif()

# Macro
target_compile_definitions(heat3D PUBLIC
    $<$<BOOL:${ENABLE_LOOP_3D}>: ENABLE_LOOP_3D>
)
