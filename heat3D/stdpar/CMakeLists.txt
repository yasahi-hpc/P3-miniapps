file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(mdspan CONFIG REQUIRED)
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(heat3D ${SOURCES})

# Compiler version
target_compile_features(heat3D PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3D PUBLIC std::mdspan layout_contiguous::layout_contiguous math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3D PUBLIC -stdpar=gpu)
    target_link_options(heat3D PUBLIC -stdpar=gpu)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3D PUBLIC -stdpar=multicore -mp)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Compiler options
target_compile_definitions(heat3D PUBLIC
    $<$<BOOL:${ENABLE_MDSPAN}>: ENABLE_MDSPAN>
)
