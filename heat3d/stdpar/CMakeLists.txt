file(GLOB SOURCES "*.cpp")

add_executable(heat3d ${SOURCES})

# Compiler version
target_compile_features(heat3d PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d PUBLIC math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3d PUBLIC -stdpar=gpu)
    target_link_options(heat3d PUBLIC -stdpar=gpu)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3d PUBLIC -stdpar=multicore -mp)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Macro
target_compile_definitions(heat3d PUBLIC $<$<BOOL:${ACCESS_VIA_RAW_POINTERS}>: ACCESS_VIA_RAW_POINTERS>)
