file(GLOB SOURCES "*.cpp")

add_executable(heat3d ${SOURCES})

# Compiler version
target_compile_features(heat3d PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d PUBLIC math_lib)
target_compile_definitions(heat3d PUBLIC SIMD)
