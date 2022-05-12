file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(Kokkos REQUIRED)

add_executable(heat3d ${SOURCES})

# Compiler version
target_compile_features(heat3d PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d PUBLIC Kokkos::kokkos math_lib)
target_compile_definitions(heat3d PUBLIC SIMD)
