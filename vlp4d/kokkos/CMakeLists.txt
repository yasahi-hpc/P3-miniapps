file(GLOB SOURCES "*.cpp")

# Compile option

# Find libraries
find_package(Kokkos REQUIRED)

add_executable(vlp4d ${SOURCES})

# Compiler version
target_compile_features(vlp4d PUBLIC cxx_std_17)

# Link Kokkos and math lib
set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_libraries(vlp4d PUBLIC Kokkos::kokkos math_lib)
elseif(BACKEND STREQUAL "HIP")
    target_link_libraries(vlp4d PUBLIC Kokkos::kokkos math_lib)
elseif(BACKEND STREQUAL "OPENMP")
    target_link_libraries(vlp4d PUBLIC Kokkos::kokkos math_lib)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Compiler options
target_compile_options(vlp4d PUBLIC -O3)

# Macro
target_compile_definitions(vlp4d PUBLIC $<$<BOOL:${MDRange3D}>: MDRange3D>)
