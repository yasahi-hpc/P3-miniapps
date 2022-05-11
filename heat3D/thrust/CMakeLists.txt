file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(mdspan CONFIG REQUIRED)

add_executable(heat3D ${SOURCES})

# Compiler version
target_compile_features(heat3D PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3D PUBLIC std::mdspan math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    target_compile_features(heat3D PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(heat3D PUBLIC --expt-relaxed-constexpr --extended-lambda)
elseif(BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    find_package(rocthrust REQUIRED CONFIG PATHS)
    target_compile_features(heat3D PUBLIC cxx_std_17)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_features(heat3D PUBLIC cxx_std_17)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Compiler options
target_compile_definitions(heat3D PUBLIC
    $<$<BOOL:${ENABLE_MDSPAN}>: ENABLE_MDSPAN>
)
