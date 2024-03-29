file(GLOB SOURCES "*.cpp")

# Find libraries
add_executable(heat3d ${SOURCES})

# Compiler version
target_compile_features(heat3d PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d PUBLIC math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    target_compile_features(heat3d PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(heat3d PUBLIC --expt-relaxed-constexpr --extended-lambda)
elseif(BACKEND STREQUAL "HIP")
    target_compile_features(heat3d PUBLIC cxx_std_17)
elseif(BACKEND STREQUAL "OPENMP")
    enable_language(CUDA)
    target_compile_features(heat3d PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(heat3d PUBLIC --expt-relaxed-constexpr --extended-lambda ${OpenMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Macro
target_compile_definitions(heat3d PUBLIC $<$<BOOL:${ACCESS_VIA_RAW_POINTERS}>: ACCESS_VIA_RAW_POINTERS>)
