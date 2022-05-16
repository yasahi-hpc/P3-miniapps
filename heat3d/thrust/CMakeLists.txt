file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(mdspan CONFIG REQUIRED)

add_executable(heat3d ${SOURCES})

# Compiler version
target_compile_features(heat3d PUBLIC cxx_std_17)

# Link mdspan
target_link_libraries(heat3d PUBLIC std::mdspan math_lib)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    target_compile_features(heat3d PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(heat3d PUBLIC --expt-relaxed-constexpr --extended-lambda)
elseif(BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    find_package(rocthrust REQUIRED CONFIG PATHS)
    target_compile_features(heat3d PUBLIC cxx_std_17)
elseif(BACKEND STREQUAL "OPENMP")
    enable_language(CUDA)
    target_compile_features(heat3d PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(heat3d PUBLIC --expt-relaxed-constexpr --extended-lambda -fopenmp)
    #target_compile_features(heat3d PUBLIC cxx_std_17)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()
