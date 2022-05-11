file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(mdspan CONFIG REQUIRED)

add_executable(vlp4d ${SOURCES})

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

# Compiler version
if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    target_compile_features(vlp4d PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(vlp4d PUBLIC --expt-relaxed-constexpr --extended-lambda)
elseif(BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    find_package(rocthrust REQUIRED CONFIG PATHS)
    target_compile_features(vlp4d PUBLIC cxx_std_17)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_features(vlp4d PUBLIC cxx_std_17)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

# Link mdspan
target_link_libraries(vlp4d PUBLIC std::mdspan math_lib)

# Compiler options
target_compile_options(vlp4d PUBLIC -O3)
