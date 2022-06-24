file(GLOB SOURCES "*.cpp")

add_executable(google_tests ${SOURCES})

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

# Compiler version
if(BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    target_compile_features(google_tests PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(google_tests PUBLIC --expt-relaxed-constexpr --extended-lambda)
elseif(BACKEND STREQUAL "HIP")
    target_compile_features(google_tests PUBLIC cxx_std_17)
elseif(BACKEND STREQUAL "OPENMP")
    enable_language(CUDA)
    target_compile_features(google_tests PUBLIC cuda_std_17)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_options(google_tests PUBLIC --expt-relaxed-constexpr --extended-lambda ${OpenMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, and OPENMP must be On.")
endif()

target_link_libraries(google_tests PUBLIC math_lib GTest::gtest_main)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(google_tests)
