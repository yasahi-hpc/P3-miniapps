file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(GTest REQUIRED CONFIG)

add_executable(lib_tests ${SOURCES})

target_compile_features(lib_tests PUBLIC cxx_std_17)
target_link_libraries(lib_tests PUBLIC math_lib GTest::gtest_main)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(lib_tests)
