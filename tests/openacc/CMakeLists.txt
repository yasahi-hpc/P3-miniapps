file(GLOB SOURCES "*.cpp")

add_executable(google_tests ${SOURCES})

target_compile_features(google_tests PUBLIC cxx_std_17)
target_link_libraries(google_tests PUBLIC math_lib GTest::gtest_main)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(google_tests)
