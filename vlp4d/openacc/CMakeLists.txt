file(GLOB SOURCES "*.cpp")

add_executable(vlp4d ${SOURCES})

# Compiler version
target_compile_features(vlp4d PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_libraries(vlp4d PUBLIC math_lib)
    target_compile_options(vlp4d PUBLIC -O3 -Minfo=accel)
else()
    target_link_libraries(vlp4d PUBLIC math_lib)
    target_compile_options(vlp4d PUBLIC -O3)
endif()
