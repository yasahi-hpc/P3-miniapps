file(GLOB SOURCES "*.cpp")

add_executable(vlp4d ${SOURCES})

target_compile_features(vlp4d PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_compile_options(vlp4d PUBLIC -O3 -stdpar=gpu)
    target_link_options(vlp4d PUBLIC -stdpar=gpu -cudalib=cufft)
else()
    target_compile_options(vlp4d PUBLIC -O3 -stdpar=multicore -mp)
endif()

target_link_libraries(vlp4d PUBLIC math_lib)
