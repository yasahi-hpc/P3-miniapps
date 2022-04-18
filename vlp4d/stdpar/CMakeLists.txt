file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(mdspan CONFIG REQUIRED)
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(vlp4d ${SOURCES})

target_compile_features(vlp4d PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_libraries(vlp4d PUBLIC std::mdspan layout_contiguous::layout_contiguous math_lib)
    target_compile_options(vlp4d PUBLIC -O3 -stdpar=gpu)
    target_link_options(vlp4d PUBLIC -stdpar=gpu -cudalib=cufft)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(vlp4d PUBLIC std::mdspan layout_contiguous::layout_contiguous math_lib)
    target_compile_options(vlp4d PUBLIC -O3 -stdpar=multicore -mp)
endif()
