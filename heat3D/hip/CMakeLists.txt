file(GLOB SOURCES "*.cpp")

# Find libraries
find_package(HIP REQUIRED)
find_package(rocthrust REQUIRED CONFIG PATHS)
find_package(mdspan CONFIG REQUIRED)
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(heat3D ${SOURCES})

target_compile_features(heat3D PUBLIC cxx_std_17)
target_link_libraries(heat3D PUBLIC std::mdspan layout_contiguous::layout_contiguous hip::device roc::rocthrust)

option(ENABLE_MDSPAN "Enable MDSPAN" OFF)
option(ENABLE_LOOP_3D "Enable 3D Loop" OFF)

target_compile_definitions(heat3D PUBLIC ENABLE_HIP
    $<$<BOOL:${ENABLE_MDSPAN}>: ENABLE_MDSPAN>
    $<$<BOOL:${ENABLE_LOOP_3D}>: ENABLE_LOOP_3D>
)

target_compile_options(heat3D PUBLIC -O3)
