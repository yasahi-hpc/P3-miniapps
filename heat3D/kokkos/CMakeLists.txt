file(GLOB SOURCES "*.cpp")

find_package(Kokkos REQUIRED)

add_executable(heat3D ${SOURCES})

target_compile_features(heat3D PUBLIC cxx_std_17)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

target_link_libraries(heat3D PUBLIC Kokkos::kokkos)

target_compile_options(heat3D PUBLIC -O3)
