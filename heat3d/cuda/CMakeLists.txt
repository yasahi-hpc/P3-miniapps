include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "No CUDA compiler found")
endif()

file(GLOB SOURCES "*.cpp")

# Find mdspan
find_package(mdspan CONFIG REQUIRED)
find_package(layout_contiguous 0.1 CONFIG REQUIRED)

add_executable(heat3D ${SOURCES})

target_compile_features(heat3D PUBLIC cxx_std_17 cuda_std_17)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
target_link_libraries(heat3D PUBLIC std::mdspan layout_contiguous::layout_contiguous)

option(ENABLE_MDSPAN "Enable MDSPAN" OFF)
option(ENABLE_LOOP_3D "Enable 3D Loop" OFF)

target_compile_definitions(heat3D PUBLIC ENABLE_CUDA
    $<$<BOOL:${ENABLE_MDSPAN}>: ENABLE_MDSPAN>
    $<$<BOOL:${ENABLE_LOOP_3D}>: ENABLE_LOOP_3D>
)

target_compile_options(heat3D PUBLIC -O3 --expt-relaxed-constexpr --extended-lambda)
