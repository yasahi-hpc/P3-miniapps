file(GLOB SOURCES "*.cpp")

add_executable(vlp4d ${SOURCES})

# Compiler version
target_compile_features(vlp4d PUBLIC cxx_std_17)

# Link libraries
target_link_libraries(vlp4d PUBLIC math_lib)

# Compiler options
target_compile_options(vlp4d PUBLIC -O3)

# Macro
target_compile_definitions(vlp4d PUBLIC $<$<BOOL:${MDRange3D}>: MDRange3D>)
