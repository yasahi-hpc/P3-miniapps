if(PROGRAMMING_MODEL STREQUAL "CUDA")
    add_subdirectory(thrust)
elseif(PROGRAMMING_MODEL STREQUAL "HIP")
    add_subdirectory(thrust)
elseif(PROGRAMMING_MODEL STREQUAL "OPENMP")
    add_subdirectory(openmp)
elseif(PROGRAMMING_MODEL STREQUAL "OPENACC")
    add_subdirectory(openacc)
elseif(PROGRAMMING_MODEL STREQUAL "STDPAR")
    add_subdirectory(stdpar)
elseif(PROGRAMMING_MODEL STREQUAL "THRUST")
    add_subdirectory(thrust)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, OPENMP, OPENACC, STDPAR, and THRUST must be On.")
endif()
