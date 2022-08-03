ARG ROCM_VERSION=5.2.0
ARG TENSORFLOW_VERSION=2.9
FROM rocm/tensorflow:rocm${ROCM_VERSION}-tf${TENSORFLOW_VERSION}-dev

WORKDIR /work
ENV CMAKE_PREFIX_PATH "$CMAKE_PREFIX_PATH;/opt/rocm/hip/;/opt/rocm/" 
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/rocm/rocfft/lib:/opt/rocm/rocthrust/lib:/opt/rocm/rocblas/lib:/opt/rocm/llvm/lib
CMD ["bash"]
