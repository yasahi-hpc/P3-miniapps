FROM rocm/tensorflow
WORKDIR /work
ENV CMAKE_PREFIX_PATH "$CMAKE_PREFIX_PATH;/opt/rocm/hip/;/opt/rocm/" 
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/rocm/rocfft/lib:/opt/rocm/rocthrust/lib:/opt/rocm/rocblas/lib:/opt/rocm/llvm/lib
CMD ["bash"]
