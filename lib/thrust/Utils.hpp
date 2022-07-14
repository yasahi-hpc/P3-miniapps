#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cstdio>

// Class helper
#define DISALLOW_COPY_AND_ASSIGN(ClassName) \
    ClassName(const ClassName&); \
    void operator=(const ClassName&);

#if defined(__CUDACC__) && defined( ENABLE_CUDA )
  inline void synchronize() {
    cudaDeviceSynchronize();
  }

  inline void setDevice(int rank) {
    int count;
    int id;
    
    cudaGetDeviceCount(&count);
    cudaSetDevice(rank % count);
    cudaGetDevice(&id);
    printf("Process%d running on GPU%d\n", rank, id);
  }
#elif defined(__HIPCC__) && defined( ENABLE_HIP )
  #include <hip/hip_runtime.h>
  inline void synchronize() {
    [[maybe_unused]] hipError_t err = hipDeviceSynchronize();
  }
  
  inline void setDevice(int rank) {
    int count;
    int id;
    hipError_t err;
    
    err = hipGetDeviceCount(&count);
    err = hipSetDevice(rank % count);
    err = hipGetDevice(&id);
    printf("Process%d running on GPU%d\n", rank, id);
  }
  
#else
  inline void synchronize() {}
  inline void setDevice(int rank) {}
#endif

#endif
